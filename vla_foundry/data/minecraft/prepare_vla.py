"""Prepare instruction-labelled Minecraft clips with raw floating-point actions.

Parallelism is by recording, so each raw log is read once and episode splits
remain disjoint. Source JSONL paths are authoritative; S3 listings are not.
"""

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import tarfile
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import yaml

from vla_foundry.data.minecraft.actions import ACTION_DIM, BUTTONS
from vla_foundry.data.minecraft.prepare import add_bytes, recording_id, recording_split
from vla_foundry.data.minecraft.raw_actions import RAW_CAMERA_SCALE, RAW_SCHEMA, load_raw_actions, raw_path


def select_clips(args):
    limits = dict(zip(("train", "val", "test"), args.max_clips, strict=True))
    counts, per_recording, seen = Counter(), Counter(), set()
    groups = defaultdict(list)
    for source in args.sources:
        with Path(source).open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                name = Path(row["video"]).name
                recording = recording_id(name)
                split = recording_split(recording, args.seed)
                if (
                    name in seen
                    or (not args.all_clips and counts[split] >= limits[split])
                    or (not args.all_clips and per_recording[recording] >= args.max_per_recording)
                    or not row.get("text_goal", "").strip()
                ):
                    continue
                seen.add(name)
                # The row's chunk_index/start_frame can describe the already split video.
                # The filename identifies the original 64-frame chunk.
                chunk_index = int(Path(name).stem.rsplit("_chunk_", 1)[1])
                groups[recording].append(
                    {
                        "video": name,
                        "text_goal": row["text_goal"],
                        "chunk_index": chunk_index,
                        "source": source,
                        "line": line_number,
                        "split": split,
                    }
                )
                counts[split] += 1
                per_recording[recording] += 1
                if sum(counts.values()) % 1000 == 0:
                    print(f"selected_clips={dict(counts)} recordings={len(groups)}", flush=True)
                if not args.all_clips and all(counts[s] >= limits[s] for s in limits):
                    break
        if not args.all_clips and all(counts[s] >= limits[s] for s in limits):
            break
    if not args.all_clips and any(counts[s] < limits[s] for s in limits):
        raise ValueError(f"Insufficient labelled clips: requested {limits}, found {dict(counts)}")
    return groups, counts


def convert_recording(task):
    root, output, recording, rows, image_size, horizon, stride, shard_samples = task
    root, output = Path(root), Path(output)
    actions, gui, noop, audit = load_raw_actions(raw_path(root, recording))
    split = rows[0]["split"]
    prefix = hashlib.sha256(recording.encode()).hexdigest()[:16]
    archive, manifests, provenance, skips = None, [], [], Counter()
    samples = 0
    try:
        for row in rows:
            path = root / "splits" / row["video"]
            try:
                with av.open(str(path)) as container:
                    fps = float(container.streams.video[0].average_rate)
                    if abs(fps - 20) > 0.1:
                        raise ValueError(f"Expected 20 FPS: {path} has {fps}")
                    frames = [
                        frame.to_image().resize((image_size, image_size)) for frame in container.decode(video=0)
                    ]
            except (av.FFmpegError, OSError, EOFError) as exc:
                # PyAV exceptions contain Cython enum members that are not
                # reliably pickleable. Record corrupt clips inside the worker
                # instead of breaking the entire ProcessPool during unpickle.
                skips["decode_error"] += 1
                provenance.append(
                    {
                        **row,
                        "recording": recording,
                        "samples": 0,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
                continue
            tick_start = row["chunk_index"] * 64
            # Raw videos can end with a final observation that has no logged action.
            # Exclude targets without labels; never pad or invent the last action.
            usable_frames = min(len(frames), max(0, len(actions) - tick_start))
            skips["unlabelled_frames"] += len(frames) - usable_frames
            made = 0
            for anchor in range(stride, usable_frames - horizon + 1, stride):
                first = tick_start + anchor
                if gui[first : first + horizon].any():
                    skips["gui_window"] += 1
                    continue
                if noop[first]:
                    skips["noop_anchor"] += 1
                    continue
                if samples % shard_samples == 0:
                    if archive:
                        archive.close()
                    stem = f"{prefix}-{samples // shard_samples:06d}"
                    archive = tarfile.open(output / split / (stem + ".tar"), "w")  # noqa: SIM115
                    manifests.append({"shard": stem, "num_sequences": 0})
                key = f"{prefix}_{row['chunk_index']:05d}_{anchor:04d}"
                for relative in (-1, 0):
                    image = io.BytesIO()
                    frames[anchor + relative].save(image, "JPEG", quality=90)
                    add_bytes(archive, f"{key}.pov_t{relative}.jpg", image.getvalue())
                lowdim = io.BytesIO()
                np.savez(
                    lowdim,
                    vpt_actions=actions[first : first + horizon],
                    past_mask=np.zeros(horizon, dtype=bool),
                    future_mask=np.ones(horizon, dtype=bool),
                )
                metadata = {
                    "recording_id": recording,
                    "video": str(path),
                    "anchor_frame": anchor,
                    "original_anchor_tick": first,
                    "anchor_relative_idx": 0,
                    "action_offset": 0,
                    "fps": fps,
                    "num_actions": len(actions),
                    "action_source": audit["raw_path"],
                }
                for suffix, data in {
                    "lowdim.npz": lowdim.getvalue(),
                    "metadata.json": json.dumps(metadata).encode(),
                    "language_instructions.json": json.dumps({"original": [row["text_goal"]]}).encode(),
                }.items():
                    add_bytes(archive, f"{key}.{suffix}", data)
                samples += 1
                made += 1
                manifests[-1]["num_sequences"] += 1
            provenance.append({**row, "recording": recording, "samples": made})
    finally:
        if archive:
            archive.close()
    return {
        "recording": recording,
        "split": split,
        "samples": samples,
        "manifest": manifests,
        "clips": len(rows),
        "skipped": dict(skips),
        "audit": audit,
        "provenance": provenance,
    }


def convert_recording_serialized(task):
    """Make all unexpected worker failures safe to transport to the parent."""
    try:
        return convert_recording(task)
    except Exception as exc:
        recording = task[2]
        raise RuntimeError(
            f"Failed converting recording {recording}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        ) from None


def load_resume_state(output, groups):
    """Validate flushed outputs and reconstruct counters for an exact resume."""
    output = Path(output)
    provenance_path = output / "selected_clips.jsonl"
    audit_path = output / "raw_actions_audit.jsonl"
    required = [provenance_path, audit_path, *(output / split / "manifest.jsonl" for split in ("train", "val", "test"))]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Cannot resume: missing output files: {missing}")

    provenance_by_recording = defaultdict(list)
    with provenance_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            recording = row.get("recording")
            if recording not in groups:
                raise ValueError(f"Cannot resume: unknown recording {recording!r} at {provenance_path}:{line_number}")
            provenance_by_recording[recording].append(row)

    completed = set()
    for recording, rows in provenance_by_recording.items():
        expected = len(groups[recording])
        if len(rows) != expected:
            raise ValueError(
                f"Cannot resume: recording {recording} has {len(rows)} provenance rows, expected {expected}"
            )
        completed.add(recording)

    ordered_recordings = sorted(groups)
    expected_prefix = set(ordered_recordings[: len(completed)])
    if completed != expected_prefix:
        raise ValueError("Cannot resume: completed recordings are not a deterministic prefix of the source corpus")

    samples, clips, recordings = Counter(), Counter(), Counter()
    for recording in ordered_recordings[: len(completed)]:
        rows = provenance_by_recording[recording]
        split = groups[recording][0]["split"]
        samples[split] += sum(int(row["samples"]) for row in rows)
        clips[split] += len(rows)
        recordings[split] += 1

    manifest_samples = Counter()
    for split in ("train", "val", "test"):
        with (output / split / "manifest.jsonl").open() as stream:
            for line in stream:
                manifest_samples[split] += int(json.loads(line)["num_sequences"])
    if manifest_samples != samples:
        raise ValueError(
            f"Cannot resume: manifest samples {dict(manifest_samples)} differ from provenance {dict(samples)}"
        )

    audit_lines = sum(1 for line in audit_path.open() if line.strip())
    if audit_lines != len(completed):
        raise ValueError(
            f"Cannot resume: raw-action audits={audit_lines}, completed recordings={len(completed)}"
        )

    # Prior skip counters were process-local in legacy partials. Preserve that
    # limitation explicitly; all newly resumed results are counted exactly.
    skips = Counter({"resume_prior_recordings_without_skip_counters": len(completed)}) if completed else Counter()
    return completed, samples, clips, recordings, skips


def convert_tasks_in_recycled_pools(tasks, workers, tasks_per_worker):
    """Yield results from finite pools, avoiding ProcessPool recycle deadlocks.

    CPython's ``max_tasks_per_child`` can leave an executor with no live workers
    when every spawned worker reaches the limit together. Explicit finite
    batches provide the same native-library memory reclamation and make the
    pool boundary observable and deterministic.
    """
    batch_size = workers * tasks_per_worker
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        print(
            f"worker_pool_batch={start // batch_size + 1} tasks={len(batch)} "
            f"remaining_after_batch={len(tasks) - start - len(batch)}",
            flush=True,
        )
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            yield from pool.map(convert_recording_serialized, batch, chunksize=1)


def prepare(args):
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Use a new output directory: {output}")
    if args.resume and not output.exists():
        raise FileNotFoundError(f"Cannot resume missing output directory: {output}")
    groups, selected = select_clips(args)
    if args.resume:
        completed, samples, clips, recordings, skips = load_resume_state(output, groups)
        print(
            f"resuming_completed_recordings={len(completed)}/{len(groups)} samples={dict(samples)}",
            flush=True,
        )
    else:
        output.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (output / split).mkdir()
        completed = set()
        samples, clips, recordings, skips = Counter(), Counter(), Counter(), Counter()
    tasks = [
        (args.data_root, str(output), recording, rows, args.image_size, args.horizon, args.stride, args.shard_samples)
        for recording, rows in sorted(groups.items())
        if recording not in completed
    ]
    output_mode = "a" if args.resume else "w"
    manifest_files = {
        split: (output / split / "manifest.jsonl").open(output_mode) for split in ("train", "val", "test")
    }
    try:
        with (
            (output / "selected_clips.jsonl").open(output_mode) as provenance,
            (output / "raw_actions_audit.jsonl").open(output_mode) as audits,
            (output / "recording_results.jsonl").open(output_mode) as recording_results,
        ):
            completed_before = len(completed)
            for number, result in enumerate(
                convert_tasks_in_recycled_pools(tasks, args.workers, args.max_tasks_per_child), completed_before + 1
            ):
                split = result["split"]
                for entry in result["manifest"]:
                    manifest_files[split].write(json.dumps(entry) + "\n")
                for entry in result["provenance"]:
                    provenance.write(json.dumps(entry) + "\n")
                audits.write(json.dumps(result["audit"]) + "\n")
                samples[split] += result["samples"]
                clips[split] += result["clips"]
                recordings[split] += 1
                skips.update(result["skipped"])
                recording_results.write(
                    json.dumps({key: value for key, value in result.items() if key not in {"manifest", "provenance"}})
                    + "\n"
                )
                for stream in (*manifest_files.values(), provenance, audits, recording_results):
                    stream.flush()
                if number % 10 == 0:
                    print(f"converted_recordings={number}/{len(groups)} samples={dict(samples)}", flush=True)
    finally:
        for stream in manifest_files.values():
            stream.close()
    for split in ("train", "val", "test"):
        if samples[split] == 0:
            raise ValueError(f"Empty {split} partition")
        # Targets are normalized explicitly by their schema; no statistics are fit on val/test.
        (output / split / "stats.json").write_text(
            json.dumps({"vpt_actions": {"mean": [0.0] * ACTION_DIM, "std": [1.0] * ACTION_DIM}}) + "\n"
        )
        (output / split / "preprocessing_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "camera_names": ["pov"],
                    "image_indices": [-1, 0],
                    "past_lowdim_steps": 0,
                    "future_lowdim_steps": args.horizon - 1,
                }
            )
        )
    summary = {
        "schema": RAW_SCHEMA,
        "buttons": list(BUTTONS),
        "action_dim": ACTION_DIM,
        "camera_scale": RAW_CAMERA_SCALE,
        "horizon": args.horizon,
        "action_offset": 0,
        "image_size": args.image_size,
        "image_indices": [-1, 0],
        "seed": args.seed,
        "clips": dict(clips),
        "samples": dict(samples),
        "recordings": dict(recordings),
        "skipped": dict(skips),
        "selected_clips": dict(selected),
        "split_method": "recording hash: 80% train, 10% val, 10% test",
        "note": (
            "Full deduplicated labelled corpus; actions from raw VPT logs"
            if args.all_clips
            else "Bounded corpus subset; labels from annotated JSONL; actions from raw VPT logs"
        ),
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "episode_splits.json").write_text(
        json.dumps({r: rows[0]["split"] for r, rows in sorted(groups.items())}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-clips", nargs=3, type=int, default=[20000, 2000, 2000])
    parser.add_argument("--max-per-recording", type=int, default=32)
    parser.add_argument(
        "--all-clips",
        action="store_true",
        help="Convert every unique labelled clip, ignoring --max-clips and --max-per-recording.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tasks-per-child", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--shard-samples", type=int, default=512)
    args = parser.parse_args()
    if (
        min(
            *args.max_clips,
            args.max_per_recording,
            args.workers,
            args.max_tasks_per_child,
            args.horizon,
            args.stride,
            args.image_size,
            args.shard_samples,
        )
        <= 0
    ):
        parser.error("Counts and sizes must be positive")
    prepare(args)
