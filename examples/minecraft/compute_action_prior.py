"""Estimate a frequency baseline using only training targets, spread by shard."""

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from vla_foundry.data.minecraft.actions import BUTTONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-shard", type=int, default=128)
    args = parser.parse_args()
    pressed, ticks, examples, shards = np.zeros(len(BUTTONS), dtype=np.int64), 0, 0, 0
    for line in args.manifest.read_text().splitlines():
        path = args.manifest.parent / (json.loads(line)["shard"] + ".tar")
        count = 0
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.name.endswith(".lowdim.npz"):
                    continue
                with np.load(io.BytesIO(archive.extractfile(member).read())) as arrays:
                    actions = arrays["vpt_actions"]
                    pressed += (actions[:, :-2] > 0).sum(axis=0)
                    ticks += len(actions)
                count += 1
                if count >= args.per_shard:
                    break
        examples += count
        shards += 1
    if not ticks:
        raise ValueError("No training actions found")
    result = {
        "buttons": list(BUTTONS),
        "press_probability": (pressed / ticks).tolist(),
        "examples": examples,
        "ticks": ticks,
        "shards": shards,
        "manifest": str(args.manifest),
        "method": f"first {args.per_shard} examples per training shard; no validation or test labels",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
