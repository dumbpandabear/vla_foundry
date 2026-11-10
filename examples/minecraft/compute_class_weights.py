"""Compute capped square-root inverse-frequency D29 weights from an audit."""

import argparse
import json
from pathlib import Path

from vla_foundry.data.minecraft.actions import D29_NAMES


def compute_weights(audit: dict, cap: float = 8.0) -> list[float]:
    counts_by_name = audit["splits"]["train"]["class_counts"]
    counts = [int(counts_by_name[name]) for name in D29_NAMES]
    maximum = max(counts)
    if maximum <= 0:
        raise ValueError("Training audit contains no D29 targets")
    return [min(cap, (maximum / count) ** 0.5) if count else 0.0 for count in counts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cap", type=float, default=8.0)
    args = parser.parse_args()
    if args.cap <= 0:
        parser.error("--cap must be positive")
    # Multinode launchers pass this value through a shell argument string.
    # Compact JSON keeps the complete list in one whitespace-delimited token.
    print(json.dumps(compute_weights(json.loads(args.audit.read_text()), args.cap), separators=(",", ":")))


if __name__ == "__main__":
    main()
