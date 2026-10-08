#!/usr/bin/env python3
"""Inventory LlamaDemo ITT marker names in one or more Perfetto trace files.

This is a dependency-free preflight check. It searches protobuf string payloads;
it does not decode timing data. Use Perfetto or trace_processor for time analysis.
"""

import argparse
import mmap
import re
from pathlib import Path


FIXED_MARKERS = (
    "Init.MPI",
    "Init.ProcessGroup",
    "Init.Model",
    "Data.Generate",
    "Forward",
    "Backward",
    "Optimizer.Step",
    "Metrics.AllReduce",
)
INDEXED_MARKERS = ("Step", "Layer", "Attn", "MLP")


def indexed_values(buffer: mmap.mmap, prefix: bytes, marker: str) -> list[int]:
    pattern = re.compile(re.escape(prefix) + marker.encode() + rb"\.(\d+)")
    return sorted({int(match.group(1)) for match in pattern.finditer(buffer)})


def format_indices(values: list[int]) -> str:
    if not values:
        return "none"
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}..{values[-1]} ({len(values)})"
    return ",".join(str(value) for value in values)


def inspect(path: Path, domain: str) -> dict[str, object]:
    prefix = f"{domain}:".encode()
    result: dict[str, object] = {"path": path, "size": path.stat().st_size}
    if result["size"] == 0:
        result.update(fixed=[], **{name.lower(): [] for name in INDEXED_MARKERS})
        return result

    with path.open("rb") as trace_file, mmap.mmap(
        trace_file.fileno(), length=0, access=mmap.ACCESS_READ
    ) as buffer:
        result["fixed"] = [
            name
            for name in FIXED_MARKERS
            if buffer.find(prefix + name.encode()) != -1
        ]
        for name in INDEXED_MARKERS:
            result[name.lower()] = indexed_values(buffer, prefix, name)
    return result


def main() -> None:
    demo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="*",
        type=Path,
        default=[
            demo_dir / "traces" / "tiny_baseline.pftrace",
            demo_dir / "traces" / "tiny_with_itt.pftrace",
        ],
        help="trace files (default: compare the two bundled traces)",
    )
    parser.add_argument("--domain", default="LlamaDemo", help="ITT domain prefix")
    args = parser.parse_args()

    missing = [path for path in args.traces if not path.is_file()]
    if missing:
        parser.error("trace not found: " + ", ".join(str(path) for path in missing))

    for path in args.traces:
        result = inspect(path, args.domain)
        print(f"{path}  ({result['size'] / (1024 * 1024):.1f} MiB)")
        fixed = result["fixed"]
        print("  fixed regions: " + (", ".join(fixed) if fixed else "none"))
        for name in INDEXED_MARKERS:
            print(f"  {name.lower():6}: {format_indices(result[name.lower()])}")


if __name__ == "__main__":
    main()
