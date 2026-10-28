#!/usr/bin/env python3
"""Compare key_averages breakdowns between narrow and wide Kineto profiler scopes."""

from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"


def parse_breakdown(path: Path) -> dict[str, int]:
    counts = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("count="):
            continue
        # format: "count=  1234  key=some::op"
        key_idx = line.index("key=")
        count = int(line[len("count="):key_idx].strip())
        key = line[key_idx + len("key="):]
        counts[key] = count
    return counts


narrow = parse_breakdown(RESULTS / "narrow_breakdown.txt")
wide   = parse_breakdown(RESULTS / "wide_breakdown.txt")

total_narrow = sum(narrow.values())
total_wide   = sum(wide.values())

print(f"Total n_events  narrow={total_narrow:,}  wide={total_wide:,}  diff={total_wide - total_narrow:+,}")
print()

all_keys = sorted(set(narrow) | set(wide))

# Only in narrow
only_narrow = {k: narrow[k] for k in all_keys if k in narrow and k not in wide}
# Only in wide
only_wide   = {k: wide[k]   for k in all_keys if k in wide   and k not in narrow}
# In both — sort by count difference
in_both = {
    k: (narrow[k], wide[k], wide[k] - narrow[k])
    for k in all_keys if k in narrow and k in wide
}

print(f"=== Only in NARROW ({len(only_narrow)} ops, sum={sum(only_narrow.values()):,}) ===")
for k, v in sorted(only_narrow.items(), key=lambda x: -x[1]):
    print(f"  {v:6d}  {k}")

print()
print(f"=== Only in WIDE ({len(only_wide)} ops, sum={sum(only_wide.values()):,}) ===")
for k, v in sorted(only_wide.items(), key=lambda x: -x[1]):
    print(f"  {v:6d}  {k}")

print()
print(f"=== In both — count changed (wide - narrow) ===")
changed = {k: v for k, v in in_both.items() if v[2] != 0}
for k, (n, w, d) in sorted(changed.items(), key=lambda x: x[1][2]):
    print(f"  {d:+6d}  narrow={n:5d}  wide={w:5d}  {k}")

print()
print(f"=== In both — count unchanged ({len(in_both) - len(changed)} ops) ===")
unchanged = {k: v for k, v in in_both.items() if v[2] == 0}
for k, (n, w, _) in sorted(unchanged.items(), key=lambda x: -x[1][0]):
    print(f"  {'=':>6}  count={n:5d}  {k}")
