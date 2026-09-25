#!/usr/bin/env python3
"""Analyze a THAPI `babeltrace_thapi trace` text dump (pytorch backend) into
the metrics this validation reports: per (hostname,vpid,vtid) entry/exit
balance, vtid/rank count, top ops by frequency, and backward-thread detection.

THAPI's own pytorch backend schema (name + overload_name only, no scope/depth)
differs from the pytorch-basic-tracing-analyze skill's RecordFunction-tracer
schema (name/scope/depth/args, `vtid = N`) -- that skill's analyze.py does not
parse this format correctly (it buckets everything under vtid "?"). This
script targets THAPI's actual raw-trace line shape:

  13:56:21.031046742 - aurora-uan-0011 - vpid: 40550, vtid: 40550 - \
    lttng_ust_pytorch:op_entry: {name: aten::empty , overload_name: memory_format }

Usage: analyze_thapi.py <raw_trace.txt> [--json]
"""
import sys
import re
import json
from collections import Counter, defaultdict

RE_LINE = re.compile(
    r'^\S+ - (?P<host>\S+) - vpid: (?P<vpid>\d+), vtid: (?P<vtid>\d+) - '
    r'lttng_ust_pytorch:(?P<kind>op_entry|op_exit): '
    r'\{name: (?P<name>[^,]*?)\s*(?:, overload_name:\s*(?P<overload>[^}]*))?\}'
)
RE_BWD = re.compile(r'Backward\d*$|autograd::engine|AccumulateGrad')


def analyze(path):
    per_key = defaultdict(lambda: {"entry": 0, "exit": 0, "bwd": 0})
    names = Counter()
    bwd_keys = set()
    hosts = set()
    total_entry = total_exit = 0
    unparsed = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = RE_LINE.match(ln)
            if not m:
                if 'op_entry' in ln or 'op_exit' in ln:
                    unparsed += 1
                continue
            host = m.group('host')
            vpid = m.group('vpid')
            vtid = m.group('vtid')
            key = f"{host}/vpid={vpid}/vtid={vtid}"
            hosts.add(host)
            kind = m.group('kind')
            rec = per_key[key]
            if kind == 'op_entry':
                rec['entry'] += 1
                total_entry += 1
                name = m.group('name').strip()
                names[name] += 1
                if RE_BWD.search(name):
                    rec['bwd'] += 1
                    bwd_keys.add(key)
            else:
                rec['exit'] += 1
                total_exit += 1

    return {
        "path": path,
        "hosts": sorted(hosts),
        "host_count": len(hosts),
        "total_events": total_entry + total_exit,
        "total_entry": total_entry,
        "total_exit": total_exit,
        "balanced": total_entry == total_exit,
        "unparsed_event_lines": unparsed,
        "keys": {k: v for k, v in sorted(per_key.items())},
        "key_count": len(per_key),
        "top_ops": names.most_common(15),
        "backward_keys": sorted(bwd_keys),
    }


def print_report(a):
    print(f"=== THAPI trace analysis: {a['path']} ===")
    print(f"hosts: {a['host_count']} ({', '.join(a['hosts'])})")
    print(f"events: {a['total_events']}  ({a['total_entry']} entry / "
          f"{a['total_exit']} exit)  balanced={a['balanced']}"
          + (f"  [unparsed op lines: {a['unparsed_event_lines']}]" if a['unparsed_event_lines'] else ""))
    print(f"distinct (host,vpid,vtid) keys: {a['key_count']}")
    for k, r in a['keys'].items():
        ok = "OK" if r['entry'] == r['exit'] else "!! UNBALANCED"
        print(f"  {k}: entry={r['entry']} exit={r['exit']} bwd={r['bwd']}  [{ok}]")
    if a['backward_keys']:
        print(f"backward runs on: {', '.join(a['backward_keys'])}")
    else:
        print("backward runs on: none detected (no *Backward0/autograd::engine/AccumulateGrad ops)")
    print("top ops:")
    for name, cnt in a['top_ops']:
        print(f"  {cnt:5d}  {name}")


if __name__ == "__main__":
    args = sys.argv[1:]
    as_json = '--json' in args
    args = [a for a in args if a != '--json']
    if not args:
        sys.exit(f"usage: {sys.argv[0]} <raw_trace.txt> [--json]")
    result = analyze(args[0])
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
