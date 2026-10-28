#!/usr/bin/env python3
"""Baseline TinyLlama DDP workload with no application-level ITT events."""

from llama3_demo_common import run


if __name__ == "__main__":
    run(description=__doc__)
