#!/usr/bin/env python3
"""The same TinyLlama DDP workload with named ITT regions enabled."""

try:
    import ittapi
except ModuleNotFoundError as exc:
    raise SystemExit(
        "The ITT entry point requires the 'ittapi' Python package. "
        "Load the platform framework environment or install ittapi, then retry."
    ) from exc

from llama3_demo_common import run


def itt_task(name: str, domain: str):
    """Adapt the Python ITT context manager to the shared workload interface."""

    return ittapi.task(name, domain=domain)


if __name__ == "__main__":
    run(task_factory=itt_task, description=__doc__)
