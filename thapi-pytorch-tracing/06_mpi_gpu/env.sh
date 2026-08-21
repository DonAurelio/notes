# Canonical environment for PyTorch RecordFunction tracing on Aurora.
# Source this at the top of every shell / remote command — the module environment
# does NOT propagate across shells or to compute nodes, so re-source it each time.
#
#   source env.sh            # frameworks only (build the tracer, run workloads)
#   source env.sh lttng      # + LTTng tooling (record/view traces)
#
# LOAD ORDER MATTERS: the lttng modules pull in oneapi, whose SYCL runtime and
# spack python would otherwise shadow the ones torch was built against (import
# torch then fails on an undefined sycl::queue symbol). Loading frameworks LAST
# puts its python + libsycl first on PATH/LD_LIBRARY_PATH, so both win.

# Make `module` available in non-interactive shells.
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/*lmod* 2>/dev/null || source /etc/profile.d/*modules* 2>/dev/null || true
fi

if [ "${1:-}" = "lttng" ]; then
  # Everything comes from one oneapi release (2025.3.1) so lttng-tools AND
  # babeltrace2 (and lttng-gen-tp via lttng-ust) resolve from the same tree.
  module load oneapi/release/2025.3.1 2>/dev/null
  module load lttng-tools/2.14.0 2>/dev/null
  module load babeltrace2/2.1.2-archive 2>/dev/null
fi

module load frameworks/2025.3.1 2>/dev/null   # LAST — its python + libsycl must win

# Expose torch's include/lib for building the tracer .so.
if command -v python >/dev/null 2>&1; then
  export TORCH_DIR="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)"
  export TORCH_INCLUDE="${TORCH_DIR}/include"
  export TORCH_LIB="${TORCH_DIR}/lib"
fi
