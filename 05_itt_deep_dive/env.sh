# Canonical environment for PyTorch on Aurora (frameworks module).
# Source this at the top of every shell / remote command -- the module
# environment does NOT propagate across shells or to compute nodes.
#
#   source env.sh
#
# LOAD ORDER MATTERS: oneapi must load before frameworks, and frameworks LAST,
# or `import torch` fails on an undefined sycl::queue symbol (oneapi's SYCL
# runtime and spack python would otherwise shadow the ones torch was built
# against). Loading frameworks last puts its python + libsycl first on
# PATH/LD_LIBRARY_PATH.

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/*lmod* 2>/dev/null || source /etc/profile.d/*modules* 2>/dev/null || true
fi

module load oneapi/release/2025.3.1 2>/dev/null
module load frameworks/2025.3.1 2>/dev/null   # LAST — its python + libsycl must win

export PATH="$HOME/.local/aurora/frameworks/2025.3.1/bin:$HOME/.local/bin:$PATH"
