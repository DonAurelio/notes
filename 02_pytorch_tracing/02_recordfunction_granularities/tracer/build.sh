#!/usr/bin/env bash
# Build the RecordFunction tracer as a preloadable shared object (LTTng backend).
#
#   source ../scripts/env.sh lttng     # torch + LTTng tooling (lttng-gen-tp)
#   ./build.sh                          # -> libtorch_tracer.so
#
# ABI NOTE: libtorch here is a RELEASE build (-DNDEBUG). The at::RecordFunction
# layout differs between debug/release, so we compile with -DNDEBUG and match
# torch's CXX11 ABI (=1) or the callback ABI is wrong.
set -euo pipefail
cd "$(dirname "$0")"

: "${TORCH_INCLUDE:?source ../scripts/env.sh lttng first (TORCH_INCLUDE unset)}"
: "${TORCH_LIB:?source ../scripts/env.sh lttng first (TORCH_LIB unset)}"
command -v lttng-gen-tp >/dev/null 2>&1 || { echo "lttng-gen-tp missing: source ../scripts/env.sh lttng"; exit 1; }

if command -v icpx >/dev/null 2>&1; then CXX=icpx; CC=icx; else CXX=g++; CC=gcc; fi
echo "[build] CXX=$CXX CC=$CC"

# The module's lttng-ust (2.14) must win over system /usr/lib64 (2.12, which
# lacks lttng_ust_probe_register). Derive its lib dir from lttng-gen-tp and
# bake it into -L and rpath so the .so binds the right liblttng-ust at runtime.
UST_ROOT="$(dirname "$(dirname "$(command -v lttng-gen-tp)")")"
UST_LIB="$UST_ROOT/lib"
echo "[build] lttng-ust lib: $UST_LIB"

# 1. Generate the CTF probe .c/.h from the .tp schema. The generated .c is the
#    ONE TU that defines the tracepoints; the tracer includes only the header.
lttng-gen-tp pytorch_tracepoints.tp

# 2. Compile the generated probe C.
"$CC" -O2 -fPIC -DNDEBUG -I. -c pytorch_tracepoints.c -o pytorch_tracepoints.o

# 3. Compile + link the preload library.
"$CXX" -std=c++17 -O2 -fPIC -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I. \
  -isystem "$TORCH_INCLUDE" \
  -isystem "$TORCH_INCLUDE/torch/csrc/api/include" \
  -shared \
  tracer.cpp pytorch_tracepoints.o \
  -L"$TORCH_LIB" -Wl,-rpath,"$TORCH_LIB" \
  -L"$UST_LIB" -Wl,-rpath,"$UST_LIB" \
  -lc10 -ltorch_cpu -ltorch \
  -llttng-ust -ldl \
  -o libtorch_tracer.so

echo "[build] wrote $(pwd)/libtorch_tracer.so"
ldd libtorch_tracer.so | grep -E 'c10|torch|lttng' || true
