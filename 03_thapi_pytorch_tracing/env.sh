#!/usr/bin/env bash
# Module recipe for THAPI pytorch-backend validation on Aurora.
# frameworks MUST load last (its python/libsycl must win on PATH/LD_LIBRARY_PATH
# over what oneapi/lttng pull in) - verified in this validation.
#
# Usage: THAPI_INSTALL=/path/to/THAPI/install source env.sh [mpi]
#   mpi arg loads mpich too (needed for experiments 02_mpi_cpu / 03_mpi_xpu / 04-05)
#   THAPI_INSTALL must point at a THAPI install prefix with `iprof`/`babeltrace_thapi`
#   built (see thapi-tracer-environment-config / thapi-tracer-build) -- no default,
#   this is a per-checkout path.

module load oneapi/release/2025.3.1
module load ruby-cast-to-yaml/0.1.1-tepbhny ruby-nokogiri/1.16.7-rnvmyfk \
            ruby-metababel/1.1.4-jlmde4f ruby-babeltrace2/0.1.5-htyldlz \
            ruby-ffi/1.17.2-obrr273 ruby-narray-ffi/1.4.4-fzwgodb
module load gmake/4.4.1 protobuf/3.28.2
module load babeltrace2/2.1.2-archive lttng-tools/2.14.0-archive

if [ "$1" = "mpi" ]; then
  module load mpich/opt/5.0.0.aurora_test.3c70a61
fi

module load frameworks

if [ -n "$THAPI_INSTALL" ]; then
  export PATH="$THAPI_INSTALL/bin:$PATH"
fi
