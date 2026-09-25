#include <ATen/record_function.h>
#include "pytorch_tracepoints.h"

namespace {

// ENTRY: fires BEFORE the kernel runs. Name-only; LTTng adds time + vpid/vtid.
std::unique_ptr<at::ObserverContext> on_entry(const at::RecordFunction& fn) {
  tracepoint(lttng_ust_pytorch, op_entry, fn.name());
  return nullptr;
}

// EXIT: fires AFTER the kernel returns.
void on_exit(const at::RecordFunction& fn, at::ObserverContext*) {
  tracepoint(lttng_ust_pytorch, op_exit, fn.name());
}

// Auto-register at library load (works under LD_PRELOAD, no python changes).
__attribute__((constructor)) void tracer_init() {
  at::addGlobalCallback(
      at::RecordFunctionCallback(&on_entry, &on_exit)
          .scopes({at::RecordScope::FUNCTION}));
}

}  // namespace
