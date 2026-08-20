// PyTorch RecordFunction tracer — one .so, six granularity aspects, selected at
// load time by environment variables so a SINGLE build covers every experiment:
//
//   TRACER_SCOPES=function|function+backward   (aspect 2 — which RecordScopes)
//   TRACER_THREAD=global|local                 (aspect 3 — callback thread reach)
//   TRACER_TOPLEVEL=0|1                         (aspect 4 — emit only depth==0)
//   TRACER_INPUTS=0|1                           (aspect 5 — render fn.inputs())
//   TRACER_SAMPLING=<0..1>                      (aspect 6 — fraction of ops sampled)
//
// The trace itself always carries scope + depth on every event (aspects 2 and 4
// are visible in the data); the toggles let us change WHAT is captured to show
// each granularity's effect. Aspect 3 is read from LTTng's vtid context.
//
// Loaded via LD_PRELOAD into an UNMODIFIED python — a __attribute__((constructor))
// installs the RecordFunction callback at load, before the workload runs.

#include <ATen/record_function.h>
#include <c10/util/ArrayRef.h>

#include <atomic>
#include <cstdlib>
#include <string>
#include <unordered_set>

#include "pytorch_tracepoints.h"

namespace {

// ---- config read once at load ------------------------------------------------
bool g_toplevel_only = false;   // aspect 4
bool g_needs_inputs = false;    // aspect 5

// aspect 4: per-thread nesting depth. thread_local so each OS thread (incl. the
// autograd backward thread) counts its own nesting independently.
thread_local int g_depth = 0;

const char* scope_name(at::RecordScope s) {
  switch (s) {
    case at::RecordScope::FUNCTION: return "FUNCTION";
    case at::RecordScope::BACKWARD_FUNCTION: return "BACKWARD_FUNCTION";
    case at::RecordScope::TORCHSCRIPT_FUNCTION: return "TORCHSCRIPT";
    case at::RecordScope::KERNEL_FUNCTION_DTYPE: return "KERNEL_DTYPE";
    case at::RecordScope::USER_SCOPE: return "USER_SCOPE";
    default: return "OTHER";
  }
}

// aspect 5: render fn.inputs() as a compact "dtype[shape]@device, ..." string.
// Only called when needsInputs(true), and only from the start callback where
// fn.inputs() is valid (they get invalidated before the end callback).
std::string render_inputs(const at::RecordFunction& fn) {
  std::string out;
  for (const c10::IValue& iv : fn.inputs()) {
    if (!out.empty()) out += ", ";
    if (iv.isTensor()) {
      const at::Tensor& t = iv.toTensor();
      if (!t.defined()) { out += "tensor(undef)"; continue; }
      out += std::string(c10::toString(t.scalar_type())) + "[";
      for (int i = 0; i < t.dim(); ++i) {
        if (i) out += ",";
        out += std::to_string(t.size(i));
      }
      out += "]@" + std::string(t.device().str());
    } else if (iv.isScalar()) {
      out += "scalar";
    } else if (iv.isIntList() || iv.isDoubleList() || iv.isList()) {
      out += "list";
    } else {
      out += iv.tagKind();
    }
  }
  return out;
}

std::unique_ptr<at::ObserverContext> on_entry(const at::RecordFunction& fn) {
  const int depth = g_depth++;                 // aspect 4: depth BEFORE increment
  if (g_toplevel_only && depth != 0) return nullptr;
  const std::string args = g_needs_inputs ? render_inputs(fn) : std::string();
  tracepoint(lttng_ust_pytorch, op_entry,
             fn.name(), scope_name(fn.scope()), depth, args.c_str());
  return nullptr;
}

void on_exit(const at::RecordFunction& fn, at::ObserverContext*) {
  const int depth = --g_depth;                 // symmetric with entry
  if (g_toplevel_only && depth != 0) return;
  tracepoint(lttng_ust_pytorch, op_exit,
             fn.name(), scope_name(fn.scope()), depth);
}

bool env_is(const char* k, const char* v) {
  const char* e = std::getenv(k);
  return e && std::string(e) == v;
}

__attribute__((constructor)) void tracer_init() {
  g_toplevel_only = env_is("TRACER_TOPLEVEL", "1");
  g_needs_inputs = env_is("TRACER_INPUTS", "1");

  // aspect 2: which scopes fire the callback.
  std::unordered_set<at::RecordScope, std::hash<at::RecordScope>> scopes = {
      at::RecordScope::FUNCTION};
  if (env_is("TRACER_SCOPES", "function+backward"))
    scopes.insert(at::RecordScope::BACKWARD_FUNCTION);

  auto cb = at::RecordFunctionCallback(&on_entry, &on_exit)
                .needsInputs(g_needs_inputs)   // aspect 5
                .scopes(scopes);               // aspect 2

  // aspect 6: sampling probability. p<1 makes the dispatcher fire the callback
  // on only ~p of ops (entry AND exit skipped together, so depth stays
  // balanced). Default 1.0 = full fidelity. NOTE: sampling only applies to
  // GLOBAL callbacks — thread-local callbacks always run.
  if (const char* p = std::getenv("TRACER_SAMPLING"))
    cb.samplingProb(std::atof(p));

  // aspect 3: global (every thread) vs thread-local (registering thread only).
  if (env_is("TRACER_THREAD", "local"))
    at::addThreadLocalCallback(cb);
  else
    at::addGlobalCallback(cb);
}

}  // namespace
