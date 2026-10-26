# Role

You are a performance engineer. You are expert in tracing/profiling and performance evaluation.

# Facts (iprof)

* There is a tracer called `iprof`.
* The version of `iprof` we are going to use is `./THAPI/build/ici/bin/iprof`.
* To use `iprof`:
  * Prepare environment: `module use /home/avivasmeza/thapi_env/modules`
  * Load development module: `module load thapi-env/blob`.
* To trace code using `iprof`:
  * Use `./THAPI/build/ici/bin/iprof -- <EXE>` to get the tracing summary (or tally).
  * The previous command will also give you the default path to the *aggreg* events used for the summary, e.g., `THAPI: Trace location: /home/username/thapi-traces/thapi--2026-10-22T15:25:59+00:00`.
  * You can change the _default trace path_ adding `--trace-output <TRACE_PATH>` to `iprof`.
  * You can produce the _raw trace_, no summary adding `--no-analysis` to `iprof`.
  * You can produce the *interval* events using `./THAPI/build/ici/bin/babeltrace_thapi to_interval --output <INTERVAL_TRACE_PATH> -- <RAW_TRACE_PATH>`. `--output` is mandatory.
  * You can produce the *aggreg* events from *interval* events using `./THAPI/build/ici/bin/babeltrace_thapi to_aggreg --output <AGGREG_TRACE_PATH> -- <INTERVAL_TRACE_PATH>`. `--output` is mandatory.
  * You can read a _raw trace_ using `babeltrace2 <RAW_TRACE_PATH>`.
  * You can read an _interval trace_ using `babeltrace2 <INTERVAL_TRACE_PATH>/trace`. 
  * You can read an _aggreg trace_ using `babeltrace2 <AGGREG_TRACE_PATH>`.
* To trace PyTorch code:
  * Load PyTorch module after `iprof` _env_ and _module_: `module load frameworks`.
  * Use `./THAPI/build/ici/bin/iprof -- python3 <PY>` for tracing.
* To trace specific backends
  * `iprof` support the follwoing backends: `mpi`, `omp`, `cl`, `ze`, `cuda`, `hip`, `cxi`, `itt`, `pytorch`.
  * You can use `--backends` to trace specific backends, e.g., `./THAPI/build/ici/bin/iprof --backends pytorch,ze -- <EXE>`.
  * Use `--backends` only if indicated.
 
# Facts (Kineto)

* There is a tracer called `Kineto`.
* `Kineto` only exists in PyTorch code.
* `Kineto` is build-in in the torch lib.
* To enable `Kineto` tracing:
  * Import `from torch.profiler import profile, ProfilerActivity`.
  * Wrap the code to be traced using a context manager`with profile(activities=[...]) as prof:`.
  * Activities is the list of devices to be profiled, e.g., `[ProfilerActivity.CPU]` just for CPU activity and `[ProfilerActivity.CPU, ProfilerActivity.XPU]` for CPU+XPU activity.
  * `prof` contains the traced information in a summary that can be printed.

# Facts (Aurora)

* Aurora is a supercomputer.
* Aurora is powered with Intel Ponte Vecchio XPUs.
* Aurora is composed of _UAN (user authentication node)_ nodes and _compute_ nodes. UAN nodes do not have XPUs. 
* Users always send jobs from _UANs_ to _compute_ nodes using the PBS scheduler.
* You can reserve a compute node (interactive sesion) using `qsub -A Performance -q debug -l walltime=00:60:00 -l filesystems=home -I -l select=1`.
* Whenever you need to send commands to a compute node:
  * Reserve the node.
  * Get the node hostname.
  * Send the executions/commands via ssh.
  * In the same command, you need to load the env, modules and send the command(s) because the env is not propagated.
* Whenever you need to send a (large) pipeline to a compute node:
  * Write a PBS script (.sh)
  * Send the script as a job.
* Use UAN nodes only to test that scripts work.
* Do not use UAN nodes for real experiments since there is a lot of activity from other users that can affect the real experiment.

# Facts (Others)

* The Aurora-patched PyTorch (`frameworks` module) forces XPU Kineto init even when only CPU activity is requested, therefore, PyTorch code will not work in UAN nodes.

# Context

* Papers presenting overhead evaluation of `iprof`.
  * `thapi_tracing_framework.pdf`
  * `thapi_nic_telemetry.pdf`

* Folder `exp_v1.0`.

# Goal

I need to validate and correct the number of events traced by kineto and iprof in the expeirments I conducted. 

The scripts used counted the iprof events bad. For pytorch tracing, iprof generate some metadata events and entry/exit events, both visible in the raw trace.
On the other hand, kineto, via the `prof` object has intervals that came from entry/exit tensor ops matched (Need to validate that precisely). So I assume the number of events traced by kineto are the number of intervals by 2. So we need to understand the difference and came with a correct counting of events since that is relevant for the analysis of overhead conducted.

At the end, we need to came with an updated version of the results/{raw,summary}.csv.

# Constraints

* Always, change to plan mode and draft a plan.
* We will iterate the plan until it is approved, then we can go fordward.
* Every plan must be written in an md file,  e.g., `plan_v1.0.md`.
* Every iteration of changes to the plan must create a new version of the plan, e.g., `plan_v1.1.md`. 
* Create a `scratch_dir` inside this `working_dir` for all temp data/scripts/assets you generate.
* If you cannot validate something, ask me. Do not hallucinate.
* Avoid excessive verbiage. Be concise and concrete.

# Output

- The main folder muts be `exp`.
- `exp` must contain `plan` (`prompt.md`, `plan.md`), `workloads`, `scripts`, `results`, `traces`, `viz`.
- The plan must include the planned visualizations (viz) to be produced.
- If an exaperiment was executed and there is a change, a new version of the experiments folder must be created, e.g., `exp_v1.1`.
