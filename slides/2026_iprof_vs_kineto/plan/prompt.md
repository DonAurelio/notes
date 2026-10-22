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


* To trace pytorch code you need to load the frameworks module after loading iprof venv. Use `module load frameworks`.
* You can use `./THAPI/build/ici/bin/iprof -- python3 foo.py` as example for tracing pytorch code.
* You can use `./THAPI/build/ici/bin/iprof --backends pytorch,ze -- python3 foo.py` just to trace torch and level zero calls.
* When executing a command, you need to load the enviroment and send the execution in the same command since env variables do not propagate.
* There is another tracer called Kineto.
* Kineto is build-in in the torch lib.
* To enable Kineto tracing/profiling, we need to anotate the code with a contex manager. `model_kineto.py` has an example.
* We are in a UAN (user authentication node).
* Kineto does not work in a UAN since the torch module provided has the xpu support bundled so event when you execute CPU code, the lib is loaded and UANs do not have XPU devices.
* Compute nodes have XPU devices.
* You can reserve a compute node for interactive sesion using `qsub -A Performance -q debug -l walltime=00:60:00 -l filesystems=home -I -l select=1`.
* I recommend to reserve the node, get the node-s hostname, then send executions/commands via ssh. You can take a look on the pytorch-basic-tracing-node-run or thapi-tracer-run skills for information on how to run in compute nodes.
* There are two papers describing, in particular, the evaluation of thapi, `thapi_tracing_framework.pdf` and `thapi_nic_telemetry.pdf`

# Goals

I need to generate a plan to compare the tracing overhead between iprof and kineto. The plan must define: workload, metrics (how overhead is going to be measured), baseline scenario and evaluation scenarios and the parameters that are going to be varied. The comparison must be as fare as posible. The overhad must be a metric that enables the comparison and should be independenent of the amoun of traced events since that can affect overhead too. Read the related papers and come with an evaluation plan indicating why the considerations.

# Constraints

* If you cannot validate something, ask me. Do not hallucinate.
* Store all assets you create in this working directory and keep them organized.
* Do not generate files or data ourside this folder.
* Avoid excessive verbiage. Be concise and concrete.

* Clean enviroment 

