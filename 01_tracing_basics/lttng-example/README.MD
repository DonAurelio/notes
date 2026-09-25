## KERNEL TRACING

Create a tracing recording session and indicate the part where recording will be placed. tracing sessions are craete with 0 channels. Channes can be created manually or they are created automatically once the user add event rules.

```bash
lttng create --output=<trace_file_path> <session_name>
```

Use the lttng command line tool to list the instrumentation points available for the kernel.

```bash
lttng list --kernel
```

Enable or disable event rules. Event rules instruct lttng to match certain events (instrumentation points).

```bash 
lttng enable-event --kernel <provider_name:tracepoint_name>
```

Start recording.

```bash
lttng start [my_session]
```

Stop and flush recording data to files.

```bash
lttng destroy [my_session]
```

## USER SPACE TRACING

For user space tracing, the trace points to be used in the user application should be defined. Tracepoints are defined in a **tracepoint provider header file (.h)**. An exmaple of a tracepoint is given as follows (file: `provider.hp`).

```c
...
LTTNG_UST_TRACEPOINT_EVENT(
    /* Tracepoint provider name */
    my_provider,

    /* Tracepoint name */
    tracepoint_test,

    LTTNG_UST_TP_ARGS(
        int, my_integer_arg,
        char *, my_string_arg
    ),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_string(my_string_field, my_string_arg)
        lttng_ust_field_integer(int, my_integer_field, my_integer_arg)
    )

)
... 
```

Once the **tracepoint provider header file (.h)** has been define, it must be imported into a **provider package source file (.c)**, an example of this file is provided as follows:

```c
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE

#include "provider.h"
```

1. Create an object file (.o) from the tracepoint provider source file (.c).

```bash
gcc -c I. package.c
```

2. Create an object file (.o) from the user application (.c).

```bash
gcc -c hello.c
```

3. Link the user application  with the tracepoint provider package, `--llttng-ust` and `libdl`

```bash
gcc -o hello hello.o package.o --lttng-ust -ldl
```

4. Run the application. 

```bash
./hello

Hello, World!
Press Enter to continue...
```

5. Create a tracing session.

```bash
lttng create my_session

Session my_session created.
Traces will be output to /home/aureavm/lttng-traces/my_session-20230117-122721
```

6. Once the application is running your will be able to list the instrumentation points available for the application.

```bash
lttng list --userspace

UST events:
-------------

PID: 7168 - Name: ./hello
      my_provider:tracepoint_test (loglevel: TRACE_DEBUG_LINE (13)) (type: tracepoint)
      lttng_ust_statedump:end (loglevel: TRACE_DEBUG_LINE (13)) (type: tracepoint)
...     
```

7. Trace just a particular event (tracepoint) by defining an event rule.

```bash
lttng enable-event --userspace my_provider:tracepoint_text

ust event my_provider_tracepoint_test created in channel channel0
```

8. Start the recording 

```bash
lttng start [my_session]
```

9. Press "Enter" in the application you run in **Step 7** so that it continues and reach the tracepoints.

10. Stop and flush recording data to files.

```bash
lttng destroy [my_session]
```

11. Since trace are recorded using the Common Trace Format (CTF), you need as special software to read the recorded traces, for instance, `babeltrace2`.

```bash 
babeltrace2 /home/aureavm/lttng-traces/my_session-20230117-122721

[12:43:06.740693703] (+?.?????????) valhalla my_provider:tracepoint_test: { cpu_id = 2 }, { my_string_field = "hi there!", my_integer_field = 23 }
[12:43:06.740696282] (+0.000002579) valhalla my_provider:tracepoint_test: { cpu_id = 2 }, { my_string_field = "./hello", my_integer_field = 0 }
[12:43:06.740705684] (+0.000009402) valhalla my_provider:tracepoint_test: { cpu_id = 2 }, { my_string_field = "i^2", my_integer_field = 1 }
```



