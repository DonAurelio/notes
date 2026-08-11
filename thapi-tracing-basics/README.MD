# DEFINITIONS 

**Profiling**

A **profiler** is suitable to identify "where" performance is lost in a given piece of code. The profiler outputs a profile, a statistical summary of observed events. The profile is used to discover which functions took the most time to execute. However the profiler won't report "why" some identified functions are the bottleneck. Bottlenecks might only occur when specific conditions are met, conditions that are sometimes impossible to capture by a statistical profiler, or impossible to reproduce with an application altered by the overhead of an event-based profiler.

**Tracing** 

Tracing is a technique used to understand what goes on in running software system. The piece of software used for tracing is called tracer, which is conceptually similar to a tape recorder. When recording, specific **instrumetation points** placed in the software source code generate **events** that are saved on a giat tape: **a trace file**. You cab record user application and operating system events at the same time, opening the posibility of resolving a wide range of problems that would otherwise be extremely challenging. 

The list of recorded events inside a trace file can be read manually like a log file for the maximum level of detail, but it's generally much more interesting to perform application specific analyses to produce reduced statistics and graphs that are useful to resolve a given problem. Trace viewrs and analyzers are specialized tools designed to do this. 

**LTTng**

The Linux Trace Toolkit: next generation is a modern toolkit for tracing Linux systems and applications. LTTng is a set of software components which interact to instrument the Linux kernel and user applications, and to control tracing (start and stop recording, create recording event rules, and the rest). Those components are bundled into the following packages:

* LTTng-tools: Libraries and command-line interface to control tracing.
* LTTng-modules: Linux kernel modules to instrument and trace the kernel.
* LTTng-UST: Libraries and Java/Python packages to instrument and trace user applications.

*Core concepts*

- An **instrumentation point** is a point, within a piece of software, which, when executed created a LTTng event. 
- A **tracepoint provider header file (.h)** contains the tracepoint definitions for a given tracepoint provider.
- A **tracepoint provider** is a set of compiled functions which provides tracepoints to an application.
- A **tracepoint provider package is an object file (`.o`)** or a shared library (`.so`) which contains one or more tracepoint providers. Its source files are:

* One or more tracepoint provider header (`.h`).
* A trace provider package source (`.c`)  

- Those functions can make LTTng emits events with user-defined fields and serialize those events as event records to one or more LTTng-UST channel sub-buffers. The `lttng_ust_tracepoint()` macro, which you insert in the source code of a user application, calls those functions (the set of compiled functions).

# References

1. [The LTTng Documentation](https://lttng.org/docs/v2.13/#doc-nuts-and-bolts)
