# NOTES 

**Profiling**

A **profiler** is suitable to identify "where" performance is lost in a given piece of code. The profiler outputs a profile, a statistical summary of observed events. The profile is used to discover which functions took the most time to execute. However the profiler won't report "why" some identified functions are the bottleneck. Bottlenecks might only occur when specific conditions are met, conditions that are sometimes impossible to capture by a statistical profiler, or impossible to reproduce with an application altered by the overhead of an event-based profiler.

**Tracing** 

Tracing is a technique used to understand what goes on in running software system. The piece of software used for tracing is called tracer, which is conceptually similar to a tape recorder. When recording, specific **instrumetation points** placed in the software source code generate **events** that are saved on a giat tape: **a trace file**. You cab record user application and operating system events at the same time, opening the posibility of resolving a wide range of problems that would otherwise be extremely challenging. 

The list of recorded events inside a trace file can be read manually like a log file for the maximum level of detail, but it's generally much more interesting to perform application specific analyses to produce reduced statistics and graphs that are useful to resolve a given problem. Trace viewrs and analyzers are specialized tools designed to do this. 

**LTTng**

The main interface for tracing control is a single command-line tool named `lttng`. The latter can create several **recording sessions**, enable and disable recording rules on the fly, filter events efficiently with custom user expressions, start and stop tracing, and much more. LTTng can write the traces on the file system or send them over the network, and keep them totally or partially. You can make LTTng execute user-defined actions when LTTng emits an event. you can view the traces once tracing becomes inactive (offline tracing analysis) or as LTTng record events (online tracing analysis).

# References

1. [The LTTng Documentation](https://lttng.org/docs/v2.13/#doc-nuts-and-bolts)