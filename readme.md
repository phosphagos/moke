# TODO List: moke and moketest

## moke

supporting multiple platform as backend.

* (T0) supports cuda;

* (T0) supports hip;

* (T1) supports cpu parallelism - multiple ISAs:

    * x86-64 with AVX512;

    * RV64 with vector extension;

* (T2) supports sycl;

* (T3) supports opencl, or vulkan compute shader;

### moke::common

* a series of common utilities for heterogenuous parallel programming.

    * common macro and type definations;

    * simple arithmetic operation templates, e.g. ceil_div, bitops;

    * device_ptr and device containers without ownership;

        * device_ptr and device_array;

        * device_tensor;

### moke::ops

* concrete operators:

    * simple unary and elem-wise operators;

    * memory accesses: gather, scatter;

    * matrix/tensor operations:

        * matmul/batch_matmul;

        * convolution;

        * resampling;

## moketest

A series of utilities for testing, including:

* Status and Error Handling:

    * `check_status`;

* Memory Handling and Containers:

    * host/device memory with ownership;

    * host/device containers with ownership:

        * host/device array with ownership;

        * host/device tensor with ownership;

    * homogenuous/heterogenuous memory container wrappers;

* I/O operation:

    * Host Memory <=> Device Memory;

    * Disk File <=> Host Memory;

    * Disk File <=> Device Memory (optional, for optimization);

* Random number generation:

    * Host random number generation with seed;

    * Device parallel random number generation with seed;

* Performance Profiling

    * Simple performance profiling based on event;

    * High resolutional profiling based on eg.`cupti`;

* Accuracy Comparison:

    * Simple binary-alignment accuracy comparator;

    * Float-point accuracy comparator with **absolute error**;

    * Float-point accuracy comparator with **relative error**;

    * DIFF3 accuracy comparator;
