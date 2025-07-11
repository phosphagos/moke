# TODO List: moke and moketest

## moke

supporting multiple platform as backend.

* [x] (P0) supports cuda;

* [x] (P0) supports hip;

* (P1) supports cpu parallelism - multiple ISAs:

    * [ ] x86-64 with AVX512;

    * [ ] RV64 with vector extension;

* [ ] (P2) supports sycl;

* [ ] (P3) supports opencl, or vulkan compute shader;

## moke::common

* a series of common utilities for heterogenuous parallel programming.

    * [x] common macro and type definations;

    * [x] simple arithmetic operation templates, e.g. ceil_div, bitops;

    * [x] runtime API wrappers:

        * [x] memory management: Device/Host `Allocator`, `Deleter` and `Pointer`;

        * [x] status and error handling: `CheckStatus`;

        * [ ] device and stream management: `Handle`;

            * [ ] stream-based async memory management;

    * core data structure - tensors:

        * [x] tensor without ownership: `DeviceTensorView` and `HostTensorView`;

        * [x] tensor with ownership: `DeviceTensor` and `HostTensor`;

## moke::test

A series of utilities for testing, including:

* I/O operation:

    * [x] Host Memory <=> Device Memory;

    * [ ] Disk File <=> Host Memory;

    * [ ] Disk File <=> Device Memory (optional, for optimization);

* Random number generation:

    * [ ] Host random number generation with seed;

    * [ ] Device parallel random number generation with seed;

* Performance Profiling

    * [ ] Simple performance profiling based on event;

    * [ ] High resolutional profiling based on eg.`cupti`;

* Accuracy Comparison:

    * [ ] Simple binary-alignment accuracy comparator;

    * [ ] Float-point accuracy comparator with **absolute error**;

    * [ ] Float-point accuracy comparator with **relative error**;

    * [ ] DIFF3 accuracy comparator;

* [ ] homogenuous/heterogenuous memory container wrappers;

## moke::ops

* concrete operators:

    * simple unary and elem-wise operators;

    * memory accesses: gather, scatter;

    * matrix/tensor operations:

        * matmul/batch_matmul;

        * convolution;

        * resampling;