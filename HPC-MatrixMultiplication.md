# HPC: Matrix Multiplication

<img src="C:\Users\14477\Desktop\prj\MarkWeave\input\assets\cfe339b4e703f9fb0e650ac60259c1f1.jpg" alt="cfe339b4e703f9fb0e650ac60259c1f1" style="zoom:25%;" />

Presenter: 

*Zhi Mengkai*, S32606

*Gao Kailong*, S326067100

***

# Section 1: What is the problem?

- Dense matrix multiplication

- The performance problem

- Research question

***

## Dense matrix multiplication

Given two square matrices, compute their product:

$$
C = AB, \qquad C_{ij} = \sum_{k=0}^{n-1} A_{ik}B_{kj}
$$

- The conventional algorithm performs approximately $2n^3$ floating-point operations.
- The matrices require $O(n^2)$ storage.
- Each output element is independent, which exposes substantial parallelism.

***

## The performance problem

Arithmetic alone does not determine execution time. The processor must move operands through a hierarchy of registers, caches, main memory, and, for CUDA, device memory.

The baseline `i-j-k` loop stores matrices in row-major order:

```c
for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j)
    {
        sum = 0;
        for (k = 0; k < n; ++k)
            sum += A[i][k] * B[k][j];
        C[i][j] = sum;
    }
```

***

## Memory-access behavior

- `A[i][k]` is read sequentially across a row.
- `B[k][j]` is read down a column with a stride of `n` elements.
- As `n` grows, the strided access to `B` uses cache lines poorly and increases memory traffic.
- More CPU threads cannot fully compensate for an inefficient access pattern.

***

## Research question

How do memory layout and parallel execution affect the performance of the same $O(n^3)$ matrix multiplication algorithm?

We isolate three factors:

1. Original `B` compared with a transposed copy `BT`
2. One CPU thread compared with OpenMP threads
3. CPU execution compared with a CUDA GPU kernel

The goal is to explain measured performance through memory-access behavior, not to claim that the benchmark kernels match production BLAS libraries.

***

# Section 2: State of The Art

- CPU matrix multiplication
- GPU matrix multiplication
- Position of this benchmark

***

## CPU matrix multiplication

High-performance GEMM implementations organize the computation around the memory hierarchy.

- **Loop reordering** changes which matrix dimension is traversed contiguously.
- **Transposition or packing** places reused values in a cache-friendly layout.
- **Cache blocking** divides matrices into tiles that fit successive cache levels.
- **Register blocking and SIMD** reuse values in registers and perform several operations per instruction.
- **Thread parallelism** distributes independent output tiles across CPU cores.

***

## CPU libraries and memory hierarchy

Libraries such as BLIS and OpenBLAS combine these techniques with architecture-specific microkernels. Goto and van de Geijn describe the core approach as layered packing, blocking, and a small optimized kernel designed around multilevel memory.

***

## GPU matrix multiplication

GPU implementations assign many output elements to threads and process the matrices in tiles.

- Threads in a block cooperatively load tiles from global memory into shared memory.
- A loaded tile can serve many multiply-accumulate operations before the next global-memory load.
- Consecutive threads should access nearby addresses so the hardware can combine their requests into fewer memory transactions.
- Optimized libraries such as cuBLAS and cuBLASLt select tuned GEMM kernels and can use specialized matrix-multiply hardware when the data type and configuration permit it.

***

## GPU memory coalescing

The important GPU distinction is that contiguous access within one thread does not guarantee efficient access across a warp. Coalescing depends on the addresses requested by neighboring threads at the same instruction.

***

## Position of this benchmark

This project uses deliberately simple kernels to expose individual effects:

| Production technique | Included here? | Purpose in this project |
| --- | --- | --- |
| Transpose or pack `B` | Yes | Compare strided and contiguous inner-loop reads |
| CPU multithreading | Yes | Measure scaling with OpenMP |
| CUDA thread parallelism | Yes | Compare CPU and GPU execution models |
| CPU SIMD | Disabled | Keep vectorization from obscuring cache effects |
| Cache or shared-memory tiling | No | Preserve a readable baseline kernel |
| Vendor GEMM library | No | Study mechanisms rather than peak performance |

***

## Original B versus transposed BT on CUDA

For the CUDA mapping used here, adjacent threads compute adjacent output columns. The original `B` layout can therefore produce coalesced reads across a warp, while `BT` gives each individual thread a contiguous inner loop but may produce strided reads across neighboring threads. The CPU transpose result cannot be assumed to predict the GPU transpose result.

***

## References

- Goto, K. and van de Geijn, R. A., [*Anatomy of High-Performance Matrix Multiplication*](https://www.cs.utexas.edu/~flame/books/TOMS.pdf)
- NVIDIA, [*CUDA Programming Guide: Memory Performance*](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#memory-performance)
- NVIDIA, [*CUDA C++ Best Practices Guide: Shared Memory in Matrix Multiplication*](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-in-matrix-multiplication-c-ab)
- NVIDIA, [*cuBLAS Documentation*](https://docs.nvidia.com/cuda/cublas/)

***

# Section 3: Methodology

***

## Controlled implementations

All six programs compute the same single-precision matrix product and use the same input generation and correctness checks.

| Version | Platform | Threads | Layout used in the inner loop |
| --- | --- | ---: | --- |
| V1 | CPU serial | 1 | Original `B[k][j]` |
| V2 | CPU serial | 1 | Transposed `BT[j][k]` |
| V3 | CPU OpenMP | 2, 4, 6, 8, 10 | Original `B[k][j]` |
| V4 | CPU OpenMP | 2, 4, 6, 8, 10 | Transposed `BT[j][k]` |
| V5 | CUDA | GPU threads in 16 x 16 blocks | Original `B[k][j]` |
| V6 | CUDA | GPU threads in 16 x 16 blocks | Transposed `BT[j][k]` |

***

## Compiler and parallel execution settings

The CPU compiler uses `-O3 -march=native` but disables loop and SLP vectorization. OpenMP parallelizes the outer `i` loop with static scheduling. The transpose uses 32 x 32 tiles and its time is recorded separately.

***

## Test matrix sizes and workload

The default matrix sizes are:

$$
n \in \{64, 256, 1024, 4096, 16384\}
$$

For each size:

- One matrix contains $n^2$ `float` values and occupies $4n^2$ bytes.
- One multiplication performs approximately $2n^3$ floating-point operations.
- At $n=16384$, three matrices require about 3 GiB and one multiplication performs about $8.8 \times 10^{12}$ floating-point operations.

The large range reveals when startup overhead, cache capacity, memory traffic, parallel scaling, and GPU transfer costs become important.

***

## Timing protocol

- Run every configuration five times.
- Retain all five raw measurements in CSV output.
- Treat the first run as a warm-up.
- Report the minimum time from runs 2 through 5.
- Compute throughput as $2n^3 / t$ and report the result in GFLOP/s.

***

## Timing boundaries

CPU timing contains only the multiplication. CUDA reports both kernel-only time and end-to-end time, where end-to-end includes host-to-device copies, kernel execution, and the device-to-host result copy. Allocation, initialization, verification, and CSV writing remain outside the timed region. Transposition time is reported separately.

***

## Correctness protocol

The inputs use a dense low-rank construction with an analytical result:

$$
A_{ik} = x_i u_k, \qquad B_{kj} = v_k y_j
$$

$$
C_{ij} = x_i y_j \sum_k u_kv_k
$$

This construction has two benefits:

- Every implementation still executes the full matrix multiplication.
- The program can verify every output element in $O(n^2)$ time instead of running another $O(n^3)$ reference multiplication.

***

## Correctness tolerance and reporting

Each of the five repetitions must pass. The tolerance is

$$
10^{-4}\max(1, |C_{ij}^{\mathrm{expected}}|)
$$

The CSV records the verification status and maximum absolute error for every configuration.

***

# Section 4: Hardware and Environment

***

# Section 5: Computational tests

***

# Section 6: Analysis

***

# Section 7: Conclusion
