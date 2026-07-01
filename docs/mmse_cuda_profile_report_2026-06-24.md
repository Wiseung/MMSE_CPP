# MMSE CUDA Profiling Report (2026-06-24)

## Scope

This report measures the current CUDA implementation in `G:\MMSE_CPP` on the local machine.

Important boundary:

- The current public API processes one LTE subframe per call.
- One call consumes `14 OFDM symbols x 1200 subcarriers`, which is one LTE 1 ms subframe under normal CP.
- Therefore:
  - `10 ms` results in this report are the aggregate cost of `10` consecutive 1 ms subframe calls.
  - `100 ms` results are the aggregate cost of `100` consecutive 1 ms subframe calls.
- This is a measurement of the current implementation as-is, not a redesigned batched 10 ms / 100 ms kernel interface.

## Test Environment

| Item               | Value                              |
| ------------------ | ---------------------------------- |
| Date               | 2026-06-24                         |
| OS                 | Windows                            |
| Repo               | `G:\MMSE_CPP`                      |
| Build              | `Release`                          |
| GPU                | NVIDIA GeForce RTX 4060 Laptop GPU |
| Driver / CUDA      | 560.94 / 12.6                      |
| Compute capability | 8.9                                |
| SM count           | 24                                 |
| Global memory      | 8,585,216,000 bytes                |
| L2 cache           | 33,554,432 bytes                   |

## Test Method

### 1. Functional pre-check

The following command was run first to ensure profiling was done on a passing build:

```powershell
ctest --test-dir build -C Release --output-on-failure -R mmse_tests
```

Result: `100% tests passed`.

### 2. Build

```powershell
cmake --build build --config Release --target mmse_cuda_profile mmse_tests
```

### 3. Profiling executable

A dedicated benchmark executable was added:

- `G:\MMSE_CPP\bench\mmse_cuda_profile.cpp`

It does the following:

- generates LTE-structured simulation data for `100` consecutive subframes
- uses full-band 20 MHz LTE structure:
  - `1200` subcarriers
  - `100` PRB
  - `14` OFDM symbols per subframe
  - normal CP
  - `2x2` MIMO
  - `2` layers
  - `64QAM` (`mod_order = 6`)
- builds CRS-bearing subframes and data RE samples consistent with the repo's current equalizer path
- measures:
  - cold `10 ms` latency
  - warm `10 ms` latency
  - repeated `10 ms` aggregate latency
  - repeated `100 ms` aggregate latency
- queries:
  - static CUDA kernel resource attributes
  - CUDA visible memory before/after init and after runs
  - process memory

### 4. Commands used

Baseline timing and memory:

```powershell
.\build\Release\mmse_cuda_profile.exe
```

Equalize kernel Nsight Compute:

```powershell
ncu --target-processes all --set basic --kernel-name-base demangled -k "regex:.*equalize_stub_kernel.*" -c 1 --profile-from-start on -f -o .\build\ncu_equalize_report .\build\Release\mmse_cuda_profile.exe --warmup-subframes 0 --iters-10ms 1 --iters-100ms 0
```

Estimate kernel Nsight Compute:

```powershell
ncu --target-processes all --set basic --kernel-name-base demangled -k "regex:.*estimate_stub_kernel.*" -c 1 --profile-from-start on -f -o .\build\ncu_estimate_report .\build\Release\mmse_cuda_profile.exe --warmup-subframes 0 --iters-10ms 1 --iters-100ms 0
```

Runtime `nvidia-smi` sampling was also performed every `200 ms` during the `100 ms` aggregate benchmark window.

## Input and Output Definition

### Input structure per 1 ms subframe

| Item                 | Value         |
| -------------------- | ------------- |
| RX antennas          | 2             |
| TX ports             | 2             |
| Layers               | 2             |
| Symbols              | 14            |
| Subcarriers          | 1200          |
| PRB                  | 100           |
| Data RE per subframe | 14,400        |
| Modulation order     | 6 bits/symbol |

### Logical data size

Input grid is planar complex float for 2 RX antennas:

- per subframe input bytes:
  - `2 RX x 2 (re/im) x 14 x 1200 x 4 bytes = 268,800 bytes`

Output is `x_hat_re`, `x_hat_im`, `sinr` for 2 layers over 14,400 RE:

- per subframe output bytes:
  - `14,400 RE x 2 layers x 3 planes x 4 bytes = 345,600 bytes`

Aggregate sizes:

| Window | Input bytes | Output bytes |
| ------ | ----------: | -----------: |
| 1 ms   |     268,800 |      345,600 |
| 10 ms  |   2,688,000 |    3,456,000 |
| 100 ms |  26,880,000 |   34,560,000 |

## Timing Results

Source:

- `G:\MMSE_CPP\build\mmse_cuda_profile_output.txt`

### Aggregate latency

| Metric                      |       Result |
| --------------------------- | -----------: |
| Cold 10 ms aggregate        |  198.0260 ms |
| Cold per-subframe           |   19.8026 ms |
| Warm 10 ms aggregate        |  121.0942 ms |
| Warm per-subframe           |   12.1094 ms |
| 10 ms average               |  120.7096 ms |
| 10 ms p50                   |  120.6548 ms |
| 10 ms p95                   |  121.5977 ms |
| 10 ms min                   |  119.3188 ms |
| 10 ms max                   |  122.7537 ms |
| 10 ms average per-subframe  |   12.0710 ms |
| 100 ms average              | 1209.7897 ms |
| 100 ms p50                  | 1209.4601 ms |
| 100 ms p95                  | 1216.0304 ms |
| 100 ms min                  | 1203.1445 ms |
| 100 ms max                  | 1218.3277 ms |
| 100 ms average per-subframe |   12.0979 ms |

### Interpretation

- Current throughput is approximately `12.07 ms` per `1 ms` LTE subframe.
- The `10 ms` aggregate workload currently takes about `120.71 ms`.
- The `100 ms` aggregate workload currently takes about `1209.79 ms`.
- The implementation is therefore not real-time for the stated LTE processing target.

## GPU Runtime Occupancy and Utilization

### `nvidia-smi` dynamic sampling during 100 ms aggregate benchmark

Source:

- `G:\MMSE_CPP\build\nvidia_smi_samples.csv`

| Metric                  |      Result |
| ----------------------- | ----------: |
| Sample count            |          85 |
| GPU util average        |      88.88% |
| GPU util peak           |         94% |
| GPU memory used average | 1432.64 MiB |
| GPU memory used peak    |    1467 MiB |

Notes:

- This is runtime device-level utilization, not per-kernel occupancy.
- Peak visible device memory during execution is far above the statically-accounted MMSE buffers, so it includes CUDA runtime/profiler/driver residency and other device allocations outside the MMSE working-set estimate.

### Nsight Compute: `equalize_stub_kernel`

Source:

- `G:\MMSE_CPP\build\ncu_equalize_report.ncu-rep`

| Metric                         |         Result |
| ------------------------------ | -------------: |
| Kernel duration                |        6.78 us |
| Grid size                      |      57 blocks |
| Block size                     |    256 threads |
| Threads launched               |         14,592 |
| Registers per thread           |             64 |
| Static shared memory           |            0 B |
| Driver shared memory per block |       1.02 KiB |
| Theoretical occupancy          |         66.67% |
| Achieved occupancy             |         37.41% |
| Achieved active warps per SM   |          17.96 |
| Waves per SM                   |           0.59 |
| Compute throughput             | 16.11% of peak |
| DRAM throughput                | 42.24% of peak |
| L1/TEX throughput              | 17.43% of peak |
| L2 throughput                  | 14.86% of peak |

Nsight warnings:

- grid too small to fill the device
- occupancy limited by registers
- workload imbalance across SM/SMSP/cache slices

### Nsight Compute: `estimate_stub_kernel`

Source:

- `G:\MMSE_CPP\build\ncu_estimate_report.ncu-rep`

| Metric                         |        Result |
| ------------------------------ | ------------: |
| Kernel duration                |      18.85 ms |
| Grid size                      |       1 block |
| Block size                     |      1 thread |
| Threads launched               |             1 |
| Registers per thread           |            54 |
| Static shared memory           |           0 B |
| Driver shared memory per block |      1.02 KiB |
| Theoretical occupancy          |        50.00% |
| Achieved occupancy             |         2.08% |
| Achieved active warps per SM   |          1.00 |
| Waves per SM                   |          0.00 |
| Compute throughput             | 0.17% of peak |
| DRAM throughput                | 0.01% of peak |
| L1/TEX throughput              | 4.10% of peak |
| L2 throughput                  | 0.14% of peak |

Nsight warnings:

- only `1` thread per block
- only `1` block in the entire grid
- severe underutilization of the GPU

### Main conclusion from kernel profiling

`equalize_stub_kernel` is not the primary latency source.

The dominant issue is `estimate_stub_kernel`, which currently runs as:

- `grid = 1`
- `block = 1`

That means the channel-estimation stage is effectively serialized onto one CUDA thread and is the main reason the end-to-end latency is around `12 ms` per subframe.

## Memory and Buffer Placement

### Logical placement

| Data                     | Placement                | Type             |
| ------------------------ | ------------------------ | ---------------- |
| Input simulation dataset | Host memory              | pageable `float` |
| Staging grids            | Host memory              | pageable `float` |
| Transport grids          | Host memory              | pinned `int16_t` |
| Output host buffers      | Host memory              | pinned `float`   |
| Device input grids       | GPU global memory        | `int16_t`        |
| Device metadata          | GPU global memory        | `CudaGridMeta`   |
| Device sigma2 state      | GPU global memory        | `float`          |
| Device estimates         | GPU global memory        | `float`          |
| Device outputs           | GPU global memory        | `float`          |
| Kernel temporaries       | registers / local memory | per-kernel       |

### Static working-set estimate from code

Source:

- profiling executable output

| Item                  |    Per slot | Total (3 slots) |
| --------------------- | ----------: | --------------: |
| Host pageable buffers | 1,022,492 B |     3,067,476 B |
| Host pinned buffers   |   537,600 B |     1,612,800 B |
| Device buffers        | 1,256,160 B |     3,768,480 B |

Derived totals:

| Item                             |       Size |
| -------------------------------- | ---------: |
| Dataset input footprint          | 25.635 MiB |
| Total host pageable ring storage |  2.925 MiB |
| Total host pinned ring storage   |  1.538 MiB |
| Total device ring storage        |  3.594 MiB |

### CUDA memory observations

From the benchmark:

| Metric                                             |          Result |
| -------------------------------------------------- | --------------: |
| GPU free before init                               | 7,443,841,024 B |
| GPU free after init                                | 7,439,646,720 B |
| GPU free after runs                                | 6,003,097,600 B |
| Init-time visible GPU allocation delta             |     4,194,304 B |
| Post-run visible GPU allocation delta vs post-init | 1,436,549,120 B |

Important note:

- The static MMSE device buffers account for about `3.59 MiB`.
- The much larger post-run visible delta from `cudaMemGetInfo` is not explained by MMSE buffers alone.
- It likely includes CUDA runtime state, allocator retention, WDDM/driver residency behavior, and profiling side effects.
- Therefore, the code-level buffer accounting is the reliable MMSE working-set estimate; `cudaMemGetInfo` should be treated as whole-context residency, not pure algorithm storage.

## Register / Shared / Local Memory Summary

| Kernel                 | Registers/thread | Static shared | Local bytes/thread | Notes                                                              |
| ---------------------- | ---------------: | ------------: | -----------------: | ------------------------------------------------------------------ |
| `estimate_stub_kernel` |               54 |           0 B |           40,000 B | very large local memory footprint due to large thread-local arrays |
| `equalize_stub_kernel` |               64 |           0 B |                0 B | occupancy limited by register count                                |

Interpretation:

- `estimate_stub_kernel` spills heavy temporary state into local memory because it declares large per-thread arrays while launching only one CUDA thread.
- `equalize_stub_kernel` is comparatively compact, and its occupancy limit is register-driven rather than shared-memory-driven.

## Output Artifacts

Generated files:

- baseline profile text:
  - `G:\MMSE_CPP\build\mmse_cuda_profile_output.txt`
- runtime-sampled profile text:
  - `G:\MMSE_CPP\build\mmse_cuda_profile_runtime_output.txt`
- runtime `nvidia-smi` samples:
  - `G:\MMSE_CPP\build\nvidia_smi_samples.csv`
- Nsight Compute reports:
  - `G:\MMSE_CPP\build\ncu_equalize_report.ncu-rep`
  - `G:\MMSE_CPP\build\ncu_estimate_report.ncu-rep`

## Overall Conclusion

The current algorithm implementation is functionally runnable on GPU, but performance is dominated by an extremely under-parallelized estimate stage.

Current measured performance:

- `~12.07 ms` per `1 ms` LTE subframe
- `~120.71 ms` per `10 ms` aggregate frame
- `~1209.79 ms` per `100 ms` aggregate window

The main bottleneck is not the equalization kernel. It is the estimate kernel, which currently runs with:

- `1 block`
- `1 thread`
- `18.85 ms` kernel time
- `2.08%` achieved occupancy

If the goal is real-time or near-real-time LTE processing, the first engineering target should be:

1. parallelize `estimate_stub_kernel`
2. remove thread-local large arrays from the single-thread estimate path
3. re-measure before changing the equalization kernel
