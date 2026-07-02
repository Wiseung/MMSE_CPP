# MMSE CUDA Profiling 报告（2026-06-24）

## 作用范围

本报告用于测量 `G:\MMSE_CPP` 当前 CUDA 实现在本机上的表现。

重要边界说明：

- 当前公开 API 仍然是“一次调用处理一个 LTE 子帧”
- 一次调用消费 `14 OFDM symbols × 1200 subcarriers`，即 normal CP 下一个 LTE `1 ms` 子帧
- 因此：
  - 本报告中的 `10 ms` 结果，是连续 `10` 次 `1 ms` 子帧调用的聚合开销
  - 本报告中的 `100 ms` 结果，是连续 `100` 次 `1 ms` 子帧调用的聚合开销
- 本报告测量的是当前实现原样行为，而不是重新设计后的 batched `10 ms / 100 ms` kernel 接口

## 测试环境

| 项目               | 数值                               |
| ------------------ | ---------------------------------- |
| 日期               | 2026-06-24                         |
| OS                 | Windows                            |
| 仓库               | `G:\MMSE_CPP`                      |
| 构建               | `Release`                          |
| GPU                | NVIDIA GeForce RTX 4060 Laptop GPU |
| Driver / CUDA      | 560.94 / 12.6                      |
| Compute capability | 8.9                                |
| SM 数              | 24                                 |
| 全局显存           | 8,585,216,000 bytes                |
| L2 cache           | 33,554,432 bytes                   |

## 测试方法

### 1. 功能预检查

在做 profiling 前，先通过下面的命令确认 profiling 基于的是一个通过测试的构建：

```powershell
ctest --test-dir build -C Release --output-on-failure -R mmse_tests
```

结果：`100% tests passed`

### 2. 构建

```powershell
cmake --build build --config Release --target mmse_cuda_profile mmse_tests
```

### 3. Profiling 可执行文件

本报告新增了一个专用 benchmark 可执行文件：

- `G:\MMSE_CPP\bench\mmse_cuda_profile.cpp`

它的行为包括：

- 生成 `100` 个连续子帧的 LTE 结构化仿真数据
- 使用 full-band 20 MHz LTE 结构：
  - `1200` 个子载波
  - `100` 个 PRB
  - 每子帧 `14` 个 OFDM symbol
  - normal CP
  - `2x2` MIMO
  - `2` 层
  - `64QAM`（`mod_order = 6`）
- 构造与仓库当前 equalizer 路径一致的 CRS-bearing 子帧与 data RE 样本
- 测量：
  - cold `10 ms` 时延
  - warm `10 ms` 时延
  - 重复 `10 ms` 聚合时延
  - 重复 `100 ms` 聚合时延
- 查询：
  - 静态 CUDA kernel 资源属性
  - init 前后以及运行后的 CUDA 可见内存
  - 进程内存

### 4. 使用命令

基线时延与内存：

```powershell
.\build\Release\mmse_cuda_profile.exe
```

`equalize` kernel 的 Nsight Compute：

```powershell
ncu --target-processes all --set basic --kernel-name-base demangled -k "regex:.*equalize_stub_kernel.*" -c 1 --profile-from-start on -f -o .\build\ncu_equalize_report .\build\Release\mmse_cuda_profile.exe --warmup-subframes 0 --iters-10ms 1 --iters-100ms 0
```

`estimate` kernel 的 Nsight Compute：

```powershell
ncu --target-processes all --set basic --kernel-name-base demangled -k "regex:.*estimate_stub_kernel.*" -c 1 --profile-from-start on -f -o .\build\ncu_estimate_report .\build\Release\mmse_cuda_profile.exe --warmup-subframes 0 --iters-10ms 1 --iters-100ms 0
```

另外，在 `100 ms` 聚合 benchmark 期间，每 `200 ms` 做了一次运行时 `nvidia-smi` 采样。

## 输入与输出定义

### 每个 `1 ms` 子帧的输入结构

| 项目           | 数值          |
| -------------- | ------------- |
| RX 天线数      | 2             |
| TX 端口数      | 2             |
| 层数           | 2             |
| Symbols        | 14            |
| Subcarriers    | 1200          |
| PRB            | 100           |
| 每子帧 Data RE | 14,400        |
| 调制阶数       | 6 bits/symbol |

### 逻辑数据尺寸

输入网格是 2 个 RX 天线的 planar complex float：

- 每子帧输入字节数：
  - `2 RX × 2 (re/im) × 14 × 1200 × 4 bytes = 268,800 bytes`

输出是 2 层、14,400 RE 上的 `x_hat_re`、`x_hat_im`、`sinr`：

- 每子帧输出字节数：
  - `14,400 RE × 2 layers × 3 planes × 4 bytes = 345,600 bytes`

聚合尺寸：

| 时间窗 | 输入字节数 | 输出字节数 |
| ------ | ---------: | ---------: |
| 1 ms   |    268,800 |    345,600 |
| 10 ms  |  2,688,000 |  3,456,000 |
| 100 ms | 26,880,000 | 34,560,000 |

## 时延结果

数据来源：

- `G:\MMSE_CPP\build\mmse_cuda_profile_output.txt`

### 聚合时延

| 指标                        |         结果 |
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

### 结果解释

- 当前吞吐大约是每个 LTE `1 ms` 子帧 `12.07 ms`
- 当前 `10 ms` 聚合工作负载大约需要 `120.71 ms`
- 当前 `100 ms` 聚合工作负载大约需要 `1209.79 ms`
- 因而该实现对既定 LTE 处理目标而言还不是实时的

## GPU 运行时占用与利用率

### `nvidia-smi` 动态采样（100 ms 聚合 benchmark 期间）

数据来源：

- `G:\MMSE_CPP\build\nvidia_smi_samples.csv`

| 指标                   |        结果 |
| ---------------------- | ----------: |
| 采样数                 |          85 |
| GPU util 平均值        |      88.88% |
| GPU util 峰值          |         94% |
| GPU memory used 平均值 | 1432.64 MiB |
| GPU memory used 峰值   |    1467 MiB |

说明：

- 这是运行时设备级利用率，而不是单 kernel occupancy
- 执行期间的峰值可见显存远高于静态统计的 MMSE buffer，因此它同时包含 CUDA runtime / profiler / driver 常驻部分，以及 MMSE 工作集之外的其它 device 分配

### Nsight Compute：`equalize_stub_kernel`

数据来源：

- `G:\MMSE_CPP\build\ncu_equalize_report.ncu-rep`

| 指标                           |           结果 |
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

Nsight 警告：

- grid 太小，无法充分填满设备
- occupancy 受寄存器数限制
- 各 SM / SMSP / cache slice 存在工作负载不均衡

### Nsight Compute：`estimate_stub_kernel`

数据来源：

- `G:\MMSE_CPP\build\ncu_estimate_report.ncu-rep`

| 指标                           |          结果 |
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

Nsight 警告：

- 每个 block 只有 `1` 个 thread
- 整个 grid 只有 `1` 个 block
- GPU 严重低利用率

### Kernel profiling 主要结论

`equalize_stub_kernel` 不是主要时延来源。

主问题在于 `estimate_stub_kernel` 当前运行为：

- `grid = 1`
- `block = 1`

这意味着信道估计阶段实际上被串行化到一个 CUDA thread 上，这正是端到端时延达到每子帧 `12 ms` 的主要原因。

## 内存与缓冲区放置

### 逻辑放置

| 数据                | 放置位置                 | 类型             |
| ------------------- | ------------------------ | ---------------- |
| 输入仿真数据集      | Host memory              | pageable `float` |
| Staging grids       | Host memory              | pageable `float` |
| Transport grids     | Host memory              | pinned `int16_t` |
| 输出 host buffers   | Host memory              | pinned `float`   |
| Device input grids  | GPU global memory        | `int16_t`        |
| Device metadata     | GPU global memory        | `CudaGridMeta`   |
| Device sigma2 state | GPU global memory        | `float`          |
| Device estimates    | GPU global memory        | `float`          |
| Device outputs      | GPU global memory        | `float`          |
| Kernel temporaries  | registers / local memory | per-kernel       |

### 代码给出的静态 working-set 估计

数据来源：

- profiling executable output

| 项目                  |     每 slot | 总计（3 slots） |
| --------------------- | ----------: | --------------: |
| Host pageable buffers | 1,022,492 B |     3,067,476 B |
| Host pinned buffers   |   537,600 B |     1,612,800 B |
| Device buffers        | 1,256,160 B |     3,768,480 B |

导出总量：

| 项目                      |       尺寸 |
| ------------------------- | ---------: |
| Dataset 输入 footprint    | 25.635 MiB |
| Host pageable ring 总存储 |  2.925 MiB |
| Host pinned ring 总存储   |  1.538 MiB |
| Device ring 总存储        |  3.594 MiB |

### CUDA 内存观察

来自 benchmark：

| 指标                                  |            结果 |
| ------------------------------------- | --------------: |
| init 前 GPU free                      | 7,443,841,024 B |
| init 后 GPU free                      | 7,439,646,720 B |
| 运行后 GPU free                       | 6,003,097,600 B |
| init 时可见 GPU 分配增量              |     4,194,304 B |
| 相对 init 后的运行后可见 GPU 分配增量 | 1,436,549,120 B |

重要说明：

- 静态 MMSE device buffers 约为 `3.59 MiB`
- `cudaMemGetInfo` 给出的更大运行后增量，并不能只由 MMSE buffers 本身解释
- 它很可能同时包含 CUDA runtime 状态、allocator 保留、WDDM / driver 常驻行为，以及 profiling 副作用
- 因此，代码级 buffer accounting 才是可信的 MMSE 工作集估计；`cudaMemGetInfo` 更适合理解整个 context 常驻量，而不是纯算法存储

## 寄存器 / Shared / Local Memory 摘要

| Kernel                 | Registers/thread | Static shared | Local bytes/thread | 说明                                                  |
| ---------------------- | ---------------: | ------------: | -----------------: | ----------------------------------------------------- |
| `estimate_stub_kernel` |               54 |           0 B |           40,000 B | 由于大规模 thread-local 数组，local memory 足迹非常大 |
| `equalize_stub_kernel` |               64 |           0 B |                0 B | occupancy 主要受寄存器数限制                          |

解释：

- `estimate_stub_kernel` 因为只发射一个 CUDA thread，却声明了大规模逐线程临时数组，所以大量临时状态被 spill 到 local memory
- `equalize_stub_kernel` 相对紧凑，occupancy 限制主要来自寄存器，而不是 shared memory

## 输出产物

生成文件：

- 基线 profile 文本：
  - `G:\MMSE_CPP\build\mmse_cuda_profile_output.txt`
- 带运行时采样的 profile 文本：
  - `G:\MMSE_CPP\build\mmse_cuda_profile_runtime_output.txt`
- 运行时 `nvidia-smi` 采样：
  - `G:\MMSE_CPP\build\nvidia_smi_samples.csv`
- Nsight Compute 报告：
  - `G:\MMSE_CPP\build\ncu_equalize_report.ncu-rep`
  - `G:\MMSE_CPP\build\ncu_estimate_report.ncu-rep`

## 总体结论

当前算法实现已经可以在 GPU 上功能性运行，但性能完全受一个严重欠并行化的 estimate 阶段支配。

当前测得性能：

- 每个 LTE `1 ms` 子帧约 `12.07 ms`
- 每个 `10 ms` 聚合窗口约 `120.71 ms`
- 每个 `100 ms` 聚合窗口约 `1209.79 ms`

主要瓶颈不是 equalization kernel，而是 estimate kernel，它当前的运行形态是：

- `1 block`
- `1 thread`
- `18.85 ms` kernel time
- `2.08%` achieved occupancy

如果目标是实时或近实时 LTE 处理，第一优先级工程目标应该是：

1. 并行化 `estimate_stub_kernel`
2. 去掉单线程 estimate 路径中的大规模 thread-local 数组
3. 在动 equalization kernel 之前先重新测量
