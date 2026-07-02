# MMSE CUDA Profiling 报告（2026-07-03）

## 作用范围

本报告更新 `G:\MMSE_CPP` 当前 checkout 在本机上的 CUDA profiling 结论。

边界保持不变：

- 当前公开 MMSE API 仍然是“一次调用处理一个 LTE `1 ms` 子帧”
- `10 ms` / `100 ms` 数字仍然表示连续重复调用 `1 ms` 接口的聚合开销
- 本报告测量的是当前实现原样行为，不是假设未来存在 batched `10 ms` / `100 ms` 接口

这次更新的关键原因是：2026-06-24 报告中的核心结论已经失效。当前 hot path 不再使用单线程 `estimate_stub_kernel`，而是拆成：

- `estimate_residual_kernel`
- `estimate_channel_kernel`
- `equalize_stub_kernel`

因此旧报告里“estimate 阶段被 `1 block × 1 thread` 串行化”的判断，不再代表当前代码。

## 测试环境

| 项目               | 数值                               |
| ------------------ | ---------------------------------- |
| 日期               | 2026-07-03                         |
| OS                 | Windows                            |
| 仓库               | `G:\MMSE_CPP`                      |
| 构建               | `Release`                          |
| GPU                | NVIDIA GeForce RTX 4060 Laptop GPU |
| Driver / CUDA      | 560.94 / 12.6                      |
| Compute capability | 8.9                                |
| SM 数              | 24                                 |
| 全局显存           | 8,585,216,000 bytes                |
| L2 cache           | 33,554,432 bytes                   |

## 验证与采集命令

功能预检查：

```powershell
ctest --test-dir build -C Release --output-on-failure -R mmse_tests
```

构建：

```powershell
cmake --build build --config Release --target mmse_cuda_profile mmse_tests
```

基线 profiling：

```powershell
.\build\Release\mmse_cuda_profile.exe --warmup-subframes 5 --iters-10ms 5 --iters-100ms 5 | Tee-Object -FilePath .\build\mmse_cuda_profile_output_2026_07_03.txt
```

Nsight Compute：

```powershell
ncu --target-processes all --set basic --kernel-name-base demangled -k "regex:.*estimate_residual_kernel.*" -c 1 --profile-from-start on -f -o .\build\ncu_estimate_residual_2026_07_03 .\build\Release\mmse_cuda_profile.exe --warmup-subframes 0 --iters-10ms 1 --iters-100ms 0
```

```powershell
ncu --target-processes all --set basic --kernel-name-base demangled -k "regex:.*estimate_channel_kernel.*" -c 1 --profile-from-start on -f -o .\build\ncu_estimate_channel_2026_07_03 .\build\Release\mmse_cuda_profile.exe --warmup-subframes 0 --iters-10ms 1 --iters-100ms 0
```

```powershell
ncu --target-processes all --set basic --kernel-name-base demangled -k "regex:.*equalize_stub_kernel.*" -c 1 --profile-from-start on -f -o .\build\ncu_equalize_2026_07_03 .\build\Release\mmse_cuda_profile.exe --warmup-subframes 0 --iters-10ms 1 --iters-100ms 0
```

## 输入与逻辑数据规模

每个 `1 ms` 子帧仍是 full-band `20 MHz LTE`：

- `1200` subcarriers
- `100` PRB
- `14` OFDM symbols
- `2x2` MIMO
- `2` layers
- `64QAM`
- `14,400` valid data RE / subframe

逻辑数据尺寸：

| 时间窗   |   输入字节数 |   输出字节数 |
| -------- | -----------: | -----------: |
| `1 ms`   |    `268,800` |    `345,600` |
| `10 ms`  |  `2,688,000` |  `3,456,000` |
| `100 ms` | `26,880,000` | `34,560,000` |

## 当前时延结果

数据来源：

- `G:\MMSE_CPP\build\mmse_cuda_profile_output_2026_07_03.txt`

### 聚合时延

| 指标                          |           结果 |
| ----------------------------- | -------------: |
| Cold `10 ms` aggregate        |  `3.979700 ms` |
| Cold per-subframe             |  `0.397970 ms` |
| Warm `10 ms` aggregate        |  `3.255600 ms` |
| Warm per-subframe             |  `0.325560 ms` |
| `10 ms` average               |  `3.566380 ms` |
| `10 ms` p50                   |  `3.583500 ms` |
| `10 ms` p95                   |  `3.593200 ms` |
| `10 ms` min                   |  `3.417500 ms` |
| `10 ms` max                   |  `3.693400 ms` |
| `10 ms` average per-subframe  |  `0.356638 ms` |
| `100 ms` average              | `35.780500 ms` |
| `100 ms` p50                  | `35.845100 ms` |
| `100 ms` p95                  | `36.538000 ms` |
| `100 ms` min                  | `34.629100 ms` |
| `100 ms` max                  | `36.548500 ms` |
| `100 ms` average per-subframe |  `0.357805 ms` |

### 与 2026-06-24 基线对比

| 指标                          |   2026-06-24 |    2026-07-03 |            变化 |
| ----------------------------- | -----------: | ------------: | --------------: |
| `10 ms` average per-subframe  | `12.0710 ms` | `0.356638 ms` | 约 `33.8x` 提速 |
| `100 ms` average per-subframe | `12.0979 ms` | `0.357805 ms` | 约 `33.8x` 提速 |

当前 full-band GPU 路径已经从“显著非实时”进入“单次 `1 ms` 子帧约 `0.36 ms`”的量级。

## Host / GPU 阶段拆分

同一份 `10 ms` 平均测量下，当前 `MmseGpuHostProfileSnapshot` 给出的每子帧均值如下：

| 指标                                  |         结果 |
| ------------------------------------- | -----------: |
| `host.phase.total_host_us`            | `313.288 us` |
| `host.phase.quantize_us`              |  `26.572 us` |
| `host.phase.layout_build_us`          |  `33.408 us` |
| `host.phase.grid_meta_pack_us`        |  `23.116 us` |
| `host.phase.grid_h2d_us`              |  `33.384 us` |
| `host.phase.estimate_launch_us`       |  `21.012 us` |
| `host.phase.equalize_launch_us`       |   `5.714 us` |
| `host.phase.outputs_d2h_us`           |  `18.110 us` |
| `host.phase.scratch_d2h_us`           | `124.768 us` |
| `host.phase.output_stage_us`          |  `23.756 us` |
| `host.phase.final_sync_us`            |   `3.448 us` |
| `host.phase.estimate_gpu_us`          |  `75.784 us` |
| `host.phase.estimate_residual_gpu_us` |  `61.732 us` |
| `host.phase.estimate_channel_gpu_us`  |  `14.051 us` |
| `host.phase.equalize_gpu_us`          |   `8.340 us` |
| `host.phase.stream_gpu_us`            | `154.234 us` |

当前含义很明确：

- `equalize` 数学本体已经只有个位数微秒
- `estimate` GPU 总开销也已经降到每子帧约 `76 us`
- 更大的固定成本在 host 侧 staging / packing / D2H / sync，尤其是 `scratch_d2h_us`

## Nsight Compute 结果

### `estimate_residual_kernel`

数据来源：

- `G:\MMSE_CPP\build\ncu_estimate_residual_2026_07_03.ncu-rep`

| 指标                         |            结果 |
| ---------------------------- | --------------: |
| Kernel duration              |      `37.15 us` |
| Grid size                    |     `64 blocks` |
| Block size                   |   `256 threads` |
| Threads launched             |        `16,384` |
| Registers per thread         |            `37` |
| Theoretical occupancy        |       `100.00%` |
| Achieved occupancy           |        `38.45%` |
| Achieved active warps per SM |         `18.46` |
| Waves per SM                 |          `0.44` |
| Compute throughput           | `1.28% of peak` |
| DRAM throughput              | `1.00% of peak` |

主要观察：

- 这已经不是单线程 kernel
- 但 grid 仍偏小，Nsight 明确提示无法填满设备
- 当前更像“小工作量 + launch/调度开销占比高”的残差统计阶段

### `estimate_channel_kernel`

数据来源：

- `G:\MMSE_CPP\build\ncu_estimate_channel_2026_07_03.ncu-rep`

| 指标                         |             结果 |
| ---------------------------- | ---------------: |
| Kernel duration              |        `9.41 us` |
| Grid size                    |     `263 blocks` |
| Block size                   |    `256 threads` |
| Threads launched             |         `67,328` |
| Registers per thread         |             `45` |
| Theoretical occupancy        |         `83.33%` |
| Achieved occupancy           |         `76.18%` |
| Achieved active warps per SM |          `36.57` |
| Waves per SM                 |           `2.19` |
| Compute throughput           | `49.52% of peak` |
| DRAM throughput              |  `4.73% of peak` |

主要观察：

- `estimate_channel_kernel` 已经具备正常的并行 launch 形态
- occupancy 接近理论值，说明旧报告中的“estimate 串行化”问题已经被根本移除
- 当前它本身不是主时延来源

### `equalize_stub_kernel`

数据来源：

- `G:\MMSE_CPP\build\ncu_equalize_2026_07_03.ncu-rep`

| 指标                         |             结果 |
| ---------------------------- | ---------------: |
| Kernel duration              |        `8.51 us` |
| Grid size                    |      `57 blocks` |
| Block size                   |    `256 threads` |
| Threads launched             |         `14,592` |
| Registers per thread         |             `70` |
| Theoretical occupancy        |         `50.00%` |
| Achieved occupancy           |         `41.35%` |
| Achieved active warps per SM |          `19.85` |
| Waves per SM                 |           `0.79` |
| Compute throughput           | `14.13% of peak` |
| DRAM throughput              | `45.69% of peak` |

主要观察：

- `equalize` 仍然是一个很短的小 kernel
- occupancy 主要继续受寄存器数限制
- 但它的绝对时长只有 `8.51 us`，不是当前端到端瓶颈

## 内存与缓冲区

静态 accounting 来自 benchmark 输出：

| 项目                   |         尺寸 |
| ---------------------- | -----------: |
| Dataset 输入 footprint | `25.635 MiB` |
| Host pageable total    |  `1.003 MiB` |
| Host pinned total      |  `1.923 MiB` |
| Device total           |  `3.978 MiB` |

CUDA 可见内存观察：

| 指标                         |              结果 |
| ---------------------------- | ----------------: |
| init 前 GPU free             | `7,443,841,024 B` |
| init 后 GPU free             | `7,437,549,568 B` |
| 运行后 GPU free              | `7,437,549,568 B` |
| init 时可见 GPU 分配增量     |     `6,291,456 B` |
| 相对 init 后的运行后额外增量 |             `0 B` |

当前解释：

- 当前 static MMSE device working set 仍然是几 MiB 量级
- 与 2026-06-24 报告里那种运行后大额额外增量相比，当前这次重测没有看到持续扩张
- 对当前代码，buffer accounting 比 `cudaMemGetInfo` 更适合描述算法工作集

## 当前结论

### 已确认的结论

1. 2026-06-24 报告中的主结论已经过时。
   当前热路径不再存在 `estimate_stub_kernel` 的 `1 block × 1 thread` 串行瓶颈。

2. 当前 full-band `1 ms` 子帧 GPU 成本约为 `0.36~0.37 ms`。
   这与 2026-07-01 预算报告里 `0.346782 ms / subframe` 的量级一致，说明当前实现已经稳定进入同一性能区间。

3. 当前端到端成本不再主要由均衡数学本体支配。
   `estimate_channel_kernel` 和 `equalize_stub_kernel` 都已经是微秒级 kernel。

4. 当前更值得关注的是框架性开销。
   从 host profile 看，下一优先级更像是：
   - `scratch_d2h`
   - host staging / packing
   - launch / sync 聚合成本

### 如果继续优化 `mmse_cuda_profile`

在当前版本上，优先顺序应改为：

1. 先检查 `scratch_d2h_us` 为什么仍接近 `125 us / subframe`
2. 再检查 output staging 与 host packing 是否还能压缩
3. 只有当这些固定成本压不下去时，再回头微调 kernel 内部细节

## 输出产物

- 基线 profile 文本：
  - `G:\MMSE_CPP\build\mmse_cuda_profile_output_2026_07_03.txt`
- Nsight Compute 报告：
  - `G:\MMSE_CPP\build\ncu_estimate_residual_2026_07_03.ncu-rep`
  - `G:\MMSE_CPP\build\ncu_estimate_channel_2026_07_03.ncu-rep`
  - `G:\MMSE_CPP\build\ncu_equalize_2026_07_03.ncu-rep`
- 历史基线报告：
  - `G:\MMSE_CPP\docs\mmse_cuda_profile_report_2026-06-24.md`
