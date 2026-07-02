# LTE MMSE 预算报告（2026-07-01）

## 作用范围

本报告回答当前 `G:\MMSE_CPP` checkout 下的一个明确问题：

- 如果一个 `10 ms` LTE 处理窗口需要完成完整物理信道集合
  （`PBCH`、`PDSCH`、`PDCCH`、`PCFICH`）的译码，
- 并且当前 CE/MMSE 模块可能会在该窗口内被调用 `5~6` 次，
- 那么这个模块本身应消耗多少时延预算，
- 以及下一步优化应该继续放在 CE/MMSE kernel 路径内，还是应转向更高层调用拓扑。

## 解释边界

本报告的重要边界如下：

- 当前公开 MMSE API 仍然是“一次调用处理一个 LTE `1 ms` 子帧”
- 因而任何 `10 ms` 判断都必须按“重复调用 `1 ms` 子帧接口”的方式推导，除非接口本身发生变化
- 本报告使用当前 wrapper 接口面：
  - `run_pbch(...)`
  - `run_pcfich(...)`
  - `run_pdcch(...)`
  - `PDSCH` 对应的通用 `run(...)`

## 当前链路认识

在当前仓库边界下，MMSE 模块覆盖：

- RE 提取
- 基于 CRS 的信道估计
- MMSE 均衡
- 均衡后软符号与 `SINR` 透传

它不覆盖以下这些下游重译码阶段：

- `PBCH` 解扰 / 速率恢复 / 卷积译码 / `MIB`
- `PDCCH` 盲检索 / 信道译码 / `DCI`
- `PDSCH` 解扰 / 速率恢复 / Turbo 译码 / `MAC PDU`
- `PCFICH` 最终 `CFI` 译码

因此，CE/MMSE 这一块必须给后续译码阶段留出足够的时延预算。

## 当前测量基线

### 1. Full-band GPU profile

使用命令：

```powershell
.\build\Release\mmse_cuda_profile.exe --warmup-subframes 5 --iters-10ms 5 --iters-100ms 0
```

在本机上测得的 full-band `1 ms` 子帧成本：

- `timing.avg_10ms_per_subframe_ms = 0.346782 ms`

测得的 host/GPU 拆分：

- `host.phase.total_host_us = 308.93 us`
- `host.phase.estimate_gpu_us = 73.86 us`
- `host.phase.equalize_gpu_us = 6.66 us`
- `host.phase.stream_gpu_us = 157.44 us`

当前含义：

- `equalize` 已经只占总开销中的很小一部分
- `estimate` 不再是早期 stub 版本中 `10+ ms` 的串行大瓶颈
- 当前开销更多由每次调用的框架性开销、staging 和同步主导，而不是均衡数学本身

### 2. 分信道 wrapper benchmark

本报告新增的 benchmark 目标：

- [mmse_channel_budget.cpp](G:\MMSE_CPP\bench\mmse_channel_budget.cpp)

使用命令：

```powershell
cmd /c "G:\MMSE_CPP\build\Release\mmse_channel_budget.exe"
```

#### CPU 路径：独立调用基线

| 指标             |       平均值 |         P95 |
| ---------------- | -----------: | ----------: |
| `PBCH`           |  `272.96 us` |  `288.7 us` |
| `PCFICH`         |  `268.95 us` |  `287.6 us` |
| `PDCCH`          |  `292.56 us` |  `310.2 us` |
| `PDSCH`          |  `387.29 us` |  `437.9 us` |
| 六次独立调用总计 | `1916.21 us` | `2088.4 us` |

#### GPU 路径：独立调用基线

| 指标             |       平均值 |         P95 |
| ---------------- | -----------: | ----------: |
| `PBCH`           |  `357.27 us` |  `407.2 us` |
| `PCFICH`         |  `354.43 us` |  `435.2 us` |
| `PDCCH`          |  `358.46 us` |  `404.7 us` |
| `PDSCH`          |  `418.31 us` |  `484.1 us` |
| 六次独立调用总计 | `2262.90 us` | `2473.5 us` |

当前观测到的提取 RE 数：

- `PBCH`: `240 RE`
- `PCFICH`: `16 RE`
- `PDCCH`: `3228 RE`
- `PDSCH`: `14400 RE / layer`

## 这些数字意味着什么

### 1. CE/MMSE kernel 路径已经进入可用区间

之前的问题是：CE/MMSE 数学路径本身是否还需要大规模 kernel 级优化。

对于当前代码和本机而言，答案是：

- 不是第一优先级

原因：

- 一个 full-band `1 ms` GPU 调用现在大约只有 `0.35 ms`
- 即使是较大的 `PDSCH` wrapper 调用也只有 `0.42 ms` 左右
- profiler 里 `equalize` 本体已经只有个位数微秒

这意味着继续做 kernel 内部微雕，不是当前收益最高的下一步。

### 2. 当前真正的问题是独立调用次数过多

当前六次调用聚合为：

- 平均 `2.263 ms`
- P95 `2.474 ms`

这已经远好于早期 `12 ms / subframe` 的基线，但对于一个干净的全链路 `10 ms` 译码预算来说仍然偏大。

主要问题在于多次重复了：

- layout build
- grid/meta packing
- wrapper validation
- channel estimation launch/state handling
- output staging / synchronization

而这些工作都发生在同一个 LTE 子帧上下文上的多个物理信道 wrapper 调用之间。

## 推荐的 LTE 10 ms 预算

### 针对完整 10 ms 译码窗口的推荐 CE/MMSE 预算

对于 `10 ms` 端到端 LTE 物理信道译码目标，我建议：

- 理想 CE/MMSE 预算：`<= 1.0 ms`
- 可接受的 CE/MMSE 上限：`<= 1.5 ms`
- 长期设计不应接受高于：`1.5 ms`

推导依据：

- 当前测得的六次调用路径 `~2.26 ms`，会占用整个 `10 ms` 墙钟预算的大约 `22.6%`
- 对一个后面还要承担大量信道译码的模块来说，这个占比太高
- `<= 1.0 ms` 可以把 MMSE 块压到总帧预算的大约 `10%`
- `<= 1.5 ms` 仍然能给控制信道和数据信道译码留下相对合理的空间

### 相对当前状态需要的改进量

使用当前测得的 GPU 六次调用聚合值：

- 当前平均：`2.263 ms`
- 目标平均：`<= 1.0 ms`
  - 需要下降约 `55.8%`
  - 需要约 `2.26x` 提速
- 上限平均：`<= 1.5 ms`
  - 需要下降约 `33.7%`
  - 需要约 `1.51x` 提速

## 推荐的下一步优化方向

### 主推荐

第一步**不要**回到 kernel 内部微雕。

优先做的是：

- 每个 LTE 子帧只做一次 channel estimate
- 在所有信道特定的提取 / 均衡路径之间复用这一份 estimate

### 为什么这是正确方向

在同一个 LTE 子帧里，`PBCH`、`PCFICH`、`PDCCH` 和 `PDSCH` 并不是彼此独立的无线场景。它们共享：

- 同一份 FFT 网格
- 同一套 CRS 上下文
- 同一个 cell id
- 同一组接收天线
- 同一个子帧时序

所以收益最高的设计，不应该是 `5~6` 次完全独立的 MMSE wrapper 调用，而应该更接近：

1. 子帧只 stage 一次
2. channel estimate 只做一次
3. 这份 estimate 复用于：
   - `PBCH`
   - `PCFICH`
   - `PDCCH`
   - `PDSCH`
4. 后面只保留每个信道自身的 RE layout / equalized output extraction

### 下一版设计的实际目标

下一轮实现应优先减少：

- 重复的 channel-estimation 工作
- 重复的 host/device setup 和同步

并验证聚合路径是否能从：

- `~2.26 ms`

降到：

- 先到 `<= 1.5 ms`
- 再视下游译码预算是否仍紧张，继续冲到 `<= 1.0 ms`

## 建议的工程里程碑

### Milestone 1

引入一个“子帧作用域”的 CE 工件：

- 一份 staged grid
- 一份 channel-estimate 结果
- 一份可复用的 per-subframe 运行时状态

### Milestone 2

暴露内部的多信道复用流程：

- `PBCH` 从缓存 estimate 提取
- `PCFICH` 从缓存 estimate 提取
- `PDCCH` 从缓存 estimate 提取
- `PDSCH` 从缓存 estimate 提取

### Milestone 3

重新 benchmark 两种形态：

- 当前独立调用路径
- shared-estimate 多信道路径

并比较：

- 总平均值
- P95
- host 侧 staging 成本
- estimate 复用收益

### Milestone 4

只有当 shared-estimate 路径仍高于 `1.5 ms` 时，才回头做：

- estimate-stage kernel 优化
- 同步裁剪
- 输出 staging 缩减

## 最终判断

对于当前机器和当前代码：

- CE/MMSE 模块在**单次调用的 kernel / 数学层面**已经足够好
- 但在**多次调用的链路级预算层面**还不够好

因此下一步目标应该是：

- **不是** “不惜一切代价把单次调用再压快”
- **而是** “把完整 `5~6` 次调用聚合压到 `1.5 ms` 以下，最好接近 `1.0 ms`”

这才是面向 `10 ms` 物理信道全链路译码目标时正确的优化方向。

## Refactor 结果更新

随后，子帧作用域的 shared-estimate refactor 被实现，并在同一台机器上针对一个“同子帧六次调用”的代表性形态重新测量：

- `PBCH`
- `PCFICH`
- `PDCCH`
- `PDSCH`
- `PDCCH`
- `PDSCH`

### Shared-estimate benchmark 结果

数据来源：

- [mmse_channel_budget.cpp](G:\MMSE_CPP\bench\mmse_channel_budget.cpp)

Refactor 后测得：

#### CPU 路径：同子帧 shared-estimate 形态

| 指标     |      平均值 |        P95 |
| -------- | ----------: | ---------: |
| `PBCH`   |  `74.54 us` |  `89.8 us` |
| `PCFICH` |  `74.15 us` |  `93.1 us` |
| `PDCCH`  | `101.28 us` | `125.0 us` |
| `PDSCH`  | `175.76 us` | `202.8 us` |
| 六次聚合 | `719.70 us` | `803.1 us` |

#### GPU 路径：同子帧 shared-estimate 形态

| 指标     |       平均值 |         P95 |
| -------- | -----------: | ----------: |
| `PBCH`   |  `155.01 us` |  `199.8 us` |
| `PCFICH` |  `146.77 us` |  `197.7 us` |
| `PDCCH`  |  `157.41 us` |  `200.8 us` |
| `PDSCH`  |  `215.49 us` |  `270.3 us` |
| 六次聚合 | `1020.30 us` | `1162.7 us` |

### 前后对比

GPU 六次调用聚合从：

- 改前：`2262.90 us`
- 改后：`1020.30 us`

改进幅度：

- 绝对下降：`1242.60 us`
- 相对下降：约 `54.9%`
- 加速比：约 `2.22x`

### 收益来源

这次收益主要来自消除同子帧内重复工作：

- 重复的 grid staging / cache preparation
- 重复的 channel-estimate 生成
- 重复的 `sigma2` 更新路径

收益**并不是**来自进一步的 equalizer kernel 微优化。

### 实现后的最终预算判断

这次 refactor 已经满足 phase-1 的硬目标：

- 目标：`gpu.six_calls.avg_us <= 1500`
- 实际：`gpu.six_calls.avg_us = 1020.30`

但还没有完全达到 stretch goal：

- stretch goal：`<= 1000 us`
- 当前差距：约 `20.3 us`

最终判断：

- CE/MMSE 模块现在已经进入可接受的 `10 ms` 全链路预算范围
- 下一步优化不再是 schedule 可行性的硬性要求
- 如果还要继续做，应把重点放在剩余的 host/device 框架性开销，而不是 equalization 数学本体
