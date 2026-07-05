# MMSE_CPP 算法原理与优化方法详解

## 1. 文档目标

本文档面向当前 `G:\MMSE_CPP` 仓库，系统说明项目中 LTE 下行信道估计与 MMSE 均衡模块的：

- 协议位置
- 输入输出接口语义
- 数学原理
- 当前 CPU / GPU 实现结构
- 已经落地并验证过的优化方法
- 当前明确不建议优先继续投入的方向

本文档不重复 API 手册，也不替代快速开始页。它更偏向“实现原理 + 性能工程说明”，用于帮助后续开发、性能分析和系统集成。

## 2. 项目在 LTE 接收链中的位置

当前项目是 LTE 下行物理层中的一个中间模块，主要完成：

1. 从频域资源网格中提取目标信道的 RE
2. 基于 CRS 做信道估计
3. 对提取出来的 RE 做 MMSE 均衡
4. 输出均衡后的复数符号和每个 RE 的 `SINR`

它当前覆盖的公开信道入口包括：

- `PDSCH` 的通用 `run(...)`
- `PBCH` 的 `run_pbch(...)`
- `PCFICH` 的 `run_pcfich(...)`
- `PDCCH` 的 `run_pdcch(...)`
- `2 Tx port PDCCH transmit-diversity` 的 `run_pdcch_td(...)`

对应公开运行时类型在 [include/mmse/mmse_equalizer.h](G:\MMSE_CPP\include\mmse\mmse_equalizer.h)。

这意味着本项目的职责边界是“equalized RE 级别输出”，而不是完整译码链。它不直接负责：

- PBCH 卷积译码 / MIB 恢复
- PDCCH 盲检索 / DCI 译码
- PDSCH 速率恢复 / HARQ 软合并 / Turbo 译码
- PCFICH 的最终 CFI 判决

不过，当前仓库已经额外提供了一层面向 `PDSCH` 的下游 helper：

- 基于 equalized `x_hat + sinr` 生成 max-log `LLR`
- 基于 LTE Gold 序列做 `PDSCH` 解扰
- 显式 caller-owned `LLR` 输出面
- 显式 `PdschDescramblingPlanCache`

这层 helper 仍然不等于真实存在的 `PDSCH` downstream context，也没有把仓库边界扩展到
完整 `PDSCH` 译码链。

因此，本项目既要保证信道估计和均衡结果正确，也要给后续软解调/译码保留足够时延预算。

## 3. 统一接口模型

从运行时结构看，CPU 和 GPU 两端共享同一套抽象：

- 输入频域网格：`PlanarGridViewF32`
- 运行描述符：`ExtractDescriptor`
- 输出视图：`EqualizerOutputView`

统一入口是：

```cpp
MmseStatus run(const PlanarGridViewF32& grid,
               const ExtractDescriptor& desc,
               EqualizerOutputView& out);
```

其上又包装出 `PBCH / PCFICH / PDCCH / PDCCH-TD` 的专用 DTO 入口。

这里最重要的设计点有两个。

第一，频域网格是“planar”布局，即实部和虚部分开存储，而不是交织复数布局。这一点直接服务于：

- CPU 侧 AVX2 向量化
- GPU 侧 coalesced memory access
- host/device 运输时的简洁拷贝路径

第二，输出不只是均衡后的复数符号，还包含每个 RE 的 `SINR`。这是项目中非常关键的契约，因为后续软解调对可靠度尺度敏感，只有 `x_hat + SINR` 成对输出，LLR 路径才完整。

## 4. 约束和支持边界

当前实现不是任意 LTE 配置的完全通用版本，而是围绕当前仓库支持面做了明确收敛。

`descriptor_supported(...)` 在 [src/mmse_equalizer_cpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_cpu.cpp) 中对配置做统一过滤，核心边界包括：

- `n_rx_ant == 2`
- `n_layers` 只支持 `1` 或 `2`
- `n_tx_ports` 在不同信道类型下有明确限制
- `pmi == -1`
- `mod_order` 只支持 `2 / 4 / 6 / 8`
- `TM4` 当前不在这条普通 MMSE 路径内支持
- `PBCH / PCFICH / PDCCH / PDSCH` 各自有独立的协议边界检查

这类收敛不是缺陷，而是当前工程策略的一部分：先在明确配置范围内把算法、性能和 API 做扎实，再逐步扩展边界。

## 5. 资源元素布局算法

### 5.1 为什么要先做 RE layout

信道估计和均衡都必须知道“哪些 RE 需要输出”。不同逻辑信道使用的 RE 集合不同：

- `PDSCH` 是数据区 RE，跳过 CRS
- `PDCCH` 只在控制区，且还要剔除 PCFICH/PHICH/CRS 等保留 RE
- `PBCH` 只在中心 6 PRB、特定 4 个符号
- `PCFICH` 只有固定 4 个 REG，对应 16 个有效 RE

因此，项目并不是简单地“全网格全 RE 均衡再过滤”，而是先构建目标 RE 布局，再只对目标 RE 输出。

### 5.2 当前布局函数

布局构建集中在 [src/mmse_equalizer_cpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_cpu.cpp)：

- `build_data_re_layout(...)`
- `build_pdcch_re_layout(...)`
- `build_pbch_re_layout(...)`
- `build_pcfich_re_layout(...)`
- `build_channel_re_layout(...)`

核心输出结构是 `ReLayout`，定义在 [src/internal/mmse_internal.h](G:\MMSE_CPP\src\internal\mmse_internal.h)，里面包含：

- `grid_indices`
- `output_slot_by_grid_re`
- `prb_segment_offsets`
- `n_re`
- `n_segments`

这套结构有三个作用：

1. 它把“物理资源网格中的 RE 索引”映射为“输出数组槽位”
2. 它给 CPU/GPU 统一提供可复用的布局元数据
3. 它给后续优化提供了稳定的“动态切片”和“按段处理”边界

### 5.3 各信道布局特点

`PDSCH` 使用 `build_data_re_layout(...)`：

- 从 `start_symbol` 开始
- 按 PRB bitmap 遍历资源
- 跳过 CRS RE

`PDCCH` 使用 `build_pdcch_re_layout(...)`：

- 只遍历 `control_symbol_count`
- 跳过 CRS
- 跳过控制区中被保留的 RE

`PBCH` 使用 `build_pbch_re_layout(...)`：

- 固定中心 6 PRB
- 固定 4 个 PBCH 符号
- 前两个 PBCH 符号中保留 CRS 空洞

`PCFICH` 使用 `build_pcfich_re_layout(...)`：

- 根据 cell ID 求出 PCFICH 的 REG 坐标
- 每个 REG 中剔除 CRS 位置

这一层的意义是：上层 wrapper 只需表达“我要哪个 LTE 信道”，而底层能把它变成明确的 RE 集合。

## 6. 信道估计算法

### 6.1 总体流程

当前 CPU 和 GPU 路径都遵循同一算法：

1. 基于 CRS 计算 LS 信道估计
2. 在每个 CRS 符号内做频域线性插值
3. 再在 OFDM 符号维度做时域插值
4. 得到完整的 `H(tx, rx, symbol, sc)`

CPU 参考实现入口是 [src/mmse_equalizer_cpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_cpu.cpp) 中的：

```cpp
void estimate_channel(const PlanarGridViewF32& grid,
                      const ExtractDescriptor& desc,
                      HGridStorage& h_full,
                      float& sigma2_estimate)
```

GPU 对应拆成两段：

- `estimate_residual_kernel`
- `estimate_channel_kernel`

在 [src/mmse_cuda_runtime.cu](G:\MMSE_CPP\src\mmse_cuda_runtime.cu)。

### 6.2 LS 估计

对每个 CRS pilot，项目当前做的是标准 LS：

```text
H_ls = Y_pilot * conj(CRS_pilot)
```

CPU 侧具体就在 `estimate_channel(...)` 的这段逻辑里：

- 根据 `cell_id / tx / symbol / pilot` 找出 CRS 位置
- 读取 `grid_at(...)`
- 乘上 `conj(crs_value(...))`

这一步的输出是离散的 pilot 域估计，不足以直接给所有数据 RE 用，因此后面要做插值。

### 6.3 频域插值

对每个 CRS 符号，项目会在每个 pilot 对之间做线性插值，把稀疏 pilot 估计扩展到所有子载波位置。

CPU 路径中这部分逻辑表现为：

- 找相邻左右 pilot
- 对边界位置做端点延拓
- 对中间位置做 `linear_interp(...)`

这一步得到的是“在 CRS 符号上的全频域信道”。

### 6.4 时域插值

随后，对每个子载波，再沿 LTE normal CP 的 14 个 OFDM symbol 做时域插值。

CPU 当前做法是：

- 只使用 `kCrsSymbols = {0, 4, 7, 11}`
- 找当前 symbol 的相邻 CRS symbol
- 调用 `lerp_symbol(...)`

最终得到完整的 `h_full`，它是后续均衡阶段共享的数据基础。

## 7. 噪声方差估计与平滑

### 7.1 sigma2 的作用

MMSE 核心矩阵里有：

```text
A = H^H H + sigma2 I
```

因此 `sigma2` 不是附属元数据，而是直接影响均衡权重和 `SINR` 的关键量。

### 7.2 当前估计方法

当前项目使用基于 pilot 残差的近似估计。

CPU 路径在 `estimate_channel(...)` 中对每个中间 pilot 计算：

- 当前 pilot 的 LS 结果
- 左右相邻 pilot 的平滑值
- 两者差值的能量

对应代码在 [src/mmse_equalizer_cpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_cpu.cpp) 的这段：

- `smooth = 0.5 * (left + right)`
- `sigma2_estimate += |ls - smooth|^2`

最后按有效残差数做均值化。

### 7.3 IIR 平滑

单子帧估计会抖动，因此项目引入了 `Sigma2State`，通过：

```cpp
update_sigma2_state(...)
```

做每个 cell 的一阶 IIR 平滑：

```text
state = alpha * previous + (1 - alpha) * current
```

同时强制不低于 `sigma2_min`。

这一步的工程意义很大：

- 避免 sigma2 过小导致数值不稳
- 避免相邻子帧间 `SINR` 抖动过大
- 为 host-owned 和 device-owned 两种 GPU 状态提供统一语义

### 7.4 device-owned sigma2

GPU 路径后续进一步支持了 `device-owned sigma2 state`。这意味着：

- `sigma2` 状态可以在 device 侧持有和更新
- 同一 cell 连续调用时不需要每次都走完整的 host 取回和更新路径

对应逻辑主要在：

- [src/mmse_equalizer_gpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_gpu.cpp)
- [src/mmse_cuda_runtime.cu](G:\MMSE_CPP\src\mmse_cuda_runtime.cu) 中的 `finalize_sigma2_kernel`

这是当前 GPU 优化中的一条重要工程线，不改变算法数学，但减少了框架性开销。

## 8. 1x2 / 2x2 MMSE 均衡原理

### 8.1 1 layer 情况

对于单层情况，当前实现使用 `equalize_1x2_scalar(...)`。

本质上这是一个 `1x2` 接收滤波问题，形式更简单：

```text
w = conj(h) / (||h||^2 + sigma2)
z = w^H y
g = ||h||^2 / (||h||^2 + sigma2)
x_hat = z / g
gamma = g / (1 - g)
```

这里已经体现了项目实现的关键设计：不是只输出 `z`，而是做了去偏置 `x_hat = z / g`，再给出 `gamma`。

### 8.2 2x2 双层情况

双层情况下使用标准 2x2 MMSE：

```text
W = (H^H H + sigma2 I)^(-1) H^H
z = W y
G = W H
g_k = Re(G_kk)
x_hat_k = z_k / g_k
gamma_k = g_k / (1 - g_k)
```

当前 CPU 标量参考实现集中在：

- `trace_equalize_2x2_scalar(...)`
- `equalize_2x2_scalar(...)`

具体在 [src/mmse_equalizer_cpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_cpu.cpp)。

实现方式是显式展开 2x2 Hermitian 矩阵的逆：

```text
A = [a11  a12]
    [a12* a22]

det = a11*a22 - |a12|^2
A^-1 = 1/det * [ a22   -a12
                -a12*   a11 ]
```

然后再计算 `W`、`z0/z1`、`g0/g1`。

### 8.3 为什么要输出去偏置后的 x_hat

这是项目数学实现里最重要的一点之一。

MMSE 的 `z = Wy` 带有偏置，若直接把 `z` 交给后续软解调，会导致可靠度缩放不正确。当前实现统一采用：

```text
x_hat = z / g
gamma = g / (1 - g)
```

这保证：

- 输出星座点是去偏置后的
- `SINR` 和输出点保持一致
- 后续 LLR 计算有完整输入

这也是为什么本项目的输出契约始终是“equalized symbol + SINR”，而不是只给 complex symbol。

## 9. PDCCH 2 Tx transmit-diversity 的处理

`PDCCH` 的 2 发端口场景并不是简单套用普通 `2x2 / 2 layer` 路径，因为这里是 transmit diversity，而不是双层 spatial multiplexing。

当前 CPU 路径通过：

- `build_pdcch_td_re_pairs(...)`
- `demap_pdcch_transmit_diversity_from_grid(...)`
- `run_pdcch_td(...)`

把同一个 TD 对偶 RE 成对处理。

GPU 路径在 `equalize_stub_kernel` 里也有单独分支：

- 识别 `td_pdcch_mode`
- 判断 `is_pdcch_td_pair_start(...)`
- 对一对 RE 一次性写出两个符号

这说明当前项目不是把所有 LTE 场景都硬塞进一个统一数学内核，而是对协议上本质不同的模式做了专门分支。

## 10. CPU 实现结构与优化

### 10.1 CPU 主执行链

CPU 上当前执行流程大致是：

1. `prepare_subframe_if_needed(...)`
2. `estimate_channel(...)`
3. `update_sigma2_state(...)`
4. `build_channel_re_layout(...)`
5. `pack_equalizer_inputs(...)`
6. 多线程 `worker_task(...)`
7. 每个 worker 根据场景走：
   - `equalize_1x2_scalar(...)`
   - `equalize_2x2_scalar(...)`
   - `equalize_2x2_avx2(...)`
   - `PDCCH TD` 专用 demap 路径

### 10.2 shared-estimate 复用

CPU 路径已经实现同一子帧上的“prepared subframe”复用。

关键点在：

- `PreparedSubframeKey`
- `make_prepared_subframe_key(...)`
- `prepared_subframe_key_equal(...)`
- `prepare_subframe_if_needed(...)`

这意味着当同一 `grid + subframe + cell + backend mode` 下连续调用：

- `run_pbch(...)`
- `run_pcfich(...)`
- `run_pdcch(...)`
- `run(...)`

时，信道估计不需要重复计算。

这条优化已经被测试覆盖，相关测试在 [tests/mmse_tests.cpp](G:\MMSE_CPP\tests\mmse_tests.cpp) 中：

- `cpu_shared_estimate_reuses_once_per_subframe`

这是当前项目最重要的“链路级优化”之一，因为它直接减少了多信道同子帧重复工作。

### 10.3 AVX2 向量化

CPU 双层路径在满足条件时走 `equalize_2x2_avx2(...)`，定义在 [src/mmse_avx2.cpp](G:\MMSE_CPP\src\mmse_avx2.cpp)。

当前策略是“跨 RE 打包”：

- 每次处理 8 个 RE
- 将 `h00/h01/h10/h11/y0/y1` 都以 SoA 方式读入
- 用 FMA 计算 `a11/a22/a12`
- 用一次倒数近似加牛顿修正求 `1/det`
- 向量化求 `W`、`z`、`g`、`x_hat`、`gamma`

这条优化的前提是 `pack_equalizer_inputs(...)` 把离散网格访问转成连续 packed 数组，因此向量化收益来自两个层面：

1. 计算层面的 SIMD
2. 访存层面的线性化

### 10.4 静态线程池

CPU 端没有采用简单的“每次调用临时起线程”，而是维护 `StaticThreadPool`。

这样做的目标不是追求极限吞吐，而是控制：

- 线程启动/销毁开销
- 高频调用下的抖动
- 工作块划分的稳定性

`run_channel_from_prepared_estimate(...)` 会把 RE 按区间切分给 worker，每个 worker 在自己的区间上执行均衡。

### 10.5 当前 CPU 优化结论

已经验证有效的 CPU 侧优化主要是：

- prepared subframe 复用
- packed input 布局
- 双层路径 AVX2
- 静态线程池并行

这些优化都不改变算法数学，只减少重复工作或提高计算密度。

## 11. GPU 实现结构与优化

### 11.1 GPU 路径总体结构

GPU 端不是简单调用一个“全能 kernel”，而是分成：

1. host staging / metadata packing
2. H2D grid 传输
3. `estimate_residual_kernel`
4. `estimate_channel_kernel`
5. `finalize_sigma2_kernel`
6. `equalize_stub_kernel`
7. outputs / scratch / completion 的 D2H
8. 轻量或深度 validation

这些步骤主要由 [src/mmse_equalizer_gpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_gpu.cpp) 中的：

- `stage_inputs(...)`
- `execute_cuda_transport_stub(...)`

来组织。

### 11.2 pinned ring buffer

GPU 路径维护 `kPinnedRingSlotCount = 3` 的 host pinned ring buffer。

每个 slot 持有：

- 传输用 grid plane
- grid metadata
- output plane
- scratch
- 事件句柄
- device buffer 句柄

这样做的目的有两个：

1. 避免重复分配 pinned host memory 和 device memory
2. 给连续调用保留流式运输和 slot 轮转基础

这是典型的“框架开销控制”优化，不改变算法，但显著影响实际时延。

### 11.3 estimate 拆成 residual 和 channel 两段

早期 profiling 报告曾显示 `estimate_stub_kernel` 是单线程大瓶颈。当前实现已经不再是那个结构。

现在 GPU estimate 被拆成：

- `estimate_residual_kernel`
  - 负责残差型 sigma2 累积
- `estimate_channel_kernel`
  - 负责完整 `H` 网格生成
- `finalize_sigma2_kernel`
  - 负责 sigma2 的 final update / IIR update

这一步是项目 GPU 线最关键的结构性改动之一。它带来的收益不是“单 kernel 微调”，而是把原来严重串行的 estimate 路径改成了并行 launch。

### 11.4 shared-estimate 与 cache-hit metadata copy

GPU 路径也实现了与 CPU 一样的同子帧 estimate 复用。

当 `PreparedSubframeKey` 命中时：

- 不再重复走完整 estimate 路径
- 只更新动态 metadata slice
- 直接复用上次 estimate 结果

命中路径的关键优化是：

- `cuda_copy_grid_meta_dynamic_h2d_async(...)`

而不是重新复制整份 grid meta。

这是已经验证过的安全优化，相关测试包括：

- `gpu_shared_estimate_skips_second_estimate_path`

### 11.5 device-owned sigma2

GPU 还进一步支持了 `device-owned sigma2`。

它的核心收益是：

- 同一 cell 连续调用时，sigma2 不需要每次都 D2H 再 H2D
- 可以把 IIR 状态更自然地保持在 device 侧

但这里有明确的工程边界：`sigma2` 状态与 cache 复用不能随意耦合，否则容易破坏跨 cell 正确性。这个坑在此前优化中已经出现过，所以当前稳定版本把状态一致性放在首位。

### 11.6 release 与 deep validation 双路径

GPU 路径不是只有一种模式。

当前有两套验证强度：

- `kReleaseSanity`
  - 轻量 spot check
  - 尽量减少 D2H 和 debug 代价
- `kTestDeepTrace`
  - 复制 estimate / scratch / completion
  - 做 CPU/GPU 样本级 trace 比对

这套分层非常重要。它避免了“为了调试而把热路径永久拖慢”的问题，也保证：

- release 路径足够轻
- 深度验证仍然能在测试中保留

### 11.7 当前 GPU 优化结论

当前已经被证明有效的 GPU 优化主要是：

1. estimate 从单线程 stub 改成 residual/channel 双 kernel
2. same-subframe shared-estimate 复用
3. cache-hit 时只拷贝 dynamic metadata slice
4. `kReleaseSanity` 下跳过不必要的 scratch/completion D2H
5. device-owned sigma2 状态
6. pinned ring slot + 预分配 device buffers

这些优化的共同点是：优先减少框架性开销，而不是继续打磨 equalize 数学本体。

## 12. 为什么当前不再优先做 equalize 数学微优化

从 [docs/mmse_cuda_profile_report_2026-07-03.md](G:\MMSE_CPP\docs\mmse_cuda_profile_report_2026-07-03.md) 和 [docs/lte_mmse_budget_report_2026-07-01.md](G:\MMSE_CPP\docs\lte_mmse_budget_report_2026-07-01.md) 可以得出当前状态：

- full-band GPU `1 ms` 子帧成本约 `0.36 ms`
- `estimate_gpu_us` 大约 `75 us`
- `equalize_gpu_us` 大约 `8 us`
- 更大的成本在 host staging / scratch D2H / stream overhead

这意味着：

- `equalize_stub_kernel` 已经是微秒级
- 继续优化 equalize 数学内核的边际收益有限
- 真正影响端到端预算的，更多是 host/device 框架开销和多信道调用拓扑

因此，本项目当前优化策略已经明确转向：

- 复用 estimate
- 裁剪 metadata copy
- 压缩 D2H / sync
- 减少同子帧多信道重复工作

而不是回头做更激进的矩阵内核微雕。

## 13. 项目中已经验证过的关键优化方法总结

下面按“是否已验证有效”总结当前优化方法。

### 13.1 已验证有效

1. 同子帧 shared-estimate 复用
   - CPU / GPU 均已落地
   - 显著减少 `PBCH / PCFICH / PDCCH / PDSCH` 连续调用时的重复信道估计

2. GPU estimate 路径并行化拆分
   - 早期单线程 stub 被移除
   - 当前 estimate 进入正常并行 kernel 形态

3. cache-hit metadata 动态切片拷贝
   - 命中 prepared subframe 时不再全量拷贝 metadata

4. release 路径裁剪 scratch/completion D2H
   - `kReleaseSanity` 下保留轻量检查，去掉不必要的深度开销

5. CPU 双层路径 AVX2
   - 基于 packed input 的跨 RE SIMD

6. device-owned sigma2 状态
   - 降低 host/device 往返负担

### 13.2 已验证但收益有限或当前非优先

1. `equalize` 内核 occupancy 微调
   - 观测上不是主要瓶颈
   - 绝对时间过短，不是当前最优先方向

2. 更激进的 per-channel layout cache
   - 曾尝试过更激进的 cache-layout 路线
   - 但与状态正确性耦合风险较高，当前稳定基线没有采用

### 13.3 当前明确的优化边界

如果后续继续做优化，优先级应是：

1. host/device 框架开销
2. `scratch_d2h` / metadata copy / sync 压缩
3. 调用拓扑级复用
4. 只有当这些都压不动时，再考虑继续打磨 equalize 内核

## 14. 正确性保障方法

### 14.1 样本级 CPU/GPU 对比

项目不是只靠“整体 BER 看起来没问题”来判断正确性，而是建立了样本级对比。

包括：

- `Equalize2x2Trace`
- `build_validation_re_samples(...)`
- GPU spot-check / trace-sample 比较

这套方法的价值在于：一旦数值漂移，可以先定位到具体 RE、具体矩阵项、具体 `g0/g1` 或 `z0/z1`，而不是直接怀疑整套算法。

### 14.2 sigma2 状态一致性测试

项目还专门覆盖了：

- `sigma2_state_persists`
- `gpu_context_sigma2_state_persists`
- `gpu_context_device_owned_sigma2_state_persists`
- `gpu_context_device_owned_sigma2_tracks_cell_id`

这说明项目已经把“数值状态在多次调用间是否正确延续”视为一等正确性问题，而不仅是单次调用输出值。

### 14.3 shared-estimate 复用测试

同样地，shared-estimate 不是仅凭 benchmark 结论接受，而是有行为级测试：

- CPU 确认同子帧只估计一次
- GPU 确认第二次调用跳过第二条 estimate 路径

这保证优化不是靠“猜测命中了缓存”，而是靠测试证明。

## 15. 当前项目的算法与优化总判断

如果把当前项目概括成一句话：

这是一个围绕 LTE 下行 `PBCH / PCFICH / PDCCH / PDSCH` 的、以 CRS 信道估计和 MMSE equalized-RE 输出为核心的工程化模块；它的主要难点已经从“MMSE 数学是否成立”转为“如何在保持协议正确性和数值稳定性的前提下，降低多信道、多次调用场景中的框架开销”。

当前项目在算法层面已经具备：

- 统一的 LTE channel layout 生成
- CRS-based channel estimation
- sigma2 估计与 IIR 平滑
- 1x2 / 2x2 MMSE equalization
- `x_hat + SINR` 完整输出契约
- PDCCH transmit-diversity 的专门处理

当前项目在优化层面已经形成清晰路线：

- 先消除重复 estimate
- 再压缩 host/device 框架开销
- 最后才考虑内核数学微优化

这条路线与当前 profiling 和 budget 报告是一致的，也是当前代码库最可信的优化边界。
