# LTE DCI 输出语义与 CE/MMSE 接口说明

## 文档目的

本文面向当前 `G:\MMSE_CPP` 仓库，说明两件事：

1. 下游把 `PDCCH` 译码结果翻译成 `DCI` 信息后，各字段通常表示什么。
2. 当前仓库中“信道估计（`estimate`）+ MMSE 均衡（`equalize`）”这一步的输入、输出、数据长度，以及 `10 ms` 窗口内的最大调用次数应如何理解。

这份文档复用现有仓库口径，不引入新的接口契约。

## 实现边界

当前仓库负责：

- LTE `PBCH / PCFICH / PDCCH / PDSCH` 的目标 RE 提取
- 基于 `CRS` 的信道估计
- `MMSE` 均衡
- 输出 equalized 软符号与 `SINR`
- `PDCCH` 的 `REG / CCE` 恢复 helper
- `PDCCH common search` 与 UE-specific 候选构造、候选 LLR 切片与速率恢复 helper
- `PDCCH` 的 `CRC-RNTI` 校验与 `DCI 1A` 解析 helper
- 默认内建尾咬卷积码译码的正式 CPU common-search 与 UE-specific `DCI 1A` 链路，可由外部回调覆盖
- 当前 `20 MHz / FDD` 范围内的 SI-RNTI 未知控制区几何搜索与调用方持有的缓存锁定

当前仓库不负责：

- 非 `DCI 1A` 的通用 `DCI` 解析 / 翻译
- GPU 版完整 PDCCH 解码阶段

因此，`DCI` 相关内容在本文中既包括**下游译码阶段的推荐解释口径**，也包括当前仓库里已经存在的
common-search、UE-specific 与 `DCI 1A` helper / CPU 链路边界；但它仍不等于仓库已经具备一套通用 LTE 控制信道接收机。

## 1. `DCI` 输出在整体链路中的位置

对 `PDCCH` 而言，基础 equalizer 接口交付的是 `equalized RE` 或 `descrambled LLR`。
调用正式 CPU blind-decode helper 时，仓库还可在当前边界内交付已通过 `CRC-RNTI` 的 `DCI 1A` 命中。

推荐按下面这条链路理解：

1. `run_pdcch(...)` 或 `run_pdcch_td(...)` 输出 `x_hat + sinr + re_grid_indices`
2. 使用仓库 helper 或下游逻辑恢复 `REG / CCE / common-search 或 UE-specific` 候选
3. 使用仓库 helper 完成软解调、解扰、速率恢复
4. 通过内建尾咬卷积码译码完成卷积译码，必要时可提供外部回调覆盖
5. 使用仓库 helper 做 `CRC-RNTI` 校验和 `DCI 1A` 解析
6. 若需要非 `DCI 1A` 格式、未知 UE RNTI 的全空间枚举或更宽 LTE PHY 配置，则继续交给仓库外下游模块

如果下游直接使用仓库 helper，则常见交接点是：

- `mmse::pdcch::BackendPdcchEqualizedIndication`
- `mmse::pdcch::BackendPdcchDescrambledLlrIndication`
- `mmse::pdcch::BackendPdcchTdEqualizedIndication`
- `mmse::pdcch::BackendPdcchTdDescrambledLlrIndication`
- `mmse::pdcch::PdcchCommonSearchDecodeResult`
- `mmse::pdcch::PdcchUeSpecificSearchResult`
- `mmse::pdcch::PdcchSiRntiGeometrySearchResult`

## 2. `DCI` 翻译输出字段含义

### 2.1 定位与上下文字段

| 字段           | 含义                        | 说明                                                                  |
| -------------- | --------------------------- | --------------------------------------------------------------------- |
| `sfn_subframe` | 无线时间索引                | 一般按 `SFN * 10 + subframe` 理解，表示这条 `DCI` 属于哪个 LTE 子帧。 |
| `cell_id`      | LTE 小区 ID                 | 译码、解扰和后续调度解释都需要绑定同一小区上下文。                    |
| `rnti`         | 通过 CRC mask 恢复出的 RNTI | 它决定这条 `DCI` 属于哪个 UE 或哪类公共控制消息。                     |
| `dci_format`   | `DCI` 格式                  | 它决定后面哪些字段存在，以及这些字段该按上行还是下行调度解释。        |

### 2.2 候选定位字段

这几个字段不是 `DCI payload` 本身，而是**控制区候选定位信息**。当前仓库已经通过
`PdcchChainMetadata` 为下游保留了透传位置：

| 字段                | 含义                                  | 仓库内来源                              |
| ------------------- | ------------------------------------- | --------------------------------------- |
| `request_id`        | 上游一次 `PDCCH` 处理请求的业务 ID    | `PdcchChainMetadata::request_id`        |
| `candidate_id`      | 搜索空间中的候选序号                  | `PdcchChainMetadata::candidate_id`      |
| `first_cce`         | 候选起始 `CCE`                        | `PdcchChainMetadata::first_cce`         |
| `aggregation_level` | 候选占用的 `CCE` 数，通常为 `1/2/4/8` | `PdcchChainMetadata::aggregation_level` |

推荐把这组字段和真正的 `DCI payload` 一起透传，因为它们对：

- 复盘盲检索命中位置
- 关联上游候选列表
- 定位误检 / 漏检

都很重要。

对当前仓库已覆盖的 `DCI 1A` helper 路径而言，这组字段会被保留在：

- `PdcchDciFormat1A::chain`
- `PdcchDciFormat1ADecodeResult::dci.chain`

### 2.3 调度字段

下表描述的是**翻译后的业务语义**。并不是每一种 `DCI format` 都会包含全部字段。

| 字段                    | 含义                      | 说明                                                                    |
| ----------------------- | ------------------------- | ----------------------------------------------------------------------- |
| `grant_direction`       | 授权方向                  | 通常表示这条 `DCI` 是下行授权还是上行授权。                             |
| `resource_assignment`   | 资源分配字段              | 表示 `PRB/VRB` 分配。下游一般再把它翻译成 `start_prb`、`n_prb` 或位图。 |
| `mcs`                   | 调制编码方案索引          | 后续会被翻译成调制阶数与编码速率，并参与 `TBS` 计算。                   |
| `rv`                    | HARQ 冗余版本             | 用于判断是初传还是哪一种重传版本。                                      |
| `ndi`                   | New Data Indicator        | 用于区分新数据与重传。                                                  |
| `harq_process`          | HARQ 进程序号             | 用于把当前调度和既有 HARQ 状态机关联起来。                              |
| `tpc`                   | 功控命令                  | 只在部分格式中出现。                                                    |
| `dai`                   | Downlink Assignment Index | 主要在 `TDD` 语义里使用。                                               |
| `pucch_tpc_or_uci_bits` | 上行控制相关字段          | 只在相关 `DCI` 格式中出现。                                             |

### 2.4 译码后常见的派生字段

很多系统会在“`DCI` 翻译输出”里顺手补齐一些派生值，便于后续 `PDSCH/PUSCH` 处理：

| 字段        | 含义       | 是否通常为原始 `DCI bit`               |
| ----------- | ---------- | -------------------------------------- |
| `start_prb` | 起始 PRB   | 否，通常由 `resource_assignment` 派生  |
| `n_prb`     | PRB 数     | 否，通常由 `resource_assignment` 派生  |
| `mod_order` | 调制阶数   | 否，通常由 `mcs` 派生                  |
| `tbs`       | 传输块大小 | 否，通常由 `mcs + n_prb + layers` 派生 |
| `is_retx`   | 是否重传   | 否，通常由 `ndi + HARQ` 状态综合判断   |

推荐在文档、日志和接口里明确区分：

- 原始 `DCI payload` 字段
- 译码后派生字段

否则很容易把“协议字段”和“实现中为了方便新增的字段”混在一起。

## 3. CE/MMSE 这一步的公共输入输出

### 3.1 公共输入

当前仓库所有入口都共享同一份 LTE `1 ms` 子帧频域网格视图：

```cpp
mmse::PlanarGridViewF32
```

当前稳定支持边界下，这份网格的固定形状是：

- `n_rx_ant in {1, 2}`
- `n_symbols == 14`
- `n_subcarriers == 1200`

也就是一个 normal-CP、20 MHz 的 LTE `1 ms` 子帧。

按当前 planar `float` 布局计算，单次调用的输入网格数据量是：

- `n_rx_ant × 2 planes(re/im) × 14 × 1200 × 4 bytes`；1Rx 为 134,400 bytes，2Rx 为 268,800 bytes

### 3.2 公共输出

CE/MMSE 阶段的公共输出本质上是两类信息：

1. 均衡后的复数软符号
   - `x_hat_re`
   - `x_hat_im`
2. 对每个 RE 或软符号的可靠度估计
   - `sinr`

其中 `PDCCH / PBCH / PCFICH` 还会额外输出 `re_grid_indices`，用于把结果重新映射回 LTE 网格位置。
对 `2Tx` 发射分集入口，输出改为一对 `re_grid_indices0 / re_grid_indices1`：每两个连续
软符号共享同一对来源 RE。

### 3.3 运行时参数分层

当前接口分成两层参数：

| 层级                   | 类型                                                                         | 作用                                                           |
| ---------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------- |
| context 初始化参数     | `MmseEqualizerCpuConfig` / `MmseEqualizerGpuConfig`                          | 选择 `backend`、线程数、CUDA stream、`sigma2` 策略等运行时策略 |
| 单次调用参数           | `ExtractDescriptor` / `PbchMmseInput` / `PcfichMmseInput` / `PdcchMmseInput` | 描述这一次要从哪份 LTE 子帧里提哪些 RE、按什么 LTE 上下文解释  |
| caller-owned 输出 view | `EqualizerOutputView` / `*MmseOutputView`                                    | 提供结果写入缓冲区                                             |
| 单次调用元数据         | `*MmseResult`                                                                | 返回本次提取到的 `RE` 数、层数、`sigma2`、带宽、控制区大小等   |

## 4. 各入口的输入、输出与数据长度

### 4.1 一览表

| 入口                  | 主要输入                                | 主要输出                                    | 当前典型数据长度                                                                             |
| --------------------- | --------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `run_pbch(...)`       | `PbchMmseInput`                         | `PbchMmseOutputView + PbchMmseResult`       | `240 RE`                                                                                     |
| `run_pcfich(...)`     | `PcfichMmseInput`                       | `PcfichMmseOutputView + PcfichMmseResult`   | `16 RE`                                                                                      |
| `run_pdcch(...)`      | `PdcchMmseInput`                        | `PdcchMmseOutputView + PdcchMmseResult`     | `20 MHz / CFI=3` 默认 CCE 对齐布局为 `3168 RE`；保留资源会按完整 CCE 组重新截断              |
| `run_pdcch_td(...)`   | `PdcchMmseInput`                        | `PdcchTdMmseOutputView + PdcchTdMmseResult` | `3168` 个 CCE 对齐源 `RE / soft symbols`                                                     |
| `run(...)`（`PDSCH`） | `PlanarGridViewF32 + ExtractDescriptor` | `EqualizerOutputView`                       | 与 `start_symbol / PRB bitmap / layers` 有关；当前 full-band benchmark 为 `14400 RE / layer` |

补充说明：

- `run_pdcch_cpu_common_search_decode(...)` 的主要输入是 `PdcchMmseInput + PdcchCommonSearchDecodeConfig`
- `run_pdcch_cpu_si_rnti_search(...)` 的主要输入是 `PdcchMmseInput + PdcchSiRntiSearchConfig`
- `run_pdcch_cpu_ue_specific_search(...)` 的主要输入是 `PdcchMmseInput + PdcchUeSpecificSearchConfig`
- `run_pdcch_cpu_si_rnti_geometry_search(...)` 的主要输入是 `PdcchSiRntiGeometrySearchRequest + PdcchSiRntiGeometrySearchCache`
- 主要输出是 `PdcchCommonSearchDecodeResult`、`PdcchSiRntiSearchResult`、`PdcchUeSpecificSearchResult` 或 `PdcchSiRntiGeometrySearchResult`
- 当前正式 helper 覆盖 common-search 与 UE-specific `DCI 1A`，默认使用内建尾咬卷积码译码

`PDSCH + 2Tx + 1 layer + TM2` 不使用通用 `run(...)`；应调用
`run_pdsch_td(PlanarGridViewF32, ExtractDescriptor, PdschTdMmseOutputView, PdschTdMmseResult)`。
它输出的连续单层软符号数等于 CRS 排除后的源 RE 数。

### 4.2 `PBCH`

输入重点字段：

- `sfn_subframe`
- `cell_id`
- `n_tx_ports`
- `tx_mode`

输出重点字段：

- `x_hat_re / x_hat_im / sinr`
- `re_grid_indices`
- `start_prb`
- `n_prb`
- `start_symbol`
- `sigma2`

当前 wrapper 固定边界：

- 中心 `6 PRB`
- `start_symbol == 7`
- 连续 `4` 个 OFDM symbols
- `expected_pbch_re == 240`

如果只按直接输出 payload 估算，不计 `meta`：

- 软符号与 `SINR`：`240 × 3 × 4 = 2,880 bytes`
- `re_grid_indices`：`240 × 2 = 480 bytes`
- 合计约 `3,360 bytes`

### 4.3 `PCFICH`

输入重点字段：

- `sfn_subframe`
- `cell_id`
- `n_tx_ports`
- `tx_mode`

输出重点字段：

- `x_hat_re / x_hat_im / sinr`
- `re_grid_indices`
- `start_symbol`
- `reg_count`
- `sigma2`

当前 wrapper 固定边界：

- `start_symbol == 0`
- `reg_count == 4`
- `expected_pcfich_re == 16`

直接输出 payload 估算：

- 软符号与 `SINR`：`16 × 3 × 4 = 192 bytes`
- `re_grid_indices`：`16 × 2 = 32 bytes`
- 合计约 `224 bytes`

### 4.4 `PDCCH`

输入重点字段：

- `sfn_subframe`
- `cell_id`
- `n_tx_ports`
- `tx_mode`
- `control_symbol_count`
- `n_prb`
- `prb_bitmap`
- `control_subframe`
- `control_re_exclusion_masks` 或 `reserved_control_res`
- `chain`

输出重点字段：

- `x_hat_re / x_hat_im / sinr`
- `re_grid_indices`
- `control_symbol_count`
- `sigma2`
- `chain`

当前 RE 布局会在 REG 交织后只保留完整的 `9 REG / CCE` 组。对 `20 MHz + normal CP +
3 control symbols + 1Tx` 的默认 `CFI=3` 测试布局，输出为：

- `3168 RE`

额外的 `PCFICH / PHICH` 保留 RE 会先从物理 REG 中排除，随后再做 CCE 对齐；因此最终数值
应以本次 `PdcchMmseResult::n_re` 为准。

按 `3168 RE` 估算直接输出 payload：

- 软符号与 `SINR`：`3168 × 3 × 4 = 38,016 bytes`
- `re_grid_indices`：`3168 × 2 = 6,336 bytes`
- 合计约 `44,352 bytes`

### 4.5 `PDCCH 2 Tx port` transmit-diversity

当 `PDCCH` 是 `2 Tx port` 发射分集场景时，应使用：

```cpp
run_pdcch_td(...)
```

而不是旧的逐 `RE` `run_pdcch(...)` 契约。

输出重点字段：

- `x_hat_re / x_hat_im / sinr`
- `re_grid_indices0`
- `re_grid_indices1`
- `n_symbols`
- `n_source_re`

当前测试路径下：

- `n_source_re == 3168`
- `n_symbols == 3168`

按直接输出 payload 估算：

- 软符号与 `SINR`：`3168 × 3 × 4 = 38,016 bytes`
- 两组索引：`3168 × 2 × 2 = 12,672 bytes`
- 合计约 `50,688 bytes`

### 4.5a `PDCCH DCI 1A` CPU 入口

当前仓库提供的正式 CPU 入口是：

```cpp
run_pdcch_cpu_common_search_decode(...)
run_pdcch_cpu_si_rnti_search(...)
run_pdcch_cpu_ue_specific_search(...)
run_pdcch_cpu_si_rnti_geometry_search(...)
```

它们的作用都不是返回单个候选，而是返回命中列表：

- `PdcchCommonSearchDecodeResult::candidate_count`
- `PdcchCommonSearchDecodeResult::hits`
- `PdcchSiRntiSearchResult::hits`
- `PdcchUeSpecificSearchResult::{candidate_count, decoded_candidate_count, crc_rnti_miss_count, semantic_reject_count, hits}`
- `PdcchSiRntiGeometrySearchResult::{status, geometry_attempt_count, geometry, decoded}`

当前固定边界：

- 仅 CPU
- common-search 与 UE-specific
- 仅 `DCI 1A`
- `run_pdcch_cpu_common_search_decode(...)` 可选接受 `PdcchTailBitingConvolutionalDecoder` 覆盖默认内建译码
- `run_pdcch_cpu_si_rnti_search(...)` 固定使用 `SI-RNTI` 语义，并仅允许外部 decoder 覆盖
- `run_pdcch_cpu_ue_specific_search(...)` 接收目标 RNTI 列表并按 `Y_k` 搜索 `L=1/2/4/8`
- `run_pdcch_cpu_si_rnti_geometry_search(...)` 当前仅支持 `20 MHz / FDD`；其 cache 在 PCI、端口、发射模式或子帧类型改变时失效

如果你需要其它 `DCI` 格式、未知 UE RNTI 的通用搜索策略或 GPU 版完整解码阶段，仍应由仓库外模块完成。

### 4.6 `PDSCH` 空间复用

`1Tx` 单层和 `2Tx + 2 layer` 空间复用的 `PDSCH` 入口是通用：

```cpp
run(const PlanarGridViewF32& grid,
    const ExtractDescriptor& desc,
    EqualizerOutputView& out)
```

输入重点字段：

- `channel_type = MmseChannelType::kPdsch`
- `start_symbol`
- `n_prb`
- `prb_bitmap`
- `n_layers`
- `mod_order`
- `n_tx_ports`
- `tx_mode`

输出重点字段：

- `x_hat_re / x_hat_im / sinr`
- `n_re_per_layer`
- `n_layers`
- `mod_order`

`PDSCH` 的 `n_re_per_layer` 不是常数，它由以下因素共同决定：

- `start_symbol`
- 激活 `PRB` 范围
- `n_layers`
- 是否有 `CRS` 空洞

当前 full-band benchmark 使用的口径是：

- `start_symbol == 1`
- `100 PRB`
- `2 layers`
- `n_re_per_layer == 14400`

对应直接输出 payload：

- `14400 RE/layer × 2 layers × 3 planes × 4 bytes = 345,600 bytes`

### 4.7 `PDSCH 2 Tx port` transmit-diversity

当 `ExtractDescriptor` 表示下面的组合时，不能调用通用 `run(...)`：

- `channel_type == MmseChannelType::kPdsch`
- `n_tx_ports == 2`
- `n_layers == 1`
- `tx_mode == 2`
- `pmi == -1`

必须调用：

```cpp
run_pdsch_td(grid, desc, out, meta)
```

该入口在同一份 `PlanarGridViewF32` 中利用 port 0/1 的 CRS 分别估计两个发送端口到每个
接收天线的信道。`build_data_re_layout(...)` 先按 symbol、PRB、tone 顺序排除两组 CRS RE，
然后将相邻、同一 OFDM symbol 内的数据 RE 两两配对；每对通过联合 Alamouti MMSE 去分集恢复
两个软符号。

输出重点字段：

- `PdschTdMmseOutputView::x_hat_re / x_hat_im / sinr`
- `PdschTdMmseOutputView::re_grid_indices0 / re_grid_indices1`
- `PdschTdMmseResult::n_symbols / n_source_re / start_symbol / prb_bitmap / sigma2`

`n_symbols == n_source_re`。输出在逻辑上是单层连续软符号矩阵；第 `2k` 和 `2k+1` 个元素
对应同一对来源 RE，且共享 `re_grid_indices0[2k] / re_grid_indices1[2k]`。调用方可构造
`n_layers == 1` 的 `EqualizerOutputView` 适配器，再复用现有 `mmse::pdsch` LLR / 解扰 helper。

## 5. `10 ms` 内最大调用次数怎么解释

### 5.1 先看接口粒度

当前仓库公开接口的固定粒度是：

- **一次调用处理一个 LTE `1 ms` 子帧**

因此本文里所有 `10 ms` 说法都表示：

- 对 `10` 个连续 `1 ms` 子帧做重复调用的聚合解释

而不是：

- 已经存在一个“单次处理 `10 ms`”的 batched API

### 5.2 协议层上界

如果把一个对齐的 LTE radio frame 视为 `10 ms`，并且只统计当前仓库这一层的 CE/MMSE 调用，则可按下面的协议层上界理解：

| 信道 / 入口                      | `10 ms` 内最大调用次数 | 说明                                                                                                                                                      |
| -------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PBCH` / `run_pbch(...)`         | `1`                    | `PBCH` 周期是 `10 ms`，当前 wrapper 固定落在 subframe `0`。                                                                                               |
| `PCFICH` / `run_pcfich(...)`     | `10`                   | 每个 downlink 子帧最多一次。若按完整 `10` 个 LTE 子帧都需处理，则上界是 `10`。                                                                            |
| `PDCCH` / `run_pdcch(...)`       | `10`                   | 每个 downlink 子帧最多一次。`1Tx` 与 `2Tx TD` 是二选一，不会在同一条真实链路上同时都算一次。                                                              |
| `PDCCH TD` / `run_pdcch_td(...)` | `10`                   | 仅当系统实际使用 `2 Tx port` PDCCH 发射分集时成立。                                                                                                       |
| `PDSCH` / `run(...)`             | **无单一固定常数**     | 如果集成层按“一子帧一个 `ExtractDescriptor`”建模，则名义上可按 `10` 理解；但如果按 grant、码字或不同 PRB 分块拆成多个 descriptor，调用次数可以大于 `10`。 |
| `PDSCH TD` / `run_pdsch_td(...)` | **无单一固定常数**     | 仅适用于 `2Tx + 1 layer + TM2`；与通用 `PDSCH run(...)` 互斥，按实际 TD grant 数量计数。                                                                  |

### 5.3 当前仓库已出现过的典型 `10 ms` 调用拓扑

当前预算报告里已经测过一条六次独立调用路径：

1. `PBCH`
2. `PCFICH`
3. `PDCCH`
4. `PDSCH`
5. `PDCCH`
6. `PDSCH`

这条链路不是协议极限，只是一个**当前工程里已经实际测量过的代表性拓扑**。

它说明：

- 在一个 `10 ms` 业务窗口里，CE/MMSE 模块很容易被多次反复调用
- 真正影响时延预算的，往往不是单个 `equalize` kernel，而是重复的 wrapper / estimate / transport 开销

除此之外，当前仓库还新增了独立的：

- `pdcch_decode_bench`

它会把 `PDCCH` CPU 链路拆成：

- `CE/MMSE`
- `REG/CCE`
- `LLR`
- `candidate gather`
- `rate recovery`
- `native_tail_biting_viterbi`
- `CRC/DCI`

并同时输出每 `1 ms` 子帧和每 `10 ms` 窗口的 `avg / p50 / p95`。其中
`native_tail_biting_viterbi` 明确表示仓库内建 `64-state` 尾咬 Viterbi 的阶段耗时；如果配置外部覆盖器，可据此对比接口边界成本。
基准还报告 SI-RNTI 几何冷启动、锁定和第五次 miss 重探的耗时及几何尝试数。

## 6. 对外说明时建议使用的口径

如果需要把这部分内容讲给项目外部调用方，推荐直接使用下面三句话：

1. 当前仓库负责的是 LTE `PBCH / PCFICH / PDCCH / PDSCH` 的 **RE 提取 + CRS 信道估计 + MMSE 均衡**，不负责最终 `DCI / MIB / TB` 译码。
2. `PDCCH` 场景下，当前仓库提供 common-search 与 UE-specific `DCI 1A` 的 CPU helper 链路，默认使用内建尾咬卷积码译码，也可由外部回调覆盖；固定 `SI-RNTI` 可直接调用专用入口，缺少可信 CFI/PHICH 几何时可调用 SI-RNTI 几何搜索入口，`candidate_id / first_cce / aggregation_level` 会和译码结果一起透传。
3. 当前 API 的基本粒度是 **一次处理一个 LTE `1 ms` 子帧**，因此 `10 ms` 的调用次数和数据量都应按 `10` 次 `1 ms` 调用去解释。

## 相关文档

- [LTE 下行信道译码总览](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md)
- [LTE Equalized Channel SDK 文档首页](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [PDCCH Chain SDK 快速开始](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [PDCCH Chain SDK API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [PDCCH Module API 集成示例](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)
- [PBCH 快速开始](/G:/MMSE_CPP/docs/pbch_chain_sdk_quick_start.md)
- [PCFICH 快速开始与 API 参考](/G:/MMSE_CPP/docs/pcfich_chain_sdk_quick_start_api_reference.md)
- [PDSCH 下游 LLR / 解扰接口面快速开始与 API 参考](/G:/MMSE_CPP/docs/pdsch_llr_downstream_quick_start_api_reference.md)
- [LTE MMSE 预算报告 (2026-07-01)](/G:/MMSE_CPP/docs/lte_mmse_budget_report_2026-07-01.md)
