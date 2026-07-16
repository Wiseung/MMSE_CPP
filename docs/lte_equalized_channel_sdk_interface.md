# LTE Equalized Channel SDK 文档

本页是由以下头文件导出的 LTE equalized-channel SDK 的稳定文档入口：

```cpp
#include "mmse/lte_chain_sdk.h"
```

当前接口版本：

- `LTE Equalized Channel SDK v1`

文档集合：

- [LTE DCI 输出语义与 CE/MMSE 接口说明](/G:/MMSE_CPP/docs/lte_dci_and_ce_mmse_reference.md)
- [PBCH 快速开始](/G:/MMSE_CPP/docs/pbch_chain_sdk_quick_start.md)
- [PCFICH 快速开始与 API 参考](/G:/MMSE_CPP/docs/pcfich_chain_sdk_quick_start_api_reference.md)
- [PDSCH 下游 LLR / 解扰接口面快速开始与 API 参考](/G:/MMSE_CPP/docs/pdsch_llr_downstream_quick_start_api_reference.md)
- [PDCCH 子页面](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [PDCCH 快速开始](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [PDCCH API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [PDCCH 版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [PDCCH 集成示例](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)

## 作用范围

当前 LTE equalized-channel SDK 覆盖：

- LTE PBCH 的 equalized RE 提取接口面
- LTE PDSCH 的下游 `LLR / descrambling` helper
- LTE PDCCH 控制区的 equalized RE 提取接口面
- LTE PDCCH common-search、UE-specific 与 SI-RNTI 几何 `DCI 1A` 的 CPU helper 链路
- LTE PCFICH 的 equalized RE 提取接口面
- PBCH、PCFICH、PDCCH 的 `4Tx x 1Rx` Td4 raw equalized output
- 基于 CRS 的信道估计
- MMSE 均衡
- 由调用方持有输出 view，并支持后端 DTO 打包
- 显式 caller-owned `LLR` 输出面与显式 `PDSCH` scrambling plan cache

当前文档深度仍然主要集中在 `PDCCH` 集成路径上。`PBCH`、`PCFICH`
和 `PDSCH` 现在已经共享同一套 `LTE` 运行时 / DTO 风格；其中 `PDSCH` 同时提供
通用空间复用入口、`2Tx` 发射分集入口和下游 `LLR / descrambling` helper 页面，但不提供
完整译码链页面。

## 推荐阅读顺序

1. 先阅读 [LTE 下行总览](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md)，建立协议背景。
2. 如果要先把 `DCI` 语义、CE/MMSE 输入输出和 `10 ms` 调用口径讲清楚，阅读 [LTE DCI 输出语义与 CE/MMSE 接口说明](/G:/MMSE_CPP/docs/lte_dci_and_ce_mmse_reference.md)。
3. 如果要集成 PBCH equalized-RE 接口，阅读 [PBCH 快速开始](/G:/MMSE_CPP/docs/pbch_chain_sdk_quick_start.md)。
4. 如果要集成 PCFICH equalized-RE 接口，阅读 [PCFICH 快速开始与 API 参考](/G:/MMSE_CPP/docs/pcfich_chain_sdk_quick_start_api_reference.md)。
5. 如果要集成 `PDSCH` 的 `2Tx + 1 layer + TM2` 发射分集路径，使用
   `run_pdsch_td(...)`；其输出可继续交给 [PDSCH 下游 LLR / 解扰接口面快速开始与 API 参考](/G:/MMSE_CPP/docs/pdsch_llr_downstream_quick_start_api_reference.md)。
6. 如果要查看当前文档最完整的信道专页，阅读 [PDCCH 子页面](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)。

## 公开头文件布局

- 统一 LTE 总头文件：
  - `#include "mmse/lte_chain_sdk.h"`
- 仅 PDCCH 总头文件：
  - `#include "mmse/pdcch_chain_sdk.h"`

## 当前信道接口面

### PBCH

- 前端 DTO 命名空间：`mmse::pbch`
- 低层输入 / 输出：
  - `PbchMmseInput`
  - `PbchMmseOutputView`
  - `PbchMmseResult`
  - 4Tx 发射分集输入 / 输出：
    - `PbchMmseInput`
    - `PbchTd4MmseOutputView`
    - `PbchTd4MmseResult`
- 运行时入口：
  - `MmseEqualizerCpuContext::run_pbch(...)`
  - `MmseEqualizerGpuContext::run_pbch(...)`
  - `MmseEqualizerCpuContext::run_pbch_td4(...)`
  - `MmseEqualizerGpuContext::run_pbch_td4(...)`
- Td4 边界：仅 `4Tx x 1Rx`、单层、`tx_mode == 2`、QPSK；固定输出 `240` 个
  source RE。当前只提供 caller-owned raw output view/result，不提供 4Tx owning backend DTO，
  也不负责 PBCH MIB 最终译码。

### PDSCH

- 下游 helper 命名空间：`mmse::pdsch`
- 低层输入 / 输出：
  - 通用空间复用输入 / 输出：`ExtractDescriptor + EqualizerOutputView`
  - 2Tx 发射分集输入 / 输出：
    - `PlanarGridViewF32 + ExtractDescriptor`
    - `PdschTdMmseOutputView`
    - `PdschTdMmseResult`
  - `PdschDescrambledLlrOutputView`
  - `PdschDescrambledLlrResult`
  - `PdschDescramblingPlanCache`
  - `BackendPdschDescrambledLlrIndication`
- 运行时入口：
  - `MmseEqualizerCpuContext::run(...)`
  - `MmseEqualizerGpuContext::run(...)`
  - `MmseEqualizerCpuContext::run_pdsch_td(...)`
  - `MmseEqualizerGpuContext::run_pdsch_td(...)`
- 选择规则：
  - `run(...)` 用于 `1Tx` 单层或 `2Tx + 2 layer` 空间复用
  - `run_pdsch_td(...)` 仅用于 `channel_type == kPdsch`、`n_tx_ports == 2`、
    `n_layers == 1`、`tx_mode == 2`、`pmi == -1` 的 Alamouti 发射分集
  - 通用 `run(...)` 会拒绝上述 `PDSCH` TD 组合，保证 CPU 与 CUDA 后端具有相同契约
- 下游 helper：
  - `prepare_pdsch_descrambling_plan(...)`
  - `build_backend_pdsch_descrambled_llr_result(...)`
  - `make_backend_pdsch_descrambled_llr_indication(...)`

### PDCCH

- 前端 DTO 命名空间：`mmse::pdcch`
- 低层输入 / 输出：
  - `PdcchMmseInput`
  - `PdcchMmseOutputView`
  - `PdcchMmseResult`
  - 新增 TD 路径：
    - `PdcchTdMmseOutputView`
    - `PdcchTdMmseResult`
    - `PdcchTd4MmseOutputView`
    - `PdcchTd4MmseResult`
    - `mmse::pdcch::BackendPdcchTd4EqualizedIndication`
- 运行时入口：
  - `MmseEqualizerCpuContext::run_pdcch(...)`
  - `MmseEqualizerGpuContext::run_pdcch(...)`
  - 新增 TD 路径：
    - `MmseEqualizerCpuContext::run_pdcch_td(...)`
    - `MmseEqualizerGpuContext::run_pdcch_td(...)`
    - `MmseEqualizerCpuContext::run_pdcch_td4(...)`
    - `MmseEqualizerGpuContext::run_pdcch_td4(...)`
  - 新增 CPU helper 路径：
    - `mmse::pdcch::run_pdcch_cpu_common_search_decode(...)`
    - `mmse::pdcch::normalize_pdcch_td4_cce_order(...)`
- Td4 边界：仅 `4Tx x 1Rx`、单层、`tx_mode == 2`、QPSK。四源 RE 输出可归一化为
  标准连续 CCE 顺序，并复用 CPU common-search、UE-specific 和 SI-RNTI 解码链。

### PCFICH

- 前端 DTO 命名空间：`mmse::pcfich`
- 低层输入 / 输出：
  - `PcfichMmseInput`
  - `PcfichMmseOutputView`
  - `PcfichMmseResult`
  - 4Tx 发射分集输入 / 输出：
    - `PcfichMmseInput`
    - `PcfichTd4MmseOutputView`
    - `PcfichTd4MmseResult`
- 运行时入口：
  - `MmseEqualizerCpuContext::run_pcfich(...)`
  - `MmseEqualizerGpuContext::run_pcfich(...)`
  - `MmseEqualizerCpuContext::run_pcfich_td4(...)`
  - `MmseEqualizerGpuContext::run_pcfich_td4(...)`
- Td4 边界：仅 `4Tx x 1Rx`、单层、`tx_mode == 2`、QPSK；固定输出 `16` 个
  source RE。当前只提供 caller-owned raw output view/result，不提供 4Tx owning backend DTO，
  也不负责 PCFICH CFI 最终译码。

### Td4 四源 RE 索引契约

`PbchTd4MmseOutputView`、`PcfichTd4MmseOutputView` 和 `PdcchTd4MmseOutputView` 使用同一
契约。对每个四元组起点 `q = 0, 4, 8, ...`，槽位 `q..q+3` 都重复记录同一组
`re_grid_indices0..3`。四个 `x_hat`/`sinr` 槽位按该 source-RE 顺序对应恢复后的 QPSK
软符号；调用方必须使用 `meta.n_symbols` 作为有效前缀，且容量至少为
`meta.n_source_re`。该契约不表示四层空间复用。

## 文档状态

- `PBCH` 现在有独立的快速开始页面。
- `PCFICH` 现在有独立的快速开始和紧凑型 API 参考页面。
- `PDSCH` 现在有独立的下游 `LLR / descrambling` helper 页面，但仓库仍没有更高层 `PDSCH` downstream context。
- `PDCCH` 仍然是当前文档最完整的子页面，并保留最详细的字段索引。
