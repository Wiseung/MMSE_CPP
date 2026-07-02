# LTE Equalized Channel SDK 文档

本页是由以下头文件导出的 LTE equalized-channel SDK 的稳定文档入口：

```cpp
#include "mmse/lte_chain_sdk.h"
```

当前接口版本：

- `LTE Equalized Channel SDK v1`

文档集合：

- [PBCH 快速开始](/G:/MMSE_CPP/docs/pbch_chain_sdk_quick_start.md)
- [PCFICH 快速开始与 API 参考](/G:/MMSE_CPP/docs/pcfich_chain_sdk_quick_start_api_reference.md)
- [PDCCH 子页面](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [PDCCH 快速开始](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [PDCCH API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [PDCCH 版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [PDCCH 集成示例](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)

## 作用范围

当前 LTE equalized-channel SDK 覆盖：

- LTE PBCH 的 equalized RE 提取接口面
- LTE PDCCH 控制区的 equalized RE 提取接口面
- LTE PCFICH 的 equalized RE 提取接口面
- 基于 CRS 的信道估计
- MMSE 均衡
- 由调用方持有输出 view，并支持后端 DTO 打包

当前文档深度仍然主要集中在 `PDCCH` 集成路径上。`PBCH` 和 `PCFICH`
现在已经共享同一套运行时与 DTO 风格，但尚未扩展出与 `PDCCH`
同等深度的独立快速开始 / 字段级参考页面。

## 推荐阅读顺序

1. 先阅读 [LTE 下行总览](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md)，建立协议背景。
2. 如果要集成 PBCH equalized-RE 接口，阅读 [PBCH 快速开始](/G:/MMSE_CPP/docs/pbch_chain_sdk_quick_start.md)。
3. 如果要集成 PCFICH equalized-RE 接口，阅读 [PCFICH 快速开始与 API 参考](/G:/MMSE_CPP/docs/pcfich_chain_sdk_quick_start_api_reference.md)。
4. 如果要查看当前文档最完整的信道专页，阅读 [PDCCH 子页面](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)。

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
- 运行时入口：
  - `MmseEqualizerCpuContext::run_pbch(...)`
  - `MmseEqualizerGpuContext::run_pbch(...)`

### PDCCH

- 前端 DTO 命名空间：`mmse::pdcch`
- 低层输入 / 输出：
  - `PdcchMmseInput`
  - `PdcchMmseOutputView`
  - `PdcchMmseResult`
  - 新增 TD 路径：
    - `PdcchTdMmseOutputView`
    - `PdcchTdMmseResult`
- 运行时入口：
  - `MmseEqualizerCpuContext::run_pdcch(...)`
  - `MmseEqualizerGpuContext::run_pdcch(...)`
  - 新增 TD 路径：
    - `MmseEqualizerCpuContext::run_pdcch_td(...)`
    - `MmseEqualizerGpuContext::run_pdcch_td(...)`

### PCFICH

- 前端 DTO 命名空间：`mmse::pcfich`
- 低层输入 / 输出：
  - `PcfichMmseInput`
  - `PcfichMmseOutputView`
  - `PcfichMmseResult`
- 运行时入口：
  - `MmseEqualizerCpuContext::run_pcfich(...)`
  - `MmseEqualizerGpuContext::run_pcfich(...)`

## 文档状态

- `PBCH` 现在有独立的快速开始页面。
- `PCFICH` 现在有独立的快速开始和紧凑型 API 参考页面。
- `PDCCH` 仍然是当前文档最完整的子页面，并保留最详细的字段索引。
