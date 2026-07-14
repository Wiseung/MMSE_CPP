# PDCCH Chain SDK 文档

本页是更大范围 LTE equalized-channel SDK 下的 PDCCH 专用子页面。

主要包含头文件：

```cpp
#include "mmse/pdcch_chain_sdk.h"
```

当前接口版本：

- `PDCCH Chain SDK v1`

父页面：

- [LTE Equalized Channel SDK 文档](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)

文档集合：

- [完整流程说明](/G:/MMSE_CPP/docs/lte_pdcch_complete_flow.md)
- [快速开始](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [集成示例](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)

## 推荐阅读顺序

1. 首次集成时先阅读 [快速开始](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
2. 如果要看“完整 `PDCCH` 接收链”和“当前仓库覆盖到哪一步”，先阅读 [完整流程说明](/G:/MMSE_CPP/docs/lte_pdcch_complete_flow.md)
3. 做字段级查询时使用 [API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
4. 在依赖兼容性保证或准备改接口前，先阅读 [版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)

## 作用范围

SDK 当前覆盖：

- LTE PDCCH 控制区 RE 提取
- 基于 CRS 的信道估计
- MMSE 均衡
- 按 RE 输出软符号和 SINR
- 通过新增的 `run_pdcch_td(...)` 接口支持 2Tx 发射分集去映射
- `REG / CCE` 恢复 helper
- `common search` 候选构造与候选 LLR 切片 helper
- UE-specific 候选构造，按目标 RNTI 和子帧枚举 `L=1/2/4/8`
- 候选级速率恢复、`CRC-RNTI` 校验和 `DCI 1A` 解析 helper
- 默认内建尾咬卷积码译码的正式 CPU common-search 与 UE-specific `DCI 1A` 链路，可由外部回调覆盖
- 1Tx 与 2Tx TD 的 GPU common-search / `DCI 1A` 一站式入口及 batch 入口
- 当前 20 MHz/FDD 边界内的 SI-RNTI 未知控制区几何搜索与调用方持有的锁定缓存

SDK 当前不覆盖：

- PCFICH 译码
- PHICH 译码
- 非 `DCI 1A` 的通用 `DCI` 译码
- GPU UE-specific、SI-RNTI geometry search、其它 DCI format 或外部 decoder callback

GPU common-search 当前固定为 `20 MHz / FDD / normal CP / regular control subframe`。2Tx 请求
自动复用 TD 去映射并保持连续 CCE 顺序；设备到主机只返回 compact candidate hits，不返回完整
equalized grid、SINR 或 LLR。

## 页面摘要

### 快速开始

当你需要下面这些内容时，先看本页：

- 最小 `include` 路径
- 最小 DTO 流程
- 正式 CPU common-search、SI-RNTI、UE-specific 和未知 SI-RNTI 几何 `DCI 1A` 链路入口
- 紧凑集成步骤
- 已验证的 demo 构建路径

链接：

- [快速开始](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)

### API 参考

当你需要下面这些内容时，先看本页：

- 字段定义
- 单位
- helper 语义
- 边界条件
- 错误码含义
- 字段索引和状态码索引

链接：

- [API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)

### 版本策略

当你需要下面这些内容时，先看本页：

- 冻结的 v1 接口面
- 允许的增量变更
- 不允许的不兼容变更
- 如何引入 `v2`

链接：

- [版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
