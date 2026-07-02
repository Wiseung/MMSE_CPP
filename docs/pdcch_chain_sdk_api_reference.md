# PDCCH Chain SDK API 参考

本页是 LTE PDCCH CE/MMSE SDK 的字段级与调用级参考文档。

主要包含头文件：

```cpp
#include "mmse/pdcch_chain_sdk.h"
```

接口版本：

- `PDCCH Chain SDK v1`

相关页面：

- [LTE Equalized Channel SDK 文档](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [文档首页](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [快速开始](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)

## 目录

- [API 摘要](#api-摘要)
- [字段索引](#字段索引)
- [状态码索引](#状态码索引)
- [1. 公开入口](#1-公开入口)
- [2. DTO 定义](#2-dto-定义)
- [3. 基础网格类型](#3-基础网格类型)
- [4. Helper 语义](#4-helper-语义)
- [5. 边界条件](#5-边界条件)
- [6. 容量要求](#6-容量要求)
- [7. 错误码](#7-错误码)
- [8. 推荐调用流程](#8-推荐调用流程)

## API 摘要

主要运行时调用：

- `MmseEqualizerCpuContext::init`
- `MmseEqualizerCpuContext::run_pdcch`
- `MmseEqualizerCpuContext::run_pdcch_td`
- `MmseEqualizerGpuContext::init`
- `MmseEqualizerGpuContext::run_pdcch`
- `MmseEqualizerGpuContext::run_pdcch_td`

主要 DTO 流程：

1. 上游构造 `mmse::pdcch::FrontendPdcchIndication`
2. helper 把它转换成 `mmse::PdcchMmseInput`
3. CE/MMSE 阶段填充 `mmse::PdcchMmseOutputView` 和 `mmse::PdcchMmseResult`
4. helper 把它们打包成 `mmse::pdcch::BackendPdcchEqualizedIndication`

新增 `2 Tx port` TD 流程：

1. 上游构造 `mmse::pdcch::FrontendPdcchIndication`
2. helper 把它转换成 `mmse::PdcchMmseInput`
3. TD CE/MMSE 阶段填充 `mmse::PdcchTdMmseOutputView` 和 `mmse::PdcchTdMmseResult`
4. helper 把它们打包成 `mmse::pdcch::BackendPdcchTdEqualizedIndication`

## 字段索引

这个索引用于快速定位字段归属和语义。

### 前端 DTO 字段

| 字段                   | 所属类型                  | 对应章节                                    |
| ---------------------- | ------------------------- | ------------------------------------------- |
| `sfn_subframe`         | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `cell_id`              | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `n_tx_ports`           | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `tx_mode`              | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `control_symbol_count` | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `n_prb`                | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `prb_bitmap`           | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `control_subframe`     | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `reserved_control_res` | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `chain`                | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |

### 保留 RE 字段

| 字段          | 所属类型            | 对应章节                              |
| ------------- | ------------------- | ------------------------------------- |
| `symbol`      | `ReservedControlRe` | [2.1](#21-mmsepdcchreservedcontrolre) |
| `prb`         | `ReservedControlRe` | [2.1](#21-mmsepdcchreservedcontrolre) |
| `tone_in_prb` | `ReservedControlRe` | [2.1](#21-mmsepdcchreservedcontrolre) |

### 链路元数据字段

| 字段                | 所属类型             | 对应章节                          |
| ------------------- | -------------------- | --------------------------------- |
| `request_id`        | `PdcchChainMetadata` | [2.2](#22-mmsepdcchchainmetadata) |
| `candidate_id`      | `PdcchChainMetadata` | [2.2](#22-mmsepdcchchainmetadata) |
| `first_cce`         | `PdcchChainMetadata` | [2.2](#22-mmsepdcchchainmetadata) |
| `aggregation_level` | `PdcchChainMetadata` | [2.2](#22-mmsepdcchchainmetadata) |

### 低层输入字段

| 字段                         | 所属类型         | 对应章节                      |
| ---------------------------- | ---------------- | ----------------------------- |
| `grid`                       | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `sfn_subframe`               | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `cell_id`                    | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `n_tx_ports`                 | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `tx_mode`                    | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `control_symbol_count`       | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `n_prb`                      | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `prb_bitmap`                 | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `control_re_exclusion_masks` | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `chain`                      | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |

### 输出 View 字段

| 字段                    | 所属类型              | 对应章节                           |
| ----------------------- | --------------------- | ---------------------------------- |
| `x_hat_re`              | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `x_hat_im`              | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `sinr`                  | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `re_grid_indices`       | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `capacity_re_per_layer` | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `capacity_re_metadata`  | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |

### 结果元数据字段

| 字段                   | 所属类型          | 对应章节                       |
| ---------------------- | ----------------- | ------------------------------ |
| `n_re`                 | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `sfn_subframe`         | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_symbols`            | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_subcarriers`        | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `cell_id`              | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_prb`                | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_tx_ports`           | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_rx_ant`             | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_layers`             | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `tx_mode`              | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `control_symbol_count` | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `mod_order`            | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `sigma2`               | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `prb_bitmap`           | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `chain`                | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |

### 后端 DTO 字段

| 字段                   | 所属类型                          | 对应章节                                            |
| ---------------------- | --------------------------------- | --------------------------------------------------- |
| `sfn_subframe`         | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `cell_id`              | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `n_prb`                | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `n_tx_ports`           | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `n_rx_ant`             | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `n_layers`             | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `tx_mode`              | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `control_symbol_count` | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `mod_order`            | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `sigma2`               | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `chain`                | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `x_hat_re`             | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `x_hat_im`             | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `sinr`                 | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `re_grid_indices`      | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |

### 网格字段

| 字段            | 所属类型            | 对应章节               |
| --------------- | ------------------- | ---------------------- |
| `re`            | `PlanarGridViewF32` | [3](#3-base-grid-type) |
| `im`            | `PlanarGridViewF32` | [3](#3-base-grid-type) |
| `n_rx_ant`      | `PlanarGridViewF32` | [3](#3-base-grid-type) |
| `n_symbols`     | `PlanarGridViewF32` | [3](#3-base-grid-type) |
| `n_subcarriers` | `PlanarGridViewF32` | [3](#3-base-grid-type) |

## 状态码索引

| 状态码                           | 摘要                                 | 详细章节            |
| -------------------------------- | ------------------------------------ | ------------------- |
| `MmseStatus::kOk`                | 调用成功                             | [7](#7-error-codes) |
| `MmseStatus::kNotInitialized`    | 使用前未初始化 context               | [7](#7-error-codes) |
| `MmseStatus::kInvalidArgument`   | 调用方提供的指针、配置或输入参数非法 | [7](#7-error-codes) |
| `MmseStatus::kUnsupportedConfig` | 请求超出当前 LTE PDCCH 支持边界      | [7](#7-error-codes) |
| `MmseStatus::kBufferTooSmall`    | 调用方输出容量不足                   | [7](#7-error-codes) |
| `MmseStatus::kInternalError`     | 运行时 / 内部传输或校验失败          | [7](#7-error-codes) |

## 1. 公开入口

CPU：

```cpp
mmse::MmseEqualizerCpuContext::init(const MmseEqualizerCpuConfig& config)
mmse::MmseEqualizerCpuContext::run_pdcch(
    const PdcchMmseInput& in,
    PdcchMmseOutputView& out,
    PdcchMmseResult& meta)
mmse::MmseEqualizerCpuContext::run_pdcch_td(
    const PdcchMmseInput& in,
    PdcchTdMmseOutputView& out,
    PdcchTdMmseResult& meta)
```

GPU：

```cpp
mmse::MmseEqualizerGpuContext::init(const MmseEqualizerGpuConfig& config)
mmse::MmseEqualizerGpuContext::run_pdcch(
    const PdcchMmseInput& in,
    PdcchMmseOutputView& out,
    PdcchMmseResult& meta)
mmse::MmseEqualizerGpuContext::run_pdcch_td(
    const PdcchMmseInput& in,
    PdcchTdMmseOutputView& out,
    PdcchTdMmseResult& meta)
```

DTO / helper 工具：

```cpp
mmse::pdcch::make_pdcch_mmse_input(...)
mmse::pdcch::make_backend_pdcch_equalized_indication(...)
mmse::pdcch::make_backend_pdcch_td_equalized_indication(...)
mmse::pdcch::decode_re_grid_index(...)
mmse::pdcch::append_pcfich_reserved_control_re_list(...)
mmse::pdcch::append_phich_reserved_control_re_list(...)
mmse::pdcch::append_fdd_phich_reserved_control_re_list(...)
```

## 2. DTO 定义

### 2.1 `mmse::pdcch::ReservedControlRe`

表示一个由上游识别为非 PDCCH 信道占用的控制区 RE。

| 字段          | 类型       | 含义                    | 单位         | 合法范围                                          |
| ------------- | ---------- | ----------------------- | ------------ | ------------------------------------------------- |
| `symbol`      | `uint32_t` | 子帧内 OFDM symbol 索引 | symbol index | `0..13`，实际通常使用 `0..control_symbol_count-1` |
| `prb`         | `uint32_t` | PRB 索引                | PRB index    | `0..99`                                           |
| `tone_in_prb` | `uint32_t` | 一个 PRB 内的 tone 索引 | tone index   | `0..11`                                           |

说明：

- 不应在这里列出 CRS RE，SDK 会在内部自动排除 CRS
- 典型用途是标记 `PCFICH` 与 `PHICH` 占用的 RE

### 2.1a `mmse::pdcch::PhichResource`

用于自动生成 FDD PHICH 保留信息的 helper 枚举。

| 枚举值      | 含义           |
| ----------- | -------------- |
| `kOneSixth` | LTE `Ng = 1/6` |
| `kHalf`     | LTE `Ng = 1/2` |
| `kOne`      | LTE `Ng = 1`   |
| `kTwo`      | LTE `Ng = 2`   |

### 2.1b `mmse::pdcch::PhichDuplexMode`

用于自动生成 PHICH 保留信息的 helper 枚举。

| 枚举值 | 含义    |
| ------ | ------- |
| `kFdd` | LTE FDD |
| `kTdd` | LTE TDD |

### 2.1c `mmse::pdcch::PhichDuration`

用于自动生成 PHICH 保留信息的 helper 枚举。

| 枚举值      | 含义                |
| ----------- | ------------------- |
| `kNormal`   | 普通 PHICH 持续长度 |
| `kExtended` | 扩展 PHICH 持续长度 |

### 2.1d `mmse::pdcch::PhichSubframeKind`

用于 PHICH helper 输入的结构化子帧类型枚举。

| 枚举值     | 含义       |
| ---------- | ---------- |
| `kRegular` | 普通子帧   |
| `kMbsfn`   | MBSFN 子帧 |

### 2.1e `mmse::pdcch::LteControlSubframeContext`

`PCFICH/PHICH` helper 共享的 LTE 控制子帧上下文。

| 字段           | 类型                | 含义                  |
| -------------- | ------------------- | --------------------- |
| `duplex_mode`  | `PhichDuplexMode`   | LTE 双工模式          |
| `subframe`     | `uint8_t`           | 子帧索引              |
| `ul_dl_config` | `uint8_t`           | LTE TDD 上下行配置    |
| `kind`         | `PhichSubframeKind` | 普通 / MBSFN 子帧类型 |

### 2.1f `mmse::pdcch::PhichReservationConfig`

用于自动生成 PHICH 保留信息的 helper 配置对象。

| 字段           | 类型                        | 含义               |
| -------------- | --------------------------- | ------------------ |
| `resource`     | `PhichResource`             | LTE `Ng`           |
| `duration`     | `PhichDuration`             | LTE PHICH duration |
| `mi`           | `uint8_t`                   | LTE PHICH `M_i`    |
| `subframe_ctx` | `LteControlSubframeContext` | 子帧上下文         |

### 2.2 `mmse::PdcchChainMetadata`

贯穿 CE/MMSE 阶段并返回给下游的透明链路元数据。

| 字段                | 类型       | 含义                       | 单位      | 合法范围     |
| ------------------- | ---------- | -------------------------- | --------- | ------------ |
| `request_id`        | `uint64_t` | 调用方自定义关联 ID        | none      | 任意         |
| `candidate_id`      | `uint32_t` | 调用方自定义 PDCCH 候选 ID | none      | 任意         |
| `first_cce`         | `uint16_t` | 调用方自定义首个 CCE 索引  | CCE index | 调用方自定义 |
| `aggregation_level` | `uint8_t`  | 调用方自定义聚合级别       | CCE count | 调用方自定义 |

说明：

- CE/MMSE 模块本身不解释这些字段
- 它们会完整保留给下游逻辑使用

### 2.3 `mmse::pdcch::FrontendPdcchIndication`

正式的上游 DTO，用于描述一次 PDCCH 均衡请求。

| 字段                   | 类型                        | 含义                      | 单位               | 合法范围                       |
| ---------------------- | --------------------------- | ------------------------- | ------------------ | ------------------------------ |
| `sfn_subframe`         | `uint32_t`                  | 无线时间索引              | SFN\*10 + subframe | 调用方给定的非负值             |
| `cell_id`              | `uint16_t`                  | LTE 物理小区 ID           | none               | `0..503`                       |
| `n_tx_ports`           | `uint8_t`                   | 本次请求的 CRS 发射端口数 | ports              | 当前 SDK 支持 `1`              |
| `tx_mode`              | `uint8_t`                   | LTE 发射模式标签          | none               | 当前 SDK 支持 `1` 或 `2`       |
| `control_symbol_count` | `uint8_t`                   | LTE 控制区大小            | OFDM symbols       | `1..3`                         |
| `n_prb`                | `uint16_t`                  | 位图中启用的 PRB 数       | PRBs               | `1..100`                       |
| `prb_bitmap`           | `array<uint16_t,7>`         | 激活 PRB 位图             | bitmap             | 必须恰好有 `n_prb` 个 bit 置位 |
| `control_subframe`     | `LteControlSubframeContext` | 共享 LTE 控制子帧上下文   | context            | helper 定义范围                |
| `reserved_control_res` | `vector<ReservedControlRe>` | 非 PDCCH 控制 RE 列表     | RE list            | 零个或多个                     |
| `chain`                | `PdcchChainMetadata`        | 透传元数据                | none               | 任意                           |

### 2.4 `mmse::PdcchMmseInput`

helper 转换后的低层模块输入。

| 字段                         | 类型                        | 含义                       | 单位               | 合法范围                       |
| ---------------------------- | --------------------------- | -------------------------- | ------------------ | ------------------------------ |
| `grid`                       | `PlanarGridViewF32`         | FFT 网格输入               | complex float32    | 见 `PlanarGridViewF32`         |
| `sfn_subframe`               | `uint32_t`                  | 无线时间索引               | SFN\*10 + subframe | 非负                           |
| `cell_id`                    | `uint16_t`                  | LTE 物理小区 ID            | none               | `0..503`                       |
| `n_tx_ports`                 | `uint8_t`                   | 发射端口数                 | ports              | 当前 SDK 支持 `1`              |
| `tx_mode`                    | `uint8_t`                   | 发射模式标签               | none               | 当前 SDK 支持 `1` 或 `2`       |
| `control_symbol_count`       | `uint8_t`                   | LTE 控制区大小             | OFDM symbols       | `1..3`                         |
| `n_prb`                      | `uint16_t`                  | 激活 PRB 数                | PRBs               | `1..100`                       |
| `prb_bitmap`                 | `array<uint16_t,7>`         | 激活 PRB 位图              | bitmap             | 必须恰好有 `n_prb` 个 bit 置位 |
| `control_subframe`           | `LteControlSubframeContext` | 共享 LTE 控制子帧上下文    | context            | helper 定义范围                |
| `control_re_exclusion_masks` | `array<uint16_t,300>`       | 按 symbol / PRB 的 RE mask | bitmask            | 每项只使用 12 bit              |
| `chain`                      | `PdcchChainMetadata`        | 透传元数据                 | none               | 任意                           |

Mask 映射：

- index = `symbol * 100 + prb`
- bit `0..11` 对应 `tone_in_prb`

### 2.5 `mmse::PdcchMmseOutputView`

`run_pdcch` 的调用方持有写入目标。

| 字段                    | 类型        | 含义                              | 单位         | 必需 |
| ----------------------- | ----------- | --------------------------------- | ------------ | ---- |
| `x_hat_re`              | `float*`    | 均衡后符号实部                    | float32      | 是   |
| `x_hat_im`              | `float*`    | 均衡后符号虚部                    | float32      | 是   |
| `sinr`                  | `float*`    | 均衡后 SINR                       | linear ratio | 是   |
| `re_grid_indices`       | `uint16_t*` | LTE 网格中的源 RE 位置            | grid index   | 是   |
| `capacity_re_per_layer` | `uint32_t`  | `x_hat_re/x_hat_im/sinr` 可写容量 | RE count     | 是   |
| `capacity_re_metadata`  | `uint32_t`  | `re_grid_indices` 可写容量        | RE count     | 是   |

### 2.6 `mmse::PdcchMmseResult`

`run_pdcch` 返回的逐调用元数据。

| 字段                   | 类型                 | 含义                       | 单位               |
| ---------------------- | -------------------- | -------------------------- | ------------------ |
| `n_re`                 | `uint32_t`           | 均衡后的 PDCCH RE 数       | RE count           |
| `sfn_subframe`         | `uint32_t`           | 无线时间索引               | SFN\*10 + subframe |
| `n_symbols`            | `uint32_t`           | 网格 symbol 维度           | symbols            |
| `n_subcarriers`        | `uint32_t`           | 网格子载波维度             | subcarriers        |
| `cell_id`              | `uint16_t`           | LTE 小区 ID                | none               |
| `n_prb`                | `uint16_t`           | 激活 PRB 数                | PRBs               |
| `n_tx_ports`           | `uint8_t`            | 发射端口数                 | ports              |
| `n_rx_ant`             | `uint8_t`            | 接收天线数                 | antennas           |
| `n_layers`             | `uint8_t`            | 输出层数                   | layers             |
| `tx_mode`              | `uint8_t`            | 发射模式                   | none               |
| `control_symbol_count` | `uint8_t`            | LTE 控制区大小             | OFDM symbols       |
| `mod_order`            | `uint8_t`            | 调制阶数，单位 bits/symbol | bits/symbol        |
| `sigma2`               | `float`              | 运行时噪声方差估计         | linear power       |
| `prb_bitmap`           | `array<uint16_t,7>`  | 激活 PRB 位图              | bitmap             |
| `chain`                | `PdcchChainMetadata` | 透传元数据                 | none               |

### 2.7 `mmse::pdcch::BackendPdcchEqualizedIndication`

helper 打包得到的正式下游 owning DTO。

| 字段                   | 类型                 | 含义               | 单位               |
| ---------------------- | -------------------- | ------------------ | ------------------ |
| `sfn_subframe`         | `uint32_t`           | 无线时间索引       | SFN\*10 + subframe |
| `cell_id`              | `uint16_t`           | LTE 小区 ID        | none               |
| `n_prb`                | `uint16_t`           | 激活 PRB 数        | PRBs               |
| `n_tx_ports`           | `uint8_t`            | 发射端口数         | ports              |
| `n_rx_ant`             | `uint8_t`            | 接收天线数         | antennas           |
| `n_layers`             | `uint8_t`            | 输出层数           | layers             |
| `tx_mode`              | `uint8_t`            | 发射模式           | none               |
| `control_symbol_count` | `uint8_t`            | LTE 控制区大小     | OFDM symbols       |
| `mod_order`            | `uint8_t`            | 调制阶数           | bits/symbol        |
| `sigma2`               | `float`              | 运行时噪声方差估计 | linear power       |
| `chain`                | `PdcchChainMetadata` | 透传元数据         | none               |
| `x_hat_re`             | `vector<float>`      | 均衡后实部         | float32            |
| `x_hat_im`             | `vector<float>`      | 均衡后虚部         | float32            |
| `sinr`                 | `vector<float>`      | 按 RE 的 SINR      | linear ratio       |
| `re_grid_indices`      | `vector<uint16_t>`   | LTE 网格源索引     | grid index         |

不变量：

- `x_hat_re.size() == x_hat_im.size() == sinr.size() == re_grid_indices.size()`

## 3. 基础网格类型

### `mmse::PlanarGridViewF32`

| 字段            | 含义                   | 单位           | 约束                 |
| --------------- | ---------------------- | -------------- | -------------------- |
| `re[0..1]`      | 每个 RX 天线的实部平面 | float32 arrays | 非空                 |
| `im[0..1]`      | 每个 RX 天线的虚部平面 | float32 arrays | 非空                 |
| `n_rx_ant`      | 接收天线数             | antennas       | 当前 SDK 要求 `2`    |
| `n_symbols`     | 网格 symbol 数         | symbols        | 当前 SDK 要求 `14`   |
| `n_subcarriers` | 网格宽度               | subcarriers    | 当前 SDK 要求 `1200` |

内存布局：

- 一个平面的索引方式是 `symbol * 1200 + subcarrier`
- 所有值都以线性 float32 复数平面存储

## 4. Helper 语义

### `make_pdcch_mmse_input(grid, frontend)`

从一份网格和一份 frontend DTO 构建 `PdcchMmseInput`。

行为：

- 复制 `FrontendPdcchIndication` 字段
- 把 `control_subframe` 复制到 `PdcchMmseInput`
- 把 `reserved_control_res` 转换成 `control_re_exclusion_masks`
- 后续 `run_pdcch` 会对这份低层请求调用集中式 `PdcchMmseInput` 校验器
- 该 helper 本身不验证 LTE 语义；语义校验发生在 `run_pdcch`

校验器族：

- `validate_lte_control_subframe_context(...)`
- `validate_phich_reservation_config(...)`
- `validate_pdcch_mmse_input(...)`

### `append_pcfich_reserved_control_re_list(...)`

构建并追加 LTE PCFICH 占用的控制 RE 到 `reserved_control_res`。

行为：

- 接收共享的 `LteControlSubframeContext`
- `FrontendPdcchIndication` 重载默认使用 `frontend.control_subframe`
- 面向当前 20 MHz normal-CP helper 支持边界
- 只标记四个 PCFICH REG 中非 CRS 的 RE
- 保留调用方已提供的条目
- 自动去重

### `append_phich_reserved_control_re_list(...)`

使用显式 `PhichReservationConfig` 构建并追加 PHICH 占用的控制 RE 到 `reserved_control_res`。

行为：

- 当 helper 支持所请求配置时，返回 `mmse::MmseStatus::kOk`
- 对于非法 helper 输入，例如 `subframe > 9`、`ul_dl_config > 6` 或 TDD `mi` 不匹配时，
  返回 `mmse::MmseStatus::kInvalidArgument`
- 当前已验证 helper 支持范围为 `FDD/TDD + normal CP + normal/extended PHICH duration`
- 在 TDD 模式下，`mi` 必须与 `subframe_ctx.ul_dl_config + subframe_ctx.subframe` 匹配
- 对 extended duration，`TDD subframe 1/6` 特例会自动选中
- 真实 `MBSFN` 子帧通过 `subframe_ctx.kind = kMbsfn` 选择
- `FrontendPdcchIndication` 的便捷重载可以从 `frontend.control_subframe` 读取同一份上下文
- 保留调用方已提供的条目
- 自动去重

### `append_fdd_phich_reserved_control_re_list(...)`

构建并追加 LTE FDD normal-duration PHICH 占用的控制 RE 到 `reserved_control_res`。

行为：

- 是 `append_phich_reserved_control_re_list(...)` 的便捷包装
- 面向当前 20 MHz FDD normal-CP helper 支持边界
- 消费 `PhichResource` 以推导 PHICH group 数
- 只标记选定 PHICH REG 内非 CRS 的 RE
- 保留调用方已提供的条目
- 自动去重

### `make_backend_pdcch_equalized_indication(meta, out)`

从下面两项构建一个 owning 的下游 DTO：

- `PdcchMmseResult`
- `PdcchMmseOutputView`

行为：

- 复制元数据
- 深拷贝 `x_hat_re`、`x_hat_im`、`sinr` 和 `re_grid_indices`
- 输出向量大小严格等于 `meta.n_re`

### `decode_re_grid_index(grid_index)`

把一个 LTE 网格索引解码为：

- `symbol`
- `subcarrier`
- `prb`
- `tone_in_prb`

公式：

- `symbol = grid_index / 1200`
- `subcarrier = grid_index % 1200`
- `prb = subcarrier / 12`
- `tone_in_prb = subcarrier % 12`

## 5. 边界条件

当前 SDK 支持边界：

- 仅 LTE
- 仅 20 MHz
- 仅 normal CP
- `n_rx_ant == 2`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- 仅 PDCCH
- `control_symbol_count in [1, 3]`
- 对 PDCCH 而言 `mod_order == 2`
- `n_layers == 1`
- `n_tx_ports == 1`
- `tx_mode == 1 or 2`

重要的不支持项：

- 传统 `run_pdcch(...)` 会拒绝 `2 Tx port` LTE PDCCH，因为其冻结契约是 per-RE
- `2 Tx port` 发射分集只通过新增的 `run_pdcch_td(...)` 支持
- SDK 不负责解码 `PCFICH` 或 `PHICH`
- 基于 helper 的自动 PHICH 保留只限于文档化的 FDD normal-CP 边界

## 6. 容量要求

调用 `run_pdcch` 之前，调用方必须提供足够的输出容量：

- `capacity_re_per_layer >= expected_n_re`
- `capacity_re_metadata >= expected_n_re`

如果容量不足，调用会返回 `kBufferTooSmall`。

如果暂时无法精确知道 `expected_n_re`，则应针对 LTE normal-CP 控制区场景做保守分配。

## 7. 错误码

### `mmse::MmseStatus::kOk`

调用成功。

### `mmse::MmseStatus::kNotInitialized`

原因：

- 在调用 `run_pdcch` 前没有先调用 `init`

处理建议：

- 先调用 `init(...)`

### `mmse::MmseStatus::kInvalidArgument`

典型原因：

- `PdcchMmseOutputView` 中存在空指针
- `MmseEqualizerCpuConfig` 或 `MmseEqualizerGpuConfig` 中配置值非法
- 网格指针布局非法

处理建议：

- 检查指针和配置取值范围

### `mmse::MmseStatus::kUnsupportedConfig`

典型原因：

- 非 LTE 网格维度
- `cell_id > 503`
- `control_symbol_count` 不在 `1..3`
- 对 PDCCH 而言 `mod_order != 2`
- `n_tx_ports != 1`
- `n_layers != 1`
- backend 选择不受支持

处理建议：

- 把请求限制在当前支持的 LTE PDCCH 边界内

### `mmse::MmseStatus::kBufferTooSmall`

典型原因：

- `capacity_re_per_layer < n_re`
- `capacity_re_metadata < n_re`

处理建议：

- 分配更大的输出缓冲区

### `mmse::MmseStatus::kInternalError`

典型原因：

- 内部传输或校验失败
- 意外的 CUDA 运行时状态不一致

处理建议：

- 视为运行时失败，并结合日志或 debug 校验路径定位

## 8. 推荐调用流程

```cpp
#include "mmse/pdcch_chain_sdk.h"

mmse::pdcch::FrontendPdcchIndication frontend = get_frontend_indication();
mmse::PlanarGridViewF32 grid = get_fft_grid();

mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);

mmse::PdcchMmseOutputView out = allocate_output_views();
mmse::PdcchMmseResult meta{};

mmse::MmseEqualizerCpuContext ctx;
ctx.init(cpu_cfg);
ctx.run_pdcch(in, out, meta);

mmse::pdcch::BackendPdcchEqualizedIndication backend =
    mmse::pdcch::make_backend_pdcch_equalized_indication(meta, out);
```
