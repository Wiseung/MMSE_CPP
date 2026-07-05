# PDSCH 下游 LLR / 解扰接口面快速开始与 API 参考

本页面向已经拿到 LTE `PDSCH` equalized 输出的调用方，说明仓库当前提供的
`soft demod + descrambling` 下游 helper。

该接口面由以下头文件导出：

```cpp
#include "mmse/lte_chain_sdk.h"
```

当前接口版本：

- `LTE Equalized Channel SDK v1`

相关页面：

- [LTE Equalized Channel SDK 文档](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [LTE 下行信道译码总览](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md)

## 作用范围

当前 `PDSCH` 下游 helper 提供：

- 基于 `EqualizerOutputView` 的 max-log `LLR` 生成
- 基于 LTE Gold 序列的 `PDSCH` 解扰
- 直接返回 `std::vector<float>` 的后端 DTO builder
- 调用方持有 `LLR` buffer 的 caller-owned 输出面
- 显式 `PdschDescramblingPlanCache`，用于复用同一 `(cell_id, rnti, sfn_subframe, codeword, llr_count)` 的 scrambling bits

当前 `PDSCH` 下游 helper 不提供：

- 速率恢复
- `HARQ` 软合并
- `Turbo` 译码
- `MAC PDU` 解析
- 仓库内置的真实 `PDSCH` 下游 context / grant context / decode context
- 隐藏的全局 cache

也就是说，这一层仍然只是：

**equalized `PDSCH` soft symbol / `SINR` -> 解扰后的 `LLR`**

而不是完整 `PDSCH` 译码链。

## 推荐用法

### 1. 直接后端 DTO builder

适合只需要一个一次性结果对象的集成方：

```cpp
#include "mmse/lte_chain_sdk.h"

mmse::ExtractDescriptor desc = get_pdsch_desc();
mmse::EqualizerOutputView eq_out = get_pdsch_equalized_output();
const std::uint16_t rnti = get_pdsch_rnti();

mmse::pdsch::BackendPdschDescrambledLlrIndication backend =
    mmse::pdsch::make_backend_pdsch_descrambled_llr_indication(desc, eq_out, rnti);
```

### 2. 显式 cache + caller-owned 输出面

适合下游真实存在 `PDSCH` grant / item / worker context，并希望复用 scrambling bits
和 `LLR` scratch buffer 的集成方：

```cpp
#include "mmse/lte_chain_sdk.h"

mmse::ExtractDescriptor desc = get_pdsch_desc();
mmse::EqualizerOutputView eq_out = get_pdsch_equalized_output();
const std::uint16_t rnti = get_pdsch_rnti();

mmse::pdsch::PdschDescramblingPlanCache plan{};
std::vector<float> llrs(max_llr_count);

mmse::pdsch::PdschDescrambledLlrOutputView llr_out{};
llr_out.llrs = llrs.data();
llr_out.capacity_llrs = static_cast<std::uint32_t>(llrs.size());

mmse::pdsch::PdschDescrambledLlrResult result{};

const mmse::MmseStatus status =
    mmse::pdsch::build_backend_pdsch_descrambled_llr_result(
        desc, eq_out, rnti, plan, llr_out, result);
if (status != mmse::MmseStatus::kOk) {
    return;
}
```

对于这条路径，`plan` 的所有权和生命周期由调用方控制。仓库当前不提供更高层的
`PDSCH` 下游 context。

## API 摘要

### 类型

#### `mmse::pdsch::PdschDescrambledLlrOutputView`

| 字段            | 类型       | 含义                   |
| --------------- | ---------- | ---------------------- |
| `llrs`          | `float*`   | 调用方持有的输出缓冲区 |
| `capacity_llrs` | `uint32_t` | 可写 `LLR` 数量上限    |

#### `mmse::pdsch::PdschDescrambledLlrResult`

| 字段                  | 含义                         |
| --------------------- | ---------------------------- |
| `sfn_subframe`        | LTE system frame/subframe id |
| `cell_id`             | LTE PCI                      |
| `n_prb`               | `PDSCH` 带宽 PRB 数          |
| `rnti`                | 当前 `PDSCH` 对应的 `RNTI`   |
| `prb_bitmap`          | PRB bitmap                   |
| `n_re_per_layer`      | 每层 equalized RE 数         |
| `llr_count_per_layer` | 每层 `LLR` 数                |
| `llr_count`           | 全部层的 `LLR` 总数          |
| `n_tx_ports`          | LTE 发射端口数               |
| `n_rx_ant`            | LTE 接收天线数               |
| `n_layers`            | 层数                         |
| `tx_mode`             | LTE 传输模式                 |
| `start_symbol`        | `PDSCH` 起始 symbol          |
| `mod_order`           | 调制阶数                     |
| `codeword`            | codeword id，默认 `0`        |
| `pmi`                 | `PMI` 透传                   |

#### `mmse::pdsch::PdschDescramblingPlanCache`

| 字段        | 含义                                     |
| ----------- | ---------------------------------------- |
| `c_init`    | 当前 cache 对应的 Gold sequence `c_init` |
| `llr_count` | 当前 cache 对应的 `LLR` 数               |
| `valid`     | 当前 cache 是否已初始化                  |
| `bits`      | 预生成并缓存的 scrambling bits           |

这个 cache 的匹配键是：

- `cell_id`
- `rnti`
- `sfn_subframe`
- `codeword`
- `llr_count`

只要这些键有任一变化，就需要重新生成 `bits`。

#### `mmse::pdsch::BackendPdschDescrambledLlrIndication`

这是仓库当前最直接的 `PDSCH` 下游后端 DTO。它会持有：

- 调度元数据
- `LLR` 向量

适合一次性打包后再传给仓库外部的下游。

### 函数

#### `prepare_pdsch_descrambling_plan(...)`

```cpp
MmseStatus prepare_pdsch_descrambling_plan(
    std::uint16_t cell_id,
    std::uint16_t rnti,
    std::uint32_t sfn_subframe,
    std::uint8_t codeword,
    std::uint32_t llr_count,
    PdschDescramblingPlanCache& plan);
```

作用：

- 根据当前 key 检查 cache 是否仍然命中
- 若未命中，则重建 `plan.bits`

#### `build_backend_pdsch_descrambled_llr_result(...)`

无 cache 版本：

```cpp
MmseStatus build_backend_pdsch_descrambled_llr_result(
    const ExtractDescriptor& desc,
    const EqualizerOutputView& out,
    std::uint16_t rnti,
    PdschDescrambledLlrOutputView& llr_out,
    PdschDescrambledLlrResult& result,
    std::uint8_t codeword = 0U);
```

显式 cache 版本：

```cpp
MmseStatus build_backend_pdsch_descrambled_llr_result(
    const ExtractDescriptor& desc,
    const EqualizerOutputView& out,
    std::uint16_t rnti,
    PdschDescramblingPlanCache& plan,
    PdschDescrambledLlrOutputView& llr_out,
    PdschDescrambledLlrResult& result,
    std::uint8_t codeword = 0U);
```

作用：

- 根据 equalized `PDSCH` 输出生成解扰后的 `LLR`
- 将元数据写入 `result`
- 将 `LLR` 写入调用方持有的 `llr_out`

#### `make_backend_pdsch_descrambled_llr_indication(...)`

无 cache 版本：

```cpp
BackendPdschDescrambledLlrIndication make_backend_pdsch_descrambled_llr_indication(
    const ExtractDescriptor& desc,
    const EqualizerOutputView& out,
    std::uint16_t rnti,
    std::uint8_t codeword = 0U);
```

显式 cache 版本：

```cpp
BackendPdschDescrambledLlrIndication make_backend_pdsch_descrambled_llr_indication(
    const ExtractDescriptor& desc,
    const EqualizerOutputView& out,
    std::uint16_t rnti,
    PdschDescramblingPlanCache& plan,
    std::uint8_t codeword = 0U);
```

作用：

- 内部分配并持有 `LLR` vector
- 返回后端 DTO

## 支持边界

当前 helper 约束：

- 输入必须已经是 equalized `PDSCH` 输出
- `mod_order` 仅支持 `2 / 4 / 6 / 8`
- `capacity_re_per_layer >= n_re_per_layer`
- caller-owned 输出面要求 `capacity_llrs >= n_layers * n_re_per_layer * mod_order`

当前 helper 的职责边界：

- 支持 `soft demod + descrambling`
- 不创建仓库内长期 `PDSCH` 下游 context
- 不管理多个 grant 间的 cache 生命周期

## 性能说明

当前仓库里，`PDSCH` 下游 helper 的对比测量入口在：

- [bench/mmse_channel_budget.cpp](G:\MMSE_CPP\bench\mmse_channel_budget.cpp)

与本接口面直接相关的 benchmark 标签包括：

- `pdsch_fused_llr_only`
- `pdsch_caller_owned_llr_only`
- `pdsch_cached_fused_llr_only`
- `pdsch_cached_caller_owned_llr_only`

在当前仓库修订上，推荐优先使用：

- 显式 `PdschDescramblingPlanCache`
- caller-owned `PdschDescrambledLlrOutputView`

因为这条组合在当前 benchmark 中是 `PDSCH` 下游 helper 的最快路径。
