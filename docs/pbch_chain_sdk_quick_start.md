# PBCH 快速开始

本页是集成 LTE PBCH equalized-RE 接口面的最快入口。该接口面由以下头文件导出：

```cpp
#include "mmse/lte_chain_sdk.h"
```

当前接口版本：

- `LTE Equalized Channel SDK v1`

相关页面：

- [LTE Equalized Channel SDK 文档](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [LTE 下行信道译码总览](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md)

## 作用范围

当前 PBCH 接口面提供：

- 面向中心 `72` 个子载波的 LTE PBCH RE 提取
- 基于 CRS 的信道估计
- MMSE 均衡
- 由调用方持有的输出 view 填充
- 面向下游透传的后端 DTO 打包
- `4Tx x 1Rx`、单层、`tx_mode == 2` 的 Td4 raw equalized output

当前 PBCH 接口面不提供：

- PBCH 解扰
- 速率恢复
- 尾咬卷积译码
- CRC 检查
- MIB 重建
- 4Tx owning backend DTO

## 最小流程

1. 上游准备一个 `mmse::pbch::FrontendPbchIndication`
2. 调用方把它转换成 `mmse::PbchMmseInput`
3. 调用方分配一个 `mmse::PbchMmseOutputView` 和一个 `mmse::PbchMmseResult`
4. 调用 `run_pbch(...)`
5. 把结果打包为 `mmse::pbch::BackendPbchEqualizedIndication`

## 最小示例

```cpp
#include "mmse/lte_chain_sdk.h"

mmse::pbch::FrontendPbchIndication frontend{};
frontend.sfn_subframe = 0;
frontend.cell_id = 0;
frontend.n_tx_ports = 1;
frontend.tx_mode = 1;

mmse::PlanarGridViewF32 grid = get_fft_grid();

mmse::PbchMmseInput in = mmse::pbch::make_pbch_mmse_input(grid, frontend);

std::vector<float> xhat_re(capacity);
std::vector<float> xhat_im(capacity);
std::vector<float> sinr(capacity);
std::vector<std::uint16_t> re_grid_indices(capacity);

mmse::PbchMmseOutputView out{};
out.x_hat_re = xhat_re.data();
out.x_hat_im = xhat_im.data();
out.sinr = sinr.data();
out.re_grid_indices = re_grid_indices.data();
out.capacity_re_per_layer = static_cast<std::uint32_t>(capacity);
out.capacity_re_metadata = static_cast<std::uint32_t>(capacity);

mmse::PbchMmseResult meta{};

mmse::MmseEqualizerCpuContext ctx;
mmse::MmseEqualizerCpuConfig cfg{};
cfg.worker_count = 1;
ctx.init(cfg);

const mmse::MmseStatus status = ctx.run_pbch(in, out, meta);
if (status != mmse::MmseStatus::kOk) {
    return;
}

mmse::pbch::BackendPbchEqualizedIndication backend =
    mmse::pbch::make_backend_pbch_equalized_indication(meta, out);
```

## 4Tx x 1Rx Td4 示例

Td4 使用相同的 `PbchMmseInput`，但要求 FFT 网格只有一个接收天线，并改用四源 RE 输出面：

```cpp
frontend.n_tx_ports = 4;
frontend.tx_mode = 2;
mmse::PbchMmseInput in = mmse::pbch::make_pbch_mmse_input(grid, frontend);

constexpr std::uint32_t capacity = 240;
std::vector<float> xhat_re(capacity), xhat_im(capacity), sinr(capacity);
std::vector<std::uint16_t> re0(capacity), re1(capacity), re2(capacity), re3(capacity);
mmse::PbchTd4MmseOutputView out{xhat_re.data(), xhat_im.data(), sinr.data(),
                re0.data(), re1.data(), re2.data(), re3.data(), capacity};
mmse::PbchTd4MmseResult meta{};

const mmse::MmseStatus status = ctx.run_pbch_td4(in, out, meta);
```

成功时 `meta.n_symbols == meta.n_source_re == 240`。对每个 `q = 0, 4, 8, ...`，
`re_grid_indices0..3[q..q+3]` 重复记录该 Td4 块的同一组四个 source RE；四个连续
`x_hat`/`sinr` 槽位是对应的 raw equalized QPSK 软符号。当前没有
`BackendPbchTd4EqualizedIndication` 或 MIB 最终译码入口，调用方应直接消费有效前缀或自行复制。

## 上游要求

上游必须提供：

- `sfn_subframe`
- `cell_id`
- `n_tx_ports`
- `tx_mode`
- 一份通过 `PlanarGridViewF32` 表示的 LTE 下行 FFT 网格

当前 wrapper 内部把 PBCH 提取边界固定为：

- 中心 `6 PRB`
- 从 symbol `7` 开始的 `4` 个 OFDM symbol
- QPSK（`mod_order == 2`）
- 单层 equalized 输出

## 下游透传

推荐下游透传对象：

- `mmse::pbch::BackendPbchEqualizedIndication`

重点字段：

- `x_hat_re`
- `x_hat_im`
- `sinr`
- `re_grid_indices`
- `start_prb`
- `n_prb`
- `start_symbol`

当前输出只表示 PBCH 的 equalized RE。后续 PBCH 译码仍然由下游外部完成。

## 容量规则

调用方必须保证：

- `capacity_re_per_layer >= expected_pbch_re`
- `capacity_re_metadata >= expected_pbch_re`

在当前 LTE 支持边界下：

- `expected_pbch_re == 240`
- `run_pbch_td4(...)` 使用 `capacity_symbols >= 240`，且 `meta.n_symbols == 240`

## 支持边界

当前已验证的支持范围：

- 仅 LTE
- 仅 20 MHz
- 仅 normal CP
- 普通 `run_pbch(...)`：`n_rx_ant == 1 or 2`，`n_tx_ports == 1`，`tx_mode == 1`
- TD2 `run_pbch_td(...)`：`n_rx_ant == 1 or 2`，`n_tx_ports == 2`，`tx_mode == 2`
- Td4 `run_pbch_td4(...)`：`n_rx_ant == 1`，`n_tx_ports == 4`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- `n_layers == 1`
- `mod_order == 2`
- Td4 额外要求 `tx_mode == 2`

## 状态码

- `MmseStatus::kOk`
  - PBCH 均衡成功完成
- `MmseStatus::kNotInitialized`
  - 尚未先调用 context `init(...)`
- `MmseStatus::kInvalidArgument`
  - 网格形状、指针或 PBCH wrapper 输入非法
- `MmseStatus::kUnsupportedConfig`
  - 请求超出当前 LTE PBCH 支持边界
- `MmseStatus::kBufferTooSmall`
  - 调用方输出容量小于提取得到的 PBCH RE 数
- `MmseStatus::kInternalError`
  - 运行时 / 内部传输失败
