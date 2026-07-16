# PCFICH 快速开始与 API 参考

本页是 LTE PCFICH equalized-RE 接口面的快速开始与字段参考页。该接口面由以下头文件导出：

```cpp
#include "mmse/lte_chain_sdk.h"
```

当前接口版本：

- `LTE Equalized Channel SDK v1`

相关页面：

- [LTE Equalized Channel SDK 文档](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [LTE 下行信道译码总览](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md)

## 作用范围

当前 PCFICH 接口面提供：

- 在 symbol `0` 内进行 LTE PCFICH REG/RE 提取
- 与仓库 helper 语义一致的、基于 CRS 感知的 occupied-RE 去除
- 基于 CRS 的信道估计
- MMSE 均衡
- 由调用方持有的输出 view 填充
- 面向下游透传的后端 DTO 打包
- `4Tx x 1Rx`、单层、`tx_mode == 2` 的 Td4 raw equalized output

当前 PCFICH 接口面不提供：

- CFI 比特解扰
- 调制解映射到最终 CFI 值
- 除 equalized RE 接口面之外的信道译码
- 4Tx owning backend DTO

## 最小流程

1. 上游准备一个 `mmse::pcfich::FrontendPcfichIndication`
2. 调用方把它转换成 `mmse::PcfichMmseInput`
3. 调用方分配一个 `mmse::PcfichMmseOutputView` 和一个 `mmse::PcfichMmseResult`
4. 调用 `run_pcfich(...)`
5. 把结果打包为 `mmse::pcfich::BackendPcfichEqualizedIndication`

## 最小示例

```cpp
#include "mmse/lte_chain_sdk.h"

mmse::pcfich::FrontendPcfichIndication frontend{};
frontend.sfn_subframe = 0;
frontend.cell_id = 0;
frontend.n_tx_ports = 1;
frontend.tx_mode = 1;

mmse::PlanarGridViewF32 grid = get_fft_grid();

mmse::PcfichMmseInput in = mmse::pcfich::make_pcfich_mmse_input(grid, frontend);

std::vector<float> xhat_re(capacity);
std::vector<float> xhat_im(capacity);
std::vector<float> sinr(capacity);
std::vector<std::uint16_t> re_grid_indices(capacity);

mmse::PcfichMmseOutputView out{};
out.x_hat_re = xhat_re.data();
out.x_hat_im = xhat_im.data();
out.sinr = sinr.data();
out.re_grid_indices = re_grid_indices.data();
out.capacity_re_per_layer = static_cast<std::uint32_t>(capacity);
out.capacity_re_metadata = static_cast<std::uint32_t>(capacity);

mmse::PcfichMmseResult meta{};

mmse::MmseEqualizerCpuContext ctx;
mmse::MmseEqualizerCpuConfig cfg{};
cfg.worker_count = 1;
ctx.init(cfg);

const mmse::MmseStatus status = ctx.run_pcfich(in, out, meta);
if (status != mmse::MmseStatus::kOk) {
    return;
}

mmse::pcfich::BackendPcfichEqualizedIndication backend =
    mmse::pcfich::make_backend_pcfich_equalized_indication(meta, out);
```

## 4Tx x 1Rx Td4 快速开始

```cpp
frontend.n_tx_ports = 4;
frontend.tx_mode = 2;
mmse::PcfichMmseInput in = mmse::pcfich::make_pcfich_mmse_input(grid, frontend);

constexpr std::uint32_t capacity = 16;
std::vector<float> xhat_re(capacity), xhat_im(capacity), sinr(capacity);
std::vector<std::uint16_t> re0(capacity), re1(capacity), re2(capacity), re3(capacity);
mmse::PcfichTd4MmseOutputView out{xhat_re.data(), xhat_im.data(), sinr.data(),
                  re0.data(), re1.data(), re2.data(), re3.data(), capacity};
mmse::PcfichTd4MmseResult meta{};

const mmse::MmseStatus status = ctx.run_pcfich_td4(in, out, meta);
```

成功时 `meta.n_symbols == meta.n_source_re == 16`。对每个 `q = 0, 4, 8, 12`，
四个索引数组在槽位 `q..q+3` 重复记录同一组 source RE，四个连续输出是对应的 raw
equalized QPSK 软符号。当前没有 `BackendPcfichTd4EqualizedIndication`，也不提供 CFI
最终判决；调用方直接消费 `meta.n_symbols` 指定的有效前缀。

## API 摘要

主要运行时调用：

- `MmseEqualizerCpuContext::run_pcfich(...)`
- `MmseEqualizerGpuContext::run_pcfich(...)`
- `MmseEqualizerCpuContext::run_pcfich_td(...)`
- `MmseEqualizerGpuContext::run_pcfich_td(...)`
- `MmseEqualizerCpuContext::run_pcfich_td4(...)`
- `MmseEqualizerGpuContext::run_pcfich_td4(...)`

主要 DTO 流程：

1. 上游构造 `mmse::pcfich::FrontendPcfichIndication`
2. helper 把它转换成 `mmse::PcfichMmseInput`
3. CE/MMSE 阶段填充 `mmse::PcfichMmseOutputView` 和 `mmse::PcfichMmseResult`
4. helper 把它们打包成 `mmse::pcfich::BackendPcfichEqualizedIndication`

## DTO 定义

### `mmse::pcfich::FrontendPcfichIndication`

| 字段           | 类型       | 含义                         |
| -------------- | ---------- | ---------------------------- |
| `sfn_subframe` | `uint32_t` | LTE system frame/subframe id |
| `cell_id`      | `uint16_t` | LTE PCI                      |
| `n_tx_ports`   | `uint8_t`  | LTE transmit antenna ports   |
| `tx_mode`      | `uint8_t`  | LTE transmit mode hint       |
| `chain`        | metadata   | caller passthrough metadata  |

### `mmse::PcfichMmseInput`

| 字段           | 类型                | 含义                         |
| -------------- | ------------------- | ---------------------------- |
| `grid`         | `PlanarGridViewF32` | LTE FFT grid                 |
| `sfn_subframe` | `uint32_t`          | LTE system frame/subframe id |
| `cell_id`      | `uint16_t`          | LTE PCI                      |
| `n_tx_ports`   | `uint8_t`           | LTE transmit antenna ports   |
| `tx_mode`      | `uint8_t`           | LTE transmit mode hint       |
| `chain`        | metadata            | caller passthrough metadata  |

### `mmse::PcfichMmseOutputView`

| 字段                    | 类型        | 含义                         |
| ----------------------- | ----------- | ---------------------------- |
| `x_hat_re`              | `float*`    | 由调用方持有的均衡后实部平面 |
| `x_hat_im`              | `float*`    | 由调用方持有的均衡后虚部平面 |
| `sinr`                  | `float*`    | 由调用方持有的 SINR 平面     |
| `re_grid_indices`       | `uint16_t*` | 由调用方持有的源 RE 索引     |
| `capacity_re_per_layer` | `uint32_t`  | 可写 equalized RE 数上限     |
| `capacity_re_metadata`  | `uint32_t`  | 可写 metadata RE 数上限      |

### `mmse::PcfichMmseResult`

| 字段            | 含义                         |
| --------------- | ---------------------------- |
| `n_re`          | 提取 / 均衡后的 PCFICH RE 数 |
| `sfn_subframe`  | LTE system frame/subframe id |
| `n_symbols`     | 网格 OFDM symbol 数          |
| `n_subcarriers` | 网格子载波数                 |
| `cell_id`       | LTE PCI                      |
| `n_prb`         | 当前支持边界下的带宽 PRB 数  |
| `start_symbol`  | 当前 PCFICH symbol 索引      |
| `reg_count`     | 当前边界下的 PCFICH REG 数   |
| `n_tx_ports`    | LTE 发射天线端口数           |
| `n_rx_ant`      | LTE 接收天线数               |
| `n_layers`      | 均衡层数                     |
| `tx_mode`       | LTE 发射模式提示             |
| `mod_order`     | 调制阶数                     |
| `sigma2`        | 运行时噪声估计               |
| `chain`         | 调用方透传元数据             |

### `mmse::PcfichTd4MmseOutputView / PcfichTd4MmseResult`

Td4 output view 以 `x_hat_re/x_hat_im/sinr` 加 `re_grid_indices0..3` 表示四源 RE 块，
容量字段为 `capacity_symbols`。result 使用 `n_symbols` 和 `n_source_re` 标记有效长度，
其余网格、端口、模式和链路元数据字段与 TD 输出语义一致。

### `mmse::pcfich::BackendPcfichEqualizedIndication`

这个后端 DTO 会复制并持有输出向量，供下游透传使用。

重点字段：

- `x_hat_re`
- `x_hat_im`
- `sinr`
- `re_grid_indices`
- `reg_count`
- `start_symbol`

## 容量规则

调用方必须保证：

- `capacity_re_per_layer >= expected_pcfich_re`
- `capacity_re_metadata >= expected_pcfich_re`

在当前 LTE 支持边界下：

- `expected_pcfich_re == 16`
- `run_pcfich_td4(...)` 使用 `capacity_symbols >= 16`，且 `meta.n_symbols == 16`

## 支持边界

当前已验证的支持范围：

- 仅 LTE
- 仅 20 MHz
- 仅 normal CP
- 普通 `run_pcfich(...)`：`n_rx_ant == 1 or 2`，`n_tx_ports == 1`，`tx_mode == 1`
- TD2 `run_pcfich_td(...)`：`n_rx_ant == 1 or 2`，`n_tx_ports == 2`，`tx_mode == 2`
- Td4 `run_pcfich_td4(...)`：`n_rx_ant == 1`，`n_tx_ports == 4`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- `n_layers == 1`
- `mod_order == 2`
- `start_symbol == 0`
- `reg_count == 4`
- Td4 额外要求 `tx_mode == 2`

## 状态码

- `MmseStatus::kOk`
  - PCFICH 均衡成功完成
- `MmseStatus::kNotInitialized`
  - 尚未先调用 context `init(...)`
- `MmseStatus::kInvalidArgument`
  - 网格形状、指针或 PCFICH wrapper 输入非法
- `MmseStatus::kUnsupportedConfig`
  - 请求超出当前 LTE PCFICH 支持边界
- `MmseStatus::kBufferTooSmall`
  - 调用方输出容量小于提取得到的 PCFICH RE 数
- `MmseStatus::kInternalError`
  - 运行时 / 内部传输失败
