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

当前 PCFICH 接口面不提供：

- CFI 比特解扰
- 调制解映射到最终 CFI 值
- 除 equalized RE 接口面之外的信道译码

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

## API 摘要

主要运行时调用：

- `MmseEqualizerCpuContext::run_pcfich(...)`
- `MmseEqualizerGpuContext::run_pcfich(...)`

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

## 支持边界

当前已验证的支持范围：

- 仅 LTE
- 仅 20 MHz
- 仅 normal CP
- `n_rx_ant == 2`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- `n_layers == 1`
- `mod_order == 2`
- `start_symbol == 0`
- `reg_count == 4`
- `n_tx_ports == 1 or 2`

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
