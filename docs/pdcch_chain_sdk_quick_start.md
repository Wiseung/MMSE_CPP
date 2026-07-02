# PDCCH Chain SDK 快速开始

本页是集成 LTE PDCCH 信道估计与 MMSE 均衡 SDK 的最快入口。该 SDK 由以下头文件导出：

```cpp
#include "mmse/pdcch_chain_sdk.h"
```

当前接口版本：

- `PDCCH Chain SDK v1`

相关页面：

- [LTE Equalized Channel SDK 文档](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [文档首页](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)

## 作用范围

SDK 当前提供：

- LTE PDCCH 控制区 RE 提取
- 基于 CRS 的信道估计
- MMSE 均衡
- 按 RE 输出软符号和 SINR
- 通过 `run_pdcch_td(...)` 新增支持 2Tx 发射分集去映射

SDK 当前不提供：

- PCFICH 译码
- PHICH 译码
- REG/CCE 重组
- 盲检索
- 信道译码

## 最小流程

1. 上游准备一个 `mmse::pdcch::FrontendPdcchIndication`
2. 调用方把它转换成 `mmse::PdcchMmseInput`
3. 调用方分配一个 `mmse::PdcchMmseOutputView` 和一个 `mmse::PdcchMmseResult`
4. 调用 `run_pdcch(...)`
5. 把输出打包成 `mmse::pdcch::BackendPdcchEqualizedIndication`

## 最小示例

```cpp
#include "mmse/pdcch_chain_sdk.h"

mmse::pdcch::FrontendPdcchIndication frontend = get_frontend_indication();
mmse::PlanarGridViewF32 grid = get_fft_grid();

frontend.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                             .subframe = 0,
                             .ul_dl_config = 0,
                             .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

mmse::pdcch::append_pcfich_reserved_control_re_list(frontend);
const mmse::MmseStatus phich_status = mmse::pdcch::append_phich_reserved_control_re_list(
    frontend,
    {.resource = mmse::pdcch::PhichResource::kOne,
     .duration = mmse::pdcch::PhichDuration::kNormal,
     .mi = 1,
     .subframe_ctx = frontend.control_subframe});
if (phich_status != mmse::MmseStatus::kOk) {
    return;
}

mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);

std::vector<float> xhat_re(capacity);
std::vector<float> xhat_im(capacity);
std::vector<float> sinr(capacity);
std::vector<std::uint16_t> re_grid_indices(capacity);

mmse::PdcchMmseOutputView out{};
out.x_hat_re = xhat_re.data();
out.x_hat_im = xhat_im.data();
out.sinr = sinr.data();
out.re_grid_indices = re_grid_indices.data();
out.capacity_re_per_layer = static_cast<std::uint32_t>(capacity);
out.capacity_re_metadata = static_cast<std::uint32_t>(capacity);

mmse::PdcchMmseResult meta{};

mmse::MmseEqualizerCpuContext ctx;
mmse::MmseEqualizerCpuConfig cfg{};
cfg.worker_count = 1;
ctx.init(cfg);

const mmse::MmseStatus status = ctx.run_pdcch(in, out, meta);
if (status != mmse::MmseStatus::kOk) {
    return;
}

mmse::pdcch::BackendPdcchEqualizedIndication backend =
    mmse::pdcch::make_backend_pdcch_equalized_indication(meta, out);
```

## 2Tx TD 示例

对于 `2 Tx port` 的 LTE PDCCH 发射分集场景，可以继续使用同一个
`FrontendPdcchIndication` 和 `PdcchMmseInput`，但输出面需要改为新增的 TD 接口，
而不是传统的单 RE 接口。

容量规则：

- `capacity_symbols >= expected_n_soft_symbols`
- 在当前控制区 TD 实现下，`expected_n_soft_symbols` 等于扣除 CRS 和 reserved-RE 后的
  有效控制区 RE 数
- 每一对相邻 RE 会生成两个 QPSK 软符号，因此 `meta.n_symbols == meta.n_source_re`

```cpp
#include "mmse/pdcch_chain_sdk.h"

mmse::pdcch::FrontendPdcchIndication frontend = get_frontend_indication();
mmse::PlanarGridViewF32 grid = get_fft_grid();

frontend.n_tx_ports = 2;
frontend.tx_mode = 2;
frontend.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                             .subframe = 0,
                             .ul_dl_config = 0,
                             .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

mmse::pdcch::append_pcfich_reserved_control_re_list(frontend);
const mmse::MmseStatus phich_status = mmse::pdcch::append_phich_reserved_control_re_list(
    frontend,
    {.resource = mmse::pdcch::PhichResource::kOne,
     .duration = mmse::pdcch::PhichDuration::kNormal,
     .mi = 1,
     .subframe_ctx = frontend.control_subframe});
if (phich_status != mmse::MmseStatus::kOk) {
    return;
}

mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);

const std::size_t capacity_symbols = conservative_td_capacity();
std::vector<float> xhat_re(capacity_symbols);
std::vector<float> xhat_im(capacity_symbols);
std::vector<float> sinr(capacity_symbols);
std::vector<std::uint16_t> re_grid_indices0(capacity_symbols);
std::vector<std::uint16_t> re_grid_indices1(capacity_symbols);

mmse::PdcchTdMmseOutputView out{};
out.x_hat_re = xhat_re.data();
out.x_hat_im = xhat_im.data();
out.sinr = sinr.data();
out.re_grid_indices0 = re_grid_indices0.data();
out.re_grid_indices1 = re_grid_indices1.data();
out.capacity_symbols = static_cast<std::uint32_t>(capacity_symbols);

mmse::PdcchTdMmseResult meta{};

mmse::MmseEqualizerGpuContext ctx;
mmse::MmseEqualizerGpuConfig cfg{};
cfg.backend = mmse::MmseGpuBackend::kCuda;
ctx.init(cfg);

const mmse::MmseStatus status = ctx.run_pdcch_td(in, out, meta);
if (status != mmse::MmseStatus::kOk) {
    return;
}

mmse::pdcch::BackendPdcchTdEqualizedIndication backend =
    mmse::pdcch::make_backend_pdcch_td_equalized_indication(meta, out);
```

其中 `backend.x_hat_re[i] / x_hat_im[i] / sinr[i]` 表示一个译码后的 QPSK 软符号，
而生成该符号用到的两个源 RE 则分别位于 `backend.re_grid_indices0[i]` 和
`backend.re_grid_indices1[i]`。

## 上游要求

上游必须提供：

- `sfn_subframe`
- `cell_id`
- `n_tx_ports`
- `tx_mode`
- `control_symbol_count`
- `n_prb`
- `prb_bitmap`
- `control_subframe`
- `reserved_control_res`
- 通过 `PdcchChainMetadata` 传入的可选链路元数据

重要规则：

- `reserved_control_res` 应该只包含非 PDCCH 控制 RE，例如 `PCFICH` 和 `PHICH`
- CRS RE 不应显式写入，因为 SDK 会在内部自动排除 CRS
- 如果上游已经知道被占用的 RE，也可以直接手工填充 `reserved_control_res`
- 对于当前 20 MHz FDD normal-CP 边界，调用方也可以使用显式共享的
  `LteControlSubframeContext` 和 `PhichReservationConfig`
- 在 TDD 模式下，`mi` 必须和所选 `subframe_ctx.ul_dl_config + subframe_ctx.subframe` 匹配
- 对 extended PHICH duration，`TDD subframe 1/6` 特例会自动选中
- 真实 `MBSFN` 子帧应通过 `subframe_ctx.kind = kMbsfn` 显式标记
- 对当前 20 MHz FDD/TDD normal-CP helper 边界，调用方也可以在
  `make_pdcch_mmse_input(...)` 之前先调用
  `append_pcfich_reserved_control_re_list(...)` 和
  `append_phich_reserved_control_re_list(...)`

## 下游透传

推荐的下游透传对象：

- `mmse::pdcch::BackendPdcchEqualizedIndication`

重点字段：

- `x_hat_re`
- `x_hat_im`
- `sinr`
- `re_grid_indices`
- `chain`

对于 `2 Tx port` 发射分集 PDCCH，请改用新增 TD 接口面：

- `mmse::PdcchTdMmseOutputView`
- `mmse::PdcchTdMmseResult`
- `mmse::MmseEqualizerCpuContext::run_pdcch_td(...)`
- `mmse::MmseEqualizerGpuContext::run_pdcch_td(...)`
- `mmse::pdcch::BackendPdcchTdEqualizedIndication`

该接口面会对每个译码后的软符号返回一组源 RE 对映射，
对应字段是 `re_grid_indices0[i]` 和 `re_grid_indices1[i]`。

下游如果需要恢复 LTE 坐标，可以这样做：

```cpp
const auto coord = mmse::pdcch::decode_re_grid_index(backend.re_grid_indices[i]);
```

## 支持边界

当前已验证的支持范围：

- 仅 LTE
- 仅 20 MHz
- 仅 normal CP
- `n_rx_ant == 2`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- `control_symbol_count in [1, 3]`
- `n_layers == 1`
- `mod_order == 2`
- `tx_mode == 1 or 2`
- `run_pdcch(...)` 对应 `n_tx_ports == 1`
- `run_pdcch_td(...)` 对应 `n_tx_ports == 2`

当前明确不支持：

- helper 层自动 `PHICH` 保留不等于支持 `PHICH` 译码

## 构建与 Demo

可编译 demo 源文件：

- [pdcch_module_demo.cpp](/G:/MMSE_CPP/bench/pdcch_module_demo.cpp)

构建并运行：

```powershell
cmake --build build --config Debug --target pdcch_module_demo
.\build\Debug\pdcch_module_demo.exe
```

## 下一步阅读

- [API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [集成示例](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)
