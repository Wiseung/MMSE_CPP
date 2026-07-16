# PDCCH Module API 集成示例

本文说明当前 `G:\MMSE_CPP` 中 PDCCH MMSE 模块的预期集成边界。

该模块只负责：

- 基于 CRS 的信道估计
- LTE 控制区内的 PDCCH RE 提取
- MMSE 均衡
- `REG / CCE` 恢复 helper
- common-search 与 UE-specific 候选构造、速率恢复、`CRC-RNTI` 和 `DCI 1A` 解析 helper
- 当前 `20 MHz / FDD` 边界内的 SI-RNTI 未知控制区几何搜索与缓存锁定
- `4Tx x 1Rx` Td4 raw equalized output、四源 RE 归一化和 CPU dispatcher

该模块不负责：

- PCFICH 译码
- PHICH 译码
- 非 `DCI 1A` 的通用 `DCI` 译码

对于 `2 Tx port` 的 LTE PDCCH 发射分集场景，请使用新增 TD 调用面
`run_pdcch_td(...)`，而不是下面展示的传统单 RE `run_pdcch(...)` 接口面。
对于 `4Tx x 1Rx` 使用 `run_pdcch_td4(...)`；该入口固定为单层、`tx_mode == 2`、QPSK。

## 1. 上游：如何填充 `control_re_exclusion_masks`

上游首先应确定：

- `control_symbol_count`，来源可以是 `PCFICH/CFI` 或可信的外部控制器
- 控制区内哪些 RE 被非 PDCCH 信道占用，例如 `PCFICH` 和 `PHICH`

然后填充 `PdcchMmseInput`，并把这些 RE 标记为排除。

```cpp
#include "mmse/pdcch_chain_sdk.h"

mmse::pdcch::FrontendPdcchIndication frontend = get_frontend_pdcch_indication();
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

mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(fft_grid_view, frontend);
```

每个 reserved RE 的预期含义：

- `symbol`：子帧内的 OFDM symbol 索引
- `prb`：`0..99` 范围内的 PRB 索引
- `tone_in_prb`：单个 PRB 内 `0..11` 的 tone 索引

当前模块会自动排除 CRS RE。上游只需要标记除此之外仍然残留的非 PDCCH 控制 RE，
典型就是 `PCFICH` 和 `PHICH`。

## 2. 模块调用

```cpp
mmse::MmseEqualizerCpuContext ctx;
mmse::MmseEqualizerCpuConfig cfg{};
cfg.worker_count = 1;
ctx.init(cfg);

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
const mmse::MmseStatus status = ctx.run_pdcch(in, out, meta);
```

对于 `2 Tx port` 发射分集，请改为分配新增 TD 输出：

```cpp
std::vector<float> xhat_re(capacity_symbols);
std::vector<float> xhat_im(capacity_symbols);
std::vector<float> sinr(capacity_symbols);
std::vector<std::uint16_t> re_grid_indices0(capacity_symbols);
std::vector<std::uint16_t> re_grid_indices1(capacity_symbols);

mmse::PdcchTdMmseOutputView td_out{};
td_out.x_hat_re = xhat_re.data();
td_out.x_hat_im = xhat_im.data();
td_out.sinr = sinr.data();
td_out.re_grid_indices0 = re_grid_indices0.data();
td_out.re_grid_indices1 = re_grid_indices1.data();
td_out.capacity_symbols = static_cast<std::uint32_t>(capacity_symbols);

mmse::PdcchTdMmseResult td_meta{};
const mmse::MmseStatus td_status = ctx.run_pdcch_td(in, td_out, td_meta);
```

对于 `4Tx x 1Rx`，分配四组索引并归一化到标准 CCE 顺序：

```cpp
std::vector<std::uint16_t> re0(capacity_symbols), re1(capacity_symbols);
std::vector<std::uint16_t> re2(capacity_symbols), re3(capacity_symbols);
mmse::PdcchTd4MmseOutputView td4_out{xhat_re.data(), xhat_im.data(), sinr.data(),
                                     re0.data(), re1.data(), re2.data(), re3.data(),
                                     static_cast<std::uint32_t>(capacity_symbols)};
mmse::PdcchTd4MmseResult td4_meta{};
const mmse::MmseStatus td4_status = ctx.run_pdcch_td4(in, td4_out, td4_meta);

const auto td4_backend =
    mmse::pdcch::make_backend_pdcch_td4_equalized_indication(td4_meta, td4_out);
mmse::pdcch::BackendPdcchEqualizedIndication cce_ordered{};
const mmse::MmseStatus normalize_status =
    mmse::pdcch::normalize_pdcch_td4_cce_order(td4_backend, cce_ordered);
```

## 3. 下游：如何消费 `re_grid_indices`

下游应先把模块输出打包成正式的 backend DTO，再消费该 DTO。

```cpp
mmse::pdcch::BackendPdcchEqualizedIndication backend =
    mmse::pdcch::make_backend_pdcch_equalized_indication(meta, out);

for (std::size_t i = 0; i < backend.re_grid_indices.size(); ++i) {
    const auto coord =
        mmse::pdcch::decode_re_grid_index(backend.re_grid_indices[i], backend.n_prb);

    DownstreamSoftRe item{};
    item.xhat_re = backend.x_hat_re[i];
    item.xhat_im = backend.x_hat_im[i];
    item.sinr = backend.sinr[i];
    item.symbol = coord.symbol;
    item.subcarrier = coord.subcarrier;
    item.prb = coord.prb;
    item.tone_in_prb = coord.tone_in_prb;
    item.request_id = backend.chain.request_id;
    item.candidate_id = backend.chain.candidate_id;
    item.first_cce = backend.chain.first_cce;
    item.aggregation_level = backend.chain.aggregation_level;

    downstream_consume(item);
}
```

对于 TD 接口面：

```cpp
mmse::pdcch::BackendPdcchTdEqualizedIndication td_backend =
    mmse::pdcch::make_backend_pdcch_td_equalized_indication(td_meta, td_out);

for (std::size_t i = 0; i < td_backend.x_hat_re.size(); ++i) {
    const auto coord0 =
        mmse::pdcch::decode_re_grid_index(td_backend.re_grid_indices0[i], td_backend.n_prb);
    const auto coord1 =
        mmse::pdcch::decode_re_grid_index(td_backend.re_grid_indices1[i], td_backend.n_prb);

    DownstreamTdSoftSymbol item{};
    item.xhat_re = td_backend.x_hat_re[i];
    item.xhat_im = td_backend.x_hat_im[i];
    item.sinr = td_backend.sinr[i];
    item.re0_symbol = coord0.symbol;
    item.re0_subcarrier = coord0.subcarrier;
    item.re1_symbol = coord1.symbol;
    item.re1_subcarrier = coord1.subcarrier;

    downstream_consume_td(item);
}
```

这就是预期的透传契约：

- 当前模块可以转换成一个正式 backend DTO
- backend DTO 持有均衡后的软符号、按 RE 的 SINR，以及 `re_grid_indices`
- 下游使用 `re_grid_indices` 恢复 REG/CCE 或其它所需顺序
- 链路元数据通过 `PdcchChainMetadata` 端到端透传
- Td4 每四个连续槽位重复同一组 `re_grid_indices0..3`；四个槽位对应两个 Alamouti 对
- `normalize_pdcch_td4_cce_order(...)` 保留软符号/SINR，只重建标准 CCE source-RE 顺序

## 4. 实用规则

- 上游应把 `control_re_exclusion_masks` 当作非 PDCCH 控制 RE 的列表
- 上游不应把 CRS RE 标进去；CRS 已由内部自动排除
- 如果上游希望由 helper 自动生成控制信道保留信息，则应通过
  `FrontendPdcchIndication::control_subframe` 传入共享的 `LteControlSubframeContext`，
  并通过 `PhichReservationConfig` 传入 `Ng + duration + mi`
- 在 TDD 模式下，`mi` 必须和选定的 `subframe_ctx.ul_dl_config + subframe_ctx.subframe` 匹配
- 对 extended duration，`TDD subframe 1/6` 特例会自动选中
- 真实 `MBSFN` 子帧应通过 `subframe_ctx.kind = kMbsfn` 标记
- 下游不应假定输出顺序天然等于 CCE 顺序
- 下游应使用 `re_grid_indices` 恢复任何需要的排序
- `run_pdcch(...)` 仍然是 `1 Tx port` 的 per-RE 接口面
- `run_pdcch_td(...)` 是新增的 `2 Tx port` 发射分集接口面，并通过
  `re_grid_indices0/re_grid_indices1` 提供 RE 对映射
- `run_pdcch_td4(...)` 是 `4Tx x 1Rx` 发射分集接口面，并通过
  `re_grid_indices0..3` 提供四源 RE 映射

## 5. 正式 CPU `DCI 1A` 验收入口

可以直接走正式 CPU 入口；默认使用内建尾咬 Viterbi，也可提供回调覆盖：

```cpp
mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(fft_grid_view, frontend);

mmse::MmseEqualizerCpuContext ctx;
mmse::MmseEqualizerCpuConfig cfg{};
cfg.worker_count = 1;
ctx.init(cfg);

mmse::pdcch::PdcchCommonSearchDecodeConfig decode_cfg{};
decode_cfg.expected_rnti = mmse::pdcch::kSiRnti;
decode_cfg.decoder = {.context = decoder_ctx, .decode = external_tail_biting_decode};

mmse::pdcch::PdcchCommonSearchDecodeResult result{};
const mmse::MmseStatus status =
    mmse::pdcch::run_pdcch_cpu_common_search_decode(ctx, in, decode_cfg, result);
if (status != mmse::MmseStatus::kOk) {
    return;
}

for (const auto& hit : result.hits) {
    downstream_consume_dci(hit.dci, hit.crc);
}
```

如果目标固定为 `SI-RNTI`，也可以改用 `run_pdcch_cpu_si_rnti_search(...)`。

这条入口当前会在仓库内顺序完成：

1. CPU dispatcher 按 `n_tx_ports` 调用 `run_pdcch(...)`、`run_pdcch_td(...)` 或
   `run_pdcch_td4(...)`
2. `REG / CCE` 恢复与 `common search` 候选构造
3. `QPSK LLR + descrambling`
4. 速率恢复
5. 内建尾咬 Viterbi；外部回调可选覆盖
6. `CRC-RNTI` 校验与 `DCI 1A` 解析

对 `2 Tx port` 和 `4Tx x 1Rx` 场景，这条入口会先用对应 normalize helper 把 TD 输出
归一化为与 `1Tx` 一致的连续 CCE 顺序，再进入候选链路。

## 6. UE-specific `DCI 1A` 入口

当上游已知一个或多个目标 UE RNTI 时，使用 `run_pdcch_cpu_ue_specific_search(...)`。该入口
复用上面的 `PdcchMmseInput`，按当前 `sfn_subframe` 的 LTE `Y_k` 递推枚举候选。

```cpp
mmse::pdcch::PdcchUeSpecificSearchConfig ue_cfg{};
ue_cfg.rntis = {0x1234U, 0x2345U};
ue_cfg.aggregation_level_mask = mmse::pdcch::kPdcchAggregationLevelMaskAll;
ue_cfg.decoder = {.context = decoder_ctx, .decode = external_tail_biting_decode};

mmse::pdcch::PdcchUeSpecificSearchResult ue_result{};
const mmse::MmseStatus ue_status =
    mmse::pdcch::run_pdcch_cpu_ue_specific_search(ctx, in, ue_cfg, ue_result);
if (ue_status != mmse::MmseStatus::kOk) {
    return;
}

for (const auto& hit : ue_result.hits) {
    downstream_consume_dci(hit.decoded.dci, hit.decoded.crc);
}
```

`rntis` 必须非空、唯一且不含 `0` 或 `kSiRnti`。该入口支持 `L=1/2/4/8`，并同时接受
1Tx、2Tx transmit-diversity 与 `4Tx x 1Rx` Td4 输入；非匹配 CRC 是正常 miss，不返回错误。

## 7. SI-RNTI 未知控制区几何入口

当调用方没有可信的 CFI、PHICH `Ng` 或 PHICH duration，但已有当前频域网格与小区上下文时，
使用 `run_pdcch_cpu_si_rnti_geometry_search(...)`。该入口只支持当前 `20 MHz / FDD / normal CP`
边界，内部枚举有效控制区几何，唯一命中后写入调用方持有的 cache。

```cpp
mmse::pdcch::PdcchSiRntiGeometrySearchRequest request{};
request.grid = fft_grid_view;
request.sfn_subframe = sfn_subframe;
request.cell_id = cell_id;
request.n_tx_ports = n_tx_ports;
request.tx_mode = tx_mode;
request.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                            .subframe = static_cast<std::uint8_t>(sfn_subframe % 10U),
                            .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

mmse::pdcch::PdcchSiRntiGeometrySearchCache geometry_cache{};
mmse::pdcch::PdcchSiRntiGeometrySearchResult geometry_result{};
const mmse::MmseStatus geometry_status =
    mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(ctx, request, {}, geometry_cache,
                                                        geometry_result);
```

`kAcquired` 表示唯一几何已锁定，`kLocked` 表示缓存几何命中，`kMiss` 表示没有合法
SI-RNTI 候选，`kAmbiguous` 表示多个几何命中且没有任何 hit 可被消费。缓存命中连续 5 次
miss 后会在第 5 次调用全量重新枚举。

## 8. 可编译 demo

仓库还提供了一个小型可编译 demo：

- 源文件：`bench/pdcch_module_demo.cpp`
- 目标：`pdcch_module_demo`

构建并运行：

```powershell
cmake --build build --config Debug --target pdcch_module_demo
.\build\Debug\pdcch_module_demo.exe
```
