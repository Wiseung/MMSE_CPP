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
- `REG / CCE` 恢复 helper
- common-search 与 UE-specific 候选构造、候选 LLR 切片与速率恢复 helper
- `CRC-RNTI` 校验、`DCI 1A` 解析 helper
- 默认内建尾咬卷积码译码的正式 CPU common-search 与 UE-specific `DCI 1A` 链路，可由外部回调覆盖
- 1Tx 与 2Tx TD 的 GPU common-search / `DCI 1A` 一站式与 batch 入口
- 当前 `20 MHz / FDD` 边界内的 SI-RNTI 未知控制区几何搜索与调用方持有的缓存锁定

SDK 当前不提供：

- PCFICH 译码
- PHICH 译码
- 非 `DCI 1A` 的通用 `DCI` 译码
- GPU UE-specific、SI-RNTI geometry search、其它 DCI format 或外部 decoder callback

## 最小流程

1. 上游准备一个 `mmse::pdcch::FrontendPdcchIndication`
2. 调用方把它转换成 `mmse::PdcchMmseInput`
3. 调用方分配一个 `mmse::PdcchMmseOutputView` 和一个 `mmse::PdcchMmseResult`
4. 调用 `run_pdcch(...)`
5. 把输出打包成 `mmse::pdcch::BackendPdcchEqualizedIndication`

如果你要直接做 CPU 侧 `DCI 1A` 验收，则最小流程可以收敛成：

1. 准备 `FrontendPdcchIndication`
2. 调用 `make_pdcch_mmse_input(...)`
3. 仅在 SDK 集成需要时提供外部尾咬卷积码回调覆盖
4. 按搜索空间调用 `run_pdcch_cpu_common_search_decode(...)`、`run_pdcch_cpu_si_rnti_search(...)` 或 `run_pdcch_cpu_ue_specific_search(...)`
5. 遍历返回结果中的 `hits`

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

这份 TD 输出还可以继续交给：

- `normalize_pdcch_td_cce_order(...)`

把 `2Tx` 去分集软符号归一化为与 `1Tx` 一致的连续 CCE 顺序，再复用后续候选构造与
`DCI 1A` 链路。

若只需要 GPU common-search / `DCI 1A` 最终命中，可直接构造
`PdcchGpuCommonSearchDecodeRequest` 并调用 `run_pdcch_gpu_common_search_decode(...)`；输入端口组合
可为 `(1, 1)` 或 `(2, 2)`。2Tx 会自动复用上述 TD 顺序，D2H 只返回 compact candidate hits。
该入口固定为 `20 MHz / FDD / normal CP / regular control subframe`，并使用 native GPU
tail-biting Viterbi。

## CPU `DCI 1A` 最小示例

最短的正式 CPU 验收路径如下；未设置 `decoder` 时会使用内建 Viterbi：

```cpp
#include "mmse/pdcch_chain_sdk.h"

mmse::pdcch::FrontendPdcchIndication frontend = get_frontend_indication();
mmse::PlanarGridViewF32 grid = get_fft_grid();

frontend.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                             .subframe = 0,
                             .ul_dl_config = 0,
                             .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
mmse::pdcch::append_pcfich_reserved_control_re_list(frontend);
mmse::pdcch::append_phich_reserved_control_re_list(frontend,
                                                   {.resource = mmse::pdcch::PhichResource::kOne,
                                                    .duration = mmse::pdcch::PhichDuration::kNormal,
                                                    .mi = 1,
                                                    .subframe_ctx = frontend.control_subframe});

mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);

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
    consume_dci_1a(hit.dci, hit.crc, hit.dci.chain.first_cce, hit.dci.chain.aggregation_level);
}
```

如果目标固定为 `SI-RNTI`，可以改用 `run_pdcch_cpu_si_rnti_search(...)`，并从 `PdcchSiRntiSearchResult::hits` 读取命中列表。

本示例的 common-search 路径固定边界是：

- 仅 CPU
- 仅 `common search`
- 仅 `DCI 1A`
- 默认使用内建尾咬卷积码译码；调用方可通过 `decoder` 提供回调覆盖

## CPU UE-specific `DCI 1A` 示例

调用方已知目标 UE 的 RNTI 时，使用 LTE UE-specific 搜索空间。候选位置由每个
RNTI 和 `sfn_subframe % 10` 的 `Y_k` 递推决定，SDK 会按启用掩码枚举 `L=1/2/4/8`。

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
    consume_dci_1a(hit.decoded.dci, hit.decoded.crc, hit.rnti, hit.first_cce,
                    hit.aggregation_level);
}
```

约束：`rntis` 必须非空、唯一，不能包含 `0` 或 `kSiRnti`；`kSiRnti` 继续使用 common-search
入口。CRC-RNTI 不匹配是正常 blind-search miss，记入 `crc_rnti_miss_count`，不视为 API 失败。

## DCI 1A 到 PDSCH Grant

对已命中的 SI-RNTI 或 C-RNTI localized DCI 1A，可通过统一头文件生成 host-owned Grant，
再适配到现有 PDSCH equalizer descriptor：

```cpp
#include "mmse/lte_chain_sdk.h"

const mmse::pdcch::PdcchDciFormat1ADecodeResult& hit = get_matched_dci_1a();

mmse::handoff::PdcchPdschHandoffConfigV1 handoff{};
handoff.start_symbol = control_symbol_count;
handoff.n_tx_ports = n_tx_ports;
handoff.n_layers = 1U;
handoff.transmission_mode = tx_mode;

mmse::handoff::PdschGrantV1 grant{};
if (mmse::handoff::make_pdsch_grant_v1(hit, handoff, grant) != mmse::MmseStatus::kOk) {
    return;
}

mmse::ExtractDescriptor pdsch_desc{};
if (mmse::handoff::make_pdsch_extract_descriptor_v1(grant, n_rx_ant, pdsch_desc) !=
    mmse::MmseStatus::kOk) {
    return;
}
```

`handoff.start_symbol`、端口、层数、TM、PMI 和码字配置来自已确认的 CFI/小区/UE 上下文，
不在 DCI 1A payload 内。首版明确拒绝 PDCCH order、distributed VRB、保留 MCS、CIF、
非 100 PRB PDSCH 网格和非单层单码字 PHY 配置。完整字段和 CMake 安装说明见
[PDCCH→PDSCH 交接 SDK V1](pdcch_pdsch_handoff_sdk_v1.md)。

## SI-RNTI 未知几何示例

当调用方拥有当前频域网格和小区上下文，但尚未获得可信 CFI/PHICH 几何时，可使用
SI-RNTI 几何搜索入口。该入口只在 `20 MHz / FDD / normal CP` 边界内有效；它枚举
有效 CFI/PHICH 保留组合，唯一 `SI-RNTI + DCI 1A` 命中后写入调用方持有的 cache。

```cpp
mmse::pdcch::PdcchSiRntiGeometrySearchRequest request{};
request.grid = grid;
request.sfn_subframe = sfn_subframe;
request.cell_id = cell_id;
request.n_tx_ports = n_tx_ports;
request.tx_mode = tx_mode;
request.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                            .subframe = static_cast<std::uint8_t>(sfn_subframe % 10U),
                            .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

mmse::pdcch::PdcchSiRntiGeometrySearchCache cache{};
mmse::pdcch::PdcchSiRntiGeometrySearchResult geometry_result{};
const mmse::MmseStatus geometry_status =
    mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(ctx, request, {}, cache,
                                                        geometry_result);
if (geometry_status != mmse::MmseStatus::kOk) {
    return;
}

if (geometry_result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kAcquired ||
    geometry_result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kLocked) {
    for (const auto& hit : geometry_result.decoded.hits) {
        consume_dci_1a(hit.decoded.dci, hit.decoded.crc, mmse::pdcch::kSiRnti,
                        hit.first_cce, hit.aggregation_level);
    }
}
```

`kAcquired` 表示本次全量枚举唯一锁定几何，`kLocked` 表示复用缓存几何命中，`kMiss` 表示
没有合法 SI-RNTI 候选，`kAmbiguous` 表示多个几何命中且 SDK 拒绝锁定。缓存针对 PCI、端口、
发射模式与子帧类型失效；锁定后连续 5 次 miss 才在第 5 次调用重新枚举。

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
- `n_rx_ant == 1 or 2`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- `control_symbol_count in [1, 3]`
- `n_layers == 1`
- `mod_order == 2`
- `tx_mode == 1 or 2`
- `run_pdcch(...)` 对应 `n_tx_ports == 1`
- `run_pdcch_td(...)` 对应 `n_tx_ports == 2`
- `run_pdcch_cpu_common_search_decode(...)` 同时接受 `1Tx` 和 `2Tx TD` 输入
- `run_pdcch_cpu_ue_specific_search(...)` 同时接受 `1Tx` 和 `2Tx TD` 输入
- 正式解码链覆盖 common-search 和 UE-specific `DCI 1A`

当前明确不支持：

- helper 层自动 `PHICH` 保留不等于支持 `PHICH` 译码
- 仓库内真实尾咬卷积码实现
- 非 `DCI 1A` 的其它 `DCI` 格式

## 构建与 Demo

可编译 demo 源文件：

- [pdcch_module_demo.cpp](/G:/MMSE_CPP/bench/pdcch_module_demo.cpp)
- [pdcch_decode_bench.cpp](/G:/MMSE_CPP/bench/pdcch_decode_bench.cpp)

构建并运行：

```powershell
cmake --build build --config Debug --target pdcch_module_demo
.\build\Debug\pdcch_module_demo.exe
```

如果你要看分阶段 CPU 时延统计：

```powershell
cmake --build build --config Release --target pdcch_decode_bench
.\build\Release\pdcch_decode_bench.exe
```

当前基准会输出每 `1 ms` 子帧和每 `10 ms` 窗口的 `avg/p50/p95`，并把卷积码阶段明确标记为
`native_tail_biting_viterbi`，并输出 `tail_biting_viterbi_us`、候选数和 CRC hit/miss 的统计。
几何搜索还输出 cold、locked 与第五次 miss reprobe 的耗时和实际几何尝试数。

## 下一步阅读

- [API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [集成示例](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)
