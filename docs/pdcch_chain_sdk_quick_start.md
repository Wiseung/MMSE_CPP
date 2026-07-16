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
- 通过 `run_pdcch_td4(...)` 支持 `4Tx x 1Rx` 发射分集 raw equalized output
- `REG / CCE` 恢复 helper
- common-search 与 UE-specific 候选构造、候选 LLR 切片与速率恢复 helper
- `CRC-RNTI` 校验、`DCI 1A` 解析 helper
- 默认内建尾咬卷积码译码的正式 CPU common-search 与 UE-specific `DCI 1A` 链路，可由外部回调覆盖
- 1Tx、2Tx TD 与 `4Tx x 1Rx` TD 的 GPU common-search / `DCI 1A` 一站式与 batch 入口
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

## 4Tx x 1Rx Td4 示例

Td4 入口只接受单接收天线、单层、`tx_mode == 2` 和 QPSK。PDCCH 输出长度随控制区和
reserved RE 变化；无法预先精确计算时，可按 normal-CP 最大控制区保守分配
`4 * 1200` 个槽位。

```cpp
frontend.n_tx_ports = 4;
frontend.tx_mode = 2;
mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);

constexpr std::uint32_t capacity = 4U * 1200U;
std::vector<float> xhat_re(capacity), xhat_im(capacity), sinr(capacity);
std::vector<std::uint16_t> re0(capacity), re1(capacity), re2(capacity), re3(capacity);
mmse::PdcchTd4MmseOutputView out{xhat_re.data(), xhat_im.data(), sinr.data(),
                                 re0.data(), re1.data(), re2.data(), re3.data(), capacity};
mmse::PdcchTd4MmseResult meta{};

const mmse::MmseStatus status = ctx.run_pdcch_td4(in, out, meta);
if (status != mmse::MmseStatus::kOk) {
    return;
}

const mmse::pdcch::BackendPdcchTd4EqualizedIndication td4 =
    mmse::pdcch::make_backend_pdcch_td4_equalized_indication(meta, out);
mmse::pdcch::BackendPdcchEqualizedIndication cce_ordered{};
if (mmse::pdcch::normalize_pdcch_td4_cce_order(td4, cce_ordered) !=
    mmse::MmseStatus::kOk) {
    return;
}
```

`meta.n_symbols == meta.n_source_re` 且为 `4` 的倍数。对每个四元组起点
`q = 0, 4, 8, ...`，`re_grid_indices0..3[q..q+3]` 重复记录同一组四个 source RE；
四个连续输出槽位是两个 Alamouti 对恢复出的四个 QPSK 软符号。归一化 helper 不改
`x_hat` 或 SINR 顺序，只把索引重建为 `re0[q], re1[q], re2[q], re3[q]` 的标准连续
CCE 顺序。

若只需要 GPU common-search / `DCI 1A` 最终命中，可直接构造
`PdcchGpuCommonSearchDecodeRequest` 并调用 `run_pdcch_gpu_common_search_decode(...)`；输入端口组合
可为 `(1, 1)`、`(2, 2)` 或 `(4, 2)`；`4Tx` 额外要求 `1Rx`。TD 请求会自动复用上述
归一化顺序，D2H 只返回 compact candidate hits。该入口固定为
`100 RB / FDD / normal CP / regular control subframe`，使用 native GPU tail-biting
Viterbi，且不支持 CIF、外部 decoder callback、UE-specific 或 SI-RNTI geometry search。

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
- `n_tx_ports` 由 CPU dispatcher 分派：1Tx 调用 `run_pdcch(...)`，2Tx 调用
  `run_pdcch_td(...)`，4Tx 调用 `run_pdcch_td4(...)`；两条 TD 路径先归一化再复用解码链

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
const auto coord =
    mmse::pdcch::decode_re_grid_index(backend.re_grid_indices[i], backend.n_prb);
```

## 支持边界

当前已验证的支持范围：

- 仅 LTE
- CPU 固定几何链路支持 `6/15/25/50/75/100 RB`；GPU common-search 固定 `100 RB`
- 仅 normal CP
- `n_rx_ant == 1 or 2`
- `n_symbols == 14`
- CPU `n_subcarriers == 12 * n_prb`；GPU common-search `n_subcarriers == 1200`
- 6 RB 的 `control_symbol_count in [1, 4]`；其它带宽为 `[1, 3]`
- `n_layers == 1`
- `mod_order == 2`
- `tx_mode == 1 or 2`
- `run_pdcch(...)` 对应 `n_tx_ports == 1`
- `run_pdcch_td(...)` 对应 `n_tx_ports == 2`
- `run_pdcch_td4(...)` 对应 `n_tx_ports == 4`，并要求 `n_rx_ant == 1`
- CPU common-search、固定 SI-RNTI、UE-specific 与 geometry dispatcher 接受 1Tx、2Tx TD
  和 `4Tx x 1Rx` Td4 输入
- 正式解码链覆盖 common-search 和 UE-specific `DCI 1A`

当前明确不支持：

- helper 层自动 `PHICH` 保留不等于支持 `PHICH` 译码
- 非 `DCI 1A` 的其它 `DCI` 格式
- GPU UE-specific、SI-RNTI geometry search、CIF 或外部 decoder callback

## 构建与 Demo

可编译 demo 源文件：

- [pdcch_module_demo.cpp](/G:/MMSE_CPP/bench/pdcch_module_demo.cpp)
- [pdcch_decode_bench.cpp](/G:/MMSE_CPP/bench/pdcch_decode_bench.cpp)

构建并运行：

```powershell
cmake --build build --config Debug --target pdcch_module_demo
.\build\Debug\pdcch_module_demo.exe
```

Release 下复跑 CPU 4Tx 与 GPU 全 sweep（GPU sweep 包含 4Tx×1Rx）：

```powershell
cmake --build build --config Release --target pdcch_decode_bench pdcch_gpu_decode_bench
.\build\Release\pdcch_decode_bench.exe --n-tx-ports 4 --workload mixed
.\build\Release\pdcch_gpu_decode_bench.exe --workload mixed
```

CPU benchmark 可选 `--n-prb 6|15|25|50|75|100` 和 `--workload mixed|random`；4Tx 会自动
使用 1Rx。GPU benchmark 支持 `--workload mixed|random`，固定 100 RB，并 sweep
1/2/4 streams、1/2/4/8/16 batch 和 1Tx/2Tx/4Tx 三种模式。`--memory-report` 当前只报告
1Tx/2Tx，不用于验证 4Tx。

如果你要看分阶段 CPU 时延统计：

```powershell
cmake --build build --config Release --target pdcch_decode_bench
.\build\Release\pdcch_decode_bench.exe
```

当前基准会输出每 `1 ms` 子帧和每 `10 ms` 窗口的 `avg/p50/p95`，并把卷积码阶段明确标记为
`native_tail_biting_viterbi`；输出键形如 `subframe_1ms.tail_biting_viterbi.avg_us`，并报告
候选数和 CRC hit/miss 统计。
几何搜索还输出 cold、locked 与第五次 miss reprobe 的耗时和实际几何尝试数。

## 下一步阅读

- [API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [版本策略](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [集成示例](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)
