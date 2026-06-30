# PDCCH Module API Example

This document shows the intended integration boundary for the current PDCCH MMSE module in
`G:\MMSE_CPP`.

The module only does:

- CRS-based channel estimation
- PDCCH RE extraction inside the LTE control region
- MMSE equalization

The module does not do:

- PCFICH decoding
- PHICH decoding
- REG/CCE regrouping
- DCI blind detection or channel decoding

For `2 Tx port` LTE PDCCH transmit diversity, use the additive TD call surface
`run_pdcch_td(...)` instead of the legacy single-RE `run_pdcch(...)` surface shown below.

## 1. Upstream: how to fill `control_re_exclusion_masks`

Upstream should first determine:

- `control_symbol_count` from PCFICH/CFI or from a trusted external controller
- which control-region REs are occupied by non-PDCCH channels such as `PCFICH` and `PHICH`

Then fill `PdcchMmseInput` and mark those REs as excluded.

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

Expected meaning of each reserved RE:

- `symbol`: OFDM symbol index inside the subframe
- `prb`: PRB index in `0..99`
- `tone_in_prb`: tone index inside one PRB in `0..11`

The current module automatically excludes CRS REs. Upstream should only mark non-PDCCH control REs
that still remain after that, typically `PCFICH` and `PHICH`.

## 2. Module call

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

For `2 Tx port` transmit diversity, allocate the additive TD output instead:

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

## 3. Downstream: how to consume `re_grid_indices`

Downstream should first pack the module output into the formal backend DTO, then consume that DTO.

```cpp
mmse::pdcch::BackendPdcchEqualizedIndication backend =
    mmse::pdcch::make_backend_pdcch_equalized_indication(meta, out);

for (std::size_t i = 0; i < backend.re_grid_indices.size(); ++i) {
    const auto coord = mmse::pdcch::decode_re_grid_index(backend.re_grid_indices[i]);

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

For the TD surface:

```cpp
mmse::pdcch::BackendPdcchTdEqualizedIndication td_backend =
    mmse::pdcch::make_backend_pdcch_td_equalized_indication(td_meta, td_out);

for (std::size_t i = 0; i < td_backend.x_hat_re.size(); ++i) {
    const auto coord0 = mmse::pdcch::decode_re_grid_index(td_backend.re_grid_indices0[i]);
    const auto coord1 = mmse::pdcch::decode_re_grid_index(td_backend.re_grid_indices1[i]);

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

This is the intended handoff contract:

- the current module can be converted into one formal backend DTO
- the backend DTO carries equalized soft symbols, per-RE SINR, and `re_grid_indices`
- downstream uses `re_grid_indices` to rebuild REG/CCE or any other required ordering
- chain metadata is preserved end to end through `PdcchChainMetadata`

## 4. Practical rules

- Upstream should treat `control_re_exclusion_masks` as the list of non-PDCCH control REs
- Upstream should not mark CRS REs there; CRS is already excluded internally
- If upstream wants helper-generated control-channel reservation, it should pass shared
  `LteControlSubframeContext` through `FrontendPdcchIndication::control_subframe`, plus
  `Ng + duration + mi` through `PhichReservationConfig`
- In TDD mode, `mi` must match the selected `subframe_ctx.ul_dl_config + subframe_ctx.subframe`
- For extended duration, `TDD subframe 1/6` special-case is selected automatically
- True `MBSFN` subframes should be marked through `subframe_ctx.kind = kMbsfn`
- Downstream should not assume output order equals CCE order
- Downstream should use `re_grid_indices` to rebuild any required ordering
- `run_pdcch(...)` remains the `1 Tx port` per-RE surface
- `run_pdcch_td(...)` is the additive `2 Tx port` transmit-diversity surface with
  `re_grid_indices0/re_grid_indices1` RE-pair mapping

## 5. Compilable demo

The repo also provides a small compilable demo:

- source: `bench/pdcch_module_demo.cpp`
- target: `pdcch_module_demo`

Build and run:

```powershell
cmake --build build --config Debug --target pdcch_module_demo
.\build\Debug\pdcch_module_demo.exe
```
