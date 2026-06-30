# PDCCH Chain SDK Quick Start

This page is the fastest path to integrate the LTE PDCCH channel-estimation and MMSE equalization
SDK exported by:

```cpp
#include "mmse/pdcch_chain_sdk.h"
```

Current interface version:

- `PDCCH Chain SDK v1`

Related pages:

- [Documentation Index](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [API Reference](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [Versioning Policy](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)

## Scope

The SDK does:

- LTE PDCCH control-region RE extraction
- CRS-based channel estimation
- MMSE equalization
- per-RE soft-symbol and SINR output

The SDK does not do:

- PCFICH decoding
- PHICH decoding
- REG/CCE regrouping
- blind detection
- channel decoding

## Minimal Flow

1. Upstream prepares one `mmse::pdcch::FrontendPdcchIndication`.
2. Caller converts it into `mmse::PdcchMmseInput`.
3. Caller allocates one `mmse::PdcchMmseOutputView` and one `mmse::PdcchMmseResult`.
4. Caller runs `run_pdcch(...)`.
5. Caller packs output into `mmse::pdcch::BackendPdcchEqualizedIndication`.

## Minimal Example

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

## Upstream Requirements

Upstream must provide:

- `sfn_subframe`
- `cell_id`
- `n_tx_ports`
- `tx_mode`
- `control_symbol_count`
- `n_prb`
- `prb_bitmap`
- `control_subframe`
- `reserved_control_res`
- optional chain metadata through `PdcchChainMetadata`

Important rule:

- `reserved_control_res` should contain non-PDCCH control REs such as `PCFICH` and `PHICH`
- CRS REs should not be listed there because the SDK excludes CRS internally
- if upstream already knows the occupied REs, it can still fill `reserved_control_res` directly
- for the current 20 MHz FDD normal-CP boundary, callers can also use
  explicit shared `LteControlSubframeContext` plus `PhichReservationConfig`
- in TDD mode, `mi` must match the selected `subframe_ctx.ul_dl_config + subframe_ctx.subframe`
- for extended PHICH duration, `TDD subframe 1/6` special-case is selected automatically
- true `MBSFN` subframes should be marked through `subframe_ctx.kind = kMbsfn`
- for the current 20 MHz FDD/TDD normal-CP helper boundary, callers can also use
  `append_pcfich_reserved_control_re_list(...)` and
  `append_phich_reserved_control_re_list(...)` before `make_pdcch_mmse_input(...)`

## Downstream Handoff

Recommended downstream handoff object:

- `mmse::pdcch::BackendPdcchEqualizedIndication`

Important fields:

- `x_hat_re`
- `x_hat_im`
- `sinr`
- `re_grid_indices`
- `chain`

Downstream can recover LTE coordinates with:

```cpp
const auto coord = mmse::pdcch::decode_re_grid_index(backend.re_grid_indices[i]);
```

## Support Boundary

Current validated support:

- LTE only
- 20 MHz only
- normal CP only
- `n_rx_ant == 2`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- `control_symbol_count in [1, 3]`
- `n_tx_ports == 1`
- `n_layers == 1`
- `mod_order == 2`
- `tx_mode == 1 or 2`

Important non-support:

- `2 Tx port` LTE PDCCH
- helper-based automatic PHICH reservation does not imply PHICH decoding support

## Build and Demo

Compilable demo source:

- [pdcch_module_demo.cpp](/G:/MMSE_CPP/bench/pdcch_module_demo.cpp)

Build and run:

```powershell
cmake --build build --config Debug --target pdcch_module_demo
.\build\Debug\pdcch_module_demo.exe
```

## Next Reading

- [API Reference](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [Versioning Policy](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [Integration Example](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)
