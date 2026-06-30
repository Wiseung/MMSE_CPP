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
- `reserved_control_res`
- optional chain metadata through `PdcchChainMetadata`

Important rule:

- `reserved_control_res` should contain non-PDCCH control REs such as `PCFICH` and `PHICH`
- CRS REs should not be listed there because the SDK excludes CRS internally

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
