# PBCH Quick Start

This page is the fastest path to integrate the LTE PBCH equalized-RE surface exported by:

```cpp
#include "mmse/lte_chain_sdk.h"
```

Current interface version:

- `LTE Equalized Channel SDK v1`

Related pages:

- [LTE Equalized Channel SDK Documentation](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [LTE Downlink Channel Decode Overview](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md)

## Scope

The current PBCH surface does:

- LTE PBCH RE extraction over the center `72` subcarriers
- CRS-based channel estimation
- MMSE equalization
- caller-owned output view filling
- backend DTO packing for downstream handoff

The current PBCH surface does not do:

- PBCH descrambling
- rate recovery
- tail-biting convolutional decoding
- CRC checking
- MIB reconstruction

## Minimal Flow

1. Upstream prepares one `mmse::pbch::FrontendPbchIndication`.
2. Caller converts it into `mmse::PbchMmseInput`.
3. Caller allocates one `mmse::PbchMmseOutputView` and one `mmse::PbchMmseResult`.
4. Caller runs `run_pbch(...)`.
5. Caller packs output into `mmse::pbch::BackendPbchEqualizedIndication`.

## Minimal Example

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

## Upstream Requirements

Upstream must provide:

- `sfn_subframe`
- `cell_id`
- `n_tx_ports`
- `tx_mode`
- one LTE downlink FFT grid through `PlanarGridViewF32`

The current wrapper internally fixes the PBCH extraction boundary to:

- center `6 PRB`
- `4` OFDM symbols starting at symbol `7`
- QPSK (`mod_order == 2`)
- single equalized layer

## Downstream Handoff

Recommended downstream handoff object:

- `mmse::pbch::BackendPbchEqualizedIndication`

Important fields:

- `x_hat_re`
- `x_hat_im`
- `sinr`
- `re_grid_indices`
- `start_prb`
- `n_prb`
- `start_symbol`

The output currently represents equalized PBCH RE only. Downstream PBCH decoding remains external.

## Capacity Rule

Caller must provide:

- `capacity_re_per_layer >= expected_pbch_re`
- `capacity_re_metadata >= expected_pbch_re`

For the current LTE support boundary:

- `expected_pbch_re == 240`

## Support Boundary

Current validated support:

- LTE only
- 20 MHz only
- normal CP only
- `n_rx_ant == 2`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- `n_layers == 1`
- `mod_order == 2`
- `tx_mode == 1 or 2`
- `n_tx_ports == 1 or 2`

## Status Codes

- `MmseStatus::kOk`
  - PBCH equalization completed successfully
- `MmseStatus::kNotInitialized`
  - context `init(...)` was not called first
- `MmseStatus::kInvalidArgument`
  - grid shape, pointers, or PBCH wrapper inputs are invalid
- `MmseStatus::kUnsupportedConfig`
  - request falls outside the current LTE PBCH support boundary
- `MmseStatus::kBufferTooSmall`
  - caller output capacity is smaller than the extracted PBCH RE count
- `MmseStatus::kInternalError`
  - runtime/internal transport failure
