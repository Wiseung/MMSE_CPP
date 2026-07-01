# PCFICH Quick Start and API Reference

This page is the quick-start and field-reference page for the LTE PCFICH equalized-RE surface
exported by:

```cpp
#include "mmse/lte_chain_sdk.h"
```

Current interface version:

- `LTE Equalized Channel SDK v1`

Related pages:

- [LTE Equalized Channel SDK Documentation](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [LTE Downlink Channel Decode Overview](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md)

## Scope

The current PCFICH surface does:

- LTE PCFICH REG/RE extraction in symbol `0`
- CRS-aware occupied-RE removal aligned with the repo's helper semantics
- CRS-based channel estimation
- MMSE equalization
- caller-owned output view filling
- backend DTO packing for downstream handoff

The current PCFICH surface does not do:

- CFI bit descrambling
- modulation demapping to final CFI value
- channel decoding beyond the equalized RE surface

## Minimal Flow

1. Upstream prepares one `mmse::pcfich::FrontendPcfichIndication`.
2. Caller converts it into `mmse::PcfichMmseInput`.
3. Caller allocates one `mmse::PcfichMmseOutputView` and one `mmse::PcfichMmseResult`.
4. Caller runs `run_pcfich(...)`.
5. Caller packs output into `mmse::pcfich::BackendPcfichEqualizedIndication`.

## Minimal Example

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

## API Summary

Primary runtime calls:

- `MmseEqualizerCpuContext::run_pcfich(...)`
- `MmseEqualizerGpuContext::run_pcfich(...)`

Primary DTO flow:

1. upstream builds `mmse::pcfich::FrontendPcfichIndication`
2. helper converts it into `mmse::PcfichMmseInput`
3. CE/MMSE stage fills `mmse::PcfichMmseOutputView` and `mmse::PcfichMmseResult`
4. helper packs them into `mmse::pcfich::BackendPcfichEqualizedIndication`

## DTO Definitions

### `mmse::pcfich::FrontendPcfichIndication`

| Field          | Type       | Meaning                      |
| -------------- | ---------- | ---------------------------- |
| `sfn_subframe` | `uint32_t` | LTE system frame/subframe id |
| `cell_id`      | `uint16_t` | LTE PCI                      |
| `n_tx_ports`   | `uint8_t`  | LTE transmit antenna ports   |
| `tx_mode`      | `uint8_t`  | LTE transmit mode hint       |
| `chain`        | metadata   | caller passthrough metadata  |

### `mmse::PcfichMmseInput`

| Field          | Type                | Meaning                      |
| -------------- | ------------------- | ---------------------------- |
| `grid`         | `PlanarGridViewF32` | LTE FFT grid                 |
| `sfn_subframe` | `uint32_t`          | LTE system frame/subframe id |
| `cell_id`      | `uint16_t`          | LTE PCI                      |
| `n_tx_ports`   | `uint8_t`           | LTE transmit antenna ports   |
| `tx_mode`      | `uint8_t`           | LTE transmit mode hint       |
| `chain`        | metadata            | caller passthrough metadata  |

### `mmse::PcfichMmseOutputView`

| Field                   | Type        | Meaning                           |
| ----------------------- | ----------- | --------------------------------- |
| `x_hat_re`              | `float*`    | caller-owned equalized real plane |
| `x_hat_im`              | `float*`    | caller-owned equalized imag plane |
| `sinr`                  | `float*`    | caller-owned SINR plane           |
| `re_grid_indices`       | `uint16_t*` | caller-owned source RE indices    |
| `capacity_re_per_layer` | `uint32_t`  | max writable equalized RE count   |
| `capacity_re_metadata`  | `uint32_t`  | max writable metadata RE count    |

### `mmse::PcfichMmseResult`

| Field           | Meaning                                   |
| --------------- | ----------------------------------------- |
| `n_re`          | extracted/equalized PCFICH RE count       |
| `sfn_subframe`  | LTE system frame/subframe id              |
| `n_symbols`     | grid OFDM symbol count                    |
| `n_subcarriers` | grid subcarrier count                     |
| `cell_id`       | LTE PCI                                   |
| `n_prb`         | current support boundary bandwidth in PRB |
| `start_symbol`  | current PCFICH symbol index               |
| `reg_count`     | number of PCFICH REG in current boundary  |
| `n_tx_ports`    | LTE transmit antenna ports                |
| `n_rx_ant`      | LTE receive antenna count                 |
| `n_layers`      | equalized layer count                     |
| `tx_mode`       | LTE transmit mode hint                    |
| `mod_order`     | modulation order                          |
| `sigma2`        | runtime noise estimate                    |
| `chain`         | caller passthrough metadata               |

### `mmse::pcfich::BackendPcfichEqualizedIndication`

This backend DTO owns copied output vectors for downstream handoff.

Important fields:

- `x_hat_re`
- `x_hat_im`
- `sinr`
- `re_grid_indices`
- `reg_count`
- `start_symbol`

## Capacity Rule

Caller must provide:

- `capacity_re_per_layer >= expected_pcfich_re`
- `capacity_re_metadata >= expected_pcfich_re`

For the current LTE support boundary:

- `expected_pcfich_re == 16`

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
- `start_symbol == 0`
- `reg_count == 4`
- `n_tx_ports == 1 or 2`

## Status Codes

- `MmseStatus::kOk`
  - PCFICH equalization completed successfully
- `MmseStatus::kNotInitialized`
  - context `init(...)` was not called first
- `MmseStatus::kInvalidArgument`
  - grid shape, pointers, or PCFICH wrapper inputs are invalid
- `MmseStatus::kUnsupportedConfig`
  - request falls outside the current LTE PCFICH support boundary
- `MmseStatus::kBufferTooSmall`
  - caller output capacity is smaller than the extracted PCFICH RE count
- `MmseStatus::kInternalError`
  - runtime/internal transport failure
