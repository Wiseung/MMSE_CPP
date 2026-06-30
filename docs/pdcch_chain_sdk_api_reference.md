# PDCCH Chain SDK API Reference

This page is the field-level and call-level reference for the LTE PDCCH CE/MMSE SDK.

Primary include:

```cpp
#include "mmse/pdcch_chain_sdk.h"
```

Interface version:

- `PDCCH Chain SDK v1`

Related pages:

- [Documentation Index](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [Quick Start](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [Versioning Policy](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)

## Table of Contents

- [API Summary](#api-summary)
- [Field Index](#field-index)
- [Status Code Index](#status-code-index)
- [1. Public entrypoints](#1-public-entrypoints)
- [2. DTO definitions](#2-dto-definitions)
- [3. Base grid type](#3-base-grid-type)
- [4. Helper semantics](#4-helper-semantics)
- [5. Boundary conditions](#5-boundary-conditions)
- [6. Capacity requirements](#6-capacity-requirements)
- [7. Error codes](#7-error-codes)
- [8. Recommended call flow](#8-recommended-call-flow)

## API Summary

Primary runtime calls:

- `MmseEqualizerCpuContext::init`
- `MmseEqualizerCpuContext::run_pdcch`
- `MmseEqualizerCpuContext::run_pdcch_td`
- `MmseEqualizerGpuContext::init`
- `MmseEqualizerGpuContext::run_pdcch`
- `MmseEqualizerGpuContext::run_pdcch_td`

Primary DTO flow:

1. upstream builds `mmse::pdcch::FrontendPdcchIndication`
2. helper converts it into `mmse::PdcchMmseInput`
3. CE/MMSE stage fills `mmse::PdcchMmseOutputView` and `mmse::PdcchMmseResult`
4. helper packs them into `mmse::pdcch::BackendPdcchEqualizedIndication`

Additive `2 Tx port` TD flow:

1. upstream builds `mmse::pdcch::FrontendPdcchIndication`
2. helper converts it into `mmse::PdcchMmseInput`
3. TD CE/MMSE stage fills `mmse::PdcchTdMmseOutputView` and `mmse::PdcchTdMmseResult`
4. helper packs them into `mmse::pdcch::BackendPdcchTdEqualizedIndication`

## Field Index

This index is intended for fast lookup of field ownership and semantics.

### Frontend DTO Fields

| Field                  | Owner type                | See section                                 |
| ---------------------- | ------------------------- | ------------------------------------------- |
| `sfn_subframe`         | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `cell_id`              | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `n_tx_ports`           | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `tx_mode`              | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `control_symbol_count` | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `n_prb`                | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `prb_bitmap`           | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `control_subframe`     | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `reserved_control_res` | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |
| `chain`                | `FrontendPdcchIndication` | [2.3](#23-mmsepdcchfrontendpdcchindication) |

### Reserved RE Fields

| Field         | Owner type          | See section                           |
| ------------- | ------------------- | ------------------------------------- |
| `symbol`      | `ReservedControlRe` | [2.1](#21-mmsepdcchreservedcontrolre) |
| `prb`         | `ReservedControlRe` | [2.1](#21-mmsepdcchreservedcontrolre) |
| `tone_in_prb` | `ReservedControlRe` | [2.1](#21-mmsepdcchreservedcontrolre) |

### Chain Metadata Fields

| Field               | Owner type           | See section                       |
| ------------------- | -------------------- | --------------------------------- |
| `request_id`        | `PdcchChainMetadata` | [2.2](#22-mmsepdcchchainmetadata) |
| `candidate_id`      | `PdcchChainMetadata` | [2.2](#22-mmsepdcchchainmetadata) |
| `first_cce`         | `PdcchChainMetadata` | [2.2](#22-mmsepdcchchainmetadata) |
| `aggregation_level` | `PdcchChainMetadata` | [2.2](#22-mmsepdcchchainmetadata) |

### Low-level Input Fields

| Field                        | Owner type       | See section                   |
| ---------------------------- | ---------------- | ----------------------------- |
| `grid`                       | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `sfn_subframe`               | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `cell_id`                    | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `n_tx_ports`                 | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `tx_mode`                    | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `control_symbol_count`       | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `n_prb`                      | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `prb_bitmap`                 | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `control_re_exclusion_masks` | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |
| `chain`                      | `PdcchMmseInput` | [2.4](#24-mmsepdcchmmseinput) |

### Output View Fields

| Field                   | Owner type            | See section                        |
| ----------------------- | --------------------- | ---------------------------------- |
| `x_hat_re`              | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `x_hat_im`              | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `sinr`                  | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `re_grid_indices`       | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `capacity_re_per_layer` | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |
| `capacity_re_metadata`  | `PdcchMmseOutputView` | [2.5](#25-mmsepdcchmmseoutputview) |

### Result Metadata Fields

| Field                  | Owner type        | See section                    |
| ---------------------- | ----------------- | ------------------------------ |
| `n_re`                 | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `sfn_subframe`         | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_symbols`            | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_subcarriers`        | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `cell_id`              | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_prb`                | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_tx_ports`           | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_rx_ant`             | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `n_layers`             | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `tx_mode`              | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `control_symbol_count` | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `mod_order`            | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `sigma2`               | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `prb_bitmap`           | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |
| `chain`                | `PdcchMmseResult` | [2.6](#26-mmsepdcchmmseresult) |

### Backend DTO Fields

| Field                  | Owner type                        | See section                                         |
| ---------------------- | --------------------------------- | --------------------------------------------------- |
| `sfn_subframe`         | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `cell_id`              | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `n_prb`                | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `n_tx_ports`           | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `n_rx_ant`             | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `n_layers`             | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `tx_mode`              | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `control_symbol_count` | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `mod_order`            | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `sigma2`               | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `chain`                | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `x_hat_re`             | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `x_hat_im`             | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `sinr`                 | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |
| `re_grid_indices`      | `BackendPdcchEqualizedIndication` | [2.7](#27-mmsepdcchbackendpdcchequalizedindication) |

### Grid Fields

| Field           | Owner type          | See section            |
| --------------- | ------------------- | ---------------------- |
| `re`            | `PlanarGridViewF32` | [3](#3-base-grid-type) |
| `im`            | `PlanarGridViewF32` | [3](#3-base-grid-type) |
| `n_rx_ant`      | `PlanarGridViewF32` | [3](#3-base-grid-type) |
| `n_symbols`     | `PlanarGridViewF32` | [3](#3-base-grid-type) |
| `n_subcarriers` | `PlanarGridViewF32` | [3](#3-base-grid-type) |

## Status Code Index

| Status code                      | Summary                                               | Detailed section    |
| -------------------------------- | ----------------------------------------------------- | ------------------- |
| `MmseStatus::kOk`                | call succeeded                                        | [7](#7-error-codes) |
| `MmseStatus::kNotInitialized`    | context was not initialized before use                | [7](#7-error-codes) |
| `MmseStatus::kInvalidArgument`   | caller-provided pointer/config/input argument invalid | [7](#7-error-codes) |
| `MmseStatus::kUnsupportedConfig` | request outside current LTE PDCCH support boundary    | [7](#7-error-codes) |
| `MmseStatus::kBufferTooSmall`    | caller output capacity insufficient                   | [7](#7-error-codes) |
| `MmseStatus::kInternalError`     | runtime/internal transport or validation failure      | [7](#7-error-codes) |

## 1. Public entrypoints

CPU:

```cpp
mmse::MmseEqualizerCpuContext::init(const MmseEqualizerCpuConfig& config)
mmse::MmseEqualizerCpuContext::run_pdcch(
    const PdcchMmseInput& in,
    PdcchMmseOutputView& out,
    PdcchMmseResult& meta)
mmse::MmseEqualizerCpuContext::run_pdcch_td(
    const PdcchMmseInput& in,
    PdcchTdMmseOutputView& out,
    PdcchTdMmseResult& meta)
```

GPU:

```cpp
mmse::MmseEqualizerGpuContext::init(const MmseEqualizerGpuConfig& config)
mmse::MmseEqualizerGpuContext::run_pdcch(
    const PdcchMmseInput& in,
    PdcchMmseOutputView& out,
    PdcchMmseResult& meta)
mmse::MmseEqualizerGpuContext::run_pdcch_td(
    const PdcchMmseInput& in,
    PdcchTdMmseOutputView& out,
    PdcchTdMmseResult& meta)
```

DTO / helper utilities:

```cpp
mmse::pdcch::make_pdcch_mmse_input(...)
mmse::pdcch::make_backend_pdcch_equalized_indication(...)
mmse::pdcch::make_backend_pdcch_td_equalized_indication(...)
mmse::pdcch::decode_re_grid_index(...)
mmse::pdcch::append_pcfich_reserved_control_re_list(...)
mmse::pdcch::append_phich_reserved_control_re_list(...)
mmse::pdcch::append_fdd_phich_reserved_control_re_list(...)
```

## 2. DTO definitions

### 2.1 `mmse::pdcch::ReservedControlRe`

One control-region RE that upstream has identified as reserved by a non-PDCCH channel.

| Field         | Type       | Meaning                               | Unit         | Valid range                                            |
| ------------- | ---------- | ------------------------------------- | ------------ | ------------------------------------------------------ |
| `symbol`      | `uint32_t` | OFDM symbol index inside the subframe | symbol index | `0..13`, practical SDK use `0..control_symbol_count-1` |
| `prb`         | `uint32_t` | PRB index                             | PRB index    | `0..99`                                                |
| `tone_in_prb` | `uint32_t` | tone index inside one PRB             | tone index   | `0..11`                                                |

Notes:

- CRS REs should not be listed here. The SDK excludes CRS internally.
- Typical use is to mark `PCFICH` and `PHICH` occupied REs.

### 2.1a `mmse::pdcch::PhichResource`

Helper enum used by automatic FDD PHICH reservation generation.

| Value       | Meaning        |
| ----------- | -------------- |
| `kOneSixth` | LTE `Ng = 1/6` |
| `kHalf`     | LTE `Ng = 1/2` |
| `kOne`      | LTE `Ng = 1`   |
| `kTwo`      | LTE `Ng = 2`   |

### 2.1b `mmse::pdcch::PhichDuplexMode`

Helper enum used by automatic PHICH reservation generation.

| Value  | Meaning |
| ------ | ------- |
| `kFdd` | LTE FDD |
| `kTdd` | LTE TDD |

### 2.1c `mmse::pdcch::PhichDuration`

Helper enum used by automatic PHICH reservation generation.

| Value       | Meaning               |
| ----------- | --------------------- |
| `kNormal`   | normal PHICH duration |
| `kExtended` | extended duration     |

### 2.1d `mmse::pdcch::PhichSubframeKind`

Structured subframe-kind enum for PHICH helper inputs.

| Value      | Meaning          |
| ---------- | ---------------- |
| `kRegular` | regular subframe |
| `kMbsfn`   | MBSFN subframe   |

### 2.1e `mmse::pdcch::LteControlSubframeContext`

Structured LTE control-subframe context shared by `PCFICH/PHICH` helpers.

| Field          | Type                | Meaning              |
| -------------- | ------------------- | -------------------- |
| `duplex_mode`  | `PhichDuplexMode`   | LTE duplex mode      |
| `subframe`     | `uint8_t`           | subframe index       |
| `ul_dl_config` | `uint8_t`           | LTE TDD UL-DL config |
| `kind`         | `PhichSubframeKind` | regular vs MBSFN     |

### 2.1f `mmse::pdcch::PhichReservationConfig`

Helper config object for automatic PHICH reservation generation.

| Field          | Type                        | Meaning            |
| -------------- | --------------------------- | ------------------ |
| `resource`     | `PhichResource`             | LTE `Ng`           |
| `duration`     | `PhichDuration`             | LTE PHICH duration |
| `mi`           | `uint8_t`                   | LTE PHICH `M_i`    |
| `subframe_ctx` | `LteControlSubframeContext` | subframe context   |

### 2.2 `mmse::PdcchChainMetadata`

Opaque chain metadata carried through the CE/MMSE stage and returned to downstream.

| Field               | Type       | Meaning                           | Unit      | Valid range    |
| ------------------- | ---------- | --------------------------------- | --------- | -------------- |
| `request_id`        | `uint64_t` | caller-defined correlation id     | none      | any            |
| `candidate_id`      | `uint32_t` | caller-defined PDCCH candidate id | none      | any            |
| `first_cce`         | `uint16_t` | caller-defined first CCE index    | CCE index | caller-defined |
| `aggregation_level` | `uint8_t`  | caller-defined aggregation level  | CCE count | caller-defined |

Notes:

- The CE/MMSE module does not interpret these fields.
- They are preserved for downstream logic.

### 2.3 `mmse::pdcch::FrontendPdcchIndication`

Formal upstream DTO used to describe one PDCCH equalization request.

| Field                  | Type                        | Meaning                                  | Unit               | Valid range                     |
| ---------------------- | --------------------------- | ---------------------------------------- | ------------------ | ------------------------------- |
| `sfn_subframe`         | `uint32_t`                  | radio time index                         | SFN\*10 + subframe | caller-defined non-negative     |
| `cell_id`              | `uint16_t`                  | LTE physical cell ID                     | none               | `0..503`                        |
| `n_tx_ports`           | `uint8_t`                   | transmit CRS port count for this request | ports              | current SDK support: `1`        |
| `tx_mode`              | `uint8_t`                   | LTE transmission mode tag                | none               | current SDK support: `1` or `2` |
| `control_symbol_count` | `uint8_t`                   | LTE control-region size                  | OFDM symbols       | `1..3`                          |
| `n_prb`                | `uint16_t`                  | PRB count enabled in bitmap              | PRBs               | `1..100`                        |
| `prb_bitmap`           | `array<uint16_t,7>`         | active PRBs                              | bitmap             | exactly `n_prb` bits set        |
| `control_subframe`     | `LteControlSubframeContext` | shared LTE control-subframe context      | context            | helper-defined                  |
| `reserved_control_res` | `vector<ReservedControlRe>` | non-PDCCH control REs                    | RE list            | zero or more                    |
| `chain`                | `PdcchChainMetadata`        | passthrough metadata                     | none               | any                             |

### 2.4 `mmse::PdcchMmseInput`

Low-level module input after helper conversion.

| Field                        | Type                        | Meaning                             | Unit               | Valid range                     |
| ---------------------------- | --------------------------- | ----------------------------------- | ------------------ | ------------------------------- |
| `grid`                       | `PlanarGridViewF32`         | FFT grid input                      | complex float32    | see `PlanarGridViewF32`         |
| `sfn_subframe`               | `uint32_t`                  | radio time index                    | SFN\*10 + subframe | non-negative                    |
| `cell_id`                    | `uint16_t`                  | LTE physical cell ID                | none               | `0..503`                        |
| `n_tx_ports`                 | `uint8_t`                   | transmit port count                 | ports              | current SDK support: `1`        |
| `tx_mode`                    | `uint8_t`                   | transmission mode tag               | none               | current SDK support: `1` or `2` |
| `control_symbol_count`       | `uint8_t`                   | LTE control region size             | OFDM symbols       | `1..3`                          |
| `n_prb`                      | `uint16_t`                  | active PRB count                    | PRBs               | `1..100`                        |
| `prb_bitmap`                 | `array<uint16_t,7>`         | active PRBs                         | bitmap             | exactly `n_prb` bits set        |
| `control_subframe`           | `LteControlSubframeContext` | shared LTE control-subframe context | context            | helper-defined                  |
| `control_re_exclusion_masks` | `array<uint16_t,300>`       | per-symbol/per-PRB RE mask          | bitmask            | 12 bits used per entry          |
| `chain`                      | `PdcchChainMetadata`        | passthrough metadata                | none               | any                             |

Mask mapping:

- index = `symbol * 100 + prb`
- bit `0..11` corresponds to `tone_in_prb`

### 2.5 `mmse::PdcchMmseOutputView`

Caller-owned write target for one `run_pdcch` invocation.

| Field                   | Type        | Meaning                                        | Unit         | Required |
| ----------------------- | ----------- | ---------------------------------------------- | ------------ | -------- |
| `x_hat_re`              | `float*`    | equalized symbol real part                     | float32      | yes      |
| `x_hat_im`              | `float*`    | equalized symbol imag part                     | float32      | yes      |
| `sinr`                  | `float*`    | post-equalization SINR                         | linear ratio | yes      |
| `re_grid_indices`       | `uint16_t*` | source RE locations in LTE grid                | grid index   | yes      |
| `capacity_re_per_layer` | `uint32_t`  | writable capacity for `x_hat_re/x_hat_im/sinr` | RE count     | yes      |
| `capacity_re_metadata`  | `uint32_t`  | writable capacity for `re_grid_indices`        | RE count     | yes      |

### 2.6 `mmse::PdcchMmseResult`

Per-call metadata returned by `run_pdcch`.

| Field                  | Type                 | Meaning                         | Unit               |
| ---------------------- | -------------------- | ------------------------------- | ------------------ |
| `n_re`                 | `uint32_t`           | number of equalized PDCCH REs   | RE count           |
| `sfn_subframe`         | `uint32_t`           | radio time index                | SFN\*10 + subframe |
| `n_symbols`            | `uint32_t`           | grid symbol dimension           | symbols            |
| `n_subcarriers`        | `uint32_t`           | grid subcarrier dimension       | subcarriers        |
| `cell_id`              | `uint16_t`           | LTE cell id                     | none               |
| `n_prb`                | `uint16_t`           | active PRB count                | PRBs               |
| `n_tx_ports`           | `uint8_t`            | transmit port count             | ports              |
| `n_rx_ant`             | `uint8_t`            | receive antenna count           | antennas           |
| `n_layers`             | `uint8_t`            | output layer count              | layers             |
| `tx_mode`              | `uint8_t`            | transmission mode               | none               |
| `control_symbol_count` | `uint8_t`            | LTE control-region size         | OFDM symbols       |
| `mod_order`            | `uint8_t`            | modulation order in bits/symbol | bits/symbol        |
| `sigma2`               | `float`              | runtime noise variance estimate | linear power       |
| `prb_bitmap`           | `array<uint16_t,7>`  | active PRBs                     | bitmap             |
| `chain`                | `PdcchChainMetadata` | passthrough metadata            | none               |

### 2.7 `mmse::pdcch::BackendPdcchEqualizedIndication`

Formal downstream owning DTO produced by helper packing.

| Field                  | Type                 | Meaning                         | Unit               |
| ---------------------- | -------------------- | ------------------------------- | ------------------ |
| `sfn_subframe`         | `uint32_t`           | radio time index                | SFN\*10 + subframe |
| `cell_id`              | `uint16_t`           | LTE cell id                     | none               |
| `n_prb`                | `uint16_t`           | active PRB count                | PRBs               |
| `n_tx_ports`           | `uint8_t`            | transmit port count             | ports              |
| `n_rx_ant`             | `uint8_t`            | receive antenna count           | antennas           |
| `n_layers`             | `uint8_t`            | output layer count              | layers             |
| `tx_mode`              | `uint8_t`            | transmission mode               | none               |
| `control_symbol_count` | `uint8_t`            | LTE control-region size         | OFDM symbols       |
| `mod_order`            | `uint8_t`            | modulation order                | bits/symbol        |
| `sigma2`               | `float`              | runtime noise variance estimate | linear power       |
| `chain`                | `PdcchChainMetadata` | passthrough metadata            | none               |
| `x_hat_re`             | `vector<float>`      | equalized real part             | float32            |
| `x_hat_im`             | `vector<float>`      | equalized imag part             | float32            |
| `sinr`                 | `vector<float>`      | per-RE SINR                     | linear ratio       |
| `re_grid_indices`      | `vector<uint16_t>`   | LTE grid source indices         | grid index         |

Invariant:

- `x_hat_re.size() == x_hat_im.size() == sinr.size() == re_grid_indices.size()`

## 3. Base grid type

### `mmse::PlanarGridViewF32`

| Field           | Meaning                    | Unit           | Constraint                  |
| --------------- | -------------------------- | -------------- | --------------------------- |
| `re[0..1]`      | real planes per RX antenna | float32 arrays | non-null                    |
| `im[0..1]`      | imag planes per RX antenna | float32 arrays | non-null                    |
| `n_rx_ant`      | receive antenna count      | antennas       | current SDK requires `2`    |
| `n_symbols`     | grid symbol count          | symbols        | current SDK requires `14`   |
| `n_subcarriers` | grid width                 | subcarriers    | current SDK requires `1200` |

Memory layout:

- one plane is indexed as `symbol * 1200 + subcarrier`
- all values are linear float32 complex planes

## 4. Helper semantics

### `make_pdcch_mmse_input(grid, frontend)`

Builds `PdcchMmseInput` from one grid and one frontend DTO.

Behavior:

- copies `FrontendPdcchIndication` fields
- copies `control_subframe` into `PdcchMmseInput`
- converts `reserved_control_res` into `control_re_exclusion_masks`
- `run_pdcch` later calls the centralized `PdcchMmseInput` validator on the low-level request
- does not validate LTE semantics by itself; validation happens in `run_pdcch`

Validator family:

- `validate_lte_control_subframe_context(...)`
- `validate_phich_reservation_config(...)`
- `validate_pdcch_mmse_input(...)`

### `append_pcfich_reserved_control_re_list(...)`

Builds and appends the LTE PCFICH-occupied control REs into `reserved_control_res`.

Behavior:

- accepts the shared `LteControlSubframeContext`
- `FrontendPdcchIndication` overload defaults to `frontend.control_subframe`
- targets the current 20 MHz normal-CP helper boundary
- marks only non-CRS REs inside the four PCFICH REGs
- preserves existing caller-provided entries
- de-duplicates repeated REs

### `append_phich_reserved_control_re_list(...)`

Builds and appends PHICH-occupied control REs into `reserved_control_res` using an explicit
`PhichReservationConfig`.

Behavior:

- returns `mmse::MmseStatus::kOk` when the helper supports the requested config
- returns `mmse::MmseStatus::kInvalidArgument` for malformed helper inputs such as
  `subframe > 9`, `ul_dl_config > 6`, or mismatched TDD `mi`
- current validated helper support is `FDD/TDD + normal CP + normal/extended PHICH duration`
- in TDD mode, `mi` must match the selected `subframe_ctx.ul_dl_config + subframe_ctx.subframe`
- for extended duration, `TDD subframe 1/6` special-case is selected automatically
- true `MBSFN` subframes are selected through `subframe_ctx.kind = kMbsfn`
- `FrontendPdcchIndication` convenience overloads can read the same context from
  `frontend.control_subframe`
- preserves existing caller-provided entries
- de-duplicates repeated REs

### `append_fdd_phich_reserved_control_re_list(...)`

Builds and appends LTE FDD normal-duration PHICH-occupied control REs into
`reserved_control_res`.

Behavior:

- convenience wrapper over `append_phich_reserved_control_re_list(...)`
- targets the current 20 MHz FDD normal-CP helper boundary
- consumes `PhichResource` to derive the PHICH group count
- marks only non-CRS REs inside the selected PHICH REGs
- preserves existing caller-provided entries
- de-duplicates repeated REs

### `make_backend_pdcch_equalized_indication(meta, out)`

Builds one owning downstream DTO from:

- `PdcchMmseResult`
- `PdcchMmseOutputView`

Behavior:

- copies metadata
- deep-copies `x_hat_re`, `x_hat_im`, `sinr`, `re_grid_indices`
- output vector sizes are exactly `meta.n_re`

### `decode_re_grid_index(grid_index)`

Decodes one LTE grid index into:

- `symbol`
- `subcarrier`
- `prb`
- `tone_in_prb`

Formula:

- `symbol = grid_index / 1200`
- `subcarrier = grid_index % 1200`
- `prb = subcarrier / 12`
- `tone_in_prb = subcarrier % 12`

## 5. Boundary conditions

Current SDK support boundary:

- LTE only
- 20 MHz only
- normal CP only
- `n_rx_ant == 2`
- `n_symbols == 14`
- `n_subcarriers == 1200`
- PDCCH only
- `control_symbol_count in [1, 3]`
- `mod_order == 2` for PDCCH
- `n_layers == 1`
- `n_tx_ports == 1`
- `tx_mode == 1 or 2`

Important non-support:

- legacy `run_pdcch(...)` rejects `2 Tx port` LTE PDCCH because its frozen contract is per-RE
- `2 Tx port` transmit-diversity is supported only through additive `run_pdcch_td(...)`
- SDK does not decode `PCFICH` or `PHICH`
- helper-based automatic PHICH reservation is limited to the documented FDD normal-CP boundary

## 6. Capacity requirements

Before calling `run_pdcch`, caller must provide enough output capacity:

- `capacity_re_per_layer >= expected_n_re`
- `capacity_re_metadata >= expected_n_re`

If capacity is too small, the call returns `kBufferTooSmall`.

If exact `expected_n_re` is unknown, allocate conservatively for LTE normal-CP control-region use.

## 7. Error codes

### `mmse::MmseStatus::kOk`

Call succeeded.

### `mmse::MmseStatus::kNotInitialized`

Cause:

- `run_pdcch` called before `init`

Action:

- call `init(...)` first

### `mmse::MmseStatus::kInvalidArgument`

Typical causes:

- null output pointers in `PdcchMmseOutputView`
- invalid config values in `MmseEqualizerCpuConfig` or `MmseEqualizerGpuConfig`
- invalid grid pointer layout

Action:

- check pointers and config ranges

### `mmse::MmseStatus::kUnsupportedConfig`

Typical causes:

- non-LTE grid dimensions
- `cell_id > 503`
- `control_symbol_count` outside `1..3`
- `mod_order != 2` for PDCCH
- `n_tx_ports != 1`
- `n_layers != 1`
- unsupported backend selection

Action:

- constrain the request to the supported LTE PDCCH boundary

### `mmse::MmseStatus::kBufferTooSmall`

Typical causes:

- `capacity_re_per_layer < n_re`
- `capacity_re_metadata < n_re`

Action:

- allocate larger output buffers

### `mmse::MmseStatus::kInternalError`

Typical causes:

- internal transport or validation failure
- unexpected CUDA runtime state mismatch

Action:

- treat as runtime failure and inspect logs / debug validation path

## 8. Recommended call flow

```cpp
#include "mmse/pdcch_chain_sdk.h"

mmse::pdcch::FrontendPdcchIndication frontend = get_frontend_indication();
mmse::PlanarGridViewF32 grid = get_fft_grid();

mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);

mmse::PdcchMmseOutputView out = allocate_output_views();
mmse::PdcchMmseResult meta{};

mmse::MmseEqualizerCpuContext ctx;
ctx.init(cpu_cfg);
ctx.run_pdcch(in, out, meta);

mmse::pdcch::BackendPdcchEqualizedIndication backend =
    mmse::pdcch::make_backend_pdcch_equalized_indication(meta, out);
```
