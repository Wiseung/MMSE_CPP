# LTE Equalized Channel SDK Documentation

This page is the stable documentation entrypoint for the LTE equalized-channel SDK exported by:

```cpp
#include "mmse/lte_chain_sdk.h"
```

Current interface version:

- `LTE Equalized Channel SDK v1`

Documentation set:

- [PBCH Quick Start](/G:/MMSE_CPP/docs/pbch_chain_sdk_quick_start.md)
- [PCFICH Quick Start and API Reference](/G:/MMSE_CPP/docs/pcfich_chain_sdk_quick_start_api_reference.md)
- [PDCCH Subpage](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [PDCCH Quick Start](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [PDCCH API Reference](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)
- [PDCCH Versioning Policy](/G:/MMSE_CPP/docs/pdcch_chain_sdk_versioning_policy.md)
- [PDCCH Integration Example](/G:/MMSE_CPP/docs/pdcch_module_api_example.md)

## Scope

The LTE equalized-channel SDK covers:

- LTE PBCH equalized RE extraction surface
- LTE PDCCH control-region equalized RE extraction surface
- LTE PCFICH equalized RE extraction surface
- CRS-based channel estimation
- MMSE equalization
- caller-owned output views plus backend DTO packing

The current documentation set is still deepest on the `PDCCH` integration path. `PBCH` and
`PCFICH` now share the same runtime and DTO style, but do not yet have standalone quick-start or
field-reference pages.

## Recommended Reading Order

1. Read [LTE downlink overview](/G:/MMSE_CPP/docs/lte_pdcch_pdsch_channel_decode_overview.md) for
   protocol context.
2. Read [PBCH Quick Start](/G:/MMSE_CPP/docs/pbch_chain_sdk_quick_start.md) when integrating the
   PBCH equalized-RE surface.
3. Read [PCFICH Quick Start and API Reference](/G:/MMSE_CPP/docs/pcfich_chain_sdk_quick_start_api_reference.md)
   when integrating the PCFICH equalized-RE surface.
4. Read [PDCCH Subpage](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md) for the most complete
   channel-specific integration surface currently documented.

## Public Header Layout

- unified LTE umbrella header:
  - `#include "mmse/lte_chain_sdk.h"`
- PDCCH-only umbrella header:
  - `#include "mmse/pdcch_chain_sdk.h"`

## Current Channel Surfaces

### PBCH

- frontend DTO namespace: `mmse::pbch`
- low-level input/output: `PbchMmseInput`, `PbchMmseOutputView`, `PbchMmseResult`
- runtime entrypoints:
  - `MmseEqualizerCpuContext::run_pbch(...)`
  - `MmseEqualizerGpuContext::run_pbch(...)`

### PDCCH

- frontend DTO namespace: `mmse::pdcch`
- low-level input/output:
  - `PdcchMmseInput`, `PdcchMmseOutputView`, `PdcchMmseResult`
  - additive TD path: `PdcchTdMmseOutputView`, `PdcchTdMmseResult`
- runtime entrypoints:
  - `MmseEqualizerCpuContext::run_pdcch(...)`
  - `MmseEqualizerGpuContext::run_pdcch(...)`
  - additive TD path:
    - `MmseEqualizerCpuContext::run_pdcch_td(...)`
    - `MmseEqualizerGpuContext::run_pdcch_td(...)`

### PCFICH

- frontend DTO namespace: `mmse::pcfich`
- low-level input/output: `PcfichMmseInput`, `PcfichMmseOutputView`, `PcfichMmseResult`
- runtime entrypoints:
  - `MmseEqualizerCpuContext::run_pcfich(...)`
  - `MmseEqualizerGpuContext::run_pcfich(...)`

## Documentation Status

- `PBCH` now has a dedicated quick-start page.
- `PCFICH` now has a dedicated quick-start and compact API-reference page.
- `PDCCH` remains the deepest documented subpage and still carries the most detailed field index.
