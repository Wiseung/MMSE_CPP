# PDCCH Chain SDK Versioning Policy

This page defines the compatibility boundary for `PDCCH Chain SDK v1`.

This repo also ships an additive parallel TD surface for `2 Tx port` LTE PDCCH:

- `MmseEqualizerCpuContext::run_pdcch_td`
- `MmseEqualizerGpuContext::run_pdcch_td`
- `PdcchTdMmseOutputView`
- `PdcchTdMmseResult`
- `BackendPdcchTdEqualizedIndication`

Those symbols are outside the frozen `v1` single-RE contract described below. The `v1` boundary
still applies specifically to `run_pdcch(...)` and its per-RE DTO semantics.

Related pages:

- [Documentation Index](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [Quick Start](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [API Reference](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)

## Version Covered

- `PDCCH Chain SDK v1`

## Frozen Interface Surface in v1

The following interface surface is considered stable and must not change incompatibly within v1:

- public umbrella header path:
  - `mmse/pdcch_chain_sdk.h`
- public DTO names:
  - `mmse::pdcch::FrontendPdcchIndication`
  - `mmse::pdcch::ReservedControlRe`
  - `mmse::pdcch::BackendPdcchEqualizedIndication`
  - `mmse::PdcchMmseInput`
  - `mmse::PdcchMmseOutputView`
  - `mmse::PdcchMmseResult`
  - `mmse::PdcchChainMetadata`
- public helper names:
  - `mmse::pdcch::make_pdcch_mmse_input`
  - `mmse::pdcch::make_backend_pdcch_equalized_indication`
  - `mmse::pdcch::decode_re_grid_index`
- runtime entrypoints:
  - `MmseEqualizerCpuContext::run_pdcch`
  - `MmseEqualizerGpuContext::run_pdcch`
- field meanings, units, and range semantics defined in the API reference
- output ordering contract:
  - `x_hat_re[i]`, `x_hat_im[i]`, `sinr[i]`, and `re_grid_indices[i]` describe the same RE
- LTE support boundary documented in the API reference
- `MmseStatus` meanings documented in the API reference

## Disallowed Changes Within v1

The following are incompatible changes and must not be made under v1:

- renaming or removing any public DTO, helper, or entrypoint listed above
- changing any documented field type
- changing documented units
  - for example changing `sinr` from linear to dB
  - or changing `sigma2` from linear power to another scale
- changing `re_grid_indices` semantics away from LTE grid indexing
- changing `decode_re_grid_index` mapping formula
- changing the meaning of `PdcchChainMetadata` passthrough fields
- changing `control_re_exclusion_masks` bit layout
- changing `run_pdcch` success/failure meaning for documented `MmseStatus` values
- silently expanding support in a way that changes current validated behavior
  - for example claiming `2 Tx port` PDCCH support without a version bump and updated contract

## Allowed Changes Within v1

The following changes are allowed without a major interface bump:

- internal implementation changes
  - performance work
  - memory layout changes internal to the library
  - CPU/GPU transport changes
- stronger validation, provided the documented valid-input surface is unchanged
- additive documentation clarifications
- additive helper functions
- additive DTO fields appended in a backward-compatible way, if and only if:
  - existing fields keep the same names, types, order meaning, and units
  - old callers that ignore the new fields continue to work
- additive `MmseStatus` handling guidance in documentation, as long as existing enum values keep
  their meanings

## Changes That Require a New Major Interface Version

The following require a new major version such as `v2`:

- changing the single-header SDK entrypoint path
- changing any frozen field type or semantic
- changing helper return semantics
- changing output indexing semantics
- changing the supported PHY scope
  - for example adding NR-specific semantics into the same DTO contract
- introducing `2 Tx port` PDCCH support with different required inputs or output interpretation

## Recommended Migration Rule

When an incompatible change is needed:

1. keep `v1` behavior intact
2. introduce a new `v2` contract in parallel
3. document the migration delta explicitly
4. only retire `v1` after downstream users have moved
