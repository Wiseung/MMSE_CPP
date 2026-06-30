# MMSE_CPP

## Local Commit Gates

Run `npm install` once after cloning. The repo uses `husky + lint-staged` to block local commits when staged files fail formatting or when native changes break the lightweight `mmse_tests` build/test smoke check.

Native gate details:

- staged `*.cpp/*.h` files are formatted locally with `clang-format`
- staged `*.md/*.json/*.yml/*.yaml` files are formatted with `prettier`
- if staged changes touch native build/test surfaces such as `src/`, `include/`, `tests/`, `bench/`, or `CMakeLists.txt`, the pre-commit hook runs a local `cmake -> build mmse_tests -> ctest` smoke check and rejects the commit on failure

## CI / CD

The repository now uses:

- `ci`: builds and tests on Windows for `main`, `codex/**`, and pull requests
- `cd`: automatically creates a release bundle and deploys to `staging` after a successful `main` CI run; `production` is intentionally manual through `workflow_dispatch`

Required GitHub Environment / repo variables before enabling real deployment:

- `production` environment with `MMSE_PRODUCTION_DEPLOY_DIR`

Optional / recommended for staging:

- `staging` environment with `MMSE_STAGING_DEPLOY_DIR`

Current deployment behavior is file-copy based: the CD workflow creates `dist/mmse_cpp-release.zip` and copies it into the configured target directory with a timestamped filename plus a `latest` copy. If `MMSE_STAGING_DEPLOY_DIR` is not configured yet, staging falls back to a runner-local drop directory so the CD path can still execute end to end.

## CUDA Runtime Policy

The GPU transport path now exposes two explicit runtime policy knobs through `MmseEqualizerGpuConfig`:

- `sigma2_ownership`
  - `kHostOwnedIir`: host owns the IIR-smoothed `sigma2` state and pushes the scalar back to device before `equalize`
  - `kDeviceOwnedState`: device owns the smoothed `sigma2` state and host only reads a summary value for sanity/inspection
- `validation_policy`
  - `kReleaseSanity`: release path keeps only lightweight finite/positive sanity checks
  - `kTestDeepTrace`: CPU-vs-GPU trace alignment remains available for debug/test-only runs

This keeps the production path lean while preserving a stricter validation mode for test scenarios.

Under `kReleaseSanity`, CUDA scratch is reduced to a 4-float header used for lightweight sanity state:
`output_slot`, `symbol`, `subcarrier`, and runtime `sigma2`.
The per-sample equalizer trace payload is only allocated and copied under `kTestDeepTrace`.

Validation sampling is now split into two layers:

- release `spot_check_sample_count`: sampled REs used only for lightweight output checks such as
  finite `xhat`, finite positive `sinr`, and header-level `sigma2` sanity
- debug `trace_sample_count`: sampled REs that additionally emit per-sample equalizer trace payload
  for CPU-vs-GPU alignment

## LTE PDCCH Adaptation

The equalizer path now supports a second LTE downlink extraction mode through
`ExtractDescriptor::channel_type`:

- `MmseChannelType::kPdsch`: existing data-region flow
- `MmseChannelType::kPdcch`: control-region flow for LTE PDCCH RE extraction

For `kPdcch`, the caller must provide:

- `control_symbol_count`: the LTE control-region size derived from PCFICH/CFI, in normal-CP
  symbols (`1..3`)
- `control_re_exclusion_masks`: per-control-symbol, per-PRB 12-bit RE masks for control-region REs
  that must be excluded before equalization, such as PCFICH/PHICH-occupied REs

Current support boundary:

- LTE only, aligned with the repo's existing CRS-based 20 MHz normal-CP design
- PDCCH extraction and MMSE equalization are supported for `1 Tx port`
- `2 Tx port` LTE PDCCH is rejected as `unsupported_config` because the repo does not yet
  implement the control-channel transmit-diversity de-mapping stage

### Module Integration Surface

For chain integration, prefer the dedicated PDCCH module API over manually building a generic
`ExtractDescriptor`.

Recommended single-header SDK entrypoint:

- `#include "mmse/pdcch_chain_sdk.h"`
- documentation index: `docs/pdcch_chain_sdk_interface.md` (`PDCCH Chain SDK v1`)
- quick start: `docs/pdcch_chain_sdk_quick_start.md`
- API reference: `docs/pdcch_chain_sdk_api_reference.md`
- versioning policy: `docs/pdcch_chain_sdk_versioning_policy.md`

Upstream-facing input:

- `PdcchMmseInput`
  - FFT grid: `grid`
  - cell/time context: `sfn_subframe`, `cell_id`
  - PDCCH region description: `control_symbol_count`, `n_prb`, `prb_bitmap`
  - non-PDCCH control RE reservation: `control_re_exclusion_masks`
  - chain passthrough metadata for later stages: `PdcchChainMetadata`

Downstream-facing output:

- equalized soft-symbol inputs: `x_hat_re`, `x_hat_im`, `sinr`
- per-output RE source mapping: `re_grid_indices`
- run metadata and passthrough fields: `PdcchMmseResult`

This keeps the current module focused on channel estimation and equalization while preserving the
resource-location and candidate metadata needed by downstream PDCCH-specific stages.
