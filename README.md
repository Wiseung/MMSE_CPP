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
