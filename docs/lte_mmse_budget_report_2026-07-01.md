# LTE MMSE Budget Report (2026-07-01)

## Scope

This report answers one specific question for the current `G:\MMSE_CPP` checkout:

- if one `10 ms` LTE processing window needs to complete decoding for the full physical-channel
  set (`PBCH`, `PDSCH`, `PDCCH`, `PCFICH`),
- and the current CE/MMSE module may be called `5-6` times inside that window,
- how much latency budget should this module consume,
- and does the next optimization step still belong inside the CE/MMSE kernel path, or in the
  higher-level call topology.

## Interpretation Boundary

Important boundary for this report:

- the current public MMSE API still processes one LTE `1 ms` subframe per call
- therefore any `10 ms` judgment must be reasoned as repeated `1 ms` subframe calls unless the
  interface changes
- this report uses the current wrapper surfaces:
  - `run_pbch(...)`
  - `run_pcfich(...)`
  - `run_pdcch(...)`
  - generic `run(...)` for `PDSCH`

## Current Chain Understanding

At the current repository boundary, the MMSE module covers:

- RE extraction
- CRS-based channel estimation
- MMSE equalization
- equalized soft-symbol and `SINR` handoff

It does not cover the downstream heavy decode stages such as:

- `PBCH` descrambling / rate recovery / convolutional decode / `MIB`
- `PDCCH` blind search / channel decode / `DCI`
- `PDSCH` descrambling / rate recovery / turbo decode / `MAC PDU`
- `PCFICH` final `CFI` decode

So the CE/MMSE block must leave enough time budget for those later stages.

## Measured Current Baseline

### 1. Full-band GPU profile

Command used:

```powershell
.\build\Release\mmse_cuda_profile.exe --warmup-subframes 5 --iters-10ms 5 --iters-100ms 0
```

Measured current full-band `1 ms` subframe cost on this machine:

- `timing.avg_10ms_per_subframe_ms = 0.346782 ms`

Measured host/GPU split:

- `host.phase.total_host_us = 308.93 us`
- `host.phase.estimate_gpu_us = 73.86 us`
- `host.phase.equalize_gpu_us = 6.66 us`
- `host.phase.stream_gpu_us = 157.44 us`

Current implication:

- `equalize` is already a very small fraction of the total
- `estimate` is no longer the old `10+ ms` serialized bottleneck from the earlier stub
- the current cost is dominated more by per-call framework / staging / synchronization overhead than
  by the equalization math itself

### 2. Per-channel wrapper benchmark

Benchmark target added for this report:

- [mmse_channel_budget.cpp](G:\MMSE_CPP\bench\mmse_channel_budget.cpp)

Command used:

```powershell
cmd /c "G:\MMSE_CPP\build\Release\mmse_channel_budget.exe"
```

#### CPU path: independent-call baseline

| Metric                |      Average |         P95 |
| --------------------- | -----------: | ----------: |
| `PBCH`                |  `272.96 us` |  `288.7 us` |
| `PCFICH`              |  `268.95 us` |  `287.6 us` |
| `PDCCH`               |  `292.56 us` |  `310.2 us` |
| `PDSCH`               |  `387.29 us` |  `437.9 us` |
| six independent calls | `1916.21 us` | `2088.4 us` |

#### GPU path: independent-call baseline

| Metric                |      Average |         P95 |
| --------------------- | -----------: | ----------: |
| `PBCH`                |  `357.27 us` |  `407.2 us` |
| `PCFICH`              |  `354.43 us` |  `435.2 us` |
| `PDCCH`               |  `358.46 us` |  `404.7 us` |
| `PDSCH`               |  `418.31 us` |  `484.1 us` |
| six independent calls | `2262.90 us` | `2473.5 us` |

Observed current extracted RE counts:

- `PBCH`: `240 RE`
- `PCFICH`: `16 RE`
- `PDCCH`: `3228 RE`
- `PDSCH`: `14400 RE / layer`

## What These Numbers Mean

### 1. The CE/MMSE kernel path is already in a usable range

The old question was whether the CE/MMSE math itself still needed major kernel-level work.

For the current code and this machine, the answer is:

- not as the first priority

Why:

- one full-band `1 ms` GPU call is already around `0.35 ms`
- even a large `PDSCH` wrapper call is only around `0.42 ms`
- `equalize` itself is only single-digit microseconds in the profiler

This means further inner-kernel micro-optimization is not the highest-leverage next step.

### 2. The current problem is the number of independent calls

The current six-call aggregate is:

- average `2.263 ms`
- p95 `2.474 ms`

That is already much smaller than the old `12 ms / subframe` baseline, but it is still too large
for a clean full-chain `10 ms` decode budget.

The main issue is repeated:

- layout build
- grid/meta packing
- wrapper validation
- channel estimation launch/state handling
- output staging / synchronization

across multiple physical-channel wrapper calls that live on the same LTE subframe context.

## Recommended LTE 10 ms Budget

### Recommended CE/MMSE budget for the full 10 ms decode window

For a `10 ms` end-to-end LTE physical-channel decode target, I recommend:

- ideal CE/MMSE budget: `<= 1.0 ms`
- acceptable CE/MMSE upper bound: `<= 1.5 ms`
- do not accept long-term design above: `1.5 ms`

Reasoning:

- the current measured six-call path at `~2.26 ms` would consume about `22.6%` of the total
  `10 ms` wall-clock budget
- that is too much for a block that still leaves all heavy channel-decoding work downstream
- a `<= 1.0 ms` target keeps the MMSE block near `10%` of the total frame budget
- a `<= 1.5 ms` upper bound still leaves reasonable space for control and data-channel decoding

### Required improvement from current measured state

Using the measured GPU six-call aggregate:

- current average: `2.263 ms`
- target average: `<= 1.0 ms`
  - required reduction: about `55.8%`
  - required speedup: about `2.26x`
- upper-bound average: `<= 1.5 ms`
  - required reduction: about `33.7%`
  - required speedup: about `1.51x`

## Recommended Next Optimization Direction

### Primary recommendation

Do **not** return first to kernel micro-surgery.

Do this first instead:

- estimate once per LTE subframe
- reuse that estimate across all channel-specific extraction/equalization paths

### Why this is the right next step

Inside one LTE subframe, `PBCH`, `PCFICH`, `PDCCH`, and `PDSCH` are not independent radio scenes.
They share:

- the same FFT grid
- the same CRS context
- the same cell id
- the same receive antennas
- the same subframe timing

So the highest-leverage design is not `5-6` fully independent MMSE wrapper calls, but something
closer to:

1. stage the subframe once
2. estimate the channel once
3. reuse the estimate for:
   - `PBCH`
   - `PCFICH`
   - `PDCCH`
   - `PDSCH`
4. only do per-channel RE layout / equalized output extraction afterward

### Practical target for the next design cut

The next implementation cut should aim to reduce:

- repeated channel-estimation work
- repeated host/device setup and synchronization

and validate that the aggregate path moves from:

- `~2.26 ms`

down to:

- `<= 1.5 ms` first
- then `<= 1.0 ms` if downstream decode budget remains tight

## Suggested Engineering Milestones

### Milestone 1

Introduce an internal subframe-scoped CE artifact:

- one staged grid
- one channel-estimate result
- one reusable per-subframe runtime state

### Milestone 2

Expose internal multi-channel reuse flow:

- `PBCH` extract from cached estimate
- `PCFICH` extract from cached estimate
- `PDCCH` extract from cached estimate
- `PDSCH` extract from cached estimate

### Milestone 3

Re-benchmark two shapes:

- current independent-call path
- shared-estimate multi-channel path

and compare:

- total average
- p95
- host-side staging cost
- estimate reuse win

### Milestone 4

Only if the shared-estimate path still stays above `1.5 ms`, return to:

- estimate-stage kernel optimization
- synchronization trimming
- output staging reduction

## Final Judgment

For the current machine and current code:

- the CE/MMSE module is already good enough at the **single-call kernel/math** level
- it is **not yet good enough** at the **multi-call chain-level budget** level

So the next target is:

- **not** “make one call faster at any cost”
- **yes** “bring the full `5-6` call aggregate below `1.5 ms`, and preferably near `1.0 ms`”

That is the right optimization target for a `10 ms` full physical-channel decode goal.

## Refactor Outcome Update

The subframe-scoped shared-estimate refactor was then implemented and remeasured on the same
machine using one same-subframe representative six-call shape:

- `PBCH`
- `PCFICH`
- `PDCCH`
- `PDSCH`
- `PDCCH`
- `PDSCH`

### Shared-estimate benchmark result

Source:

- [mmse_channel_budget.cpp](G:\MMSE_CPP\bench\mmse_channel_budget.cpp)

Measured result after the refactor:

#### CPU path: shared-estimate same-subframe shape

| Metric             |     Average |        P95 |
| ------------------ | ----------: | ---------: |
| `PBCH`             |  `74.54 us` |  `89.8 us` |
| `PCFICH`           |  `74.15 us` |  `93.1 us` |
| `PDCCH`            | `101.28 us` | `125.0 us` |
| `PDSCH`            | `175.76 us` | `202.8 us` |
| six-call aggregate | `719.70 us` | `803.1 us` |

#### GPU path: shared-estimate same-subframe shape

| Metric             |      Average |         P95 |
| ------------------ | -----------: | ----------: |
| `PBCH`             |  `155.01 us` |  `199.8 us` |
| `PCFICH`           |  `146.77 us` |  `197.7 us` |
| `PDCCH`            |  `157.41 us` |  `200.8 us` |
| `PDSCH`            |  `215.49 us` |  `270.3 us` |
| six-call aggregate | `1020.30 us` | `1162.7 us` |

### Before vs after

GPU six-call aggregate moved from:

- before: `2262.90 us`
- after: `1020.30 us`

Improvement:

- absolute reduction: `1242.60 us`
- relative reduction: about `54.9%`
- speedup: about `2.22x`

### What caused the win

The win came from eliminating repeated same-subframe work:

- repeated grid staging / cache preparation
- repeated channel-estimate generation
- repeated `sigma2` update path

The win did **not** come from further equalizer kernel micro-optimization.

### Final budget judgment after implementation

The refactor now satisfies the phase-1 hard target:

- target: `gpu.six_calls.avg_us <= 1500`
- result: `gpu.six_calls.avg_us = 1020.30`

It does not fully meet the stretch goal yet:

- stretch goal: `<= 1000 us`
- current gap: about `20.3 us`

Final judgment:

- the CE/MMSE module is now inside the acceptable `10 ms` full-chain budget range
- the next optimization step no longer needs to be mandatory for schedule viability
- if further work is desired, it should focus on the remaining host/device framework overhead rather
  than on the equalization math itself
