# PDCCH Module API Example

This document shows the intended integration boundary for the current PDCCH MMSE module in
`G:\MMSE_CPP`.

The module only does:

- CRS-based channel estimation
- PDCCH RE extraction inside the LTE control region
- MMSE equalization

The module does not do:

- PCFICH decoding
- PHICH decoding
- REG/CCE regrouping
- DCI blind detection or channel decoding

## 1. Upstream: how to fill `control_re_exclusion_masks`

Upstream should first determine:

- `control_symbol_count` from PCFICH/CFI or from a trusted external controller
- which control-region REs are occupied by non-PDCCH channels such as `PCFICH` and `PHICH`

Then fill `PdcchMmseInput` and mark those REs as excluded.

```cpp
#include "mmse/pdcch_chain_sdk.h"

mmse::pdcch::FrontendPdcchIndication frontend = get_frontend_pdcch_indication();

mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(fft_grid_view, frontend);
```

Expected meaning of each reserved RE:

- `symbol`: OFDM symbol index inside the subframe
- `prb`: PRB index in `0..99`
- `tone_in_prb`: tone index inside one PRB in `0..11`

The current module automatically excludes CRS REs. Upstream should only mark non-PDCCH control REs
that still remain after that, typically `PCFICH` and `PHICH`.

## 2. Module call

```cpp
mmse::MmseEqualizerCpuContext ctx;
mmse::MmseEqualizerCpuConfig cfg{};
cfg.worker_count = 1;
ctx.init(cfg);

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
const mmse::MmseStatus status = ctx.run_pdcch(in, out, meta);
```

## 3. Downstream: how to consume `re_grid_indices`

Downstream should first pack the module output into the formal backend DTO, then consume that DTO.

```cpp
mmse::pdcch::BackendPdcchEqualizedIndication backend =
    mmse::pdcch::make_backend_pdcch_equalized_indication(meta, out);

for (std::size_t i = 0; i < backend.re_grid_indices.size(); ++i) {
    const auto coord = mmse::pdcch::decode_re_grid_index(backend.re_grid_indices[i]);

    DownstreamSoftRe item{};
    item.xhat_re = backend.x_hat_re[i];
    item.xhat_im = backend.x_hat_im[i];
    item.sinr = backend.sinr[i];
    item.symbol = coord.symbol;
    item.subcarrier = coord.subcarrier;
    item.prb = coord.prb;
    item.tone_in_prb = coord.tone_in_prb;
    item.request_id = backend.chain.request_id;
    item.candidate_id = backend.chain.candidate_id;
    item.first_cce = backend.chain.first_cce;
    item.aggregation_level = backend.chain.aggregation_level;

    downstream_consume(item);
}
```

This is the intended handoff contract:

- the current module can be converted into one formal backend DTO
- the backend DTO carries equalized soft symbols, per-RE SINR, and `re_grid_indices`
- downstream uses `re_grid_indices` to rebuild REG/CCE or any other required ordering
- chain metadata is preserved end to end through `PdcchChainMetadata`

## 4. Practical rules

- Upstream should treat `control_re_exclusion_masks` as the list of non-PDCCH control REs
- Upstream should not mark CRS REs there; CRS is already excluded internally
- Downstream should not assume output order equals CCE order
- Downstream should use `re_grid_indices` to rebuild any required ordering
- Current support is limited to `1 Tx port` LTE PDCCH

## 5. Compilable demo

The repo also provides a small compilable demo:

- source: `bench/pdcch_module_demo.cpp`
- target: `pdcch_module_demo`

Build and run:

```powershell
cmake --build build --config Debug --target pdcch_module_demo
.\build\Debug\pdcch_module_demo.exe
```
