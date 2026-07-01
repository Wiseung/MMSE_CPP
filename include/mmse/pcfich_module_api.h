#pragma once

#include "mmse/pcfich_chain_dto.h"

namespace mmse::pcfich {

inline MmseStatus validate_pcfich_mmse_input(const PcfichMmseInput& in) {
    if (in.grid.n_rx_ant != kLteNumRxAntV1 || in.grid.n_symbols != kLteNumSymbolsNormalCp ||
        in.grid.n_subcarriers != kLteNumSubcarriers20MHz) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::uint32_t rx = 0; rx < kLteNumRxAntV1; ++rx) {
        if (in.grid.re[rx] == nullptr || in.grid.im[rx] == nullptr) {
            return MmseStatus::kInvalidArgument;
        }
    }
    if (in.n_tx_ports == 0U || in.n_tx_ports > kLteNumTxPortsV1) {
        return MmseStatus::kInvalidArgument;
    }
    if (in.tx_mode != 1U && in.tx_mode != 2U) {
        return MmseStatus::kInvalidArgument;
    }
    return MmseStatus::kOk;
}

inline PcfichMmseInput make_pcfich_mmse_input(const PlanarGridViewF32& grid,
                                              const FrontendPcfichIndication& frontend) {
    PcfichMmseInput in{};
    in.grid = grid;
    in.sfn_subframe = frontend.sfn_subframe;
    in.cell_id = frontend.cell_id;
    in.n_tx_ports = frontend.n_tx_ports;
    in.tx_mode = frontend.tx_mode;
    in.chain = frontend.chain;
    return in;
}

inline BackendPcfichEqualizedIndication
make_backend_pcfich_equalized_indication(const PcfichMmseResult& meta,
                                         const PcfichMmseOutputView& out) {
    BackendPcfichEqualizedIndication backend{};
    backend.sfn_subframe = meta.sfn_subframe;
    backend.cell_id = meta.cell_id;
    backend.n_prb = meta.n_prb;
    backend.start_symbol = meta.start_symbol;
    backend.reg_count = meta.reg_count;
    backend.n_tx_ports = meta.n_tx_ports;
    backend.n_rx_ant = meta.n_rx_ant;
    backend.n_layers = meta.n_layers;
    backend.tx_mode = meta.tx_mode;
    backend.mod_order = meta.mod_order;
    backend.sigma2 = meta.sigma2;
    backend.chain = meta.chain;
    backend.x_hat_re.assign(out.x_hat_re, out.x_hat_re + meta.n_re);
    backend.x_hat_im.assign(out.x_hat_im, out.x_hat_im + meta.n_re);
    backend.sinr.assign(out.sinr, out.sinr + meta.n_re);
    backend.re_grid_indices.assign(out.re_grid_indices, out.re_grid_indices + meta.n_re);
    return backend;
}

} // namespace mmse::pcfich
