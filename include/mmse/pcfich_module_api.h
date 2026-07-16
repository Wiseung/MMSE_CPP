#pragma once

#include "mmse/lte_descrambling.h"
#include "mmse/lte_soft_demod.h"
#include "mmse/pcfich_chain_dto.h"

namespace mmse::pcfich {

inline MmseStatus validate_pcfich_mmse_input(const PcfichMmseInput& in) {
    if (in.cell_id >= kLteNumCellIds) {
        return MmseStatus::kInvalidArgument;
    }
    if (in.grid.n_rx_ant == 0U || in.grid.n_rx_ant > kMmseV1MaxNumRxAntennas ||
        in.grid.n_symbols != kLteNumSymbolsNormalCp ||
        in.grid.n_subcarriers != kLteNumSubcarriers20MHz) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::uint32_t rx = 0; rx < in.grid.n_rx_ant; ++rx) {
        if (in.grid.re[rx] == nullptr || in.grid.im[rx] == nullptr) {
            return MmseStatus::kInvalidArgument;
        }
    }
    if (in.n_tx_ports == 0U || in.n_tx_ports > kMmseV1MaxNumCrsTxPorts) {
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

inline BackendPcfichDescrambledLlrIndication
make_backend_pcfich_descrambled_llr_indication(const BackendPcfichEqualizedIndication& backend) {
    BackendPcfichDescrambledLlrIndication llr_backend{};
    llr_backend.sfn_subframe = backend.sfn_subframe;
    llr_backend.cell_id = backend.cell_id;
    llr_backend.n_prb = backend.n_prb;
    llr_backend.start_symbol = backend.start_symbol;
    llr_backend.reg_count = backend.reg_count;
    llr_backend.n_tx_ports = backend.n_tx_ports;
    llr_backend.n_rx_ant = backend.n_rx_ant;
    llr_backend.n_layers = backend.n_layers;
    llr_backend.tx_mode = backend.tx_mode;
    llr_backend.mod_order = backend.mod_order;
    llr_backend.sigma2 = backend.sigma2;
    llr_backend.chain = backend.chain;
    llr_backend.re_grid_indices = backend.re_grid_indices;

    const std::uint8_t n_layers = backend.n_layers == 0U ? 1U : backend.n_layers;
    const std::uint32_t n_re_per_layer =
        static_cast<std::uint32_t>(backend.x_hat_re.size() / n_layers);
    (void)mmse::lte::build_max_log_llrs(backend.x_hat_re.data(), backend.x_hat_im.data(),
                                        backend.sinr.data(), n_re_per_layer, n_re_per_layer,
                                        n_layers, backend.mod_order, llr_backend.llrs);
    mmse::lte::descramble_llrs_inplace(
        llr_backend.llrs.data(), static_cast<std::uint32_t>(llr_backend.llrs.size()),
        mmse::lte::pcfich_c_init(backend.cell_id, backend.sfn_subframe));
    return llr_backend;
}

inline BackendPcfichTdEqualizedIndication
make_backend_pcfich_td_equalized_indication(const PcfichTdMmseResult& meta,
                                            const PcfichTdMmseOutputView& out) {
    BackendPcfichTdEqualizedIndication backend{};
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
    backend.x_hat_re.assign(out.x_hat_re, out.x_hat_re + meta.n_symbols);
    backend.x_hat_im.assign(out.x_hat_im, out.x_hat_im + meta.n_symbols);
    backend.sinr.assign(out.sinr, out.sinr + meta.n_symbols);
    backend.re_grid_indices0.assign(out.re_grid_indices0, out.re_grid_indices0 + meta.n_symbols);
    backend.re_grid_indices1.assign(out.re_grid_indices1, out.re_grid_indices1 + meta.n_symbols);
    return backend;
}

inline BackendPcfichTdDescrambledLlrIndication make_backend_pcfich_td_descrambled_llr_indication(
    const BackendPcfichTdEqualizedIndication& backend) {
    BackendPcfichTdDescrambledLlrIndication llr_backend{};
    llr_backend.sfn_subframe = backend.sfn_subframe;
    llr_backend.cell_id = backend.cell_id;
    llr_backend.n_prb = backend.n_prb;
    llr_backend.start_symbol = backend.start_symbol;
    llr_backend.reg_count = backend.reg_count;
    llr_backend.n_tx_ports = backend.n_tx_ports;
    llr_backend.n_rx_ant = backend.n_rx_ant;
    llr_backend.n_layers = backend.n_layers;
    llr_backend.tx_mode = backend.tx_mode;
    llr_backend.mod_order = backend.mod_order;
    llr_backend.sigma2 = backend.sigma2;
    llr_backend.chain = backend.chain;
    llr_backend.re_grid_indices0 = backend.re_grid_indices0;
    llr_backend.re_grid_indices1 = backend.re_grid_indices1;
    if (backend.n_layers != 1U || backend.mod_order != 2U ||
        backend.x_hat_im.size() != backend.x_hat_re.size() ||
        backend.sinr.size() != backend.x_hat_re.size()) {
        return llr_backend;
    }
    (void)mmse::lte::build_max_log_llrs(backend.x_hat_re.data(), backend.x_hat_im.data(),
                                        backend.sinr.data(),
                                        static_cast<std::uint32_t>(backend.x_hat_re.size()),
                                        static_cast<std::uint32_t>(backend.x_hat_re.size()), 1U,
                                        backend.mod_order, llr_backend.llrs);
    mmse::lte::descramble_llrs_inplace(
        llr_backend.llrs.data(), static_cast<std::uint32_t>(llr_backend.llrs.size()),
        mmse::lte::pcfich_c_init(backend.cell_id, backend.sfn_subframe));
    return llr_backend;
}

} // namespace mmse::pcfich
