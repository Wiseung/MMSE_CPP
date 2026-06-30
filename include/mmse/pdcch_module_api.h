#pragma once

#include <cstdint>

#include "mmse/constants.h"
#include "mmse/pdcch_chain_dto.h"
#include "mmse/types.h"

namespace mmse::pdcch {

struct ReCoord {
    std::uint32_t symbol = 0;
    std::uint32_t subcarrier = 0;
    std::uint32_t prb = 0;
    std::uint32_t tone_in_prb = 0;
};

inline constexpr std::size_t control_mask_index(std::uint32_t symbol, std::uint32_t prb) {
    return static_cast<std::size_t>(symbol) * kLteNumPrb20MHz + static_cast<std::size_t>(prb);
}

inline void clear_control_re_exclusion_masks(PdcchMmseInput& in) {
    in.control_re_exclusion_masks.fill(0U);
}

inline void exclude_control_re(PdcchMmseInput& in, std::uint32_t symbol, std::uint32_t prb,
                               std::uint32_t tone_in_prb) {
    if (symbol >= kLteMaxControlSymbolsNormalCp || prb >= kLteNumPrb20MHz ||
        tone_in_prb >= kLteNumSubcarriersPerPrb) {
        return;
    }
    in.control_re_exclusion_masks[control_mask_index(symbol, prb)] |=
        static_cast<std::uint16_t>(1U << tone_in_prb);
}

inline bool is_control_re_excluded(const PdcchMmseInput& in, std::uint32_t symbol,
                                   std::uint32_t prb, std::uint32_t tone_in_prb) {
    if (symbol >= kLteMaxControlSymbolsNormalCp || prb >= kLteNumPrb20MHz ||
        tone_in_prb >= kLteNumSubcarriersPerPrb) {
        return false;
    }
    return (in.control_re_exclusion_masks[control_mask_index(symbol, prb)] &
            static_cast<std::uint16_t>(1U << tone_in_prb)) != 0U;
}

template <typename Range>
inline void apply_reserved_control_re_list(PdcchMmseInput& in, const Range& reserved_control_res) {
    clear_control_re_exclusion_masks(in);
    for (const auto& re : reserved_control_res) {
        exclude_control_re(in, re.symbol, re.prb, re.tone_in_prb);
    }
}

inline PdcchMmseInput make_pdcch_mmse_input(const PlanarGridViewF32& grid,
                                            const FrontendPdcchIndication& frontend) {
    PdcchMmseInput in{};
    in.grid = grid;
    in.sfn_subframe = frontend.sfn_subframe;
    in.cell_id = frontend.cell_id;
    in.n_tx_ports = frontend.n_tx_ports;
    in.tx_mode = frontend.tx_mode;
    in.control_symbol_count = frontend.control_symbol_count;
    in.n_prb = frontend.n_prb;
    in.prb_bitmap = frontend.prb_bitmap;
    in.chain = frontend.chain;
    apply_reserved_control_re_list(in, frontend.reserved_control_res);
    return in;
}

inline BackendPdcchEqualizedIndication
make_backend_pdcch_equalized_indication(const PdcchMmseResult& meta,
                                        const PdcchMmseOutputView& out) {
    BackendPdcchEqualizedIndication backend{};
    backend.sfn_subframe = meta.sfn_subframe;
    backend.cell_id = meta.cell_id;
    backend.n_prb = meta.n_prb;
    backend.n_tx_ports = meta.n_tx_ports;
    backend.n_rx_ant = meta.n_rx_ant;
    backend.n_layers = meta.n_layers;
    backend.tx_mode = meta.tx_mode;
    backend.control_symbol_count = meta.control_symbol_count;
    backend.mod_order = meta.mod_order;
    backend.sigma2 = meta.sigma2;
    backend.chain = meta.chain;
    backend.x_hat_re.assign(out.x_hat_re, out.x_hat_re + meta.n_re);
    backend.x_hat_im.assign(out.x_hat_im, out.x_hat_im + meta.n_re);
    backend.sinr.assign(out.sinr, out.sinr + meta.n_re);
    backend.re_grid_indices.assign(out.re_grid_indices, out.re_grid_indices + meta.n_re);
    return backend;
}

inline ReCoord decode_re_grid_index(std::uint16_t grid_index) {
    const std::uint32_t symbol = grid_index / kLteNumSubcarriers20MHz;
    const std::uint32_t subcarrier = grid_index % kLteNumSubcarriers20MHz;
    return {
        .symbol = symbol,
        .subcarrier = subcarrier,
        .prb = subcarrier / kLteNumSubcarriersPerPrb,
        .tone_in_prb = subcarrier % kLteNumSubcarriersPerPrb,
    };
}

} // namespace mmse::pdcch
