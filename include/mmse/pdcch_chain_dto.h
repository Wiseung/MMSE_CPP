#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "mmse/constants.h"
#include "mmse/types.h"

namespace mmse::pdcch {

struct ReservedControlRe {
    std::uint32_t symbol = 0;
    std::uint32_t prb = 0;
    std::uint32_t tone_in_prb = 0;
};

struct FrontendPdcchIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 1;
    std::uint8_t tx_mode = 1;
    std::uint8_t control_symbol_count = 0;
    std::uint16_t n_prb = 0;
    std::array<std::uint16_t, 7> prb_bitmap{};
    LteControlSubframeContext control_subframe{};
    std::vector<ReservedControlRe> reserved_control_res{};
    PdcchChainMetadata chain{};
};

struct BackendPdcchEqualizedIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PdcchChainMetadata chain{};
    std::vector<float> x_hat_re{};
    std::vector<float> x_hat_im{};
    std::vector<float> sinr{};
    std::vector<std::uint16_t> re_grid_indices{};
};

struct BackendPdcchTdEqualizedIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PdcchChainMetadata chain{};
    std::vector<float> x_hat_re{};
    std::vector<float> x_hat_im{};
    std::vector<float> sinr{};
    std::vector<std::uint16_t> re_grid_indices0{};
    std::vector<std::uint16_t> re_grid_indices1{};
};

} // namespace mmse::pdcch
