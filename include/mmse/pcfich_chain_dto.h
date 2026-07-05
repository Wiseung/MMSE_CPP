#pragma once

#include <cstdint>
#include <vector>

#include "mmse/types.h"

namespace mmse::pcfich {

struct FrontendPcfichIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 1;
    std::uint8_t tx_mode = 1;
    PcfichChainMetadata chain{};
};

struct BackendPcfichEqualizedIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t reg_count = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PcfichChainMetadata chain{};
    std::vector<float> x_hat_re{};
    std::vector<float> x_hat_im{};
    std::vector<float> sinr{};
    std::vector<std::uint16_t> re_grid_indices{};
};

struct BackendPcfichDescrambledLlrIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t reg_count = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PcfichChainMetadata chain{};
    std::vector<float> llrs{};
    std::vector<std::uint16_t> re_grid_indices{};
};

} // namespace mmse::pcfich
