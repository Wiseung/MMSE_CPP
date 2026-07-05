#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mmse::pdsch {

struct PdschDescrambledLlrOutputView {
    float* llrs = nullptr;
    std::uint32_t capacity_llrs = 0;
};

struct PdschDescrambledLlrResult {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint16_t rnti = 0;
    std::array<std::uint16_t, 7> prb_bitmap{};
    std::uint32_t n_re_per_layer = 0;
    std::uint32_t llr_count_per_layer = 0;
    std::uint32_t llr_count = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t mod_order = 0;
    std::uint8_t codeword = 0;
    std::int8_t pmi = -1;
};

struct PdschDescramblingPlanCache {
    std::uint32_t c_init = 0;
    std::uint32_t llr_count = 0;
    bool valid = false;
    std::vector<std::uint8_t> bits{};
};

struct BackendPdschDescrambledLlrIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint16_t rnti = 0;
    std::array<std::uint16_t, 7> prb_bitmap{};
    std::uint32_t n_re_per_layer = 0;
    std::uint32_t llr_count_per_layer = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t mod_order = 0;
    std::uint8_t codeword = 0;
    std::int8_t pmi = -1;
    std::vector<float> llrs{};
};

} // namespace mmse::pdsch
