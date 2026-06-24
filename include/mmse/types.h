#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace mmse {

enum class MmseStatus : std::uint8_t {
    kOk = 0,
    kNotInitialized,
    kInvalidArgument,
    kUnsupportedConfig,
    kBufferTooSmall,
    kInternalError,
};

struct PlanarGridViewF32 {
    std::array<const float*, 2> re{};
    std::array<const float*, 2> im{};
    std::uint32_t n_rx_ant = 0;
    std::uint32_t n_symbols = 0;
    std::uint32_t n_subcarriers = 0;
};

struct ExtractDescriptor {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t mod_order = 0;
    std::uint16_t n_prb = 0;
    std::array<std::uint16_t, 7> prb_bitmap{};
    std::int8_t pmi = -1;
};

struct EqualizerOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint32_t capacity_re_per_layer = 0;
    std::uint32_t n_re_per_layer = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t mod_order = 0;
};

struct MmseEqualizerCpuConfig {
    std::uint32_t worker_count = 1;
    float sigma2_iir_alpha = 0.8F;
    float sigma2_min = 1.0e-4F;
    float det_floor = 1.0e-6F;
    float g_min = 1.0e-4F;
    float gamma_max = 1.0e4F;
};

}  // namespace mmse
