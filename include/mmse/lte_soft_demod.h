#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "mmse/lte_descrambling.h"
#include "mmse/types.h"

namespace mmse::lte {

inline constexpr std::uint32_t llr_count_per_layer(std::uint32_t n_re_per_layer,
                                                   std::uint8_t mod_order) noexcept {
    return n_re_per_layer * static_cast<std::uint32_t>(mod_order);
}

inline constexpr std::uint32_t total_llr_count(std::uint32_t n_re_per_layer, std::uint8_t n_layers,
                                               std::uint8_t mod_order) noexcept {
    return static_cast<std::uint32_t>(n_layers) * llr_count_per_layer(n_re_per_layer, mod_order);
}

inline constexpr bool supported_mod_order(std::uint8_t mod_order) noexcept {
    return mod_order == 2U || mod_order == 4U || mod_order == 6U || mod_order == 8U;
}

namespace detail {

inline constexpr float kInvSqrt2 = 0.70710678118654752440F;
inline constexpr float kInvSqrt10 = 0.31622776601683793319F;
inline constexpr float kInvSqrt42 = 0.15430334996209191000F;

inline float qam_symbol_norm(std::uint8_t mod_order) noexcept {
    switch (mod_order) {
    case 2U:
        return std::sqrt(2.0F);
    case 4U:
        return std::sqrt(10.0F);
    case 6U:
        return std::sqrt(42.0F);
    case 8U:
        return std::sqrt(170.0F);
    default:
        return 1.0F;
    }
}

inline std::uint32_t gray_to_binary(std::uint32_t gray) noexcept {
    std::uint32_t binary = gray;
    while (gray > 0U) {
        gray >>= 1U;
        binary ^= gray;
    }
    return binary;
}

inline float axis_level_from_gray(std::uint32_t gray, std::uint8_t axis_bits,
                                  float symbol_norm) noexcept {
    const std::uint32_t binary = gray_to_binary(gray);
    const std::uint32_t max_level = (1U << axis_bits) - 1U;
    const std::int32_t signed_level =
        static_cast<std::int32_t>(max_level) - 2 * static_cast<std::int32_t>(binary);
    return static_cast<float>(signed_level) / symbol_norm;
}

inline std::uint8_t gray_bit(std::uint32_t gray, std::uint8_t axis_bits,
                             std::uint8_t axis_bit_index) noexcept {
    const std::uint8_t shift = static_cast<std::uint8_t>(axis_bits - 1U - axis_bit_index);
    return static_cast<std::uint8_t>((gray >> shift) & 1U);
}

inline float sq_dist(float x, float level, float gamma) noexcept {
    const float delta = x - level;
    return gamma * delta * delta;
}

inline float min2(float a, float b) noexcept {
    return a < b ? a : b;
}

inline float min4(float a, float b, float c, float d) noexcept {
    return min2(min2(a, b), min2(c, d));
}

inline void build_axis_llrs_16qam(float x, float gamma, float& bit0, float& bit1) noexcept {
    constexpr float l1 = 1.0F * kInvSqrt10;
    constexpr float l3 = 3.0F * kInvSqrt10;
    const float ax = std::fabs(x);
    const float d1 = sq_dist(ax, l1, gamma);
    const float d3 = sq_dist(ax, l3, gamma);
    const float opp = gamma * (ax + l1) * (ax + l1);
    const float same = min2(d1, d3);
    bit0 = std::signbit(x) ? (opp - same) : (same - opp);
    bit1 = d3 - d1;
}

inline void build_axis_llrs_64qam(float x, float gamma, float& bit0, float& bit1,
                                  float& bit2) noexcept {
    constexpr float l1 = 1.0F * kInvSqrt42;
    constexpr float l3 = 3.0F * kInvSqrt42;
    constexpr float l5 = 5.0F * kInvSqrt42;
    constexpr float l7 = 7.0F * kInvSqrt42;
    const float ax = std::fabs(x);
    const float d1 = sq_dist(ax, l1, gamma);
    const float d3 = sq_dist(ax, l3, gamma);
    const float d5 = sq_dist(ax, l5, gamma);
    const float d7 = sq_dist(ax, l7, gamma);
    const float opp = gamma * (ax + l1) * (ax + l1);
    const float same = min4(d1, d3, d5, d7);
    bit0 = std::signbit(x) ? (opp - same) : (same - opp);
    bit1 = min2(d5, d7) - min2(d1, d3);
    bit2 = min2(d1, d7) - min2(d3, d5);
}

inline void assign_descrambled_llr(float value, float* dst, std::size_t idx, std::uint32_t& x1,
                                   std::uint32_t& x2) noexcept {
    dst[idx] = xor_sign_mask(value, (x1 ^ x2) & 1U);
    gold_step(x1, x2);
}

inline void build_max_log_llrs_qpsk(const float* x_hat_re, const float* x_hat_im, const float* sinr,
                                    std::uint32_t capacity_re_per_layer,
                                    std::uint32_t n_re_per_layer, std::uint8_t n_layers,
                                    std::vector<float>& llrs) {
    const std::uint32_t per_layer_llr_count = llr_count_per_layer(n_re_per_layer, 2U);
    if (llrs.size() != static_cast<std::size_t>(n_layers) * per_layer_llr_count) {
        llrs.resize(static_cast<std::size_t>(n_layers) * per_layer_llr_count);
    }

    constexpr float kScale = -2.0F / kInvSqrt2;
    for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
        const std::size_t layer_re_base = static_cast<std::size_t>(layer) * capacity_re_per_layer;
        const std::size_t layer_llr_base = static_cast<std::size_t>(layer) * per_layer_llr_count;
        for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
            const std::size_t re_index = layer_re_base + re;
            const float gamma = std::max(sinr[re_index], 0.0F);
            const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 2U;
            llrs[llr_base + 0U] = kScale * gamma * x_hat_re[re_index];
            llrs[llr_base + 1U] = kScale * gamma * x_hat_im[re_index];
        }
    }
}

inline void build_max_log_llrs_16qam(const float* x_hat_re, const float* x_hat_im,
                                     const float* sinr, std::uint32_t capacity_re_per_layer,
                                     std::uint32_t n_re_per_layer, std::uint8_t n_layers,
                                     std::vector<float>& llrs) {
    const std::uint32_t per_layer_llr_count = llr_count_per_layer(n_re_per_layer, 4U);
    if (llrs.size() != static_cast<std::size_t>(n_layers) * per_layer_llr_count) {
        llrs.resize(static_cast<std::size_t>(n_layers) * per_layer_llr_count);
    }

    for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
        const std::size_t layer_re_base = static_cast<std::size_t>(layer) * capacity_re_per_layer;
        const std::size_t layer_llr_base = static_cast<std::size_t>(layer) * per_layer_llr_count;
        for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
            const std::size_t re_index = layer_re_base + re;
            const float x_re = x_hat_re[re_index];
            const float x_im = x_hat_im[re_index];
            const float gamma = std::max(sinr[re_index], 0.0F);
            float li0 = 0.0F;
            float li1 = 0.0F;
            float lq0 = 0.0F;
            float lq1 = 0.0F;
            build_axis_llrs_16qam(x_re, gamma, li0, li1);
            build_axis_llrs_16qam(x_im, gamma, lq0, lq1);

            const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 4U;
            llrs[llr_base + 0U] = li0;
            llrs[llr_base + 1U] = lq0;
            llrs[llr_base + 2U] = li1;
            llrs[llr_base + 3U] = lq1;
        }
    }
}

inline void build_max_log_llrs_64qam(const float* x_hat_re, const float* x_hat_im,
                                     const float* sinr, std::uint32_t capacity_re_per_layer,
                                     std::uint32_t n_re_per_layer, std::uint8_t n_layers,
                                     std::vector<float>& llrs) {
    const std::uint32_t per_layer_llr_count = llr_count_per_layer(n_re_per_layer, 6U);
    if (llrs.size() != static_cast<std::size_t>(n_layers) * per_layer_llr_count) {
        llrs.resize(static_cast<std::size_t>(n_layers) * per_layer_llr_count);
    }

    for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
        const std::size_t layer_re_base = static_cast<std::size_t>(layer) * capacity_re_per_layer;
        const std::size_t layer_llr_base = static_cast<std::size_t>(layer) * per_layer_llr_count;
        for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
            const std::size_t re_index = layer_re_base + re;
            const float x_re = x_hat_re[re_index];
            const float x_im = x_hat_im[re_index];
            const float gamma = std::max(sinr[re_index], 0.0F);
            float li0 = 0.0F;
            float li1 = 0.0F;
            float li2 = 0.0F;
            float lq0 = 0.0F;
            float lq1 = 0.0F;
            float lq2 = 0.0F;
            build_axis_llrs_64qam(x_re, gamma, li0, li1, li2);
            build_axis_llrs_64qam(x_im, gamma, lq0, lq1, lq2);

            const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 6U;
            llrs[llr_base + 0U] = li0;
            llrs[llr_base + 1U] = lq0;
            llrs[llr_base + 2U] = li1;
            llrs[llr_base + 3U] = lq1;
            llrs[llr_base + 4U] = li2;
            llrs[llr_base + 5U] = lq2;
        }
    }
}

} // namespace detail

inline MmseStatus build_max_log_llrs(const float* x_hat_re, const float* x_hat_im,
                                     const float* sinr, std::uint32_t capacity_re_per_layer,
                                     std::uint32_t n_re_per_layer, std::uint8_t n_layers,
                                     std::uint8_t mod_order, std::vector<float>& llrs);

inline MmseStatus
build_max_log_descrambled_llrs(const float* x_hat_re, const float* x_hat_im, const float* sinr,
                               std::uint32_t capacity_re_per_layer, std::uint32_t n_re_per_layer,
                               std::uint8_t n_layers, std::uint8_t mod_order, std::uint32_t c_init,
                               float* llrs, std::uint32_t llr_capacity) {
    if (!supported_mod_order(mod_order) || n_layers == 0U ||
        capacity_re_per_layer < n_re_per_layer) {
        return MmseStatus::kInvalidArgument;
    }
    if (n_re_per_layer != 0U && (x_hat_re == nullptr || x_hat_im == nullptr || sinr == nullptr)) {
        return MmseStatus::kInvalidArgument;
    }

    const std::uint32_t required_llr_count = total_llr_count(n_re_per_layer, n_layers, mod_order);
    if (required_llr_count != 0U && llrs == nullptr) {
        return MmseStatus::kInvalidArgument;
    }
    if (llr_capacity < required_llr_count) {
        return MmseStatus::kBufferTooSmall;
    }

    std::uint32_t x1 = 0U;
    std::uint32_t x2 = 0U;
    detail::init_gold(c_init, x1, x2);

    switch (mod_order) {
    case 2U: {
        constexpr float kScale = -2.0F / detail::kInvSqrt2;
        for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
            const std::size_t layer_re_base =
                static_cast<std::size_t>(layer) * capacity_re_per_layer;
            const std::size_t layer_llr_base =
                static_cast<std::size_t>(layer) * llr_count_per_layer(n_re_per_layer, 2U);
            for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
                const std::size_t re_index = layer_re_base + re;
                const float gamma = std::max(sinr[re_index], 0.0F);
                const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 2U;
                detail::assign_descrambled_llr(kScale * gamma * x_hat_re[re_index], llrs,
                                               llr_base + 0U, x1, x2);
                detail::assign_descrambled_llr(kScale * gamma * x_hat_im[re_index], llrs,
                                               llr_base + 1U, x1, x2);
            }
        }
        return MmseStatus::kOk;
    }
    case 4U: {
        for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
            const std::size_t layer_re_base =
                static_cast<std::size_t>(layer) * capacity_re_per_layer;
            const std::size_t layer_llr_base =
                static_cast<std::size_t>(layer) * llr_count_per_layer(n_re_per_layer, 4U);
            for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
                const std::size_t re_index = layer_re_base + re;
                const float x_re = x_hat_re[re_index];
                const float x_im = x_hat_im[re_index];
                const float gamma = std::max(sinr[re_index], 0.0F);
                float li0 = 0.0F;
                float li1 = 0.0F;
                float lq0 = 0.0F;
                float lq1 = 0.0F;
                detail::build_axis_llrs_16qam(x_re, gamma, li0, li1);
                detail::build_axis_llrs_16qam(x_im, gamma, lq0, lq1);
                const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 4U;
                detail::assign_descrambled_llr(li0, llrs, llr_base + 0U, x1, x2);
                detail::assign_descrambled_llr(lq0, llrs, llr_base + 1U, x1, x2);
                detail::assign_descrambled_llr(li1, llrs, llr_base + 2U, x1, x2);
                detail::assign_descrambled_llr(lq1, llrs, llr_base + 3U, x1, x2);
            }
        }
        return MmseStatus::kOk;
    }
    case 6U: {
        for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
            const std::size_t layer_re_base =
                static_cast<std::size_t>(layer) * capacity_re_per_layer;
            const std::size_t layer_llr_base =
                static_cast<std::size_t>(layer) * llr_count_per_layer(n_re_per_layer, 6U);
            for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
                const std::size_t re_index = layer_re_base + re;
                const float x_re = x_hat_re[re_index];
                const float x_im = x_hat_im[re_index];
                const float gamma = std::max(sinr[re_index], 0.0F);
                float li0 = 0.0F;
                float li1 = 0.0F;
                float li2 = 0.0F;
                float lq0 = 0.0F;
                float lq1 = 0.0F;
                float lq2 = 0.0F;
                detail::build_axis_llrs_64qam(x_re, gamma, li0, li1, li2);
                detail::build_axis_llrs_64qam(x_im, gamma, lq0, lq1, lq2);
                const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 6U;
                detail::assign_descrambled_llr(li0, llrs, llr_base + 0U, x1, x2);
                detail::assign_descrambled_llr(lq0, llrs, llr_base + 1U, x1, x2);
                detail::assign_descrambled_llr(li1, llrs, llr_base + 2U, x1, x2);
                detail::assign_descrambled_llr(lq1, llrs, llr_base + 3U, x1, x2);
                detail::assign_descrambled_llr(li2, llrs, llr_base + 4U, x1, x2);
                detail::assign_descrambled_llr(lq2, llrs, llr_base + 5U, x1, x2);
            }
        }
        return MmseStatus::kOk;
    }
    default:
        break;
    }

    std::vector<float> temp_llrs{};
    if (build_max_log_llrs(x_hat_re, x_hat_im, sinr, capacity_re_per_layer, n_re_per_layer,
                           n_layers, mod_order, temp_llrs) != MmseStatus::kOk) {
        return MmseStatus::kInvalidArgument;
    }
    descramble_llrs_inplace(temp_llrs.data(), static_cast<std::uint32_t>(temp_llrs.size()), c_init);
    std::copy(temp_llrs.begin(), temp_llrs.end(), llrs);
    return MmseStatus::kOk;
}

inline MmseStatus build_max_log_descrambled_llrs(
    const float* x_hat_re, const float* x_hat_im, const float* sinr,
    std::uint32_t capacity_re_per_layer, std::uint32_t n_re_per_layer, std::uint8_t n_layers,
    std::uint8_t mod_order, const std::uint8_t* scrambling_bits, std::uint32_t scrambling_bit_count,
    float* llrs, std::uint32_t llr_capacity) {
    if (!supported_mod_order(mod_order) || n_layers == 0U ||
        capacity_re_per_layer < n_re_per_layer) {
        return MmseStatus::kInvalidArgument;
    }
    if (n_re_per_layer != 0U && (x_hat_re == nullptr || x_hat_im == nullptr || sinr == nullptr)) {
        return MmseStatus::kInvalidArgument;
    }

    const std::uint32_t required_llr_count = total_llr_count(n_re_per_layer, n_layers, mod_order);
    if (required_llr_count != 0U && (llrs == nullptr || scrambling_bits == nullptr)) {
        return MmseStatus::kInvalidArgument;
    }
    if (llr_capacity < required_llr_count || scrambling_bit_count < required_llr_count) {
        return MmseStatus::kBufferTooSmall;
    }

    auto assign_cached = [&](float value, std::size_t idx) {
        llrs[idx] = detail::xor_sign_mask(value, scrambling_bits[idx] & 1U);
    };

    switch (mod_order) {
    case 2U: {
        constexpr float kScale = -2.0F / detail::kInvSqrt2;
        for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
            const std::size_t layer_re_base =
                static_cast<std::size_t>(layer) * capacity_re_per_layer;
            const std::size_t layer_llr_base =
                static_cast<std::size_t>(layer) * llr_count_per_layer(n_re_per_layer, 2U);
            for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
                const std::size_t re_index = layer_re_base + re;
                const float gamma = std::max(sinr[re_index], 0.0F);
                const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 2U;
                assign_cached(kScale * gamma * x_hat_re[re_index], llr_base + 0U);
                assign_cached(kScale * gamma * x_hat_im[re_index], llr_base + 1U);
            }
        }
        return MmseStatus::kOk;
    }
    case 4U: {
        for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
            const std::size_t layer_re_base =
                static_cast<std::size_t>(layer) * capacity_re_per_layer;
            const std::size_t layer_llr_base =
                static_cast<std::size_t>(layer) * llr_count_per_layer(n_re_per_layer, 4U);
            for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
                const std::size_t re_index = layer_re_base + re;
                const float x_re = x_hat_re[re_index];
                const float x_im = x_hat_im[re_index];
                const float gamma = std::max(sinr[re_index], 0.0F);
                float li0 = 0.0F;
                float li1 = 0.0F;
                float lq0 = 0.0F;
                float lq1 = 0.0F;
                detail::build_axis_llrs_16qam(x_re, gamma, li0, li1);
                detail::build_axis_llrs_16qam(x_im, gamma, lq0, lq1);

                const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 4U;
                assign_cached(li0, llr_base + 0U);
                assign_cached(lq0, llr_base + 1U);
                assign_cached(li1, llr_base + 2U);
                assign_cached(lq1, llr_base + 3U);
            }
        }
        return MmseStatus::kOk;
    }
    case 6U: {
        for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
            const std::size_t layer_re_base =
                static_cast<std::size_t>(layer) * capacity_re_per_layer;
            const std::size_t layer_llr_base =
                static_cast<std::size_t>(layer) * llr_count_per_layer(n_re_per_layer, 6U);
            for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
                const std::size_t re_index = layer_re_base + re;
                const float x_re = x_hat_re[re_index];
                const float x_im = x_hat_im[re_index];
                const float gamma = std::max(sinr[re_index], 0.0F);
                float li0 = 0.0F;
                float li1 = 0.0F;
                float li2 = 0.0F;
                float lq0 = 0.0F;
                float lq1 = 0.0F;
                float lq2 = 0.0F;
                detail::build_axis_llrs_64qam(x_re, gamma, li0, li1, li2);
                detail::build_axis_llrs_64qam(x_im, gamma, lq0, lq1, lq2);

                const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * 6U;
                assign_cached(li0, llr_base + 0U);
                assign_cached(lq0, llr_base + 1U);
                assign_cached(li1, llr_base + 2U);
                assign_cached(lq1, llr_base + 3U);
                assign_cached(li2, llr_base + 4U);
                assign_cached(lq2, llr_base + 5U);
            }
        }
        return MmseStatus::kOk;
    }
    default:
        break;
    }

    std::vector<float> temp_llrs{};
    if (build_max_log_llrs(x_hat_re, x_hat_im, sinr, capacity_re_per_layer, n_re_per_layer,
                           n_layers, mod_order, temp_llrs) != MmseStatus::kOk) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::uint32_t i = 0; i < required_llr_count; ++i) {
        llrs[i] = detail::xor_sign_mask(temp_llrs[i], scrambling_bits[i] & 1U);
    }
    return MmseStatus::kOk;
}

inline MmseStatus build_max_log_descrambled_llrs(const float* x_hat_re, const float* x_hat_im,
                                                 const float* sinr,
                                                 std::uint32_t capacity_re_per_layer,
                                                 std::uint32_t n_re_per_layer,
                                                 std::uint8_t n_layers, std::uint8_t mod_order,
                                                 std::uint32_t c_init, std::vector<float>& llrs) {
    const std::uint32_t required_llr_count = total_llr_count(n_re_per_layer, n_layers, mod_order);
    if (llrs.size() != required_llr_count) {
        llrs.resize(required_llr_count);
    }
    return build_max_log_descrambled_llrs(x_hat_re, x_hat_im, sinr, capacity_re_per_layer,
                                          n_re_per_layer, n_layers, mod_order, c_init, llrs.data(),
                                          required_llr_count);
}

inline MmseStatus build_max_log_llrs(const float* x_hat_re, const float* x_hat_im,
                                     const float* sinr, std::uint32_t capacity_re_per_layer,
                                     std::uint32_t n_re_per_layer, std::uint8_t n_layers,
                                     std::uint8_t mod_order, std::vector<float>& llrs) {
    if (!supported_mod_order(mod_order) || n_layers == 0U ||
        capacity_re_per_layer < n_re_per_layer) {
        return MmseStatus::kInvalidArgument;
    }
    if (n_re_per_layer != 0U && (x_hat_re == nullptr || x_hat_im == nullptr || sinr == nullptr)) {
        return MmseStatus::kInvalidArgument;
    }

    switch (mod_order) {
    case 2U:
        detail::build_max_log_llrs_qpsk(x_hat_re, x_hat_im, sinr, capacity_re_per_layer,
                                        n_re_per_layer, n_layers, llrs);
        return MmseStatus::kOk;
    case 4U:
        detail::build_max_log_llrs_16qam(x_hat_re, x_hat_im, sinr, capacity_re_per_layer,
                                         n_re_per_layer, n_layers, llrs);
        return MmseStatus::kOk;
    case 6U:
        detail::build_max_log_llrs_64qam(x_hat_re, x_hat_im, sinr, capacity_re_per_layer,
                                         n_re_per_layer, n_layers, llrs);
        return MmseStatus::kOk;
    default:
        break;
    }

    const std::uint8_t axis_bits = static_cast<std::uint8_t>(mod_order / 2U);
    const std::uint32_t axis_cardinality = 1U << axis_bits;
    const std::uint32_t per_layer_llr_count = llr_count_per_layer(n_re_per_layer, mod_order);
    const float symbol_norm = detail::qam_symbol_norm(mod_order);

    if (llrs.size() != static_cast<std::size_t>(n_layers) * per_layer_llr_count) {
        llrs.resize(static_cast<std::size_t>(n_layers) * per_layer_llr_count);
    }
    for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
        const std::size_t layer_re_base = static_cast<std::size_t>(layer) * capacity_re_per_layer;
        const std::size_t layer_llr_base = static_cast<std::size_t>(layer) * per_layer_llr_count;
        for (std::uint32_t re = 0; re < n_re_per_layer; ++re) {
            const std::size_t re_index = layer_re_base + re;
            const float x_re = x_hat_re[re_index];
            const float x_im = x_hat_im[re_index];
            const float gamma = std::max(sinr[re_index], 0.0F);

            std::array<float, 4> min_i0{};
            std::array<float, 4> min_i1{};
            std::array<float, 4> min_q0{};
            std::array<float, 4> min_q1{};
            min_i0.fill(std::numeric_limits<float>::infinity());
            min_i1.fill(std::numeric_limits<float>::infinity());
            min_q0.fill(std::numeric_limits<float>::infinity());
            min_q1.fill(std::numeric_limits<float>::infinity());

            for (std::uint32_t gray = 0; gray < axis_cardinality; ++gray) {
                const float axis_level = detail::axis_level_from_gray(gray, axis_bits, symbol_norm);
                const float dist_i = gamma * (x_re - axis_level) * (x_re - axis_level);
                const float dist_q = gamma * (x_im - axis_level) * (x_im - axis_level);
                for (std::uint8_t axis_bit = 0; axis_bit < axis_bits; ++axis_bit) {
                    if (detail::gray_bit(gray, axis_bits, axis_bit) == 0U) {
                        min_i0[axis_bit] = std::min(min_i0[axis_bit], dist_i);
                        min_q0[axis_bit] = std::min(min_q0[axis_bit], dist_q);
                    } else {
                        min_i1[axis_bit] = std::min(min_i1[axis_bit], dist_i);
                        min_q1[axis_bit] = std::min(min_q1[axis_bit], dist_q);
                    }
                }
            }

            const std::size_t llr_base = layer_llr_base + static_cast<std::size_t>(re) * mod_order;
            for (std::uint8_t axis_bit = 0; axis_bit < axis_bits; ++axis_bit) {
                // Match the repo formula in mmse_equalizer_tdd.md: min(X^0) - min(X^1).
                llrs[llr_base + 2U * axis_bit] = min_i0[axis_bit] - min_i1[axis_bit];
                llrs[llr_base + 2U * axis_bit + 1U] = min_q0[axis_bit] - min_q1[axis_bit];
            }
        }
    }

    return MmseStatus::kOk;
}

} // namespace mmse::lte
