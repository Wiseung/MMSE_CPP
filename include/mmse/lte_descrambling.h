#pragma once

#include <bit>
#include <cstdint>

namespace mmse::lte {

inline constexpr std::uint32_t kDescramblingNc = 1600U;

constexpr std::uint32_t subframe_from_sfn_subframe(std::uint32_t sfn_subframe) noexcept {
    return sfn_subframe % 10U;
}

constexpr std::uint32_t pbch_c_init(std::uint16_t cell_id) noexcept {
    return static_cast<std::uint32_t>(cell_id);
}

constexpr std::uint32_t pdcch_c_init(std::uint16_t cell_id, std::uint32_t sfn_subframe) noexcept {
    return (subframe_from_sfn_subframe(sfn_subframe) << 9U) + static_cast<std::uint32_t>(cell_id);
}

constexpr std::uint32_t pcfich_c_init(std::uint16_t cell_id, std::uint32_t sfn_subframe) noexcept {
    const std::uint32_t subframe = subframe_from_sfn_subframe(sfn_subframe);
    return (((subframe + 1U) * (2U * static_cast<std::uint32_t>(cell_id) + 1U)) << 9U) +
           static_cast<std::uint32_t>(cell_id);
}

constexpr std::uint32_t pdsch_c_init(std::uint16_t cell_id, std::uint16_t rnti,
                                     std::uint32_t sfn_subframe,
                                     std::uint8_t codeword = 0U) noexcept {
    const std::uint32_t subframe = subframe_from_sfn_subframe(sfn_subframe);
    return (static_cast<std::uint32_t>(rnti) << 14U) +
           (static_cast<std::uint32_t>(codeword) << 13U) + (subframe << 9U) +
           static_cast<std::uint32_t>(cell_id);
}

namespace detail {

inline void gold_step(std::uint32_t& x1, std::uint32_t& x2) noexcept {
    const std::uint32_t x1_feedback = ((x1 >> 3U) ^ x1) & 1U;
    const std::uint32_t x2_feedback = ((x2 >> 3U) ^ (x2 >> 2U) ^ (x2 >> 1U) ^ x2) & 1U;
    x1 = (x1 >> 1U) | (x1_feedback << 30U);
    x2 = (x2 >> 1U) | (x2_feedback << 30U);
}

inline void warm_up_gold(std::uint32_t& x1, std::uint32_t& x2) noexcept {
    for (std::uint32_t n = 0; n < kDescramblingNc; ++n) {
        gold_step(x1, x2);
    }
}

inline void init_gold(std::uint32_t c_init, std::uint32_t& x1, std::uint32_t& x2) noexcept {
    x1 = 1U;
    x2 = c_init & 0x7FFFFFFFU;
    warm_up_gold(x1, x2);
}

inline float xor_sign_mask(float value, std::uint32_t scramble_bit) noexcept {
    const std::uint32_t sign_mask = scramble_bit << 31U;
    return std::bit_cast<float>(std::bit_cast<std::uint32_t>(value) ^ sign_mask);
}

} // namespace detail

inline void generate_scrambling_bits(std::uint32_t c_init, std::uint8_t* bits,
                                     std::uint32_t count) noexcept {
    std::uint32_t x1 = 0U;
    std::uint32_t x2 = 0U;
    detail::init_gold(c_init, x1, x2);
    std::uint32_t n = 0U;
    for (; n + 8U <= count; n += 8U) {
        bits[n + 0U] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 1U] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 2U] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 3U] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 4U] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 5U] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 6U] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 7U] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
    }
    for (; n < count; ++n) {
        bits[n] = static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
    }
}

inline void descramble_bits_inplace(std::uint8_t* bits, std::uint32_t count,
                                    std::uint32_t c_init) noexcept {
    std::uint32_t x1 = 0U;
    std::uint32_t x2 = 0U;
    detail::init_gold(c_init, x1, x2);
    std::uint32_t n = 0U;
    for (; n + 8U <= count; n += 8U) {
        bits[n + 0U] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 1U] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 2U] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 3U] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 4U] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 5U] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 6U] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        bits[n + 7U] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
    }
    for (; n < count; ++n) {
        bits[n] ^= static_cast<std::uint8_t>((x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
    }
}

inline void descramble_llrs_inplace(float* llrs, std::uint32_t count,
                                    std::uint32_t c_init) noexcept {
    std::uint32_t x1 = 0U;
    std::uint32_t x2 = 0U;
    detail::init_gold(c_init, x1, x2);
    std::uint32_t n = 0U;
    for (; n + 8U <= count; n += 8U) {
        llrs[n + 0U] = detail::xor_sign_mask(llrs[n + 0U], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        llrs[n + 1U] = detail::xor_sign_mask(llrs[n + 1U], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        llrs[n + 2U] = detail::xor_sign_mask(llrs[n + 2U], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        llrs[n + 3U] = detail::xor_sign_mask(llrs[n + 3U], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        llrs[n + 4U] = detail::xor_sign_mask(llrs[n + 4U], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        llrs[n + 5U] = detail::xor_sign_mask(llrs[n + 5U], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        llrs[n + 6U] = detail::xor_sign_mask(llrs[n + 6U], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
        llrs[n + 7U] = detail::xor_sign_mask(llrs[n + 7U], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
    }
    for (; n < count; ++n) {
        llrs[n] = detail::xor_sign_mask(llrs[n], (x1 ^ x2) & 1U);
        detail::gold_step(x1, x2);
    }
}

} // namespace mmse::lte
