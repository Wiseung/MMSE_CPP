#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "mmse/pdcch_module_api.h"

namespace mmse::pdcch {
namespace {

constexpr std::uint32_t kConstraintLength = 7U;
constexpr std::uint32_t kStateCount = 1U << (kConstraintLength - 1U);
constexpr std::uint32_t kTrellisBranchCount = 2U * kStateCount;
constexpr std::uint32_t kBranchOutputClassCount = 1U << kPdcchConvolutionalCodeRate;
constexpr std::uint32_t kHighPredecessorStateBit = kStateCount / 2U;
constexpr std::uint32_t kMaxCodewordBitCount = kPdcchMaxDciPayloadBits + kPdcchCrcBitCount;
constexpr std::array<std::uint8_t, kPdcchConvolutionalCodeRate> kGeneratorPolynomials = {
    0133U,
    0171U,
    0165U,
};

constexpr std::uint8_t parity(std::uint32_t value) noexcept {
    std::uint8_t result = 0U;
    while (value != 0U) {
        result ^= static_cast<std::uint8_t>(value & 1U);
        value >>= 1U;
    }
    return result;
}

using BranchOutputSigns = std::array<std::int8_t, kPdcchConvolutionalCodeRate>;

constexpr std::array<BranchOutputSigns, kTrellisBranchCount> make_branch_output_signs() noexcept {
    std::array<BranchOutputSigns, kTrellisBranchCount> output_signs{};
    for (std::uint32_t shift_register = 0U; shift_register < kTrellisBranchCount;
         ++shift_register) {
        for (std::uint32_t stream = 0U; stream < kPdcchConvolutionalCodeRate; ++stream) {
            output_signs[shift_register][stream] =
                parity(shift_register & kGeneratorPolynomials[stream]) != 0U ? 1 : -1;
        }
    }
    return output_signs;
}

constexpr std::array<BranchOutputSigns, kTrellisBranchCount> kBranchOutputSigns =
    make_branch_output_signs();

constexpr std::uint8_t branch_output_class(const BranchOutputSigns& signs) noexcept {
    std::uint8_t result = 0U;
    for (std::uint32_t stream = 0U; stream < kPdcchConvolutionalCodeRate; ++stream) {
        result |= static_cast<std::uint8_t>(signs[stream] > 0 ? 1U << stream : 0U);
    }
    return result;
}

constexpr std::array<std::uint8_t, kTrellisBranchCount> make_branch_output_classes() noexcept {
    std::array<std::uint8_t, kTrellisBranchCount> classes{};
    for (std::uint32_t shift_register = 0U; shift_register < kTrellisBranchCount;
         ++shift_register) {
        classes[shift_register] = branch_output_class(kBranchOutputSigns[shift_register]);
    }
    return classes;
}

constexpr std::array<std::uint8_t, kTrellisBranchCount> kBranchOutputClasses =
    make_branch_output_classes();

constexpr std::array<std::uint8_t, kStateCount / 2U>
make_low_predecessor_zero_output_classes() noexcept {
    std::array<std::uint8_t, kStateCount / 2U> classes{};
    for (std::uint32_t low_predecessor = 0U; low_predecessor < kStateCount / 2U;
         ++low_predecessor) {
        classes[low_predecessor] = kBranchOutputClasses[low_predecessor << 1U];
    }
    return classes;
}

constexpr std::array<std::uint8_t, kStateCount / 2U> kLowPredecessorZeroOutputClasses =
    make_low_predecessor_zero_output_classes();

double branch_metric(const float* llrs, std::uint32_t shift_register) noexcept {
    const BranchOutputSigns& signs = kBranchOutputSigns[shift_register];
    return static_cast<double>(signs[0]) * static_cast<double>(llrs[0]) +
           static_cast<double>(signs[1]) * static_cast<double>(llrs[1]) +
           static_cast<double>(signs[2]) * static_cast<double>(llrs[2]);
}

} // namespace

MmseStatus decode_pdcch_tail_biting_convolutional(const PdcchRateRecoveredLlr& recovered,
                                                  std::vector<std::uint8_t>& decoded_bits) {
    decoded_bits.clear();
    if (recovered.codeword_bit_count == 0U ||
        recovered.codeword_bit_count > kPdcchMaxDciPayloadBits + kPdcchCrcBitCount ||
        recovered.soft_bit_polarity != PdcchSoftBitPolarity::kNegativeFavorsZero ||
        recovered.llr_order != PdcchConvolutionalLlrOrder::kLteRateRecoveredTriplets ||
        recovered.convolutional_llrs.size() !=
            static_cast<std::size_t>(recovered.codeword_bit_count) * kPdcchConvolutionalCodeRate) {
        return MmseStatus::kInvalidArgument;
    }
    const std::uint32_t bit_count = recovered.codeword_bit_count;
    const double negative_infinity = -std::numeric_limits<double>::infinity();
    double best_metric = negative_infinity;
    std::array<std::uint8_t, kMaxCodewordBitCount> best_bits;
    std::array<std::uint8_t, kMaxCodewordBitCount> candidate_bits{};
    std::array<std::array<double, kBranchOutputClassCount>, kMaxCodewordBitCount> branch_metrics;
    std::array<std::uint64_t, kMaxCodewordBitCount> survivor_high_predecessor;

    for (std::uint32_t bit = 0U; bit < bit_count; ++bit) {
        const float* const llrs =
            recovered.convolutional_llrs.data() + bit * kPdcchConvolutionalCodeRate;
        if (!std::isfinite(llrs[0]) || !std::isfinite(llrs[1]) || !std::isfinite(llrs[2])) {
            return MmseStatus::kInvalidArgument;
        }
        branch_metrics[bit] = {
            branch_metric(llrs, 0U), branch_metric(llrs, 2U), branch_metric(llrs, 7U),
            branch_metric(llrs, 5U), branch_metric(llrs, 4U), branch_metric(llrs, 6U),
            branch_metric(llrs, 3U), branch_metric(llrs, 1U),
        };
    }

    for (std::uint32_t initial_state = 0U; initial_state < kStateCount; ++initial_state) {
        std::array<double, kStateCount> metric_storage;
        std::array<double, kStateCount> next_metric_storage;
        auto* metrics = &metric_storage;
        auto* next_metrics = &next_metric_storage;
        metrics->fill(negative_infinity);
        (*metrics)[initial_state] = 0.0;

        for (std::uint32_t bit = 0U; bit < bit_count; ++bit) {
            const auto& bit_metrics = branch_metrics[bit];
            std::uint64_t high_predecessor_word = 0U;
            for (std::uint32_t target_state = 0U; target_state < kStateCount; target_state += 2U) {
                const std::uint32_t low_predecessor = target_state >> 1U;
                const std::uint32_t high_predecessor = low_predecessor | kHighPredecessorStateBit;
                const std::uint8_t low_zero_class =
                    kLowPredecessorZeroOutputClasses[low_predecessor];
                const std::uint8_t complement_class = low_zero_class ^ 0x7U;
                const double direct_branch_metric = bit_metrics[low_zero_class];
                const double complement_branch_metric = bit_metrics[complement_class];
                const double low_predecessor_metric = (*metrics)[low_predecessor];
                const double high_predecessor_metric = (*metrics)[high_predecessor];
                const double low_zero_metric = low_predecessor_metric + direct_branch_metric;
                const double high_zero_metric = high_predecessor_metric + complement_branch_metric;

                // The former source-state order retained the lower predecessor on equal metrics.
                if (high_zero_metric > low_zero_metric) {
                    (*next_metrics)[target_state] = high_zero_metric;
                    high_predecessor_word |= std::uint64_t{1U} << target_state;
                } else {
                    (*next_metrics)[target_state] = low_zero_metric;
                }

                const double low_one_metric = low_predecessor_metric + complement_branch_metric;
                const double high_one_metric = high_predecessor_metric + direct_branch_metric;
                if (high_one_metric > low_one_metric) {
                    (*next_metrics)[target_state + 1U] = high_one_metric;
                    high_predecessor_word |= std::uint64_t{1U} << (target_state + 1U);
                } else {
                    (*next_metrics)[target_state + 1U] = low_one_metric;
                }
            }
            survivor_high_predecessor[bit] = high_predecessor_word;
            std::swap(metrics, next_metrics);
        }

        if ((*metrics)[initial_state] <= best_metric) {
            continue;
        }
        std::uint32_t state = initial_state;
        for (std::uint32_t bit = bit_count; bit > 0U; --bit) {
            candidate_bits[bit - 1U] = static_cast<std::uint8_t>(state & 1U);
            const bool high_predecessor =
                (survivor_high_predecessor[bit - 1U] & (std::uint64_t{1U} << state)) != 0U;
            state = (state >> 1U) | (high_predecessor ? kHighPredecessorStateBit : 0U);
        }
        if (state != initial_state) {
            continue;
        }
        best_metric = (*metrics)[initial_state];
        best_bits = candidate_bits;
    }

    if (!std::isfinite(best_metric)) {
        return MmseStatus::kInternalError;
    }
    decoded_bits.assign(best_bits.begin(), best_bits.begin() + bit_count);
    return MmseStatus::kOk;
}

} // namespace mmse::pdcch