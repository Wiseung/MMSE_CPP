#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "mmse/lte_descrambling.h"
#include "mmse/lte_soft_demod.h"
#include "mmse/mmse_equalizer.h"
#include "mmse/pbch_chain_dto.h"
#include "mmse/pbch_module_api.h"
#include "mmse/pdsch_chain_dto.h"
#include "mmse/pdsch_module_api.h"
#include "mmse/pdcch_chain_dto.h"
#include "mmse/pdcch_chain_sdk.h"
#include "mmse/pdcch_module_api.h"
#include "mmse/pcfich_chain_dto.h"
#include "mmse/pcfich_module_api.h"
#include "internal/mmse_cuda_runtime.h"
#include "internal/mmse_internal.h"

using mmse::EqualizerOutputView;
using mmse::ExtractDescriptor;
using mmse::MmseCpuBackend;
using mmse::MmseEqualizerCpuConfig;
using mmse::MmseEqualizerCpuContext;
using mmse::MmseEqualizerGpuConfig;
using mmse::MmseEqualizerGpuContext;
using mmse::MmseGpuBackend;
using mmse::MmseStatus;
using mmse::PlanarGridViewF32;
using namespace mmse;
using mmse::detail::Complex32;

namespace {

constexpr std::uint32_t kValidationSampleCount = 12U;

struct TestFailure {
    std::string message;
};

void expect_true(bool cond, const std::string& message) {
    if (!cond) {
        throw TestFailure{message};
    }
}

void expect_near(float lhs, float rhs, float eps, const std::string& message) {
    if (std::fabs(lhs - rhs) > eps) {
        throw TestFailure{message + " lhs=" + std::to_string(lhs) + " rhs=" + std::to_string(rhs)};
    }
}

void expect_relative_near(float lhs, float rhs, float rel_tol, float abs_tol,
                          const std::string& message) {
    const float abs_err = std::fabs(lhs - rhs);
    const float scale = std::max(std::fabs(rhs), 1.0F);
    if (abs_err > std::max(abs_tol, rel_tol * scale)) {
        throw TestFailure{message + " lhs=" + std::to_string(lhs) + " rhs=" + std::to_string(rhs)};
    }
}

std::uint32_t build_validation_samples(const ExtractDescriptor& desc,
                                       const mmse::detail::ReLayout& layout,
                                       std::array<std::uint32_t, kValidationSampleCount>& samples) {
    samples.fill(0U);
    return mmse::detail::build_validation_re_samples(
        layout, desc.start_symbol, kLteNumSymbolsNormalCp, kLteNumSubcarriers20MHz, samples.data(),
        static_cast<std::uint32_t>(samples.size()));
}

void expect_gpu_matches_cpu_samples(const ExtractDescriptor& desc,
                                    const EqualizerOutputView& gpu_out,
                                    const EqualizerOutputView& cpu_out,
                                    const std::string& label_prefix) {
    mmse::detail::ReLayout layout{};
    mmse::detail::build_data_re_layout(desc, layout);
    std::array<std::uint32_t, kValidationSampleCount> samples{};
    const std::uint32_t sample_count = build_validation_samples(desc, layout, samples);
    for (std::uint32_t layer = 0; layer < cpu_out.n_layers; ++layer) {
        for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
            const std::uint32_t re = samples[sample];
            if (re >= cpu_out.n_re_per_layer) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(layer) * cpu_out.capacity_re_per_layer + re;
            const std::string prefix =
                label_prefix + " layer=" + std::to_string(layer) + " re=" + std::to_string(re);
            expect_relative_near(gpu_out.x_hat_re[idx], cpu_out.x_hat_re[idx], 5.0e-2F, 1.0e-4F,
                                 prefix + " xhat real");
            expect_relative_near(gpu_out.x_hat_im[idx], cpu_out.x_hat_im[idx], 5.0e-2F, 1.0e-4F,
                                 prefix + " xhat imag");
            expect_relative_near(gpu_out.sinr[idx], cpu_out.sinr[idx], 7.0e-2F, 1.0e-4F,
                                 prefix + " sinr");
        }
    }
}

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

struct TwoLayerCase {
    Complex32 h00;
    Complex32 h01;
    Complex32 h10;
    Complex32 h11;
    Complex32 x0;
    Complex32 x1;
};

GridBuffers make_zero_grid(std::uint32_t n_subcarriers = kLteNumSubcarriers20MHz) {
    GridBuffers buffers;
    for (std::uint32_t rx = 0; rx < 2; ++rx) {
        buffers.re[rx].assign(kLteNumSymbolsNormalCp * n_subcarriers, 0.0F);
        buffers.im[rx].assign(kLteNumSymbolsNormalCp * n_subcarriers, 0.0F);
    }
    return buffers;
}

PlanarGridViewF32 make_grid_view(const GridBuffers& buffers) {
    PlanarGridViewF32 grid{};
    grid.re = {buffers.re[0].data(), buffers.re[1].data()};
    grid.im = {buffers.im[0].data(), buffers.im[1].data()};
    grid.n_rx_ant = 2;
    grid.n_symbols = kLteNumSymbolsNormalCp;
    grid.n_subcarriers = static_cast<std::uint32_t>(buffers.re[0].size() / kLteNumSymbolsNormalCp);
    return grid;
}

PlanarGridViewF32 make_single_rx_grid_view(const GridBuffers& buffers) {
    PlanarGridViewF32 grid{};
    grid.re = {buffers.re[0].data(), nullptr};
    grid.im = {buffers.im[0].data(), nullptr};
    grid.n_rx_ant = 1U;
    grid.n_symbols = kLteNumSymbolsNormalCp;
    grid.n_subcarriers = static_cast<std::uint32_t>(buffers.re[0].size() / kLteNumSymbolsNormalCp);
    return grid;
}

ExtractDescriptor make_fullband_desc() {
    ExtractDescriptor desc{};
    desc.cell_id = 0;
    desc.n_tx_ports = 2;
    desc.n_rx_ant = 2;
    desc.n_layers = 2;
    desc.start_symbol = 1;
    desc.control_symbol_count = 0;
    desc.mod_order = 2;
    desc.n_prb = 100;
    desc.tx_mode = 2;
    desc.pmi = -1;
    desc.sfn_subframe = 3;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    return desc;
}

ExtractDescriptor make_pdcch_desc(std::uint16_t n_prb = kLteNumPrb20MHz,
                                  std::uint8_t control_symbol_count = 3U) {
    ExtractDescriptor desc{};
    desc.cell_id = 0;
    desc.n_tx_ports = 1;
    desc.n_rx_ant = 2;
    desc.n_layers = 1;
    desc.tx_mode = 1;
    desc.channel_type = MmseChannelType::kPdcch;
    desc.start_symbol = 0;
    desc.control_symbol_count = control_symbol_count;
    desc.mod_order = 2;
    desc.n_prb = n_prb;
    desc.sfn_subframe = 3;
    desc.prb_bitmap.fill(0U);
    for (std::uint32_t prb = 0U; prb < n_prb; ++prb) {
        desc.prb_bitmap[prb / 16U] |= static_cast<std::uint16_t>(1U << (prb % 16U));
    }
    desc.control_re_exclusion_masks.fill(0U);
    return desc;
}

ExtractDescriptor make_pbch_desc() {
    ExtractDescriptor desc{};
    desc.cell_id = 0;
    desc.n_tx_ports = 1;
    desc.n_rx_ant = 2;
    desc.n_layers = 1;
    desc.tx_mode = 1;
    desc.channel_type = MmseChannelType::kPbch;
    desc.start_symbol = kLtePbchStartSymbolNormalCp;
    desc.control_symbol_count = 0;
    desc.mod_order = 2;
    desc.n_prb = kLtePbchNumPrb;
    desc.sfn_subframe = 0;
    desc.prb_bitmap.fill(0U);
    for (std::uint32_t prb = kLtePbchStartPrb; prb < kLtePbchStartPrb + kLtePbchNumPrb; ++prb) {
        desc.prb_bitmap[prb / 16U] |= static_cast<std::uint16_t>(1U << (prb % 16U));
    }
    return desc;
}

ExtractDescriptor make_pcfich_desc() {
    ExtractDescriptor desc{};
    desc.cell_id = 0;
    desc.n_tx_ports = 1;
    desc.n_rx_ant = 2;
    desc.n_layers = 1;
    desc.tx_mode = 1;
    desc.channel_type = MmseChannelType::kPcfich;
    desc.start_symbol = 0;
    desc.control_symbol_count = 1;
    desc.mod_order = 2;
    desc.n_prb = 100;
    desc.sfn_subframe = 0;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    return desc;
}

TwoLayerCase make_two_layer_case() {
    return {
        .h00 = {0.90F, 0.10F},
        .h01 = {0.25F, -0.30F},
        .h10 = {0.10F, 0.35F},
        .h11 = {1.05F, -0.20F},
        .x0 = {0.75F, -0.25F},
        .x1 = {-0.35F, 0.60F},
    };
}

std::vector<std::uint8_t> reference_gold_sequence(std::uint32_t c_init, std::uint32_t count) {
    std::array<std::uint8_t, 31> x1{};
    std::array<std::uint8_t, 31> x2{};
    x1[0] = 1U;
    for (std::uint32_t i = 0; i < x2.size(); ++i) {
        x2[i] = static_cast<std::uint8_t>((c_init >> i) & 1U);
    }

    std::vector<std::uint8_t> seq(count, 0U);
    for (std::uint32_t n = 0; n < mmse::lte::kDescramblingNc + count; ++n) {
        if (n >= mmse::lte::kDescramblingNc) {
            seq[n - mmse::lte::kDescramblingNc] = static_cast<std::uint8_t>(x1[0] ^ x2[0]);
        }

        const std::uint8_t x1_feedback = static_cast<std::uint8_t>(x1[3] ^ x1[0]);
        const std::uint8_t x2_feedback = static_cast<std::uint8_t>(x2[3] ^ x2[2] ^ x2[1] ^ x2[0]);
        for (std::uint32_t i = 0; i + 1U < x1.size(); ++i) {
            x1[i] = x1[i + 1U];
            x2[i] = x2[i + 1U];
        }
        x1.back() = x1_feedback;
        x2.back() = x2_feedback;
    }

    return seq;
}

Complex32 reference_crs_value(std::uint16_t cell_id, std::uint8_t subframe, std::uint8_t port,
                              std::uint8_t symbol, std::uint32_t pilot_index) {
    const std::uint32_t slot = 2U * subframe + (symbol >= 7U ? 1U : 0U);
    const std::uint32_t slot_symbol = symbol % 7U;
    const std::uint32_t c_init =
        (1U << 10U) * (7U * (slot + 1U) + slot_symbol + 1U) * (2U * cell_id + 1U) + 2U * cell_id +
        1U;
    const std::vector<std::uint8_t> bits = reference_gold_sequence(c_init, 2U * (pilot_index + 1U));
    constexpr float scale = 0.7071067811865475F;
    return {scale * (bits[2U * pilot_index] == 0U ? 1.0F : -1.0F),
            scale * (bits[2U * pilot_index + 1U] == 0U ? 1.0F : -1.0F)};
}

void expect_bits_match_binary_string(const std::vector<std::uint8_t>& bits, const char* expected,
                                     const std::string& label) {
    const std::size_t expected_size = std::char_traits<char>::length(expected);
    expect_true(bits.size() == expected_size, label + " size");
    for (std::size_t i = 0; i < expected_size; ++i) {
        const std::uint8_t expected_bit = expected[i] == '1' ? 1U : 0U;
        if (bits[i] != expected_bit) {
            throw TestFailure{label + " mismatch at bit " + std::to_string(i)};
        }
    }
}

std::vector<float> reference_build_max_log_llrs(const float* x_hat_re, const float* x_hat_im,
                                                const float* sinr,
                                                std::uint32_t capacity_re_per_layer,
                                                std::uint32_t n_re_per_layer, std::uint8_t n_layers,
                                                std::uint8_t mod_order) {
    const std::uint8_t axis_bits = static_cast<std::uint8_t>(mod_order / 2U);
    const std::uint32_t axis_cardinality = 1U << axis_bits;
    const std::uint32_t per_layer_llr_count = n_re_per_layer * mod_order;
    const float symbol_norm = [&] {
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
    }();

    auto gray_to_binary = [](std::uint32_t gray) {
        std::uint32_t binary = gray;
        while (gray > 0U) {
            gray >>= 1U;
            binary ^= gray;
        }
        return binary;
    };

    std::vector<float> llrs(static_cast<std::size_t>(n_layers) * per_layer_llr_count, 0.0F);
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
                const std::uint32_t binary = gray_to_binary(gray);
                const std::uint32_t max_level = axis_cardinality - 1U;
                const std::int32_t signed_level =
                    static_cast<std::int32_t>(max_level) - 2 * static_cast<std::int32_t>(binary);
                const float axis_level = static_cast<float>(signed_level) / symbol_norm;
                const float dist_i = gamma * (x_re - axis_level) * (x_re - axis_level);
                const float dist_q = gamma * (x_im - axis_level) * (x_im - axis_level);
                for (std::uint8_t axis_bit = 0; axis_bit < axis_bits; ++axis_bit) {
                    const std::uint8_t shift = static_cast<std::uint8_t>(axis_bits - 1U - axis_bit);
                    if (((gray >> shift) & 1U) == 0U) {
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
                llrs[llr_base + 2U * axis_bit] = min_i0[axis_bit] - min_i1[axis_bit];
                llrs[llr_base + 2U * axis_bit + 1U] = min_q0[axis_bit] - min_q1[axis_bit];
            }
        }
    }
    return llrs;
}

void expect_sign(float value, int expected_sign, const std::string& message) {
    if (expected_sign < 0) {
        expect_true(value < 0.0F, message);
        return;
    }
    if (expected_sign > 0) {
        expect_true(value > 0.0F, message);
        return;
    }
    expect_near(value, 0.0F, 1.0e-6F, message);
}

std::vector<float> reference_pdcch_convolutional_rate_match(const std::vector<float>& input,
                                                            std::uint32_t encoded_bit_count) {
    constexpr std::uint32_t kColumns = 32U;
    constexpr std::array<std::uint8_t, kColumns> kPermutation = {
        1U, 17U, 9U, 25U, 5U, 21U, 13U, 29U, 3U, 19U, 11U, 27U, 7U, 23U, 15U, 31U,
        0U, 16U, 8U, 24U, 4U, 20U, 12U, 28U, 2U, 18U, 10U, 26U, 6U, 22U, 14U, 30U,
    };
    expect_true(input.size() % mmse::pdcch::kPdcchConvolutionalCodeRate == 0U,
                "reference convolutional input must contain three streams");

    const std::uint32_t codeword_bit_count =
        static_cast<std::uint32_t>(input.size() / mmse::pdcch::kPdcchConvolutionalCodeRate);
    const std::uint32_t row_count = (codeword_bit_count + kColumns - 1U) / kColumns;
    const std::uint32_t interleaver_size = row_count * kColumns;
    const std::uint32_t dummy_bit_count = interleaver_size - codeword_bit_count;
    std::vector<float> collected(mmse::pdcch::kPdcchConvolutionalCodeRate * interleaver_size, 0.0F);
    std::vector<std::uint8_t> present(collected.size(), 0U);

    std::uint32_t collection_index = 0U;
    for (std::uint32_t stream = 0U; stream < mmse::pdcch::kPdcchConvolutionalCodeRate; ++stream) {
        for (std::uint32_t column = 0U; column < kColumns; ++column) {
            for (std::uint32_t row = 0U; row < row_count; ++row) {
                const std::uint32_t padded_bit = row * kColumns + kPermutation[column];
                if (padded_bit >= dummy_bit_count) {
                    collected[collection_index] =
                        input[(padded_bit - dummy_bit_count) *
                                  mmse::pdcch::kPdcchConvolutionalCodeRate +
                              stream];
                    present[collection_index] = 1U;
                }
                ++collection_index;
            }
        }
    }

    std::vector<float> output;
    output.reserve(encoded_bit_count);
    collection_index = 0U;
    while (output.size() < encoded_bit_count) {
        if (present[collection_index] != 0U) {
            output.push_back(collected[collection_index]);
        }
        collection_index = (collection_index + 1U) % collected.size();
    }
    return output;
}

std::uint8_t reference_parity(std::uint32_t value) {
    return static_cast<std::uint8_t>(std::popcount(value) & 1U);
}

std::vector<std::uint8_t>
reference_pdcch_tail_biting_convolutional_encode(const std::vector<std::uint8_t>& bits) {
    constexpr std::array<std::uint8_t, 3> kGenerators = {0133U, 0171U, 0165U};
    expect_true(bits.size() >= 6U, "reference tail-biting encoder input length");
    std::uint32_t state = 0U;
    for (std::uint32_t delay = 0U; delay < 6U; ++delay) {
        expect_true(bits[bits.size() - 1U - delay] <= 1U,
                    "reference tail-biting encoder binary input");
        state |= static_cast<std::uint32_t>(bits[bits.size() - 1U - delay]) << delay;
    }

    std::vector<std::uint8_t> encoded{};
    encoded.reserve(bits.size() * kGenerators.size());
    for (const std::uint8_t bit : bits) {
        expect_true(bit <= 1U, "reference tail-biting encoder binary bit");
        const std::uint32_t shift_register = (state << 1U) | bit;
        for (const std::uint8_t generator : kGenerators) {
            encoded.push_back(reference_parity(shift_register & generator));
        }
        state = shift_register & 0x3FU;
    }
    return encoded;
}

std::vector<float> reference_llrs_from_encoded_bits(const std::vector<std::uint8_t>& encoded,
                                                    float magnitude) {
    std::vector<float> llrs{};
    llrs.reserve(encoded.size());
    for (const std::uint8_t bit : encoded) {
        expect_true(bit <= 1U, "reference encoded bit must be binary");
        llrs.push_back(bit != 0U ? magnitude : -magnitude);
    }
    return llrs;
}

void append_bits(std::vector<std::uint8_t>& bits, std::uint32_t value, std::uint8_t width) {
    for (std::uint8_t bit = 0U; bit < width; ++bit) {
        const std::uint8_t shift = static_cast<std::uint8_t>(width - 1U - bit);
        bits.push_back(static_cast<std::uint8_t>((value >> shift) & 1U));
    }
}

std::vector<std::uint8_t> append_pdcch_crc_rnti_mask(const std::vector<std::uint8_t>& payload,
                                                     std::uint16_t rnti) {
    std::vector<std::uint8_t> result = payload;
    const std::uint16_t masked_crc =
        static_cast<std::uint16_t>(mmse::pdcch::calculate_pdcch_crc16(
                                       payload.data(), static_cast<std::uint32_t>(payload.size())) ^
                                   rnti);
    append_bits(result, masked_crc, mmse::pdcch::kPdcchCrcBitCount);
    return result;
}

std::uint16_t reference_type2_riv(std::uint16_t n_prb, std::uint16_t start_prb,
                                  std::uint16_t allocated_prb_count) {
    expect_true(n_prb != 0U && allocated_prb_count != 0U &&
                    start_prb + allocated_prb_count <= n_prb,
                "reference type2 RIV arguments");
    if (allocated_prb_count - 1U <= n_prb / 2U) {
        return static_cast<std::uint16_t>(n_prb * (allocated_prb_count - 1U) + start_prb);
    }
    return static_cast<std::uint16_t>(n_prb * (n_prb - allocated_prb_count + 1U) + n_prb - 1U -
                                      start_prb);
}

struct TailBitingDecoderProbe {
    std::vector<float> expected_llrs{};
    std::vector<std::uint8_t> decoded_bits{};
    MmseStatus return_status = MmseStatus::kOk;
    std::uint32_t call_count = 0U;
};

MmseStatus
probe_tail_biting_decoder(void* context,
                          const mmse::pdcch::PdcchTailBitingConvolutionalDecodeRequest& request) {
    auto* probe = static_cast<TailBitingDecoderProbe*>(context);
    if (probe == nullptr || request.convolutional_llrs == nullptr ||
        request.decoded_bits == nullptr ||
        request.convolutional_llr_count != probe->expected_llrs.size() ||
        request.decoded_bit_count * mmse::pdcch::kPdcchConvolutionalCodeRate !=
            request.convolutional_llr_count ||
        request.soft_bit_polarity != mmse::pdcch::PdcchSoftBitPolarity::kNegativeFavorsZero ||
        request.llr_order != mmse::pdcch::PdcchConvolutionalLlrOrder::kLteRateRecoveredTriplets ||
        !request.tail_biting) {
        return MmseStatus::kInternalError;
    }
    for (std::uint32_t i = 0U; i < request.convolutional_llr_count; ++i) {
        if (std::fabs(request.convolutional_llrs[i] - probe->expected_llrs[i]) > 1.0e-6F) {
            return MmseStatus::kInternalError;
        }
    }

    ++probe->call_count;
    if (probe->return_status != MmseStatus::kOk) {
        return probe->return_status;
    }
    if (!probe->decoded_bits.empty()) {
        if (probe->decoded_bits.size() != request.decoded_bit_count) {
            return MmseStatus::kInternalError;
        }
        std::copy(probe->decoded_bits.begin(), probe->decoded_bits.end(), request.decoded_bits);
        return MmseStatus::kOk;
    }
    for (std::uint32_t bit = 0U; bit < request.decoded_bit_count; ++bit) {
        request.decoded_bits[bit] = static_cast<std::uint8_t>(bit & 1U);
    }
    return MmseStatus::kOk;
}

struct FixedTailBitingDecoder {
    std::vector<std::uint8_t> decoded_bits{};
    std::uint32_t call_count = 0U;
};

MmseStatus
fixed_tail_biting_decoder(void* context,
                          const mmse::pdcch::PdcchTailBitingConvolutionalDecodeRequest& request) {
    auto* decoder = static_cast<FixedTailBitingDecoder*>(context);
    if (decoder == nullptr || request.convolutional_llrs == nullptr ||
        request.decoded_bits == nullptr || !request.tail_biting ||
        request.decoded_bit_count != decoder->decoded_bits.size() ||
        request.convolutional_llr_count !=
            request.decoded_bit_count * mmse::pdcch::kPdcchConvolutionalCodeRate) {
        return MmseStatus::kInternalError;
    }
    std::copy(decoder->decoded_bits.begin(), decoder->decoded_bits.end(), request.decoded_bits);
    ++decoder->call_count;
    return MmseStatus::kOk;
}

struct SelectiveTailBitingDecoder {
    std::vector<float> expected_llrs{};
    std::vector<std::uint8_t> matched_bits{};
    std::vector<std::uint8_t> miss_bits{};
    std::uint32_t call_count = 0U;
};

MmseStatus selective_tail_biting_decoder(
    void* context, const mmse::pdcch::PdcchTailBitingConvolutionalDecodeRequest& request) {
    auto* decoder = static_cast<SelectiveTailBitingDecoder*>(context);
    if (decoder == nullptr || request.convolutional_llrs == nullptr ||
        request.decoded_bits == nullptr || !request.tail_biting ||
        request.convolutional_llr_count != decoder->expected_llrs.size() ||
        request.decoded_bit_count != decoder->matched_bits.size() ||
        decoder->miss_bits.size() != decoder->matched_bits.size()) {
        return MmseStatus::kInternalError;
    }
    bool matches = true;
    for (std::uint32_t llr = 0U; llr < request.convolutional_llr_count; ++llr) {
        if (std::fabs(request.convolutional_llrs[llr] - decoder->expected_llrs[llr]) > 1.0e-4F) {
            matches = false;
            break;
        }
    }
    const std::vector<std::uint8_t>& output = matches ? decoder->matched_bits : decoder->miss_bits;
    std::copy(output.begin(), output.end(), request.decoded_bits);
    ++decoder->call_count;
    return MmseStatus::kOk;
}

void fill_identity_channel(GridBuffers& buffers, const ExtractDescriptor& desc, float data0,
                           float data1) {
    mmse::detail::ensure_crs_tables();
    const std::uint32_t subframe = mmse::detail::subframe_from_descriptor(desc);
    const std::uint32_t n_subcarriers =
        static_cast<std::uint32_t>(buffers.re[0].size() / kLteNumSymbolsNormalCp);
    for (std::uint32_t symbol = 0; symbol < kLteNumSymbolsNormalCp; ++symbol) {
        for (std::uint32_t sc = 0; sc < n_subcarriers; ++sc) {
            const std::size_t idx = static_cast<std::size_t>(symbol) * n_subcarriers + sc;
            const bool is_data_symbol = symbol >= desc.start_symbol;
            const bool is_port0_crs =
                mmse::detail::is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc) &&
                sc % 6U == mmse::detail::crs_frequency_offset(desc.cell_id, 0U,
                                                              static_cast<std::uint8_t>(symbol));
            const bool is_port1_crs =
                desc.n_tx_ports > 1U &&
                mmse::detail::is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc) &&
                sc % 6U == mmse::detail::crs_frequency_offset(desc.cell_id, 1U,
                                                              static_cast<std::uint8_t>(symbol));
            if (is_port0_crs) {
                const std::uint32_t pilot =
                    (sc - mmse::detail::crs_frequency_offset(desc.cell_id, 0U,
                                                             static_cast<std::uint8_t>(symbol))) /
                    6U;
                const auto& crs = mmse::detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(subframe),
                     .port = 0U,
                     .crs_symbol_index = static_cast<std::uint8_t>(
                         std::find(mmse::detail::kCrsSymbols.begin(),
                                   mmse::detail::kCrsSymbols.end(), symbol) -
                         mmse::detail::kCrsSymbols.begin())},
                    pilot);
                buffers.re[0][idx] = crs.re;
                buffers.im[0][idx] = crs.im;
            } else if (is_port1_crs) {
                const std::uint32_t pilot =
                    (sc - mmse::detail::crs_frequency_offset(desc.cell_id, 1U,
                                                             static_cast<std::uint8_t>(symbol))) /
                    6U;
                const auto& crs = mmse::detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(subframe),
                     .port = 1U,
                     .crs_symbol_index = static_cast<std::uint8_t>(
                         std::find(mmse::detail::kCrsSymbols.begin(),
                                   mmse::detail::kCrsSymbols.end(), symbol) -
                         mmse::detail::kCrsSymbols.begin())},
                    pilot);
                buffers.re[1][idx] = crs.re;
                buffers.im[1][idx] = crs.im;
            } else if (is_data_symbol) {
                buffers.re[0][idx] = data0;
                buffers.im[0][idx] = 0.0F;
                buffers.re[1][idx] = data1;
                buffers.im[1][idx] = 0.0F;
            }
        }
    }
}

void fill_constant_mimo_channel(GridBuffers& buffers, const ExtractDescriptor& desc, Complex32 h00,
                                Complex32 h01, Complex32 h10, Complex32 h11, Complex32 x0,
                                Complex32 x1) {
    mmse::detail::ensure_crs_tables();
    const std::uint32_t subframe = mmse::detail::subframe_from_descriptor(desc);

    for (std::uint32_t symbol = 0; symbol < kLteNumSymbolsNormalCp; ++symbol) {
        for (std::uint32_t sc = 0; sc < kLteNumSubcarriers20MHz; ++sc) {
            const std::size_t idx = static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz + sc;
            const bool is_data_symbol = symbol >= desc.start_symbol;
            const bool is_port0_crs =
                mmse::detail::is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc) &&
                sc % 6U == mmse::detail::crs_frequency_offset(desc.cell_id, 0U,
                                                              static_cast<std::uint8_t>(symbol));
            const bool is_port1_crs =
                mmse::detail::is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc) &&
                sc % 6U == mmse::detail::crs_frequency_offset(desc.cell_id, 1U,
                                                              static_cast<std::uint8_t>(symbol));

            Complex32 y_rx0{};
            Complex32 y_rx1{};
            if (is_port0_crs) {
                const std::uint32_t pilot =
                    (sc - mmse::detail::crs_frequency_offset(desc.cell_id, 0U,
                                                             static_cast<std::uint8_t>(symbol))) /
                    6U;
                const auto& crs = mmse::detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(subframe),
                     .port = 0U,
                     .crs_symbol_index = static_cast<std::uint8_t>(
                         std::find(mmse::detail::kCrsSymbols.begin(),
                                   mmse::detail::kCrsSymbols.end(), symbol) -
                         mmse::detail::kCrsSymbols.begin())},
                    pilot);
                y_rx0 = mmse::detail::cmul(h00, crs);
                y_rx1 = mmse::detail::cmul(h10, crs);
            } else if (is_port1_crs) {
                const std::uint32_t pilot =
                    (sc - mmse::detail::crs_frequency_offset(desc.cell_id, 1U,
                                                             static_cast<std::uint8_t>(symbol))) /
                    6U;
                const auto& crs = mmse::detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(subframe),
                     .port = 1U,
                     .crs_symbol_index = static_cast<std::uint8_t>(
                         std::find(mmse::detail::kCrsSymbols.begin(),
                                   mmse::detail::kCrsSymbols.end(), symbol) -
                         mmse::detail::kCrsSymbols.begin())},
                    pilot);
                y_rx0 = mmse::detail::cmul(h01, crs);
                y_rx1 = mmse::detail::cmul(h11, crs);
            } else if (is_data_symbol) {
                y_rx0 =
                    mmse::detail::cadd(mmse::detail::cmul(h00, x0), mmse::detail::cmul(h01, x1));
                y_rx1 =
                    mmse::detail::cadd(mmse::detail::cmul(h10, x0), mmse::detail::cmul(h11, x1));
            }

            buffers.re[0][idx] = y_rx0.re;
            buffers.im[0][idx] = y_rx0.im;
            buffers.re[1][idx] = y_rx1.re;
            buffers.im[1][idx] = y_rx1.im;
        }
    }
}

void fill_td_pair(GridBuffers& buffers, std::uint16_t grid0, std::uint16_t grid1, Complex32 h00,
                  Complex32 h01, Complex32 h10, Complex32 h11, Complex32 s0, Complex32 s1) {
    const std::uint32_t symbol0 = grid0 / kLteNumSubcarriers20MHz;
    const std::uint32_t sc0 = grid0 % kLteNumSubcarriers20MHz;
    const std::uint32_t symbol1 = grid1 / kLteNumSubcarriers20MHz;
    const std::uint32_t sc1 = grid1 % kLteNumSubcarriers20MHz;
    const Complex32 y0_k0 =
        mmse::detail::cadd(mmse::detail::cmul(h00, s0), mmse::detail::cmul(h01, s1));
    const Complex32 y1_k0 =
        mmse::detail::cadd(mmse::detail::cmul(h10, s0), mmse::detail::cmul(h11, s1));
    const Complex32 y0_k1 = mmse::detail::csub(mmse::detail::cmul(h00, mmse::detail::cconj(s1)),
                                               mmse::detail::cmul(h01, mmse::detail::cconj(s0)));
    const Complex32 y1_k1 = mmse::detail::csub(mmse::detail::cmul(h10, mmse::detail::cconj(s1)),
                                               mmse::detail::cmul(h11, mmse::detail::cconj(s0)));

    buffers.re[0][grid0] = y0_k0.re;
    buffers.im[0][grid0] = y0_k0.im;
    buffers.re[1][grid0] = y1_k0.re;
    buffers.im[1][grid0] = y1_k0.im;

    buffers.re[0][grid1] = y0_k1.re;
    buffers.im[0][grid1] = y0_k1.im;
    buffers.re[1][grid1] = y1_k1.re;
    buffers.im[1][grid1] = y1_k1.im;
}

void fill_pdcch_td_pair(GridBuffers& buffers, const ExtractDescriptor& desc, std::uint16_t grid0,
                        std::uint16_t grid1, Complex32 h00, Complex32 h01, Complex32 h10,
                        Complex32 h11, Complex32 s0, Complex32 s1) {
    const std::uint32_t n_subcarriers = lte_downlink_subcarrier_count(desc.n_prb);
    const std::uint32_t symbol0 = grid0 / n_subcarriers;
    const std::uint32_t symbol1 = grid1 / n_subcarriers;
    expect_true(symbol0 == symbol1, "td pair must stay in one symbol");
    fill_td_pair(buffers, grid0, grid1, h00, h01, h10, h11, s0, s1);
}

void fill_td_layout(GridBuffers& buffers, const mmse::detail::ReLayout& layout, Complex32 h00,
                    Complex32 h01, Complex32 h10, Complex32 h11, Complex32 s0, Complex32 s1) {
    expect_true((layout.n_re & 1U) == 0U, "td layout must have even RE count");
    for (std::uint32_t i = 0U; i < layout.n_re; i += 2U) {
        fill_td_pair(buffers, layout.grid_indices[i], layout.grid_indices[i + 1U], h00, h01, h10,
                     h11, s0, s1);
    }
}

void fill_pdcch_td_layout(GridBuffers& buffers, const ExtractDescriptor& desc,
                          const mmse::detail::ReLayout& layout, Complex32 h00, Complex32 h01,
                          Complex32 h10, Complex32 h11, Complex32 s0, Complex32 s1) {
    expect_true((layout.n_re & 1U) == 0U, "td layout must have even RE count");
    for (std::uint32_t i = 0; i < layout.n_re; i += 2U) {
        fill_pdcch_td_pair(buffers, desc, layout.grid_indices[i], layout.grid_indices[i + 1U], h00,
                           h01, h10, h11, s0, s1);
    }
}

void test_reject_invalid_descriptor() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    GridBuffers buffers = make_zero_grid();
    auto grid = make_grid_view(buffers);
    auto desc = make_fullband_desc();
    desc.n_layers = 3;

    std::vector<float> xre(20000U);
    std::vector<float> xim(20000U);
    std::vector<float> sinr(20000U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), 10000U};
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kUnsupportedConfig,
                "unsupported descriptor should fail");
}

void test_buffer_too_small() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    GridBuffers buffers = make_zero_grid();
    auto grid = make_grid_view(buffers);
    auto desc = make_fullband_desc();

    std::vector<float> xre(16U);
    std::vector<float> xim(16U);
    std::vector<float> sinr(16U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), 8U};
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kBufferTooSmall, "tiny buffer should fail");
}

void test_single_layer_identity_channel_equalization() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    config.sigma2_min = 1.0e-5F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    GridBuffers buffers = make_zero_grid();
    auto desc = make_fullband_desc();
    desc.n_layers = 1;
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> xre(cap);
    std::vector<float> xim(cap);
    std::vector<float> sinr(cap);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kOk, "run must succeed");
    expect_true(out.n_re_per_layer > 0U, "must output REs");
    expect_true(out.n_re_per_layer == 14400U, "full-band RE count");
    expect_near(out.x_hat_re[0], 1.0F, 1.0e-3F, "layer0 xhat");
    expect_near(out.x_hat_im[0], 0.0F, 1.0e-5F, "layer0 imag");
    if (!(out.sinr[0] > 10.0F)) {
        throw TestFailure{"high SINR expected actual=" + std::to_string(out.sinr[0])};
    }
}

void test_single_rx_single_layer_equalization_and_pdcch() {
    GridBuffers buffers = make_zero_grid();
    auto desc = make_fullband_desc();
    desc.n_tx_ports = 1U;
    desc.n_rx_ant = 1U;
    desc.n_layers = 1U;
    desc.tx_mode = 1U;
    fill_identity_channel(buffers, desc, 0.75F, 0.0F);
    const PlanarGridViewF32 single_rx_grid = make_single_rx_grid_view(buffers);

    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.backend = MmseCpuBackend::kScalar;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "single-rx cpu init");

    constexpr std::uint32_t kCapacity = 20000U;
    std::vector<float> cpu_re(kCapacity), cpu_im(kCapacity), cpu_sinr(kCapacity);
    EqualizerOutputView cpu_out{cpu_re.data(), cpu_im.data(), cpu_sinr.data(), kCapacity};
    expect_true(cpu.run(single_rx_grid, desc, cpu_out) == MmseStatus::kOk,
                "single-rx single-layer cpu run");
    expect_near(cpu_re[0], 0.75F, 1.0e-3F, "single-rx cpu xhat real");
    expect_near(cpu_im[0], 0.0F, 1.0e-5F, "single-rx cpu xhat imag");
    expect_true(cpu_sinr[0] > 10.0F, "single-rx cpu sinr");

    auto unsupported_desc = desc;
    unsupported_desc.n_tx_ports = 2U;
    unsupported_desc.n_layers = 2U;
    expect_true(cpu.run(single_rx_grid, unsupported_desc, cpu_out) ==
                    MmseStatus::kUnsupportedConfig,
                "single-rx must reject two-layer spatial multiplexing");

    auto pdcch_desc = make_pdcch_desc();
    pdcch_desc.n_rx_ant = 1U;
    PdcchMmseInput pdcch_in{};
    pdcch_in.grid = single_rx_grid;
    pdcch_in.sfn_subframe = pdcch_desc.sfn_subframe;
    pdcch_in.cell_id = pdcch_desc.cell_id;
    pdcch_in.n_tx_ports = pdcch_desc.n_tx_ports;
    pdcch_in.tx_mode = pdcch_desc.tx_mode;
    pdcch_in.control_symbol_count = pdcch_desc.control_symbol_count;
    pdcch_in.n_prb = pdcch_desc.n_prb;
    pdcch_in.prb_bitmap = pdcch_desc.prb_bitmap;
    pdcch_in.control_re_exclusion_masks = pdcch_desc.control_re_exclusion_masks;
    pdcch_in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                                 .subframe =
                                     static_cast<std::uint8_t>(pdcch_desc.sfn_subframe % 10U),
                                 .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    std::vector<float> pdcch_re(3168U), pdcch_im(3168U), pdcch_sinr(3168U);
    std::vector<std::uint16_t> pdcch_indices(3168U);
    PdcchMmseOutputView pdcch_out{pdcch_re.data(),      pdcch_im.data(), pdcch_sinr.data(),
                                  pdcch_indices.data(), 3168U,           3168U};
    PdcchMmseResult pdcch_meta{};
    expect_true(cpu.run_pdcch(pdcch_in, pdcch_out, pdcch_meta) == MmseStatus::kOk,
                "single-rx pdcch run");
    expect_true(pdcch_meta.n_rx_ant == 1U && pdcch_meta.n_re == 3168U, "single-rx pdcch metadata");

    GridBuffers pbch_buffers = make_zero_grid();
    const auto pbch_desc = make_pbch_desc();
    fill_identity_channel(pbch_buffers, pbch_desc, 0.5F, 0.0F);
    PbchMmseInput pbch_in{};
    pbch_in.grid = make_single_rx_grid_view(pbch_buffers);
    pbch_in.sfn_subframe = pbch_desc.sfn_subframe;
    pbch_in.cell_id = pbch_desc.cell_id;
    pbch_in.n_tx_ports = pbch_desc.n_tx_ports;
    pbch_in.tx_mode = pbch_desc.tx_mode;
    std::vector<float> pbch_re(240U), pbch_im(240U), pbch_sinr(240U);
    std::vector<std::uint16_t> pbch_indices(240U);
    PbchMmseOutputView pbch_out{pbch_re.data(),      pbch_im.data(), pbch_sinr.data(),
                                pbch_indices.data(), 240U,           240U};
    PbchMmseResult pbch_meta{};
    expect_true(cpu.run_pbch(pbch_in, pbch_out, pbch_meta) == MmseStatus::kOk,
                "single-rx pbch run");
    expect_true(pbch_meta.n_rx_ant == 1U && pbch_meta.n_re == 240U, "single-rx pbch metadata");

    GridBuffers pcfich_buffers = make_zero_grid();
    const auto pcfich_desc = make_pcfich_desc();
    fill_identity_channel(pcfich_buffers, pcfich_desc, 0.5F, 0.0F);
    PcfichMmseInput pcfich_in{};
    pcfich_in.grid = make_single_rx_grid_view(pcfich_buffers);
    pcfich_in.sfn_subframe = pcfich_desc.sfn_subframe;
    pcfich_in.cell_id = pcfich_desc.cell_id;
    pcfich_in.n_tx_ports = pcfich_desc.n_tx_ports;
    pcfich_in.tx_mode = pcfich_desc.tx_mode;
    std::vector<float> pcfich_re(16U), pcfich_im(16U), pcfich_sinr(16U);
    std::vector<std::uint16_t> pcfich_indices(16U);
    PcfichMmseOutputView pcfich_out{
        pcfich_re.data(), pcfich_im.data(), pcfich_sinr.data(), pcfich_indices.data(), 16U, 16U};
    PcfichMmseResult pcfich_meta{};
    expect_true(cpu.run_pcfich(pcfich_in, pcfich_out, pcfich_meta) == MmseStatus::kOk,
                "single-rx pcfich run");
    expect_true(pcfich_meta.n_rx_ant == 1U && pcfich_meta.n_re == 16U, "single-rx pcfich metadata");

    GridBuffers td_buffers = make_zero_grid();
    auto td_desc = make_pdcch_desc();
    td_desc.n_rx_ant = 1U;
    td_desc.n_tx_ports = 2U;
    td_desc.tx_mode = 2U;
    fill_identity_channel(td_buffers, td_desc, 0.0F, 0.0F);
    mmse::detail::ReLayout td_layout{};
    const std::uint32_t td_re_count = mmse::detail::build_pdcch_re_layout(td_desc, td_layout);
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};
    fill_pdcch_td_layout(td_buffers, td_desc, td_layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                         {0.0F, 0.0F}, s0, s1);
    PdcchMmseInput td_in = pdcch_in;
    td_in.grid = make_single_rx_grid_view(td_buffers);
    td_in.n_tx_ports = td_desc.n_tx_ports;
    td_in.tx_mode = td_desc.tx_mode;
    std::vector<float> td_re(td_re_count), td_im(td_re_count), td_sinr(td_re_count);
    std::vector<std::uint16_t> td_grid0(td_re_count), td_grid1(td_re_count);
    PdcchTdMmseOutputView td_out{td_re.data(),    td_im.data(),    td_sinr.data(),
                                 td_grid0.data(), td_grid1.data(), td_re_count};
    PdcchTdMmseResult td_meta{};
    expect_true(cpu.run_pdcch_td(td_in, td_out, td_meta) == MmseStatus::kOk,
                "single-rx pdcch transmit-diversity run");
    expect_near(td_re[0], s0.re, 1.0e-3F, "single-rx td symbol0 real");
    expect_near(td_im[0], s0.im, 1.0e-3F, "single-rx td symbol0 imag");
    expect_near(td_re[1], s1.re, 1.0e-3F, "single-rx td symbol1 real");
    expect_near(td_im[1], s1.im, 1.0e-3F, "single-rx td symbol1 imag");

    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kAuto;
    gpu_config.sigma2_min = cpu_config.sigma2_min;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "single-rx gpu auto init");
    std::vector<float> gpu_re(kCapacity), gpu_im(kCapacity), gpu_sinr(kCapacity);
    EqualizerOutputView gpu_out{gpu_re.data(), gpu_im.data(), gpu_sinr.data(), kCapacity};
    expect_true(gpu.run(single_rx_grid, desc, gpu_out) == MmseStatus::kOk,
                "single-rx gpu auto run");
    expect_near(gpu_re[0], cpu_re[0], 1.0e-3F, "single-rx gpu xhat real");
    expect_near(gpu_im[0], cpu_im[0], 1.0e-5F, "single-rx gpu xhat imag");
    expect_near(gpu_sinr[0], cpu_sinr[0], 1.0e-2F, "single-rx gpu sinr");
}

void test_two_layer_constant_channel_matches_golden() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    config.sigma2_min = 1.0e-3F;
    config.det_floor = 1.0e-6F;
    config.g_min = 1.0e-4F;
    config.gamma_max = 1.0e4F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    const TwoLayerCase c = make_two_layer_case();
    fill_constant_mimo_channel(buffers, desc, c.h00, c.h01, c.h10, c.h11, c.x0, c.x1);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> xre(cap * 2U);
    std::vector<float> xim(cap * 2U);
    std::vector<float> sinr(cap * 2U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kOk, "two-layer run");
    expect_true(out.n_re_per_layer == 14400U, "two-layer RE count");

    const Complex32 y0 =
        mmse::detail::cadd(mmse::detail::cmul(c.h00, c.x0), mmse::detail::cmul(c.h01, c.x1));
    const Complex32 y1 =
        mmse::detail::cadd(mmse::detail::cmul(c.h10, c.x0), mmse::detail::cmul(c.h11, c.x1));
    mmse::detail::HGridStorage h_full{};
    float sigma2_estimate = 0.0F;
    mmse::detail::estimate_channel(grid, desc, h_full, sigma2_estimate);
    mmse::detail::Sigma2State sigma2_state{};
    const float sigma2_runtime =
        mmse::detail::update_sigma2_state(sigma2_state, sigma2_estimate, config);

    const auto golden0 =
        mmse::detail::equalize_2x2_scalar(c.h00, c.h01, c.h10, c.h11, y0, y1, sigma2_runtime,
                                          config.det_floor, config.g_min, config.gamma_max, 0U);
    const auto golden1 =
        mmse::detail::equalize_2x2_scalar(c.h00, c.h01, c.h10, c.h11, y0, y1, sigma2_runtime,
                                          config.det_floor, config.g_min, config.gamma_max, 1U);

    float max_x0_re = 0.0F;
    float max_x0_im = 0.0F;
    float max_x1_re = 0.0F;
    float max_x1_im = 0.0F;
    float max_g0 = 0.0F;
    float max_g1 = 0.0F;
    for (std::uint32_t i = 0; i < out.n_re_per_layer; ++i) {
        max_x0_re = std::max(max_x0_re, std::fabs(out.x_hat_re[i] - golden0.xhat.re));
        max_x0_im = std::max(max_x0_im, std::fabs(out.x_hat_im[i] - golden0.xhat.im));
        max_g0 = std::max(max_g0, std::fabs(out.sinr[i] - golden0.gamma));

        const std::size_t layer1 = out.capacity_re_per_layer + i;
        max_x1_re = std::max(max_x1_re, std::fabs(out.x_hat_re[layer1] - golden1.xhat.re));
        max_x1_im = std::max(max_x1_im, std::fabs(out.x_hat_im[layer1] - golden1.xhat.im));
        max_g1 = std::max(max_g1, std::fabs(out.sinr[layer1] - golden1.gamma));
    }

    expect_true(max_x0_re <= 1.0e-3F, "layer0 real max error");
    expect_true(max_x0_im <= 1.0e-3F, "layer0 imag max error");
    expect_true(max_x1_re <= 1.0e-3F, "layer1 real max error");
    expect_true(max_x1_im <= 1.0e-3F, "layer1 imag max error");
    if (!(max_g0 <= 6.0e-2F)) {
        throw TestFailure{"layer0 sinr max error=" + std::to_string(max_g0)};
    }
    if (!(max_g1 <= 6.0e-2F)) {
        throw TestFailure{"layer1 sinr max error=" + std::to_string(max_g1)};
    }
}

void test_two_layer_repeated_runs_are_stable() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    config.sigma2_min = 1.0e-3F;
    config.det_floor = 1.0e-6F;
    config.g_min = 1.0e-4F;
    config.gamma_max = 1.0e4F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    const TwoLayerCase c = make_two_layer_case();
    fill_constant_mimo_channel(buffers, desc, c.h00, c.h01, c.h10, c.h11, c.x0, c.x1);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> xre_a(cap * 2U);
    std::vector<float> xim_a(cap * 2U);
    std::vector<float> sinr_a(cap * 2U);
    std::vector<float> xre_b(cap * 2U);
    std::vector<float> xim_b(cap * 2U);
    std::vector<float> sinr_b(cap * 2U);
    EqualizerOutputView out_a{xre_a.data(), xim_a.data(), sinr_a.data(), cap};
    EqualizerOutputView out_b{xre_b.data(), xim_b.data(), sinr_b.data(), cap};

    expect_true(ctx.run(grid, desc, out_a) == MmseStatus::kOk, "first run");
    expect_true(ctx.run(grid, desc, out_b) == MmseStatus::kOk, "second run");
    for (std::uint32_t i = 0; i < out_a.n_re_per_layer * 2U; ++i) {
        expect_near(xre_a[i], xre_b[i], 1.0e-6F, "stable xhat real");
        expect_near(xim_a[i], xim_b[i], 1.0e-6F, "stable xhat imag");
        expect_near(sinr_a[i], sinr_b[i], 1.0e-6F, "stable sinr");
    }
}

void test_two_layer_avx2_context_matches_scalar_context() {
    if (!mmse::detail::cpu_supports_avx2()) {
        std::cout << "[SKIP] two_layer_avx2_context_matches_scalar_context\n";
        return;
    }

    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    const TwoLayerCase c = make_two_layer_case();
    fill_constant_mimo_channel(buffers, desc, c.h00, c.h01, c.h10, c.h11, c.x0, c.x1);
    auto grid = make_grid_view(buffers);

    MmseEqualizerCpuConfig scalar_cfg{};
    scalar_cfg.worker_count = 1;
    scalar_cfg.sigma2_min = 1.0e-3F;
    scalar_cfg.det_floor = 1.0e-6F;
    scalar_cfg.g_min = 1.0e-4F;
    scalar_cfg.gamma_max = 1.0e4F;
    scalar_cfg.backend = MmseCpuBackend::kScalar;

    MmseEqualizerCpuConfig avx2_cfg = scalar_cfg;
    avx2_cfg.backend = MmseCpuBackend::kAvx2;

    MmseEqualizerCpuContext scalar_ctx;
    MmseEqualizerCpuContext avx2_ctx;
    expect_true(scalar_ctx.init(scalar_cfg) == MmseStatus::kOk, "scalar ctx init");
    expect_true(avx2_ctx.init(avx2_cfg) == MmseStatus::kOk, "avx2 ctx init");

    const std::uint32_t cap = 20000U;
    std::vector<float> scalar_re(cap * 2U);
    std::vector<float> scalar_im(cap * 2U);
    std::vector<float> scalar_sinr(cap * 2U);
    std::vector<float> avx2_re(cap * 2U);
    std::vector<float> avx2_im(cap * 2U);
    std::vector<float> avx2_sinr(cap * 2U);
    EqualizerOutputView scalar_out{scalar_re.data(), scalar_im.data(), scalar_sinr.data(), cap};
    EqualizerOutputView avx2_out{avx2_re.data(), avx2_im.data(), avx2_sinr.data(), cap};

    expect_true(scalar_ctx.run(grid, desc, scalar_out) == MmseStatus::kOk, "scalar ctx run");
    expect_true(avx2_ctx.run(grid, desc, avx2_out) == MmseStatus::kOk, "avx2 ctx run");
    expect_true(scalar_out.n_re_per_layer == avx2_out.n_re_per_layer, "matching RE count");

    for (std::uint32_t i = 0; i < scalar_out.n_re_per_layer * 2U; ++i) {
        expect_near(avx2_re[i], scalar_re[i], 1.0e-5F, "ctx avx2/scalar xhat real");
        expect_near(avx2_im[i], scalar_im[i], 1.0e-5F, "ctx avx2/scalar xhat imag");
        expect_near(avx2_sinr[i], scalar_sinr[i], 6.0e-2F, "ctx avx2/scalar sinr");
    }
}

void test_two_layer_scalar_golden_matches() {
    const TwoLayerCase c = make_two_layer_case();
    const Complex32 y0 =
        mmse::detail::cadd(mmse::detail::cmul(c.h00, c.x0), mmse::detail::cmul(c.h01, c.x1));
    const Complex32 y1 =
        mmse::detail::cadd(mmse::detail::cmul(c.h10, c.x0), mmse::detail::cmul(c.h11, c.x1));
    const float sigma2 = 1.0e-3F;
    const float det_floor = 1.0e-6F;
    const float g_min = 1.0e-4F;
    const float gamma_max = 1.0e4F;

    const auto eq0 = mmse::detail::equalize_2x2_scalar(c.h00, c.h01, c.h10, c.h11, y0, y1, sigma2,
                                                       det_floor, g_min, gamma_max, 0U);
    const auto eq1 = mmse::detail::equalize_2x2_scalar(c.h00, c.h01, c.h10, c.h11, y0, y1, sigma2,
                                                       det_floor, g_min, gamma_max, 1U);

    expect_near(eq0.xhat.re, c.x0.re, 2.0e-2F, "scalar golden layer0 real");
    expect_near(eq0.xhat.im, c.x0.im, 2.0e-2F, "scalar golden layer0 imag");
    expect_near(eq1.xhat.re, c.x1.re, 2.0e-2F, "scalar golden layer1 real");
    expect_near(eq1.xhat.im, c.x1.im, 2.0e-2F, "scalar golden layer1 imag");
    expect_true(eq0.gamma > 0.0F, "scalar golden layer0 sinr positive");
    expect_true(eq1.gamma > 0.0F, "scalar golden layer1 sinr positive");
}

void test_two_layer_avx2_matches_scalar_kernel() {
    if (!mmse::detail::cpu_supports_avx2()) {
        std::cout << "[SKIP] two_layer_avx2_matches_scalar_kernel\n";
        return;
    }

    const TwoLayerCase c = make_two_layer_case();
    const Complex32 y0 =
        mmse::detail::cadd(mmse::detail::cmul(c.h00, c.x0), mmse::detail::cmul(c.h01, c.x1));
    const Complex32 y1 =
        mmse::detail::cadd(mmse::detail::cmul(c.h10, c.x0), mmse::detail::cmul(c.h11, c.x1));
    constexpr std::uint32_t lanes = 8U;
    constexpr float sigma2 = 1.0e-3F;
    constexpr float det_floor = 1.0e-6F;
    constexpr float g_min = 1.0e-4F;
    constexpr float gamma_max = 1.0e4F;

    mmse::detail::PackedEqualizerInputs packed{};
    for (std::uint32_t i = 0; i < lanes; ++i) {
        packed.h00_re[i] = c.h00.re;
        packed.h00_im[i] = c.h00.im;
        packed.h01_re[i] = c.h01.re;
        packed.h01_im[i] = c.h01.im;
        packed.h10_re[i] = c.h10.re;
        packed.h10_im[i] = c.h10.im;
        packed.h11_re[i] = c.h11.re;
        packed.h11_im[i] = c.h11.im;
        packed.y0_re[i] = y0.re;
        packed.y0_im[i] = y0.im;
        packed.y1_re[i] = y1.re;
        packed.y1_im[i] = y1.im;
    }

    std::array<float, lanes> out_re0{};
    std::array<float, lanes> out_im0{};
    std::array<float, lanes> out_sinr0{};
    std::array<float, lanes> out_re1{};
    std::array<float, lanes> out_im1{};
    std::array<float, lanes> out_sinr1{};
    mmse::detail::equalize_2x2_avx2(packed, 0U, lanes, sigma2, det_floor, g_min, gamma_max,
                                    out_re0.data(), out_im0.data(), out_sinr0.data(),
                                    out_re1.data(), out_im1.data(), out_sinr1.data());

    const auto eq0 = mmse::detail::equalize_2x2_scalar(c.h00, c.h01, c.h10, c.h11, y0, y1, sigma2,
                                                       det_floor, g_min, gamma_max, 0U);
    const auto eq1 = mmse::detail::equalize_2x2_scalar(c.h00, c.h01, c.h10, c.h11, y0, y1, sigma2,
                                                       det_floor, g_min, gamma_max, 1U);

    for (std::uint32_t i = 0; i < lanes; ++i) {
        expect_near(out_re0[i], eq0.xhat.re, 1.0e-5F, "avx2 layer0 real");
        expect_near(out_im0[i], eq0.xhat.im, 1.0e-5F, "avx2 layer0 imag");
        expect_near(out_sinr0[i], eq0.gamma, 6.0e-2F, "avx2 layer0 sinr");
        expect_near(out_re1[i], eq1.xhat.re, 1.0e-5F, "avx2 layer1 real");
        expect_near(out_im1[i], eq1.xhat.im, 1.0e-5F, "avx2 layer1 imag");
        expect_near(out_sinr1[i], eq1.gamma, 6.0e-2F, "avx2 layer1 sinr");
    }
}

void test_sigma2_state_persists() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    config.sigma2_iir_alpha = 0.5F;
    config.sigma2_min = 1.0e-6F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    auto desc = make_fullband_desc();
    GridBuffers clean = make_zero_grid();
    fill_identity_channel(clean, desc, 1.0F, 1.0F);
    auto clean_grid = make_grid_view(clean);

    GridBuffers noisy = clean;
    for (std::size_t i = 0; i < noisy.re[0].size(); i += 17U) {
        noisy.re[0][i] += 0.2F;
        noisy.im[1][i] -= 0.1F;
    }
    auto noisy_grid = make_grid_view(noisy);

    const std::uint32_t cap = 20000U;
    std::vector<float> xre(cap * 2U);
    std::vector<float> xim(cap * 2U);
    std::vector<float> sinr(cap * 2U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};

    expect_true(ctx.run(noisy_grid, desc, out) == MmseStatus::kOk, "noisy run");
    const float sinr_noisy = out.sinr[0];
    expect_true(ctx.run(clean_grid, desc, out) == MmseStatus::kOk, "clean run");
    const float sinr_after = out.sinr[0];
    expect_true(sinr_after < 1.0e6F, "clamped finite sinr");
    expect_true(sinr_after > sinr_noisy, "cleaner second pass should improve sinr");
}

void test_single_layer_path() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    auto desc = make_fullband_desc();
    desc.n_layers = 1;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.75F, 0.0F);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> xre(cap);
    std::vector<float> xim(cap);
    std::vector<float> sinr(cap);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kOk, "single-layer run");
    expect_near(out.x_hat_re[0], 0.75F, 1.0e-3F, "single-layer xhat");
}

void test_pdcch_layout_excludes_crs_and_reserved_res() {
    const auto desc = make_pdcch_desc();
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    expect_true(n_re == 3168U, "pdcch RE count without extra reserved REs");
    for (std::uint32_t i = 0; i < n_re; ++i) {
        const std::uint32_t grid_idx = layout.grid_indices[i];
        const std::uint32_t symbol = grid_idx / kLteNumSubcarriers20MHz;
        const std::uint32_t sc = grid_idx % kLteNumSubcarriers20MHz;
        expect_true(symbol < desc.control_symbol_count, "pdcch symbol must stay in control region");
        expect_true(!mmse::detail::is_crs_re(desc, static_cast<std::uint8_t>(symbol), sc),
                    "pdcch layout must exclude CRS");
    }

    auto reserved = desc;
    reserved.control_re_exclusion_masks[kLteNumPrb20MHz] = 0x000FU;
    const std::uint32_t n_re_reserved = mmse::detail::build_pdcch_re_layout(reserved, layout);
    expect_true(n_re_reserved == n_re,
                "CCE alignment may retain the same PDCCH RE count after excluding one REG");
    for (std::uint32_t i = 0; i < n_re_reserved; ++i) {
        const std::uint32_t grid_idx = layout.grid_indices[i];
        expect_true(grid_idx < kLteNumSubcarriers20MHz || grid_idx > kLteNumSubcarriers20MHz + 3U,
                    "reserved control REs must be excluded");
    }
}

void test_pbch_layout_matches_lte_center_72_subcarriers() {
    ExtractDescriptor desc = make_pbch_desc();
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_re = mmse::detail::build_channel_re_layout(desc, layout);
    expect_true(n_re == 240U, "pbch layout should expose 240 RE");
    for (std::uint32_t i = 0; i < n_re; ++i) {
        const std::uint32_t grid_idx = layout.grid_indices[i];
        const std::uint32_t symbol = grid_idx / kLteNumSubcarriers20MHz;
        const std::uint32_t sc = grid_idx % kLteNumSubcarriers20MHz;
        expect_true(symbol >= kLtePbchStartSymbolNormalCp &&
                        symbol < kLtePbchStartSymbolNormalCp + kLtePbchNumSymbols,
                    "pbch symbol range");
        expect_true(sc >= kLtePbchStartPrb * kLteNumSubcarriersPerPrb &&
                        sc < (kLtePbchStartPrb + kLtePbchNumPrb) * kLteNumSubcarriersPerPrb,
                    "pbch subcarrier range");
    }
}

void test_pcfich_layout_matches_four_regs_without_crs() {
    ExtractDescriptor desc = make_pcfich_desc();
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_re = mmse::detail::build_channel_re_layout(desc, layout);
    expect_true(n_re == 16U, "pcfich layout should expose 16 RE after CRS removal");
    for (std::uint32_t i = 0; i < n_re; ++i) {
        const std::uint32_t grid_idx = layout.grid_indices[i];
        const std::uint32_t symbol = grid_idx / kLteNumSubcarriers20MHz;
        const std::uint32_t sc = grid_idx % kLteNumSubcarriers20MHz;
        expect_true(symbol == 0U, "pcfich should stay in symbol 0");
        expect_true(!mmse::detail::is_crs_re(desc, 0U, sc), "pcfich layout must exclude CRS");
    }
}

void test_single_port_lte_control_reference_vector() {
    const auto pdcch_desc = make_pdcch_desc();
    mmse::detail::ReLayout pdcch_layout{};
    expect_true(mmse::detail::build_pdcch_re_layout(pdcch_desc, pdcch_layout) == 3168U,
                "single-port pdcch reference RE count");
    constexpr std::array<std::uint16_t, 36> kFirstCceGridIndices = {
        601U,  602U,  604U,  605U,  1U,    2U,    4U,    5U,    901U,  902U,  904U,  905U,
        301U,  302U,  304U,  305U,  3148U, 3149U, 3150U, 3151U, 2548U, 2549U, 2550U, 2551U,
        3448U, 3449U, 3450U, 3451U, 2848U, 2849U, 2850U, 2851U, 3072U, 3073U, 3074U, 3075U,
    };
    for (std::uint32_t i = 0U; i < kFirstCceGridIndices.size(); ++i) {
        expect_true(pdcch_layout.grid_indices[i] == kFirstCceGridIndices[i],
                    "single-port pdcch first CCE REG order");
    }

    const auto pbch_desc = make_pbch_desc();
    mmse::detail::ReLayout pbch_layout{};
    expect_true(mmse::detail::build_pbch_re_layout(pbch_desc, pbch_layout) == 240U,
                "single-port pbch reference RE count");
    constexpr std::array<std::uint16_t, 8> kPbchPrefixGridIndices = {
        8965U, 8966U, 8968U, 8969U, 8971U, 8972U, 8974U, 8975U,
    };
    for (std::uint32_t i = 0U; i < kPbchPrefixGridIndices.size(); ++i) {
        expect_true(pbch_layout.grid_indices[i] == kPbchPrefixGridIndices[i],
                    "single-port pbch CRS holes");
    }

    const auto pcfich_desc = make_pcfich_desc();
    mmse::detail::ReLayout pcfich_layout{};
    expect_true(mmse::detail::build_pcfich_re_layout(pcfich_desc, pcfich_layout) == 16U,
                "single-port pcfich reference RE count");
    constexpr std::array<std::uint16_t, 16> kPcfichGridIndices = {
        1U, 2U, 4U, 5U, 301U, 302U, 304U, 305U, 601U, 602U, 604U, 605U, 901U, 902U, 904U, 905U,
    };
    for (std::uint32_t i = 0U; i < kPcfichGridIndices.size(); ++i) {
        expect_true(pcfich_layout.grid_indices[i] == kPcfichGridIndices[i],
                    "single-port pcfich REG CRS holes");
    }

    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, pdcch_desc, 0.75F, 0.0F);
    const PlanarGridViewF32 grid = make_grid_view(buffers);
    static mmse::detail::HGridStorage h_full{};
    float sigma2_estimate = 0.0F;
    mmse::detail::estimate_channel(grid, pdcch_desc, h_full, sigma2_estimate);
    expect_near(sigma2_estimate, 0.0F, 1.0e-7F,
                "single-port noise estimate must ignore absent port-one CRS");

    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1U;
    config.backend = MmseCpuBackend::kScalar;
    config.sigma2_min = 1.0e-5F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "single-port reference cpu init");

    std::vector<float> xre(3168U), xim(3168U), sinr(3168U);
    std::vector<std::uint16_t> indices(3168U);
    PdcchMmseOutputView out{xre.data(), xim.data(), sinr.data(), indices.data(), 3168U, 3168U};
    PdcchMmseInput input{};
    input.grid = grid;
    input.sfn_subframe = pdcch_desc.sfn_subframe;
    input.cell_id = pdcch_desc.cell_id;
    input.n_tx_ports = pdcch_desc.n_tx_ports;
    input.tx_mode = pdcch_desc.tx_mode;
    input.control_symbol_count = pdcch_desc.control_symbol_count;
    input.n_prb = pdcch_desc.n_prb;
    input.prb_bitmap = pdcch_desc.prb_bitmap;
    input.control_re_exclusion_masks = pdcch_desc.control_re_exclusion_masks;
    input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                              .subframe = static_cast<std::uint8_t>(pdcch_desc.sfn_subframe % 10U),
                              .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    PdcchMmseResult meta{};
    expect_true(ctx.run_pdcch(input, out, meta) == MmseStatus::kOk,
                "single-port reference pdcch run");
    expect_near(meta.sigma2, config.sigma2_min, 1.0e-7F, "single-port sigma2 floor");
    expect_near(xre[0], 0.75F, 1.0e-3F, "single-port equalized symbol");
    expect_near(xim[0], 0.0F, 1.0e-5F, "single-port equalized symbol imaginary part");
    expect_true(sinr[0] > 0.99F * config.gamma_max, "single-port equalized SINR");
    for (std::uint32_t i = 0U; i < kFirstCceGridIndices.size(); ++i) {
        expect_true(indices[i] == kFirstCceGridIndices[i], "single-port pdcch output CCE order");
    }
}

void test_pdcch_cpu_run_supports_single_port_and_generic_two_port_layout() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    config.sigma2_min = 1.0e-5F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    GridBuffers buffers = make_zero_grid();
    auto desc = make_pdcch_desc();
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 8000U;
    std::vector<float> xre(cap);
    std::vector<float> xim(cap);
    std::vector<float> sinr(cap);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kOk, "single-port pdcch run");
    expect_true(out.n_re_per_layer == 3168U, "single-port pdcch RE count");
    expect_true(out.mod_order == 2U, "pdcch mod order must stay qpsk");

    auto two_port = desc;
    two_port.n_tx_ports = 2U;
    two_port.tx_mode = 2U;
    expect_true(ctx.run(grid, two_port, out) == MmseStatus::kUnsupportedConfig,
                "generic run must reject two-port pdcch transmit diversity");
}

void test_pbch_frontend_dto_builds_mmse_input() {
    GridBuffers buffers = make_zero_grid();
    const PlanarGridViewF32 grid = make_grid_view(buffers);

    mmse::pbch::FrontendPbchIndication frontend{};
    frontend.sfn_subframe = 5U;
    frontend.cell_id = 23U;
    frontend.n_tx_ports = 1U;
    frontend.tx_mode = 1U;
    frontend.chain.request_id = 111U;

    const PbchMmseInput in = mmse::pbch::make_pbch_mmse_input(grid, frontend);
    expect_true(in.grid.n_symbols == grid.n_symbols, "pbch dto grid passthrough");
    expect_true(in.sfn_subframe == frontend.sfn_subframe, "pbch dto subframe");
    expect_true(in.cell_id == frontend.cell_id, "pbch dto cell");
    expect_true(in.chain.request_id == frontend.chain.request_id, "pbch dto chain");
}

void test_pcfich_frontend_dto_builds_mmse_input() {
    GridBuffers buffers = make_zero_grid();
    const PlanarGridViewF32 grid = make_grid_view(buffers);

    mmse::pcfich::FrontendPcfichIndication frontend{};
    frontend.sfn_subframe = 6U;
    frontend.cell_id = 41U;
    frontend.n_tx_ports = 1U;
    frontend.tx_mode = 1U;
    frontend.chain.request_id = 222U;

    const PcfichMmseInput in = mmse::pcfich::make_pcfich_mmse_input(grid, frontend);
    expect_true(in.grid.n_symbols == grid.n_symbols, "pcfich dto grid passthrough");
    expect_true(in.sfn_subframe == frontend.sfn_subframe, "pcfich dto subframe");
    expect_true(in.cell_id == frontend.cell_id, "pcfich dto cell");
    expect_true(in.chain.request_id == frontend.chain.request_id, "pcfich dto chain");
}

void test_pbch_backend_dto_packs_equalized_output() {
    PbchMmseResult meta{};
    meta.n_re = 3U;
    meta.sfn_subframe = 7U;
    meta.cell_id = 10U;
    meta.start_prb = kLtePbchStartPrb;
    meta.n_prb = kLtePbchNumPrb;
    meta.start_symbol = kLtePbchStartSymbolNormalCp;
    meta.n_tx_ports = 1U;
    meta.n_rx_ant = 2U;
    meta.n_layers = 1U;
    meta.tx_mode = 1U;
    meta.mod_order = 2U;
    meta.sigma2 = 0.25F;
    meta.chain.request_id = 333U;

    std::array<float, 3> xre = {1.0F, 2.0F, 3.0F};
    std::array<float, 3> xim = {0.1F, 0.2F, 0.3F};
    std::array<float, 3> sinr = {10.0F, 11.0F, 12.0F};
    std::array<std::uint16_t, 3> indices = {100U, 101U, 102U};
    PbchMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices = indices.data();

    const auto backend = mmse::pbch::make_backend_pbch_equalized_indication(meta, out);
    expect_true(backend.re_grid_indices.size() == 3U, "pbch backend dto size");
    expect_true(backend.x_hat_re[1] == 2.0F, "pbch backend dto xhat");
    expect_true(backend.sigma2 == meta.sigma2, "pbch backend dto sigma2");
    expect_true(backend.chain.request_id == meta.chain.request_id, "pbch backend dto chain");
}

void test_pcfich_backend_dto_packs_equalized_output() {
    PcfichMmseResult meta{};
    meta.n_re = 3U;
    meta.sfn_subframe = 8U;
    meta.cell_id = 11U;
    meta.n_prb = kLteNumPrb20MHz;
    meta.start_symbol = 0U;
    meta.reg_count = static_cast<std::uint8_t>(kLtePcfichNumRegs);
    meta.n_tx_ports = 1U;
    meta.n_rx_ant = 2U;
    meta.n_layers = 1U;
    meta.tx_mode = 1U;
    meta.mod_order = 2U;
    meta.sigma2 = 0.5F;
    meta.chain.request_id = 444U;

    std::array<float, 3> xre = {4.0F, 5.0F, 6.0F};
    std::array<float, 3> xim = {0.4F, 0.5F, 0.6F};
    std::array<float, 3> sinr = {13.0F, 14.0F, 15.0F};
    std::array<std::uint16_t, 3> indices = {10U, 11U, 12U};
    PcfichMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices = indices.data();

    const auto backend = mmse::pcfich::make_backend_pcfich_equalized_indication(meta, out);
    expect_true(backend.re_grid_indices.size() == 3U, "pcfich backend dto size");
    expect_true(backend.x_hat_im[2] == 0.6F, "pcfich backend dto imag");
    expect_true(backend.reg_count == meta.reg_count, "pcfich backend dto reg count");
    expect_true(backend.chain.request_id == meta.chain.request_id, "pcfich backend dto chain");
}

void test_pbch_cpu_run_returns_equalized_pbch_surface() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    expect_true(ctx.init(config) == MmseStatus::kOk, "pbch cpu init");

    GridBuffers buffers = make_zero_grid();
    auto desc = make_pbch_desc();
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);

    PbchMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.chain.request_id = 555U;

    std::vector<float> xre(300U);
    std::vector<float> xim(300U);
    std::vector<float> sinr(300U);
    std::vector<std::uint16_t> indices(300U);
    PbchMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices = indices.data();
    out.capacity_re_per_layer = 300U;
    out.capacity_re_metadata = 300U;

    PbchMmseResult meta{};
    expect_true(ctx.run_pbch(in, out, meta) == MmseStatus::kOk, "pbch cpu run");
    expect_true(meta.n_re == 240U, "pbch cpu n_re");
    expect_true(meta.start_prb == kLtePbchStartPrb, "pbch cpu start prb");
    expect_true(meta.chain.request_id == in.chain.request_id, "pbch cpu chain");
}

void test_pcfich_cpu_run_returns_equalized_pcfich_surface() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    expect_true(ctx.init(config) == MmseStatus::kOk, "pcfich cpu init");

    GridBuffers buffers = make_zero_grid();
    auto desc = make_pcfich_desc();
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);

    PcfichMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.chain.request_id = 666U;

    std::vector<float> xre(64U);
    std::vector<float> xim(64U);
    std::vector<float> sinr(64U);
    std::vector<std::uint16_t> indices(64U);
    PcfichMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices = indices.data();
    out.capacity_re_per_layer = 64U;
    out.capacity_re_metadata = 64U;

    PcfichMmseResult meta{};
    expect_true(ctx.run_pcfich(in, out, meta) == MmseStatus::kOk, "pcfich cpu run");
    expect_true(meta.n_re == 16U, "pcfich cpu n_re");
    expect_true(meta.reg_count == kLtePcfichNumRegs, "pcfich cpu reg count");
    expect_true(meta.chain.request_id == in.chain.request_id, "pcfich cpu chain");
}

void test_crs_values_match_independent_reference() {
    mmse::detail::ensure_crs_tables();
    struct CrsCase {
        std::uint16_t cell_id;
        std::uint8_t subframe;
        std::uint8_t port;
        std::uint8_t symbol_index;
        std::uint32_t pilot;
    };
    const std::array cases = {
        CrsCase{0U, 0U, 0U, 0U, 0U},
        CrsCase{1U, 3U, 1U, 1U, 17U},
        CrsCase{42U, 7U, 0U, 2U, 86U},
        CrsCase{503U, 9U, 1U, 3U, 199U},
    };
    for (const CrsCase& item : cases) {
        const std::uint8_t symbol = mmse::detail::kCrsSymbols[item.symbol_index];
        const Complex32 expected =
            reference_crs_value(item.cell_id, item.subframe, item.port, symbol, item.pilot);
        const Complex32 actual = mmse::detail::crs_value({.cell_id = item.cell_id,
                                                          .subframe = item.subframe,
                                                          .port = item.port,
                                                          .crs_symbol_index = item.symbol_index},
                                                         item.pilot);
        expect_near(actual.re, expected.re, 1.0e-6F, "independent CRS real");
        expect_near(actual.im, expected.im, 1.0e-6F, "independent CRS imag");
    }
}

void test_channel_interpolation_extrapolates_linearly() {
    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 1U;
    desc.n_rx_ant = 1U;
    desc.tx_mode = 1U;
    desc.sfn_subframe = 6U;
    GridBuffers buffers = make_zero_grid();
    const auto channel_at = [](std::uint32_t symbol, std::uint32_t sc) {
        return Complex32{
            0.25F + 0.02F * static_cast<float>(symbol) + 0.001F * static_cast<float>(sc),
            -0.15F + 0.01F * static_cast<float>(symbol) - 0.0005F * static_cast<float>(sc)};
    };
    for (std::uint32_t cs = 0U; cs < mmse::detail::kCrsSymbols.size(); ++cs) {
        const std::uint8_t symbol = mmse::detail::kCrsSymbols[cs];
        for (std::uint32_t pilot = 0U; pilot < kLteNumPilotTonesPerCrsSymbol; ++pilot) {
            const std::uint32_t sc = mmse::detail::crs_subcarrier(desc.cell_id, 0U, symbol, pilot);
            const Complex32 sample = mmse::detail::cmul(
                channel_at(symbol, sc),
                mmse::detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                     .port = 0U,
                     .crs_symbol_index = static_cast<std::uint8_t>(cs)},
                    pilot));
            const std::size_t index =
                static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz + sc;
            buffers.re[0][index] = sample.re;
            buffers.im[0][index] = sample.im;
        }
    }
    static mmse::detail::HGridStorage h_full{};
    float sigma2 = 0.0F;
    mmse::detail::estimate_channel(make_single_rx_grid_view(buffers), desc, h_full, sigma2);
    for (const auto [symbol, sc] : std::array<std::pair<std::uint32_t, std::uint32_t>, 4>{
             {{0U, 0U}, {4U, 1199U}, {12U, 17U}, {13U, 1199U}}}) {
        const Complex32 expected = channel_at(symbol, sc);
        const Complex32 actual =
            h_full[(static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz) + sc];
        expect_near(actual.re, expected.re, 2.0e-5F, "linear interpolation real");
        expect_near(actual.im, expected.im, 2.0e-5F, "linear interpolation imag");
    }
    expect_near(sigma2, 0.0F, 1.0e-6F, "linear channel noise estimate");
}

void test_noise_estimate_has_two_thirds_calibration() {
    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 1U;
    desc.n_rx_ant = 1U;
    desc.tx_mode = 1U;
    GridBuffers buffers = make_zero_grid();
    constexpr float kSigma2 = 0.04F;
    constexpr float kAxis = 0.1414213562373095F;
    std::uint32_t state = 0x13579BDFU;
    for (std::uint32_t cs = 0U; cs < mmse::detail::kCrsSymbols.size(); ++cs) {
        const std::uint8_t symbol = mmse::detail::kCrsSymbols[cs];
        for (std::uint32_t pilot = 0U; pilot < kLteNumPilotTonesPerCrsSymbol; ++pilot) {
            state = state * 1664525U + 1013904223U;
            const float noise_re = (state & 1U) == 0U ? kAxis : -kAxis;
            const float noise_im = (state & 2U) == 0U ? kAxis : -kAxis;
            const std::uint32_t sc = mmse::detail::crs_subcarrier(desc.cell_id, 0U, symbol, pilot);
            const Complex32 ls{1.0F + noise_re, noise_im};
            const Complex32 sample = mmse::detail::cmul(
                ls, mmse::detail::crs_value(
                        {.cell_id = desc.cell_id,
                         .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                         .port = 0U,
                         .crs_symbol_index = static_cast<std::uint8_t>(cs)},
                        pilot));
            const std::size_t index =
                static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz + sc;
            buffers.re[0][index] = sample.re;
            buffers.im[0][index] = sample.im;
        }
    }
    static mmse::detail::HGridStorage h_full{};
    float sigma2 = 0.0F;
    mmse::detail::estimate_channel(make_single_rx_grid_view(buffers), desc, h_full, sigma2);
    expect_relative_near(sigma2, kSigma2, 0.12F, 1.0e-4F, "two-thirds calibrated AWGN estimate");
}

void test_generation_invalidates_cpu_subframe_cache() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1U;
    expect_true(ctx.init(config) == MmseStatus::kOk, "generation cache cpu init");
    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);
    PlanarGridViewF32 grid = make_grid_view(buffers);
    grid.generation = 100U;
    std::vector<float> xre(40000U), xim(40000U), sinr(40000U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), 20000U};
    mmse::detail::debug_reset_estimate_channel_call_count();
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kOk, "generation cache first run");
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kOk, "generation cache reuse run");
    expect_true(mmse::detail::debug_get_estimate_channel_call_count() == 1U,
                "unchanged generation must reuse estimate");
    ++grid.generation;
    buffers.re[0][0] += 0.125F;
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kOk, "generation cache invalidation run");
    expect_true(mmse::detail::debug_get_estimate_channel_call_count() == 2U,
                "new generation must refresh estimate");
}

void test_pbch_and_pcfich_td_cpu_golden_and_dto() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1U;
    config.sigma2_min = 1.0e-5F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "td special-channel cpu init");
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};

    auto pbch_desc = make_pbch_desc();
    pbch_desc.n_tx_ports = 2U;
    pbch_desc.n_rx_ant = 1U;
    pbch_desc.tx_mode = 2U;
    GridBuffers pbch_buffers = make_zero_grid();
    fill_identity_channel(pbch_buffers, pbch_desc, 0.0F, 0.0F);
    mmse::detail::ReLayout pbch_layout{};
    const std::uint32_t pbch_re = mmse::detail::build_pbch_re_layout(pbch_desc, pbch_layout);
    fill_td_layout(pbch_buffers, pbch_layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                   {0.0F, 0.0F}, s0, s1);
    PbchMmseInput pbch_in{.grid = make_single_rx_grid_view(pbch_buffers),
                          .sfn_subframe = pbch_desc.sfn_subframe,
                          .cell_id = pbch_desc.cell_id,
                          .n_tx_ports = 2U,
                          .tx_mode = 2U};
    std::vector<float> pbch_xre(pbch_re), pbch_xim(pbch_re), pbch_sinr(pbch_re);
    std::vector<std::uint16_t> pbch_re0(pbch_re), pbch_re1(pbch_re);
    PbchTdMmseOutputView pbch_out{pbch_xre.data(), pbch_xim.data(), pbch_sinr.data(),
                                  pbch_re0.data(), pbch_re1.data(), pbch_re};
    PbchTdMmseResult pbch_meta{};
    expect_true(ctx.run_pbch_td(pbch_in, pbch_out, pbch_meta) == MmseStatus::kOk,
                "pbch td cpu run");
    expect_true(pbch_meta.n_symbols == pbch_re && pbch_meta.n_source_re == pbch_re,
                "pbch td metadata count");
    expect_near(pbch_xre[0], s0.re, 1.0e-3F, "pbch td symbol zero real");
    expect_near(pbch_xim[1], s1.im, 1.0e-3F, "pbch td symbol one imag");
    expect_true(pbch_re0[0] == pbch_layout.grid_indices[0] &&
                    pbch_re1[1] == pbch_layout.grid_indices[1],
                "pbch td source pair metadata");
    const auto pbch_backend =
        mmse::pbch::make_backend_pbch_td_equalized_indication(pbch_meta, pbch_out);
    const auto pbch_llr = mmse::pbch::make_backend_pbch_td_descrambled_llr_indication(pbch_backend);
    expect_true(pbch_llr.llrs.size() == pbch_re * 2U &&
                    pbch_llr.re_grid_indices0 == pbch_backend.re_grid_indices0,
                "pbch td LLR DTO");

    auto pcfich_desc = make_pcfich_desc();
    pcfich_desc.n_tx_ports = 2U;
    pcfich_desc.tx_mode = 2U;
    GridBuffers pcfich_buffers = make_zero_grid();
    fill_identity_channel(pcfich_buffers, pcfich_desc, 0.0F, 0.0F);
    mmse::detail::ReLayout pcfich_layout{};
    const std::uint32_t pcfich_re =
        mmse::detail::build_pcfich_re_layout(pcfich_desc, pcfich_layout);
    fill_td_layout(pcfich_buffers, pcfich_layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                   {1.0F, 0.0F}, s0, s1);
    PcfichMmseInput pcfich_in{.grid = make_grid_view(pcfich_buffers),
                              .sfn_subframe = pcfich_desc.sfn_subframe,
                              .cell_id = pcfich_desc.cell_id,
                              .n_tx_ports = 2U,
                              .tx_mode = 2U};
    std::vector<float> pcfich_xre(pcfich_re), pcfich_xim(pcfich_re), pcfich_sinr(pcfich_re);
    std::vector<std::uint16_t> pcfich_re0(pcfich_re), pcfich_re1(pcfich_re);
    PcfichTdMmseOutputView pcfich_out{pcfich_xre.data(), pcfich_xim.data(), pcfich_sinr.data(),
                                      pcfich_re0.data(), pcfich_re1.data(), pcfich_re};
    PcfichTdMmseResult pcfich_meta{};
    expect_true(ctx.run_pcfich_td(pcfich_in, pcfich_out, pcfich_meta) == MmseStatus::kOk,
                "pcfich td cpu run");
    expect_true(pcfich_meta.n_symbols == pcfich_re && pcfich_meta.n_source_re == pcfich_re,
                "pcfich td metadata count");
    expect_near(pcfich_xre[0], s0.re, 1.0e-3F, "pcfich td symbol zero real");
    expect_near(pcfich_xim[1], s1.im, 1.0e-3F, "pcfich td symbol one imag");
    expect_true(pcfich_re0[0] == pcfich_layout.grid_indices[0] &&
                    pcfich_re1[1] == pcfich_layout.grid_indices[1],
                "pcfich td source pair metadata");
    const auto pcfich_backend =
        mmse::pcfich::make_backend_pcfich_td_equalized_indication(pcfich_meta, pcfich_out);
    const auto pcfich_llr =
        mmse::pcfich::make_backend_pcfich_td_descrambled_llr_indication(pcfich_backend);
    expect_true(pcfich_llr.llrs.size() == pcfich_re * 2U &&
                    pcfich_llr.re_grid_indices1 == pcfich_backend.re_grid_indices1,
                "pcfich td LLR DTO");
}

void test_gpu_cuda_pbch_and_pcfich_td_match_cpu() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "td special-channel cpu parity init");
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    gpu_config.sigma2_min = cpu_config.sigma2_min;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "td special-channel gpu parity init");
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};

    auto pbch_desc = make_pbch_desc();
    pbch_desc.n_tx_ports = 2U;
    pbch_desc.n_rx_ant = 1U;
    pbch_desc.tx_mode = 2U;
    GridBuffers pbch_buffers = make_zero_grid();
    fill_identity_channel(pbch_buffers, pbch_desc, 0.0F, 0.0F);
    mmse::detail::ReLayout pbch_layout{};
    const std::uint32_t pbch_count = mmse::detail::build_pbch_re_layout(pbch_desc, pbch_layout);
    fill_td_layout(pbch_buffers, pbch_layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                   {0.0F, 0.0F}, s0, s1);
    PbchMmseInput pbch_in{.grid = make_single_rx_grid_view(pbch_buffers),
                          .sfn_subframe = pbch_desc.sfn_subframe,
                          .cell_id = pbch_desc.cell_id,
                          .n_tx_ports = 2U,
                          .tx_mode = 2U};
    std::vector<float> pbch_cpu_re(pbch_count), pbch_cpu_im(pbch_count), pbch_cpu_sinr(pbch_count);
    std::vector<float> pbch_gpu_re(pbch_count), pbch_gpu_im(pbch_count), pbch_gpu_sinr(pbch_count);
    std::vector<std::uint16_t> pbch_cpu_re0(pbch_count), pbch_cpu_re1(pbch_count);
    std::vector<std::uint16_t> pbch_gpu_re0(pbch_count), pbch_gpu_re1(pbch_count);
    PbchTdMmseOutputView pbch_cpu_out{pbch_cpu_re.data(),   pbch_cpu_im.data(),
                                      pbch_cpu_sinr.data(), pbch_cpu_re0.data(),
                                      pbch_cpu_re1.data(),  pbch_count};
    PbchTdMmseOutputView pbch_gpu_out{pbch_gpu_re.data(),   pbch_gpu_im.data(),
                                      pbch_gpu_sinr.data(), pbch_gpu_re0.data(),
                                      pbch_gpu_re1.data(),  pbch_count};
    PbchTdMmseResult pbch_cpu_meta{}, pbch_gpu_meta{};
    expect_true(cpu.run_pbch_td(pbch_in, pbch_cpu_out, pbch_cpu_meta) == MmseStatus::kOk,
                "pbch td cpu parity run");
    expect_true(gpu.run_pbch_td(pbch_in, pbch_gpu_out, pbch_gpu_meta) == MmseStatus::kOk,
                "pbch td gpu parity run");
    expect_near(pbch_gpu_meta.sigma2, pbch_cpu_meta.sigma2, 1.0e-5F, "pbch td sigma2 parity");
    for (std::uint32_t i = 0U; i < pbch_count; ++i) {
        expect_near(pbch_gpu_re[i], pbch_cpu_re[i], 1.0e-4F, "pbch td gpu xhat real");
        expect_near(pbch_gpu_im[i], pbch_cpu_im[i], 1.0e-4F, "pbch td gpu xhat imag");
        expect_near(pbch_gpu_sinr[i], pbch_cpu_sinr[i], 1.0e-3F, "pbch td gpu sinr");
        expect_true(pbch_gpu_re0[i] == pbch_cpu_re0[i] && pbch_gpu_re1[i] == pbch_cpu_re1[i],
                    "pbch td gpu source pair");
    }

    auto pcfich_desc = make_pcfich_desc();
    pcfich_desc.n_tx_ports = 2U;
    pcfich_desc.tx_mode = 2U;
    GridBuffers pcfich_buffers = make_zero_grid();
    fill_identity_channel(pcfich_buffers, pcfich_desc, 0.0F, 0.0F);
    mmse::detail::ReLayout pcfich_layout{};
    const std::uint32_t pcfich_count =
        mmse::detail::build_pcfich_re_layout(pcfich_desc, pcfich_layout);
    fill_td_layout(pcfich_buffers, pcfich_layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                   {1.0F, 0.0F}, s0, s1);
    PcfichMmseInput pcfich_in{.grid = make_grid_view(pcfich_buffers),
                              .sfn_subframe = pcfich_desc.sfn_subframe,
                              .cell_id = pcfich_desc.cell_id,
                              .n_tx_ports = 2U,
                              .tx_mode = 2U};
    std::vector<float> pcfich_cpu_re(pcfich_count), pcfich_cpu_im(pcfich_count),
        pcfich_cpu_sinr(pcfich_count), pcfich_gpu_re(pcfich_count), pcfich_gpu_im(pcfich_count),
        pcfich_gpu_sinr(pcfich_count);
    std::vector<std::uint16_t> pcfich_cpu_re0(pcfich_count), pcfich_cpu_re1(pcfich_count),
        pcfich_gpu_re0(pcfich_count), pcfich_gpu_re1(pcfich_count);
    PcfichTdMmseOutputView pcfich_cpu_out{pcfich_cpu_re.data(),   pcfich_cpu_im.data(),
                                          pcfich_cpu_sinr.data(), pcfich_cpu_re0.data(),
                                          pcfich_cpu_re1.data(),  pcfich_count};
    PcfichTdMmseOutputView pcfich_gpu_out{pcfich_gpu_re.data(),   pcfich_gpu_im.data(),
                                          pcfich_gpu_sinr.data(), pcfich_gpu_re0.data(),
                                          pcfich_gpu_re1.data(),  pcfich_count};
    PcfichTdMmseResult pcfich_cpu_meta{}, pcfich_gpu_meta{};
    expect_true(cpu.run_pcfich_td(pcfich_in, pcfich_cpu_out, pcfich_cpu_meta) == MmseStatus::kOk,
                "pcfich td cpu parity run");
    expect_true(gpu.run_pcfich_td(pcfich_in, pcfich_gpu_out, pcfich_gpu_meta) == MmseStatus::kOk,
                "pcfich td gpu parity run");
    expect_near(pcfich_gpu_meta.sigma2, pcfich_cpu_meta.sigma2, 1.0e-5F, "pcfich td sigma2 parity");
    for (std::uint32_t i = 0U; i < pcfich_count; ++i) {
        expect_near(pcfich_gpu_re[i], pcfich_cpu_re[i], 1.0e-4F, "pcfich td gpu xhat real");
        expect_near(pcfich_gpu_im[i], pcfich_cpu_im[i], 1.0e-4F, "pcfich td gpu xhat imag");
        expect_near(pcfich_gpu_sinr[i], pcfich_cpu_sinr[i], 1.0e-3F, "pcfich td gpu sinr");
        expect_true(pcfich_gpu_re0[i] == pcfich_cpu_re0[i] &&
                        pcfich_gpu_re1[i] == pcfich_cpu_re1[i],
                    "pcfich td gpu source pair");
    }
#endif
}

void test_cpu_shared_estimate_reuses_once_per_subframe() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    expect_true(ctx.init(config) == MmseStatus::kOk, "shared-estimate cpu init");

    GridBuffers buffers = make_zero_grid();
    auto pdsch_desc = make_fullband_desc();
    pdsch_desc.sfn_subframe = 0U;
    auto pbch_in = PbchMmseInput{.grid = make_grid_view(buffers),
                                 .sfn_subframe = 0U,
                                 .cell_id = pdsch_desc.cell_id,
                                 .n_tx_ports = 1U,
                                 .tx_mode = 1U};
    auto pcfich_in = PcfichMmseInput{.grid = make_grid_view(buffers),
                                     .sfn_subframe = 0U,
                                     .cell_id = pdsch_desc.cell_id,
                                     .n_tx_ports = 1U,
                                     .tx_mode = 1U};
    auto pdcch_desc = make_pdcch_desc();
    pdcch_desc.sfn_subframe = 0U;
    PdcchMmseInput pdcch_in{};
    pdcch_in.grid = make_grid_view(buffers);
    pdcch_in.sfn_subframe = pdcch_desc.sfn_subframe;
    pdcch_in.cell_id = pdcch_desc.cell_id;
    pdcch_in.n_tx_ports = 1U;
    pdcch_in.tx_mode = 1U;
    pdcch_in.control_symbol_count = pdcch_desc.control_symbol_count;
    pdcch_in.n_prb = pdcch_desc.n_prb;
    pdcch_in.prb_bitmap = pdcch_desc.prb_bitmap;
    pdcch_in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                                 .subframe = 0U,
                                 .ul_dl_config = 0U,
                                 .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

    std::vector<float> pdsch_re(40000U);
    std::vector<float> pdsch_im(40000U);
    std::vector<float> pdsch_sinr(40000U);
    EqualizerOutputView pdsch_out{pdsch_re.data(), pdsch_im.data(), pdsch_sinr.data(), 20000U};

    std::vector<float> pbch_re(256U), pbch_im(256U), pbch_sinr(256U);
    std::vector<std::uint16_t> pbch_idx(256U);
    PbchMmseOutputView pbch_out{pbch_re.data(),  pbch_im.data(), pbch_sinr.data(),
                                pbch_idx.data(), 256U,           256U};
    PbchMmseResult pbch_meta{};

    std::vector<float> pcfich_re(32U), pcfich_im(32U), pcfich_sinr(32U);
    std::vector<std::uint16_t> pcfich_idx(32U);
    PcfichMmseOutputView pcfich_out{
        pcfich_re.data(), pcfich_im.data(), pcfich_sinr.data(), pcfich_idx.data(), 32U, 32U};
    PcfichMmseResult pcfich_meta{};

    std::vector<float> pdcch_re(4000U), pdcch_im(4000U), pdcch_sinr(4000U);
    std::vector<std::uint16_t> pdcch_idx(4000U);
    PdcchMmseOutputView pdcch_out{pdcch_re.data(),  pdcch_im.data(), pdcch_sinr.data(),
                                  pdcch_idx.data(), 4000U,           4000U};
    PdcchMmseResult pdcch_meta{};

    mmse::detail::debug_reset_estimate_channel_call_count();
    expect_true(ctx.run_pbch(pbch_in, pbch_out, pbch_meta) == MmseStatus::kOk, "pbch run");
    expect_true(ctx.run_pcfich(pcfich_in, pcfich_out, pcfich_meta) == MmseStatus::kOk,
                "pcfich run");
    expect_true(ctx.run_pdcch(pdcch_in, pdcch_out, pdcch_meta) == MmseStatus::kOk, "pdcch run");
    expect_true(ctx.run(make_grid_view(buffers), pdsch_desc, pdsch_out) == MmseStatus::kOk,
                "pdsch run");
    expect_true(mmse::detail::debug_get_estimate_channel_call_count() == 2U,
                "estimate should be reused only for matching channel-estimation contracts");
}

void test_gpu_shared_estimate_skips_second_estimate_path() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kCuda;
    expect_true(gpu.init(config) == MmseStatus::kOk, "gpu shared-estimate init");

    GridBuffers buffers = make_zero_grid();
    auto pdsch_desc = make_fullband_desc();
    pdsch_desc.sfn_subframe = 0U;
    fill_identity_channel(buffers, pdsch_desc, 1.0F, 1.0F);
    auto grid = make_grid_view(buffers);

    PbchMmseInput pbch_in{};
    pbch_in.grid = grid;
    pbch_in.sfn_subframe = 0U;
    pbch_in.cell_id = pdsch_desc.cell_id;
    pbch_in.n_tx_ports = 1U;
    pbch_in.tx_mode = 1U;

    auto pdcch_desc = make_pdcch_desc();
    pdcch_desc.sfn_subframe = 0U;
    PdcchMmseInput pdcch_in{};
    pdcch_in.grid = grid;
    pdcch_in.sfn_subframe = 0U;
    pdcch_in.cell_id = pdcch_desc.cell_id;
    pdcch_in.n_tx_ports = 1U;
    pdcch_in.tx_mode = 1U;
    pdcch_in.control_symbol_count = pdcch_desc.control_symbol_count;
    pdcch_in.n_prb = pdcch_desc.n_prb;
    pdcch_in.prb_bitmap = pdcch_desc.prb_bitmap;
    pdcch_in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                                 .subframe = 0U,
                                 .ul_dl_config = 0U,
                                 .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

    std::vector<float> pbch_re(256U), pbch_im(256U), pbch_sinr(256U);
    std::vector<std::uint16_t> pbch_idx(256U);
    PbchMmseOutputView pbch_out{pbch_re.data(),  pbch_im.data(), pbch_sinr.data(),
                                pbch_idx.data(), 256U,           256U};
    PbchMmseResult pbch_meta{};

    std::vector<float> pdcch_re(4000U), pdcch_im(4000U), pdcch_sinr(4000U);
    std::vector<std::uint16_t> pdcch_idx(4000U);
    PdcchMmseOutputView pdcch_out{pdcch_re.data(),  pdcch_im.data(), pdcch_sinr.data(),
                                  pdcch_idx.data(), 4000U,           4000U};
    PdcchMmseResult pdcch_meta{};

    expect_true(gpu.run_pbch(pbch_in, pbch_out, pbch_meta) == MmseStatus::kOk, "gpu pbch run");
    const MmseGpuHostProfileSnapshot first = gpu.last_host_profile();
    expect_true(gpu.run_pdcch(pdcch_in, pdcch_out, pdcch_meta) == MmseStatus::kOk, "gpu pdcch run");
    const MmseGpuHostProfileSnapshot second = gpu.last_host_profile();

    expect_true(first.estimate_launch_us > 0.0, "first run should estimate");
    expect_true(second.estimate_launch_us < 1.0, "second run should skip estimate launch");
    expect_true(second.grid_h2d_us < 1.0, "second run should skip grid h2d");
    expect_true(second.sigma2_d2h_us < 1.0, "second run should skip sigma2 d2h");
#endif
}

void test_pdcch_td_cpu_run_recovers_qpsk_symbols() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    config.sigma2_min = 1.0e-5F;
    config.det_floor = 1.0e-6F;
    config.g_min = 1.0e-4F;
    config.gamma_max = 1.0e4F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);

    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    if (n_source_re != 3168U) {
        throw TestFailure{"two-port pdcch RE count actual=" + std::to_string(n_source_re)};
    }
    expect_true((n_source_re & 1U) == 0U, "two-port pdcch RE count must be even");
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};
    fill_pdcch_td_layout(buffers, desc, layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                         {1.0F, 0.0F}, s0, s1);

    PdcchMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.control_symbol_count = desc.control_symbol_count;
    in.n_prb = desc.n_prb;
    in.prb_bitmap = desc.prb_bitmap;
    in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                           .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                           .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

    std::vector<float> xre(n_source_re);
    std::vector<float> xim(n_source_re);
    std::vector<float> sinr(n_source_re);
    std::vector<std::uint16_t> grid0(n_source_re);
    std::vector<std::uint16_t> grid1(n_source_re);
    PdcchTdMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices0 = grid0.data();
    out.re_grid_indices1 = grid1.data();
    out.capacity_symbols = n_source_re;

    PdcchTdMmseResult meta{};
    const MmseStatus status = ctx.run_pdcch_td(in, out, meta);
    if (status != MmseStatus::kOk) {
        throw TestFailure{"pdcch td cpu run status=" + std::string(to_string(status))};
    }

    expect_true(meta.n_source_re == n_source_re, "td source re count");
    expect_true(meta.n_symbols == n_source_re, "td soft-symbol count");
    expect_true(grid0[0] == layout.grid_indices[0] && grid1[0] == layout.grid_indices[1],
                "td first symbol pair indices");
    expect_true(grid0[1] == layout.grid_indices[0] && grid1[1] == layout.grid_indices[1],
                "td second symbol pair indices");
    expect_near(xre[0], s0.re, 1.0e-3F, "td symbol0 real");
    expect_near(xim[0], s0.im, 1.0e-3F, "td symbol0 imag");
    expect_near(xre[1], s1.re, 1.0e-3F, "td symbol1 real");
    expect_near(xim[1], s1.im, 1.0e-3F, "td symbol1 imag");
    expect_true(sinr[0] > 10.0F, "td symbol0 sinr");
    expect_true(sinr[1] > 10.0F, "td symbol1 sinr");
}

void test_pdcch_td_scalar_demap_recovers_qpsk_symbols() {
    const Complex32 h00{1.0F, 0.0F};
    const Complex32 h01{0.0F, 0.0F};
    const Complex32 h10{0.0F, 0.0F};
    const Complex32 h11{1.0F, 0.0F};
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};

    const Complex32 y0_k0 =
        mmse::detail::cadd(mmse::detail::cmul(h00, s0), mmse::detail::cmul(h01, s1));
    const Complex32 y1_k0 =
        mmse::detail::cadd(mmse::detail::cmul(h10, s0), mmse::detail::cmul(h11, s1));
    const Complex32 y0_k1 = mmse::detail::csub(mmse::detail::cmul(h00, mmse::detail::cconj(s1)),
                                               mmse::detail::cmul(h01, mmse::detail::cconj(s0)));
    const Complex32 y1_k1 = mmse::detail::csub(mmse::detail::cmul(h10, mmse::detail::cconj(s1)),
                                               mmse::detail::cmul(h11, mmse::detail::cconj(s0)));

    const auto eq = mmse::detail::demap_pdcch_transmit_diversity_scalar(
        h00, h01, h10, h11, h00, h01, h10, h11, y0_k0, y1_k0, y0_k1, y1_k1, 1.0e-5F, 1.0e-6F,
        1.0e-4F, 1.0e4F);
    expect_near(eq.symbol0.xhat.re, s0.re, 1.0e-3F, "td scalar symbol0 real");
    expect_near(eq.symbol0.xhat.im, s0.im, 1.0e-3F, "td scalar symbol0 imag");
    expect_near(eq.symbol1.xhat.re, s1.re, 1.0e-3F, "td scalar symbol1 real");
    expect_near(eq.symbol1.xhat.im, s1.im, 1.0e-3F, "td scalar symbol1 imag");
}

void test_pdcch_run_rejects_two_port_contract() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    config.sigma2_min = 1.0e-5F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    GridBuffers buffers = make_zero_grid();
    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);

    PdcchMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.control_symbol_count = desc.control_symbol_count;
    in.n_prb = desc.n_prb;
    in.prb_bitmap = desc.prb_bitmap;
    in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                           .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                           .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

    std::vector<float> xre(4000U), xim(4000U), sinr(4000U);
    std::vector<std::uint16_t> grid_idx(4000U);
    PdcchMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices = grid_idx.data();
    out.capacity_re_per_layer = 4000U;
    out.capacity_re_metadata = 4000U;
    PdcchMmseResult meta{};

    expect_true(ctx.run_pdcch(in, out, meta) == MmseStatus::kUnsupportedConfig,
                "legacy run_pdcch contract should still reject two-port td");
}

void test_pdcch_module_api_returns_chain_metadata_and_re_indices() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    config.sigma2_min = 1.0e-5F;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    GridBuffers buffers = make_zero_grid();
    auto desc = make_pdcch_desc();
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);

    PdcchMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.control_symbol_count = desc.control_symbol_count;
    in.n_prb = desc.n_prb;
    in.prb_bitmap = desc.prb_bitmap;
    in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                           .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                           .ul_dl_config = 0U,
                           .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    in.control_re_exclusion_masks = desc.control_re_exclusion_masks;
    in.chain.request_id = 77U;
    in.chain.candidate_id = 5U;
    in.chain.first_cce = 2U;
    in.chain.aggregation_level = 4U;

    const std::uint32_t cap = 8000U;
    std::vector<float> xre(cap);
    std::vector<float> xim(cap);
    std::vector<float> sinr(cap);
    std::vector<std::uint16_t> indices(cap);
    PdcchMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices = indices.data();
    out.capacity_re_per_layer = cap;
    out.capacity_re_metadata = cap;

    PdcchMmseResult meta{};
    const MmseStatus status = ctx.run_pdcch(in, out, meta);
    if (status != MmseStatus::kOk) {
        throw TestFailure{"pdcch module api run status=" + std::string(to_string(status))};
    }
    expect_true(meta.n_re == 3168U, "pdcch module api RE count");
    expect_true(meta.chain.request_id == in.chain.request_id, "request id passthrough");
    expect_true(meta.chain.candidate_id == in.chain.candidate_id, "candidate id passthrough");
    expect_true(meta.chain.first_cce == in.chain.first_cce, "first cce passthrough");
    expect_true(meta.chain.aggregation_level == in.chain.aggregation_level,
                "aggregation passthrough");
    expect_true(meta.control_symbol_count == in.control_symbol_count, "cfi passthrough");
    expect_true(meta.mod_order == 2U, "pdcch api qpsk");
    expect_true(indices[0] < kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz,
                "grid index must be valid");
}

void test_pdcch_mmse_input_validator() {
    GridBuffers buffers = make_zero_grid();
    auto desc = make_pdcch_desc();
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);

    PdcchMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.control_symbol_count = desc.control_symbol_count;
    in.n_prb = desc.n_prb;
    in.prb_bitmap = desc.prb_bitmap;
    in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                           .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                           .ul_dl_config = 0U,
                           .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

    expect_true(mmse::pdcch::validate_pdcch_mmse_input(in) == MmseStatus::kOk,
                "pdcch input validator should accept consistent input");

    expect_true(mmse::pdcch::validate_lte_control_subframe_context(
                    in.control_subframe, in.sfn_subframe) == MmseStatus::kOk,
                "lte control subframe validator should accept consistent input");

    in.control_symbol_count = 0U;
    expect_true(mmse::pdcch::validate_pdcch_mmse_input(in) == MmseStatus::kInvalidArgument,
                "pdcch input validator should reject invalid control symbol count");
}

void test_pdcch_run_rejects_inconsistent_control_subframe() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    GridBuffers buffers = make_zero_grid();
    auto desc = make_pdcch_desc();
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);

    PdcchMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.control_symbol_count = desc.control_symbol_count;
    in.n_prb = desc.n_prb;
    in.prb_bitmap = desc.prb_bitmap;
    in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                           .subframe = 9U,
                           .ul_dl_config = 0U,
                           .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

    const std::uint32_t cap = 8000U;
    std::vector<float> xre(cap);
    std::vector<float> xim(cap);
    std::vector<float> sinr(cap);
    std::vector<std::uint16_t> indices(cap);
    PdcchMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices = indices.data();
    out.capacity_re_per_layer = cap;
    out.capacity_re_metadata = cap;

    PdcchMmseResult meta{};
    expect_true(ctx.run_pdcch(in, out, meta) == MmseStatus::kInvalidArgument,
                "run_pdcch should reject mismatched control_subframe.subframe");

    in.control_subframe.subframe = static_cast<std::uint8_t>(in.sfn_subframe % 10U);
    in.control_subframe.ul_dl_config = 1U;
    expect_true(ctx.run_pdcch(in, out, meta) == MmseStatus::kInvalidArgument,
                "run_pdcch should reject fdd control_subframe with ul_dl_config");
}

void test_pdcch_helper_mask_and_grid_index_decode() {
    PdcchMmseInput in{};
    mmse::pdcch::clear_control_re_exclusion_masks(in);
    expect_true(!mmse::pdcch::is_control_re_excluded(in, 1U, 10U, 3U), "mask should start clear");
    mmse::pdcch::exclude_control_re(in, 1U, 10U, 3U);
    expect_true(mmse::pdcch::is_control_re_excluded(in, 1U, 10U, 3U),
                "mask helper should mark reserved RE");

    const std::uint16_t grid_index =
        static_cast<std::uint16_t>(2U * kLteNumSubcarriers20MHz + 123U);
    const auto coord = mmse::pdcch::decode_re_grid_index(grid_index);
    expect_true(coord.symbol == 2U, "decoded symbol");
    expect_true(coord.subcarrier == 123U, "decoded subcarrier");
    expect_true(coord.prb == 10U, "decoded prb");
    expect_true(coord.tone_in_prb == 3U, "decoded tone");
}

void test_pdcch_helper_apply_reserved_re_list() {
    PdcchMmseInput in{};
    const std::array reserved = {
        mmse::pdcch::ReservedControlRe{.symbol = 0U, .prb = 0U, .tone_in_prb = 1U},
        mmse::pdcch::ReservedControlRe{.symbol = 2U, .prb = 5U, .tone_in_prb = 11U},
    };

    mmse::pdcch::apply_reserved_control_re_list(in, reserved);
    expect_true(mmse::pdcch::is_control_re_excluded(in, 0U, 0U, 1U),
                "first reserved RE should be applied");
    expect_true(mmse::pdcch::is_control_re_excluded(in, 2U, 5U, 11U),
                "second reserved RE should be applied");
    expect_true(!mmse::pdcch::is_control_re_excluded(in, 0U, 0U, 0U),
                "non-reserved RE should remain clear");
}

void test_pdcch_control_region_builds_regs_and_cces() {
    constexpr std::uint32_t kRegCount = 18U;
    std::array<std::uint16_t, kRegCount * mmse::pdcch::kPdcchRePerReg> grid_indices{};
    for (std::uint32_t re = 0U; re < grid_indices.size(); ++re) {
        grid_indices[re] = static_cast<std::uint16_t>(re + 100U);
    }

    mmse::pdcch::PdcchControlRegion control_region{};
    expect_true(mmse::pdcch::build_pdcch_control_region(
                    17U, 1U, grid_indices.data(), static_cast<std::uint32_t>(grid_indices.size()),
                    control_region) == MmseStatus::kOk,
                "pdcch control-region builder should accept ordered REs");
    expect_true(control_region.cell_id == 17U, "pdcch control-region cell id");
    expect_true(control_region.control_symbol_count == 1U, "pdcch control-region CFI");
    expect_true(control_region.n_source_re == grid_indices.size(),
                "pdcch control-region source RE count");
    expect_true(control_region.regs.size() == kRegCount, "pdcch control-region REG count");
    expect_true(control_region.cces.size() == 2U, "pdcch control-region CCE count");
    expect_true(control_region.n_unassigned_reg == 0U, "pdcch control-region complete CCEs");
    expect_true(control_region.regs[1].source_re_indices[0] == 4U,
                "pdcch control-region REG source offset");
    expect_true(control_region.regs[1].grid_indices[3] == 107U,
                "pdcch control-region REG grid index");
    expect_true(control_region.cces[1].reg_indices[0] == 9U &&
                    control_region.cces[1].reg_indices[8] == 17U,
                "pdcch control-region CCE REG range");
}

void test_pdcch_control_region_rejects_invalid_re_indices() {
    std::array<std::uint16_t, 8> unordered = {0U, 1U, 2U, 3U, 3U, 5U, 6U, 7U};
    mmse::pdcch::PdcchControlRegion control_region{};
    expect_true(mmse::pdcch::build_pdcch_control_region(
                    0U, 1U, unordered.data(), static_cast<std::uint32_t>(unordered.size()),
                    control_region) == MmseStatus::kInvalidArgument,
                "pdcch control-region builder should reject duplicate RE indices");

    std::array<std::uint16_t, 8> outside_control_region = {
        0U, 1U, 2U, 3U, 4U, 5U, 6U, static_cast<std::uint16_t>(kLteNumSubcarriers20MHz)};
    expect_true(mmse::pdcch::build_pdcch_control_region(
                    0U, 1U, outside_control_region.data(),
                    static_cast<std::uint32_t>(outside_control_region.size()),
                    control_region) == MmseStatus::kInvalidArgument,
                "pdcch control-region builder should reject REs outside the control region");
}

void test_pdcch_common_search_candidates_follow_lte_levels() {
    constexpr std::uint32_t kRegCount = 810U;
    std::array<std::uint16_t, kRegCount * mmse::pdcch::kPdcchRePerReg> grid_indices{};
    for (std::uint32_t re = 0U; re < grid_indices.size(); ++re) {
        grid_indices[re] = static_cast<std::uint16_t>(re);
    }

    mmse::pdcch::PdcchControlRegion control_region{};
    expect_true(mmse::pdcch::build_pdcch_control_region(
                    0U, 3U, grid_indices.data(), static_cast<std::uint32_t>(grid_indices.size()),
                    control_region) == MmseStatus::kOk,
                "pdcch common search test control-region build");
    expect_true(control_region.cces.size() == 90U, "pdcch common search CCE count");

    std::vector<mmse::pdcch::PdcchCommonSearchCandidate> candidates;
    mmse::pdcch::build_pdcch_common_search_candidates(control_region, candidates);
    expect_true(candidates.size() == 6U, "pdcch common search candidate count");
    expect_true(candidates[0].candidate_id == 0U && candidates[0].first_cce == 0U &&
                    candidates[0].aggregation_level == 4U &&
                    candidates[0].encoded_bit_count == 288U,
                "pdcch common search first aggregation-four candidate");
    expect_true(candidates[3].candidate_id == 3U && candidates[3].first_cce == 12U &&
                    candidates[3].aggregation_level == 4U &&
                    candidates[3].encoded_bit_count == 288U,
                "pdcch common search last aggregation-four candidate");
    expect_true(candidates[4].candidate_id == 4U && candidates[4].first_cce == 0U &&
                    candidates[4].aggregation_level == 8U &&
                    candidates[4].encoded_bit_count == 576U,
                "pdcch common search first aggregation-eight candidate");
    expect_true(candidates[5].candidate_id == 5U && candidates[5].first_cce == 8U &&
                    candidates[5].aggregation_level == 8U &&
                    candidates[5].encoded_bit_count == 576U,
                "pdcch common search last aggregation-eight candidate");
}

void test_pdcch_ue_specific_search_candidates_follow_lte_levels() {
    constexpr std::uint32_t kRegCount = 810U;
    std::array<std::uint16_t, kRegCount * mmse::pdcch::kPdcchRePerReg> grid_indices{};
    for (std::uint32_t re = 0U; re < grid_indices.size(); ++re) {
        grid_indices[re] = static_cast<std::uint16_t>(re);
    }

    mmse::pdcch::PdcchControlRegion control_region{};
    expect_true(mmse::pdcch::build_pdcch_control_region(
                    0U, 3U, grid_indices.data(), static_cast<std::uint32_t>(grid_indices.size()),
                    control_region) == MmseStatus::kOk &&
                    control_region.cces.size() == 90U,
                "pdcch UE-specific search control-region build");

    mmse::pdcch::PdcchUeSpecificSearchConfig config{};
    config.rntis = {0x1234U};
    std::vector<mmse::pdcch::PdcchUeSpecificSearchCandidate> candidates{};
    expect_true(mmse::pdcch::build_pdcch_ue_specific_search_candidates(
                    control_region, 0U, config, candidates) == MmseStatus::kOk &&
                    candidates.size() == 16U,
                "pdcch UE-specific search should build all LTE levels");
    constexpr std::array<std::uint16_t, 16> kExpectedFirstCceAtSf0 = {
        73U, 74U, 75U, 76U, 77U, 78U, 56U, 58U, 60U, 62U, 64U, 66U, 36U, 40U, 72U, 80U,
    };
    constexpr std::array<std::uint8_t, 16> kExpectedLevels = {
        1U, 1U, 1U, 1U, 1U, 1U, 2U, 2U, 2U, 2U, 2U, 2U, 4U, 4U, 8U, 8U,
    };
    for (std::uint32_t candidate = 0U; candidate < candidates.size(); ++candidate) {
        expect_true(candidates[candidate].rnti == 0x1234U &&
                        candidates[candidate].search_candidate.candidate_id == candidate &&
                        candidates[candidate].search_candidate.first_cce ==
                            kExpectedFirstCceAtSf0[candidate] &&
                        candidates[candidate].search_candidate.aggregation_level ==
                            kExpectedLevels[candidate] &&
                        candidates[candidate].search_candidate.encoded_bit_count ==
                            72U * kExpectedLevels[candidate],
                    "pdcch UE-specific candidate ordering and metadata");
    }

    expect_true(mmse::pdcch::build_pdcch_ue_specific_search_candidates(
                    control_region, 5U, config, candidates) == MmseStatus::kOk &&
                    candidates[0].search_candidate.first_cce == 27U &&
                    candidates[6].search_candidate.first_cce == 54U &&
                    candidates[12].search_candidate.first_cce == 28U &&
                    candidates[14].search_candidate.first_cce == 56U,
                "pdcch UE-specific search must use the subframe-specific Yk");
    std::vector<mmse::pdcch::PdcchUeSpecificSearchCandidate> next_frame_candidates{};
    expect_true(mmse::pdcch::build_pdcch_ue_specific_search_candidates(
                    control_region, 15U, config, next_frame_candidates) == MmseStatus::kOk &&
                    next_frame_candidates.size() == candidates.size() &&
                    std::equal(candidates.begin(), candidates.end(), next_frame_candidates.begin(),
                               [](const auto& lhs, const auto& rhs) {
                                   return lhs.rnti == rhs.rnti &&
                                          lhs.search_candidate.first_cce ==
                                              rhs.search_candidate.first_cce &&
                                          lhs.search_candidate.aggregation_level ==
                                              rhs.search_candidate.aggregation_level;
                               }),
                "pdcch UE-specific search must repeat candidate positions each radio frame");

    constexpr std::uint32_t kSmallRegCount = 72U;
    std::array<std::uint16_t, kSmallRegCount * mmse::pdcch::kPdcchRePerReg> small_grid_indices{};
    for (std::uint32_t re = 0U; re < small_grid_indices.size(); ++re) {
        small_grid_indices[re] = static_cast<std::uint16_t>(re);
    }
    mmse::pdcch::PdcchControlRegion small_control_region{};
    expect_true(mmse::pdcch::build_pdcch_control_region(
                    0U, 1U, small_grid_indices.data(),
                    static_cast<std::uint32_t>(small_grid_indices.size()),
                    small_control_region) == MmseStatus::kOk &&
                    small_control_region.cces.size() == 8U,
                "pdcch small UE-specific search control-region build");
    expect_true(mmse::pdcch::build_pdcch_ue_specific_search_candidates(
                    small_control_region, 0U, config, candidates) == MmseStatus::kOk &&
                    candidates.size() == 13U,
                "pdcch UE-specific search must truncate candidate levels by available CCEs");
    constexpr std::array<std::uint16_t, 13> kExpectedFirstCceAtSmallRegion = {
        5U, 6U, 7U, 0U, 1U, 2U, 2U, 4U, 6U, 0U, 4U, 0U, 0U,
    };
    for (std::uint32_t candidate = 0U; candidate < candidates.size(); ++candidate) {
        expect_true(candidates[candidate].search_candidate.first_cce ==
                        kExpectedFirstCceAtSmallRegion[candidate],
                    "pdcch UE-specific search must retain standard small-region placement");
    }

    config.rntis = {};
    expect_true(mmse::pdcch::build_pdcch_ue_specific_search_candidates(
                    control_region, 0U, config, candidates) == MmseStatus::kInvalidArgument,
                "pdcch UE-specific search must reject an empty RNTI list");
    config.rntis = {0x1234U, 0x1234U};
    expect_true(mmse::pdcch::build_pdcch_ue_specific_search_candidates(
                    control_region, 0U, config, candidates) == MmseStatus::kInvalidArgument,
                "pdcch UE-specific search must reject duplicate RNTIs");
    config.rntis = {mmse::pdcch::kSiRnti};
    expect_true(mmse::pdcch::build_pdcch_ue_specific_search_candidates(
                    control_region, 0U, config, candidates) == MmseStatus::kInvalidArgument,
                "pdcch UE-specific search must reserve SI-RNTI for common search");
}

void test_pdcch_common_search_candidate_llrs_follow_cce_offsets() {
    constexpr std::uint32_t kCceCount = 8U;
    constexpr std::uint32_t kReCount =
        kCceCount * mmse::pdcch::kPdcchRegsPerCce * mmse::pdcch::kPdcchRePerReg;
    mmse::pdcch::BackendPdcchEqualizedIndication backend{};
    backend.sfn_subframe = 7U;
    backend.cell_id = 19U;
    backend.n_layers = 1U;
    backend.control_symbol_count = 1U;
    backend.mod_order = 2U;
    backend.chain.request_id = 880U;
    backend.x_hat_re.resize(kReCount);
    backend.x_hat_im.resize(kReCount);
    backend.sinr.resize(kReCount, 1.0F);
    backend.re_grid_indices.resize(kReCount);
    for (std::uint32_t re = 0U; re < kReCount; ++re) {
        backend.x_hat_re[re] = static_cast<float>(re + 1U) * 0.01F;
        backend.x_hat_im[re] = -static_cast<float>(re + 1U) * 0.02F;
        backend.re_grid_indices[re] = static_cast<std::uint16_t>(re);
    }

    std::vector<float> expected_full_llrs{};
    expect_true(mmse::lte::build_max_log_descrambled_llrs(
                    backend.x_hat_re.data(), backend.x_hat_im.data(), backend.sinr.data(), kReCount,
                    kReCount, 1U, 2U,
                    mmse::lte::pdcch_c_init(backend.cell_id, backend.sfn_subframe),
                    expected_full_llrs) == MmseStatus::kOk,
                "pdcch candidate test full LLR build");

    std::vector<mmse::pdcch::PdcchCandidateLlr> candidate_llrs;
    expect_true(mmse::pdcch::build_pdcch_common_search_candidate_llrs(backend, candidate_llrs) ==
                    MmseStatus::kOk,
                "pdcch candidate LLR builder should succeed");
    expect_true(candidate_llrs.size() == 3U, "pdcch candidate LLR count");
    expect_true(candidate_llrs[1].chain.request_id == backend.chain.request_id &&
                    candidate_llrs[1].chain.candidate_id == 1U &&
                    candidate_llrs[1].chain.first_cce == 4U &&
                    candidate_llrs[1].chain.aggregation_level == 4U &&
                    candidate_llrs[1].encoded_bit_count == 288U,
                "pdcch aggregation-four candidate metadata");
    expect_true(candidate_llrs[1].llrs.size() == 288U,
                "pdcch aggregation-four candidate LLR count");
    for (std::uint32_t llr = 0U; llr < candidate_llrs[1].llrs.size(); ++llr) {
        expect_near(candidate_llrs[1].llrs[llr], expected_full_llrs[288U + llr], 1.0e-5F,
                    "pdcch aggregation-four candidate LLR slice");
    }
    expect_true(candidate_llrs[2].chain.candidate_id == 2U &&
                    candidate_llrs[2].chain.first_cce == 0U &&
                    candidate_llrs[2].chain.aggregation_level == 8U &&
                    candidate_llrs[2].encoded_bit_count == 576U,
                "pdcch aggregation-eight candidate metadata");
    expect_true(candidate_llrs[2].llrs == expected_full_llrs,
                "pdcch aggregation-eight candidate must retain full control-region LLRs");
}

void test_pdcch_common_search_candidate_llrs_reject_invalid_backend() {
    mmse::pdcch::BackendPdcchEqualizedIndication backend{};
    backend.n_layers = 2U;
    backend.mod_order = 2U;
    backend.x_hat_re = {0.0F};
    backend.x_hat_im = {0.0F};
    backend.sinr = {1.0F};
    backend.re_grid_indices = {0U};
    std::vector<mmse::pdcch::PdcchCandidateLlr> candidate_llrs;
    expect_true(mmse::pdcch::build_pdcch_common_search_candidate_llrs(backend, candidate_llrs) ==
                    MmseStatus::kInvalidArgument,
                "pdcch candidate LLR builder should reject unsupported PDCCH layers");
}

void test_pdcch_convolutional_rate_recovery_combines_repetitions() {
    constexpr std::uint32_t kDciPayloadBits = 16U;
    constexpr std::uint32_t kCodewordBits = kDciPayloadBits + mmse::pdcch::kPdcchCrcBitCount;
    std::vector<float> expected_convolutional_llrs(
        kCodewordBits * mmse::pdcch::kPdcchConvolutionalCodeRate, 0.0F);
    for (std::uint32_t i = 0U; i < expected_convolutional_llrs.size(); ++i) {
        expected_convolutional_llrs[i] = static_cast<float>(i + 1U) * 0.25F;
    }

    mmse::pdcch::PdcchCandidateLlr candidate{};
    candidate.sfn_subframe = 7U;
    candidate.cell_id = 19U;
    candidate.chain = {
        .request_id = 551U, .candidate_id = 2U, .first_cce = 8U, .aggregation_level = 4U};
    candidate.encoded_bit_count = 288U;
    candidate.llrs = reference_pdcch_convolutional_rate_match(expected_convolutional_llrs,
                                                              candidate.encoded_bit_count);

    mmse::pdcch::PdcchRateRecoveredLlr recovered{};
    expect_true(mmse::pdcch::recover_pdcch_convolutional_rate_matched_llrs(
                    candidate, kDciPayloadBits, recovered) == MmseStatus::kOk,
                "pdcch convolutional rate recovery should succeed");
    expect_true(recovered.sfn_subframe == candidate.sfn_subframe &&
                    recovered.cell_id == candidate.cell_id &&
                    recovered.chain.request_id == candidate.chain.request_id &&
                    recovered.chain.candidate_id == candidate.chain.candidate_id &&
                    recovered.codeword_bit_count == kCodewordBits,
                "pdcch rate recovery metadata passthrough");
    expect_true(recovered.convolutional_llrs.size() == expected_convolutional_llrs.size(),
                "pdcch rate recovery output length");
    for (std::uint32_t i = 0U; i < recovered.convolutional_llrs.size(); ++i) {
        expect_near(recovered.convolutional_llrs[i], 3.0F * expected_convolutional_llrs[i], 1.0e-5F,
                    "pdcch rate recovery soft combine");
    }
}

void test_pdcch_convolutional_rate_recovery_handles_dummy_bits() {
    constexpr std::uint32_t kDciPayloadBits = 17U;
    constexpr std::uint32_t kCodewordBits = kDciPayloadBits + mmse::pdcch::kPdcchCrcBitCount;
    std::vector<float> convolutional_llrs(kCodewordBits * mmse::pdcch::kPdcchConvolutionalCodeRate,
                                          1.0F);

    mmse::pdcch::PdcchCandidateLlr candidate{};
    candidate.chain.aggregation_level = 4U;
    candidate.encoded_bit_count = 288U;
    candidate.llrs =
        reference_pdcch_convolutional_rate_match(convolutional_llrs, candidate.encoded_bit_count);

    mmse::pdcch::PdcchRateRecoveredLlr recovered{};
    expect_true(mmse::pdcch::recover_pdcch_convolutional_rate_matched_llrs(
                    candidate, kDciPayloadBits, recovered) == MmseStatus::kOk,
                "pdcch rate recovery with dummy bits should succeed");
    for (const float llr : recovered.convolutional_llrs) {
        expect_true(llr == 2.0F || llr == 3.0F,
                    "pdcch dummy-bit recovery must only combine observed soft bits");
    }
}

void test_pdcch_convolutional_rate_recovery_rejects_invalid_candidate() {
    mmse::pdcch::PdcchCandidateLlr candidate{};
    candidate.chain.aggregation_level = 4U;
    candidate.encoded_bit_count = 288U;
    candidate.llrs.assign(287U, 1.0F);
    mmse::pdcch::PdcchRateRecoveredLlr recovered{};
    expect_true(mmse::pdcch::recover_pdcch_convolutional_rate_matched_llrs(
                    candidate, 16U, recovered) == MmseStatus::kInvalidArgument,
                "pdcch rate recovery should reject inconsistent candidate LLR size");
}

void test_pdcch_tail_biting_decoder_adapter_contract() {
    constexpr std::uint32_t kDciPayloadBits = 16U;
    constexpr std::uint32_t kCodewordBits = kDciPayloadBits + mmse::pdcch::kPdcchCrcBitCount;
    mmse::pdcch::PdcchRateRecoveredLlr recovered{};
    recovered.codeword_bit_count = kCodewordBits;
    recovered.convolutional_llrs.resize(kCodewordBits * mmse::pdcch::kPdcchConvolutionalCodeRate);
    for (std::uint32_t i = 0U; i < recovered.convolutional_llrs.size(); ++i) {
        recovered.convolutional_llrs[i] = static_cast<float>(i) * 0.5F;
    }

    TailBitingDecoderProbe probe{.expected_llrs = recovered.convolutional_llrs};
    const mmse::pdcch::PdcchTailBitingConvolutionalDecoder decoder{
        .context = &probe,
        .decode = probe_tail_biting_decoder,
    };
    std::vector<std::uint8_t> decoded_bits;
    expect_true(mmse::pdcch::invoke_pdcch_tail_biting_convolutional_decoder(
                    recovered, decoder, decoded_bits) == MmseStatus::kOk,
                "pdcch tail-biting adapter should succeed");
    expect_true(probe.call_count == 1U && decoded_bits.size() == kCodewordBits,
                "pdcch tail-biting adapter output size");
    for (std::uint32_t bit = 0U; bit < decoded_bits.size(); ++bit) {
        expect_true(decoded_bits[bit] == static_cast<std::uint8_t>(bit & 1U),
                    "pdcch tail-biting adapter bit output");
    }

    probe.return_status = MmseStatus::kInternalError;
    decoded_bits.assign(1U, 1U);
    expect_true(mmse::pdcch::invoke_pdcch_tail_biting_convolutional_decoder(
                    recovered, decoder, decoded_bits) == MmseStatus::kInternalError,
                "pdcch tail-biting adapter should propagate external decoder failure");
    expect_true(decoded_bits.empty(), "pdcch tail-biting adapter should clear failed output");
}

void test_pdcch_native_tail_biting_decoder_all_initial_states() {
    constexpr std::uint32_t kBitCount = 44U;
    for (std::uint32_t initial_state = 0U; initial_state < 64U; ++initial_state) {
        std::vector<std::uint8_t> bits(kBitCount, 0U);
        for (std::uint32_t bit = 0U; bit + 6U < kBitCount; ++bit) {
            bits[bit] = static_cast<std::uint8_t>(((bit * 13U + initial_state * 7U) >> 1U) & 1U);
        }
        for (std::uint32_t delay = 0U; delay < 6U; ++delay) {
            bits[kBitCount - 1U - delay] = static_cast<std::uint8_t>((initial_state >> delay) & 1U);
        }

        mmse::pdcch::PdcchRateRecoveredLlr recovered{};
        recovered.codeword_bit_count = kBitCount;
        recovered.convolutional_llrs = reference_llrs_from_encoded_bits(
            reference_pdcch_tail_biting_convolutional_encode(bits), 6.0F);
        std::vector<std::uint8_t> decoded{};
        expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(recovered, decoded) ==
                            MmseStatus::kOk &&
                        decoded == bits,
                    "native tail-biting decoder must close every initial state");
    }
}

void test_pdcch_native_tail_biting_decoder_length_boundaries() {
    constexpr std::array<std::uint32_t, 16> kBitCounts = {
        6U,   7U,   8U,   15U,
        31U,  32U,  33U,  43U,
        44U,  63U,  64U,  65U,
        127U, 128U, 143U, mmse::pdcch::kPdcchMaxDciPayloadBits + mmse::pdcch::kPdcchCrcBitCount,
    };
    for (const std::uint32_t bit_count : kBitCounts) {
        for (const std::uint32_t pattern : {0U, 1U}) {
            std::vector<std::uint8_t> bits(bit_count, 0U);
            for (std::uint32_t bit = 0U; bit < bit_count; ++bit) {
                bits[bit] = static_cast<std::uint8_t>(
                    ((bit * 29U + bit_count * 13U + pattern * 17U) ^ (bit >> 1U)) & 1U);
            }
            mmse::pdcch::PdcchRateRecoveredLlr recovered{};
            recovered.codeword_bit_count = bit_count;
            recovered.convolutional_llrs = reference_llrs_from_encoded_bits(
                reference_pdcch_tail_biting_convolutional_encode(bits), 12.0F);
            std::vector<std::uint8_t> decoded(bit_count, 1U);
            expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(recovered, decoded) ==
                                MmseStatus::kOk &&
                            decoded == bits,
                        "native tail-biting decoder must preserve length-boundary codewords");
        }
    }
}

void test_pdcch_native_tail_biting_decoder_llr_and_invalid_inputs() {
    std::vector<std::uint8_t> bits(44U, 0U);
    for (std::uint32_t bit = 0U; bit < bits.size(); ++bit) {
        bits[bit] = static_cast<std::uint8_t>((bit * 5U + 1U) & 1U);
    }
    const std::vector<std::uint8_t> encoded =
        reference_pdcch_tail_biting_convolutional_encode(bits);
    mmse::pdcch::PdcchRateRecoveredLlr recovered{};
    recovered.codeword_bit_count = static_cast<std::uint32_t>(bits.size());
    recovered.convolutional_llrs = reference_llrs_from_encoded_bits(encoded, 8.0F);
    for (std::uint32_t i = 0U; i < recovered.convolutional_llrs.size(); i += 17U) {
        recovered.convolutional_llrs[i] *= 0.25F;
    }
    std::vector<std::uint8_t> decoded{};
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(recovered, decoded) ==
                        MmseStatus::kOk &&
                    decoded == bits,
                "native tail-biting decoder must tolerate finite LLR perturbations");

    recovered.soft_bit_polarity = static_cast<mmse::pdcch::PdcchSoftBitPolarity>(1U);
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(recovered, decoded) ==
                    MmseStatus::kInvalidArgument,
                "native tail-biting decoder must validate LLR polarity");
    expect_true(decoded.empty(), "native tail-biting decoder must clear failed polarity output");
    recovered.soft_bit_polarity = mmse::pdcch::PdcchSoftBitPolarity::kNegativeFavorsZero;
    recovered.convolutional_llrs.pop_back();
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(recovered, decoded) ==
                    MmseStatus::kInvalidArgument,
                "native tail-biting decoder must validate LLR length");
    expect_true(decoded.empty(), "native tail-biting decoder must clear failed length output");
    recovered.convolutional_llrs = reference_llrs_from_encoded_bits(encoded, 8.0F);
    recovered.convolutional_llrs[0] = std::numeric_limits<float>::quiet_NaN();
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(recovered, decoded) ==
                    MmseStatus::kInvalidArgument,
                "native tail-biting decoder must reject non-finite LLRs");
    expect_true(decoded.empty(), "native tail-biting decoder must clear failed non-finite output");
}

void test_pdcch_native_tail_biting_decoder_boundaries_tie_break_and_failures() {
    const auto expect_invalid_and_cleared = [](const mmse::pdcch::PdcchRateRecoveredLlr& invalid,
                                               const std::string& label) {
        std::vector<std::uint8_t> decoded = {1U, 1U};
        expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(invalid, decoded) ==
                        MmseStatus::kInvalidArgument,
                    label + " status");
        expect_true(decoded.empty(), label + " output clear");
    };

    mmse::pdcch::PdcchRateRecoveredLlr fixed_vector{};
    fixed_vector.codeword_bit_count = 6U;
    fixed_vector.convolutional_llrs = {
        7.0F, -6.0F, 8.0F, -8.0F, -7.0F, 6.0F, 6.0F, -8.0F, -7.0F,
        8.0F, -6.0F, 7.0F, -7.0F, -8.0F, 6.0F, 6.0F, -7.0F, -8.0F,
    };
    const std::vector<std::uint8_t> fixed_bits = {1U, 0U, 1U, 1U, 0U, 1U};
    std::vector<std::uint8_t> decoded{};
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(fixed_vector, decoded) ==
                        MmseStatus::kOk &&
                    decoded == fixed_bits,
                "native tail-biting decoder must match fixed soft vector");

    mmse::pdcch::PdcchRateRecoveredLlr minimum_length{};
    minimum_length.codeword_bit_count = 1U;
    minimum_length.convolutional_llrs = {0.0F, 0.0F, 0.0F};
    decoded.assign(1U, 1U);
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(minimum_length, decoded) ==
                        MmseStatus::kOk &&
                    decoded == std::vector<std::uint8_t>{0U},
                "native tail-biting decoder must support minimum codeword length");

    for (const float tied_llr : {0.0F, -0.0F}) {
        for (std::uint32_t bit_count = 1U;
             bit_count <= mmse::pdcch::kPdcchMaxDciPayloadBits + mmse::pdcch::kPdcchCrcBitCount;
             ++bit_count) {
            mmse::pdcch::PdcchRateRecoveredLlr tied_llrs{};
            tied_llrs.codeword_bit_count = bit_count;
            tied_llrs.convolutional_llrs.assign(
                bit_count * mmse::pdcch::kPdcchConvolutionalCodeRate, tied_llr);
            decoded.assign(bit_count, 1U);
            expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(tied_llrs, decoded) ==
                                MmseStatus::kOk &&
                            decoded == std::vector<std::uint8_t>(bit_count, 0U),
                        "native tail-biting decoder must retain lower-predecessor tie break");
        }
    }

    mmse::pdcch::PdcchRateRecoveredLlr finite_extremes{};
    finite_extremes.codeword_bit_count =
        mmse::pdcch::kPdcchMaxDciPayloadBits + mmse::pdcch::kPdcchCrcBitCount;
    constexpr std::array<float, 6> kFiniteExtremes = {
        std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),        -std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),        std::numeric_limits<float>::lowest(),
    };
    finite_extremes.convolutional_llrs.resize(finite_extremes.codeword_bit_count *
                                              mmse::pdcch::kPdcchConvolutionalCodeRate);
    for (std::uint32_t llr = 0U; llr < finite_extremes.convolutional_llrs.size(); ++llr) {
        finite_extremes.convolutional_llrs[llr] = kFiniteExtremes[llr % kFiniteExtremes.size()];
    }
    std::vector<std::uint8_t> first_extreme_decode{};
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(
                    finite_extremes, first_extreme_decode) == MmseStatus::kOk &&
                    first_extreme_decode.size() == finite_extremes.codeword_bit_count &&
                    std::all_of(first_extreme_decode.begin(), first_extreme_decode.end(),
                                [](std::uint8_t bit) { return bit <= 1U; }),
                "native tail-biting decoder must accept finite subnormal and extreme LLRs");
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(finite_extremes, decoded) ==
                        MmseStatus::kOk &&
                    decoded == first_extreme_decode,
                "native tail-biting decoder must be deterministic for finite extreme LLRs");

    std::vector<std::uint8_t> maximum_bits(
        mmse::pdcch::kPdcchMaxDciPayloadBits + mmse::pdcch::kPdcchCrcBitCount, 0U);
    for (std::uint32_t bit = 0U; bit < maximum_bits.size(); ++bit) {
        maximum_bits[bit] = static_cast<std::uint8_t>(((bit * 37U + 11U) >> 2U) & 1U);
    }
    mmse::pdcch::PdcchRateRecoveredLlr maximum_length{};
    maximum_length.codeword_bit_count = static_cast<std::uint32_t>(maximum_bits.size());
    maximum_length.convolutional_llrs = reference_llrs_from_encoded_bits(
        reference_pdcch_tail_biting_convolutional_encode(maximum_bits), 8.0F);
    expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(maximum_length, decoded) ==
                        MmseStatus::kOk &&
                    decoded == maximum_bits,
                "native tail-biting decoder must support maximum codeword length");

    mmse::pdcch::PdcchRateRecoveredLlr empty{};
    expect_invalid_and_cleared(empty, "native tail-biting zero length");

    auto oversized = fixed_vector;
    oversized.codeword_bit_count =
        mmse::pdcch::kPdcchMaxDciPayloadBits + mmse::pdcch::kPdcchCrcBitCount + 1U;
    expect_invalid_and_cleared(oversized, "native tail-biting oversized length");

    auto invalid_order = fixed_vector;
    invalid_order.llr_order = static_cast<mmse::pdcch::PdcchConvolutionalLlrOrder>(1U);
    expect_invalid_and_cleared(invalid_order, "native tail-biting LLR order");

    auto infinite_llr = fixed_vector;
    infinite_llr.convolutional_llrs[0] = std::numeric_limits<float>::infinity();
    expect_invalid_and_cleared(infinite_llr, "native tail-biting infinite LLR");

    auto negative_infinite_llr = fixed_vector;
    negative_infinite_llr.convolutional_llrs[0] = -std::numeric_limits<float>::infinity();
    expect_invalid_and_cleared(negative_infinite_llr, "native tail-biting negative infinite LLR");
}

void test_pdcch_crc16_and_rnti_mask_check() {
    std::vector<std::uint8_t> check_bits{};
    for (const char character : std::string{"123456789"}) {
        append_bits(check_bits, static_cast<std::uint8_t>(character), 8U);
    }
    expect_true(mmse::pdcch::calculate_pdcch_crc16(
                    check_bits.data(), static_cast<std::uint32_t>(check_bits.size())) == 0x31C3U,
                "pdcch CRC16 should match the 123456789 check value");

    const std::vector<std::uint8_t> payload = {1U, 0U, 1U, 1U, 0U, 0U, 1U};
    const std::vector<std::uint8_t> decoded =
        append_pdcch_crc_rnti_mask(payload, mmse::pdcch::kSiRnti);
    mmse::pdcch::PdcchCrcRntiCheck check{};
    expect_true(mmse::pdcch::check_pdcch_crc_rnti(decoded.data(),
                                                  static_cast<std::uint32_t>(decoded.size()),
                                                  mmse::pdcch::kSiRnti, check) == MmseStatus::kOk,
                "pdcch CRC-RNTI check should succeed");
    expect_true(check.matches_expected_rnti && check.unmasked_rnti == mmse::pdcch::kSiRnti,
                "pdcch CRC-RNTI must recover SI-RNTI");
    expect_true(mmse::pdcch::check_pdcch_crc_rnti(decoded.data(),
                                                  static_cast<std::uint32_t>(decoded.size()),
                                                  0x1234U, check) == MmseStatus::kOk &&
                    !check.matches_expected_rnti && check.unmasked_rnti == mmse::pdcch::kSiRnti,
                "pdcch CRC-RNTI mismatch must remain a non-error blind-search result");
}

void test_pdcch_dci_format1a_fdd_si_rnti_parse() {
    const mmse::pdcch::PdcchDciFormat1AConfig config{};
    const std::uint16_t riv = reference_type2_riv(100U, 10U, 20U);
    std::vector<std::uint8_t> payload{};
    append_bits(payload, 1U, 1U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, riv, mmse::pdcch::pdcch_dci_riv_bit_count(config.n_prb));
    append_bits(payload, 7U, 5U);
    append_bits(payload, 5U, 3U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, 2U, 2U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, 1U, 1U);
    expect_true(payload.size() == mmse::pdcch::pdcch_dci_format1a_payload_bit_count(config),
                "FDD DCI 1A payload size");

    const std::vector<std::uint8_t> decoded =
        append_pdcch_crc_rnti_mask(payload, mmse::pdcch::kSiRnti);
    mmse::pdcch::PdcchDciFormat1ADecodeResult result{};
    const PdcchChainMetadata chain{
        .request_id = 900U, .candidate_id = 2U, .first_cce = 8U, .aggregation_level = 4U};
    expect_true(mmse::pdcch::validate_and_parse_pdcch_dci_format1a(
                    decoded.data(), static_cast<std::uint32_t>(decoded.size()),
                    mmse::pdcch::kSiRnti, 17U, 19U, chain, config, result) == MmseStatus::kOk,
                "FDD SI-RNTI DCI 1A decode should succeed");
    expect_true(result.matched && result.crc.matches_expected_rnti,
                "FDD SI-RNTI DCI 1A CRC result");
    expect_true(!result.dci.is_pdcch_order && !result.dci.distributed_vrb_assignment &&
                    result.dci.resource_indication_value == riv && result.dci.start_prb == 10U &&
                    result.dci.n_prb == 20U && result.dci.mcs_tbs_index == 7U &&
                    result.dci.harq_process == 5U && result.dci.redundancy_version == 2U &&
                    result.dci.n_prb_1a_is_three && result.dci.n_prb_1a == 3U &&
                    result.dci.chain.request_id == chain.request_id,
                "FDD SI-RNTI DCI 1A parsed grant fields");
}

void test_pdcch_dci_format1a_tdd_distributed_parse() {
    mmse::pdcch::PdcchDciFormat1AConfig config{};
    config.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd;
    const std::uint16_t riv = reference_type2_riv(100U, 7U, 30U);
    std::vector<std::uint8_t> payload{};
    append_bits(payload, 1U, 1U);
    append_bits(payload, 1U, 1U);
    append_bits(payload, riv, mmse::pdcch::pdcch_dci_riv_bit_count(config.n_prb));
    append_bits(payload, 12U, 5U);
    append_bits(payload, 9U, 4U);
    append_bits(payload, 1U, 1U);
    append_bits(payload, 3U, 2U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, 2U, 2U);
    expect_true(payload.size() == mmse::pdcch::pdcch_dci_format1a_payload_bit_count(config),
                "TDD DCI 1A payload size");

    const std::vector<std::uint8_t> decoded =
        append_pdcch_crc_rnti_mask(payload, mmse::pdcch::kSiRnti);
    mmse::pdcch::PdcchDciFormat1ADecodeResult result{};
    expect_true(mmse::pdcch::validate_and_parse_pdcch_dci_format1a(
                    decoded.data(), static_cast<std::uint32_t>(decoded.size()),
                    mmse::pdcch::kSiRnti, 9U, 19U, {}, config, result) == MmseStatus::kOk,
                "TDD SI-RNTI DCI 1A decode should succeed");
    expect_true(result.matched && result.dci.distributed_vrb_assignment &&
                    result.dci.n_gap_is_two && result.dci.start_prb == 7U &&
                    result.dci.n_prb == 30U && result.dci.mcs_tbs_index == 12U &&
                    result.dci.harq_process == 9U && result.dci.redundancy_version == 3U &&
                    result.dci.downlink_assignment_index == 2U,
                "TDD SI-RNTI DCI 1A parsed distributed fields");
}

void test_pdcch_dci_format1a_pdcch_order_and_crc_miss() {
    const mmse::pdcch::PdcchDciFormat1AConfig config{};
    std::vector<std::uint8_t> payload{};
    append_bits(payload, 1U, 1U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, (1U << mmse::pdcch::pdcch_dci_riv_bit_count(config.n_prb)) - 1U,
                mmse::pdcch::pdcch_dci_riv_bit_count(config.n_prb));
    append_bits(payload, 42U, 6U);
    append_bits(payload, 9U, 4U);
    while (payload.size() < mmse::pdcch::pdcch_dci_format1a_payload_bit_count(config)) {
        payload.push_back(0U);
    }

    std::vector<std::uint8_t> decoded = append_pdcch_crc_rnti_mask(payload, mmse::pdcch::kSiRnti);
    mmse::pdcch::PdcchDciFormat1ADecodeResult result{};
    expect_true(mmse::pdcch::validate_and_parse_pdcch_dci_format1a(
                    decoded.data(), static_cast<std::uint32_t>(decoded.size()),
                    mmse::pdcch::kSiRnti, 0U, 0U, {}, config, result) == MmseStatus::kOk &&
                    result.matched && result.dci.is_pdcch_order &&
                    result.dci.preamble_index == 42U && result.dci.prach_mask_index == 9U,
                "PDCCH order must parse after SI-RNTI CRC validation");

    decoded.back() ^= 1U;
    expect_true(mmse::pdcch::validate_and_parse_pdcch_dci_format1a(
                    decoded.data(), static_cast<std::uint32_t>(decoded.size()),
                    mmse::pdcch::kSiRnti, 0U, 0U, {}, config, result) == MmseStatus::kOk &&
                    !result.matched && !result.crc.matches_expected_rnti,
                "CRC miss must not attempt DCI payload parsing");
}

void test_pdcch_dci_format1a_adapter_end_to_end() {
    const mmse::pdcch::PdcchDciFormat1AConfig config{};
    const std::uint16_t riv = reference_type2_riv(100U, 22U, 10U);
    std::vector<std::uint8_t> payload{};
    append_bits(payload, 1U, 1U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, riv, mmse::pdcch::pdcch_dci_riv_bit_count(config.n_prb));
    append_bits(payload, 4U, 5U);
    append_bits(payload, 1U, 3U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, 0U, 2U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, 0U, 1U);
    const std::vector<std::uint8_t> decoded_bits =
        append_pdcch_crc_rnti_mask(payload, mmse::pdcch::kSiRnti);

    mmse::pdcch::PdcchRateRecoveredLlr recovered{};
    recovered.sfn_subframe = 27U;
    recovered.cell_id = 11U;
    recovered.chain = {
        .request_id = 12345U, .candidate_id = 1U, .first_cce = 4U, .aggregation_level = 4U};
    recovered.codeword_bit_count = static_cast<std::uint32_t>(decoded_bits.size());
    recovered.convolutional_llrs.assign(
        recovered.codeword_bit_count * mmse::pdcch::kPdcchConvolutionalCodeRate, 0.0F);

    TailBitingDecoderProbe probe{
        .expected_llrs = recovered.convolutional_llrs,
        .decoded_bits = decoded_bits,
    };
    const mmse::pdcch::PdcchTailBitingConvolutionalDecoder decoder{
        .context = &probe,
        .decode = probe_tail_biting_decoder,
    };
    mmse::pdcch::PdcchDciFormat1ADecodeResult result{};
    expect_true(mmse::pdcch::decode_pdcch_dci_format1a_with_adapter(
                    recovered, decoder, mmse::pdcch::kSiRnti, config, result) == MmseStatus::kOk,
                "pdcch DCI 1A adapter end-to-end decode should succeed");
    expect_true(probe.call_count == 1U && result.matched && result.crc.matches_expected_rnti &&
                    result.dci.start_prb == 22U && result.dci.n_prb == 10U &&
                    result.dci.mcs_tbs_index == 4U && result.dci.chain.request_id == 12345U,
                "pdcch DCI 1A adapter end-to-end result");
}

void test_pdcch_type2_riv_decodes_long_allocation() {
    const std::uint16_t riv = reference_type2_riv(100U, 5U, 80U);
    std::uint16_t start_prb = 0U;
    std::uint16_t allocated_prb_count = 0U;
    expect_true(mmse::pdcch::decode_pdcch_type2_riv(riv, 100U, start_prb, allocated_prb_count) ==
                        MmseStatus::kOk &&
                    start_prb == 5U && allocated_prb_count == 80U,
                "type2 RIV must decode the alternative long-allocation form");
}

void test_pdcch_helper_builds_pcfich_reserved_res() {
    const auto reserved = mmse::pdcch::build_pcfich_reserved_control_re_list(0U);
    const auto reserved_with_ctx = mmse::pdcch::build_pcfich_reserved_control_re_list(
        0U, {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
             .subframe = 1U,
             .ul_dl_config = 1U,
             .kind = mmse::pdcch::LteControlSubframeKind::kMbsfn});
    expect_true(reserved.size() == 16U, "pcfich should reserve 16 REs after CRS removal");
    expect_true(reserved_with_ctx.size() == reserved.size(),
                "pcfich shared-context helper should preserve RE count");
    for (std::size_t i = 0; i < reserved.size(); ++i) {
        expect_true(reserved_with_ctx[i].symbol == reserved[i].symbol,
                    "pcfich shared-context symbol");
        expect_true(reserved_with_ctx[i].prb == reserved[i].prb, "pcfich shared-context prb");
        expect_true(reserved_with_ctx[i].tone_in_prb == reserved[i].tone_in_prb,
                    "pcfich shared-context tone");
    }

    PdcchMmseInput in{};
    mmse::pdcch::apply_reserved_control_re_list(in, reserved);
    expect_true(mmse::pdcch::is_control_re_excluded(in, 0U, 0U, 1U), "pcfich reg 0 tone 1");
    expect_true(mmse::pdcch::is_control_re_excluded(in, 0U, 0U, 2U), "pcfich reg 0 tone 2");
    expect_true(mmse::pdcch::is_control_re_excluded(in, 0U, 0U, 4U), "pcfich reg 0 tone 4");
    expect_true(mmse::pdcch::is_control_re_excluded(in, 0U, 0U, 5U), "pcfich reg 0 tone 5");
    expect_true(!mmse::pdcch::is_control_re_excluded(in, 0U, 0U, 0U), "pcfich must not mark CRS");
    expect_true(!mmse::pdcch::is_control_re_excluded(in, 0U, 0U, 3U), "pcfich must not mark CRS");
}

void test_pdcch_helper_builds_fdd_phich_reserved_res_properties() {
    const auto reserved =
        mmse::pdcch::build_fdd_phich_reserved_control_re_list(0U, mmse::pdcch::PhichResource::kOne);
    expect_true(reserved.size() == 156U, "fdd normal PHICH reserved RE count");
    for (const auto& re : reserved) {
        expect_true(re.symbol == 0U, "fdd normal PHICH should stay in symbol 0");
        expect_true(re.prb < kLteNumPrb20MHz, "fdd normal PHICH PRB range");
        expect_true(re.tone_in_prb < kLteNumSubcarriersPerPrb, "fdd normal PHICH tone range");
        expect_true(re.tone_in_prb % 6U != 0U && re.tone_in_prb % 6U != 3U,
                    "fdd normal PHICH should not mark CRS");
    }
}

void test_pdcch_helper_phich_config_contract() {
    std::vector<mmse::pdcch::ReservedControlRe> reserved{};
    const MmseStatus ok = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kNormal});
    expect_true(ok == MmseStatus::kOk, "fdd normal phich config should be accepted");
    expect_true(reserved.size() == 156U, "fdd normal phich config should preserve helper output");

    reserved.clear();
    const MmseStatus tdd_status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kNormal,
         .mi = 1U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                          .subframe = 0U,
                          .ul_dl_config = 3U}});
    expect_true(tdd_status == MmseStatus::kOk, "tdd normal phich config should be accepted");
    expect_true(reserved.size() == 156U, "tdd mi=1 phich config should match base helper output");

    reserved.clear();
    const MmseStatus tdd_mi2_status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kNormal,
         .mi = 2U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                          .subframe = 5U,
                          .ul_dl_config = 0U}});
    expect_true(tdd_mi2_status == MmseStatus::kOk, "tdd mi=2 phich config should be accepted");
    expect_true(reserved.size() > 156U, "tdd mi=2 phich config should reserve more REs");

    reserved.clear();
    const MmseStatus tdd_zero_status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kNormal,
         .mi = 0U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                          .subframe = 2U,
                          .ul_dl_config = 0U}});
    expect_true(tdd_zero_status == MmseStatus::kOk,
                "tdd subframe without phich should accept mi=0");
    expect_true(reserved.empty(), "tdd subframe without phich should append no REs");

    reserved.clear();
    const MmseStatus bad_subframe_status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kNormal,
         .mi = 1U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                          .subframe = 10U,
                          .ul_dl_config = 0U}});
    expect_true(bad_subframe_status == MmseStatus::kInvalidArgument,
                "phich helper should reject out-of-range subframe");
    expect_true(reserved.empty(), "invalid subframe should not append REs");

    const MmseStatus bad_tdd_mi_status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kNormal,
         .mi = 1U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                          .subframe = 2U,
                          .ul_dl_config = 0U}});
    expect_true(bad_tdd_mi_status == MmseStatus::kInvalidArgument,
                "tdd phich helper should reject mismatched mi/subframe");
    expect_true(reserved.empty(), "mismatched mi/subframe should not append REs");

    const MmseStatus bad_tdd_cfg_status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kNormal,
         .mi = 1U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                          .subframe = 0U,
                          .ul_dl_config = 7U}});
    expect_true(bad_tdd_cfg_status == MmseStatus::kInvalidArgument,
                "tdd phich helper should reject out-of-range ul-dl config");
    expect_true(reserved.empty(), "invalid ul-dl config should not append REs");

    const MmseStatus extended_status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kExtended,
         .mi = 1U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                          .subframe = 0U,
                          .ul_dl_config = 0U}});
    expect_true(extended_status == MmseStatus::kOk,
                "extended fdd phich helper contract should be accepted");
    expect_true(!reserved.empty(), "extended fdd phich helper should append REs");

    bool saw_symbol1_or_2 = false;
    for (const auto& re : reserved) {
        expect_true(re.symbol <= 2U, "extended phich should stay within first three symbols");
        if (re.symbol == 1U || re.symbol == 2U) {
            saw_symbol1_or_2 = true;
        }
    }
    expect_true(saw_symbol1_or_2, "extended fdd phich should touch later control symbols");
}

void test_pdcch_helper_extended_phich_special_case_distribution() {
    std::vector<mmse::pdcch::ReservedControlRe> reserved{};
    const MmseStatus status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kExtended,
         .mi = 1U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                          .subframe = 1U,
                          .ul_dl_config = 1U,
                          .kind = mmse::pdcch::LteControlSubframeKind::kMbsfn}});
    expect_true(status == MmseStatus::kOk, "extended special-case phich helper should be accepted");
    expect_true(!reserved.empty(), "extended special-case phich helper should append REs");

    for (const auto& re : reserved) {
        expect_true(re.symbol <= 1U, "extended special-case phich should stay within symbols 0/1");
    }
}

void test_pdcch_helper_extended_tdd_sf1_automatic_special_case() {
    std::vector<mmse::pdcch::ReservedControlRe> reserved{};
    const MmseStatus status = mmse::pdcch::append_phich_reserved_control_re_list(
        reserved, 0U,
        {.resource = mmse::pdcch::PhichResource::kOne,
         .duration = mmse::pdcch::PhichDuration::kExtended,
         .mi = 1U,
         .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                          .subframe = 1U,
                          .ul_dl_config = 1U}});
    expect_true(status == MmseStatus::kOk,
                "extended tdd subframe 1 should auto-select special-case mapping");
    expect_true(!reserved.empty(), "extended tdd subframe 1 should append REs");
    for (const auto& re : reserved) {
        expect_true(re.symbol <= 1U,
                    "extended tdd subframe 1 special-case should stay within symbols 0/1");
    }
}

void test_pdcch_helper_tdd_phich_affects_layout() {
    auto desc = make_pdcch_desc();
    mmse::detail::ReLayout layout{};
    const std::uint32_t base_n_re = mmse::detail::build_pdcch_re_layout(desc, layout);

    mmse::pdcch::FrontendPdcchIndication frontend{};
    frontend.cell_id = desc.cell_id;
    frontend.control_symbol_count = desc.control_symbol_count;
    frontend.n_tx_ports = desc.n_tx_ports;
    frontend.tx_mode = desc.tx_mode;
    frontend.n_prb = desc.n_prb;
    frontend.prb_bitmap = desc.prb_bitmap;
    mmse::pdcch::append_pcfich_reserved_control_re_list(frontend);
    expect_true(mmse::pdcch::append_phich_reserved_control_re_list(
                    frontend, {.resource = mmse::pdcch::PhichResource::kOne,
                               .duration = mmse::pdcch::PhichDuration::kNormal,
                               .mi = 2U,
                               .subframe_ctx = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                                                .subframe = 0U,
                                                .ul_dl_config = 0U}}) == MmseStatus::kOk,
                "tdd phich helper should append to frontend");

    PdcchMmseInput in{};
    mmse::pdcch::apply_reserved_control_re_list(in, frontend.reserved_control_res);
    desc.control_re_exclusion_masks = in.control_re_exclusion_masks;
    const std::uint32_t reserved_n_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    expect_true(reserved_n_re < base_n_re - 172U,
                "tdd mi=2 helper should exclude more REs than fdd");
}

void test_pdcch_frontend_helper_auto_reserved_res_reduce_layout() {
    auto desc = make_pdcch_desc();
    mmse::detail::ReLayout layout{};
    const std::uint32_t base_n_re = mmse::detail::build_pdcch_re_layout(desc, layout);

    mmse::pdcch::FrontendPdcchIndication frontend{};
    frontend.cell_id = desc.cell_id;
    frontend.control_symbol_count = desc.control_symbol_count;
    frontend.n_tx_ports = desc.n_tx_ports;
    frontend.tx_mode = desc.tx_mode;
    frontend.n_prb = desc.n_prb;
    frontend.prb_bitmap = desc.prb_bitmap;
    frontend.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                                 .subframe = 0U,
                                 .ul_dl_config = 0U,
                                 .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    mmse::pdcch::append_pcfich_reserved_control_re_list(frontend);
    mmse::pdcch::append_fdd_phich_reserved_control_re_list(frontend,
                                                           mmse::pdcch::PhichResource::kOne);

    PdcchMmseInput in{};
    mmse::pdcch::apply_reserved_control_re_list(in, frontend.reserved_control_res);
    desc.control_re_exclusion_masks = in.control_re_exclusion_masks;

    const std::uint32_t reserved_n_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    expect_true(frontend.reserved_control_res.size() == 172U,
                "pcfich plus phich should add expected unique reserved REs");
    expect_true(reserved_n_re < base_n_re,
                "auto reserved RE helper should reduce the CCE-aligned layout");
    expect_true((reserved_n_re % (mmse::pdcch::kPdcchRePerReg * mmse::pdcch::kPdcchRegsPerCce)) ==
                    0U,
                "auto reserved RE layout must contain complete CCEs");
    for (std::uint32_t i = 0U; i < reserved_n_re; ++i) {
        const auto coord = mmse::pdcch::decode_re_grid_index(layout.grid_indices[i]);
        for (const auto& reserved_re : frontend.reserved_control_res) {
            expect_true(coord.symbol != reserved_re.symbol || coord.prb != reserved_re.prb ||
                            coord.tone_in_prb != reserved_re.tone_in_prb,
                        "auto reserved RE helper must remove every reserved PDCCH RE");
        }
    }
}

void test_pdcch_frontend_dto_builds_mmse_input() {
    GridBuffers buffers = make_zero_grid();
    auto grid = make_grid_view(buffers);

    mmse::pdcch::FrontendPdcchIndication frontend{};
    frontend.sfn_subframe = 9U;
    frontend.cell_id = 22U;
    frontend.n_tx_ports = 1U;
    frontend.tx_mode = 1U;
    frontend.control_symbol_count = 3U;
    frontend.n_prb = 100U;
    frontend.prb_bitmap.fill(0xFFFFU);
    frontend.prb_bitmap.back() = 0x000FU;
    frontend.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                                 .subframe = 9U,
                                 .ul_dl_config = 0U,
                                 .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    frontend.chain.request_id = 55U;
    frontend.chain.candidate_id = 6U;
    frontend.chain.first_cce = 3U;
    frontend.chain.aggregation_level = 4U;
    frontend.reserved_control_res.push_back({.symbol = 0U, .prb = 1U, .tone_in_prb = 2U});

    const PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);
    expect_true(in.grid.n_symbols == grid.n_symbols, "dto grid passthrough");
    expect_true(in.sfn_subframe == frontend.sfn_subframe, "dto subframe passthrough");
    expect_true(in.cell_id == frontend.cell_id, "dto cell passthrough");
    expect_true(in.control_symbol_count == frontend.control_symbol_count, "dto cfi passthrough");
    expect_true(in.control_subframe.subframe == frontend.control_subframe.subframe,
                "dto control_subframe subframe passthrough");
    expect_true(in.control_subframe.ul_dl_config == frontend.control_subframe.ul_dl_config,
                "dto control_subframe ul_dl_config passthrough");
    expect_true(in.control_subframe.kind == frontend.control_subframe.kind,
                "dto control_subframe kind passthrough");
    expect_true(in.chain.request_id == frontend.chain.request_id, "dto chain passthrough");
    expect_true(mmse::pdcch::is_control_re_excluded(in, 0U, 1U, 2U), "dto reserved RE application");
}

void test_pdcch_frontend_control_subframe_drives_helpers() {
    mmse::pdcch::FrontendPdcchIndication frontend{};
    frontend.cell_id = 0U;
    frontend.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd,
                                 .subframe = 1U,
                                 .ul_dl_config = 1U,
                                 .kind = mmse::pdcch::LteControlSubframeKind::kMbsfn};

    mmse::pdcch::append_pcfich_reserved_control_re_list(frontend);
    const MmseStatus phich_status = mmse::pdcch::append_phich_reserved_control_re_list(
        frontend, mmse::pdcch::PhichResource::kOne, mmse::pdcch::PhichDuration::kExtended, 1U);
    expect_true(phich_status == MmseStatus::kOk,
                "frontend control_subframe should drive phich helper");
    expect_true(!frontend.reserved_control_res.empty(),
                "frontend control_subframe-driven helpers should append reserved REs");
}

void test_pdcch_backend_dto_packs_equalized_output() {
    PdcchMmseResult meta{};
    meta.n_re = 3U;
    meta.sfn_subframe = 9U;
    meta.cell_id = 21U;
    meta.n_prb = 100U;
    meta.n_tx_ports = 1U;
    meta.n_rx_ant = 2U;
    meta.n_layers = 1U;
    meta.tx_mode = 1U;
    meta.control_symbol_count = 3U;
    meta.mod_order = 2U;
    meta.sigma2 = 0.125F;
    meta.chain.request_id = 88U;
    meta.chain.candidate_id = 4U;
    meta.chain.first_cce = 6U;
    meta.chain.aggregation_level = 8U;

    std::array<float, 3> xre = {1.0F, 2.0F, 3.0F};
    std::array<float, 3> xim = {0.1F, 0.2F, 0.3F};
    std::array<float, 3> sinr = {10.0F, 11.0F, 12.0F};
    std::array<std::uint16_t, 3> grid_idx = {4U, 5U, 6U};

    PdcchMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices = grid_idx.data();

    const auto backend = mmse::pdcch::make_backend_pdcch_equalized_indication(meta, out);
    expect_true(backend.re_grid_indices.size() == 3U, "backend dto size");
    expect_true(backend.x_hat_re[1] == 2.0F, "backend dto xhat real");
    expect_true(backend.x_hat_im[2] == 0.3F, "backend dto xhat imag");
    expect_true(backend.sinr[0] == 10.0F, "backend dto sinr");
    expect_true(backend.chain.request_id == meta.chain.request_id, "backend dto chain");
    expect_true(backend.sigma2 == meta.sigma2, "backend dto sigma2");
}

void test_pdcch_td_backend_dto_packs_equalized_output() {
    PdcchTdMmseResult meta{};
    meta.n_symbols = 2U;
    meta.n_source_re = 2U;
    meta.sfn_subframe = 9U;
    meta.cell_id = 21U;
    meta.n_prb = 100U;
    meta.n_tx_ports = 2U;
    meta.n_rx_ant = 2U;
    meta.n_layers = 1U;
    meta.tx_mode = 2U;
    meta.control_symbol_count = 3U;
    meta.mod_order = 2U;
    meta.sigma2 = 0.125F;
    meta.chain.request_id = 88U;

    std::array<float, 2> xre = {1.0F, 2.0F};
    std::array<float, 2> xim = {0.1F, 0.2F};
    std::array<float, 2> sinr = {10.0F, 11.0F};
    std::array<std::uint16_t, 2> grid_idx0 = {4U, 4U};
    std::array<std::uint16_t, 2> grid_idx1 = {5U, 5U};

    PdcchTdMmseOutputView out{};
    out.x_hat_re = xre.data();
    out.x_hat_im = xim.data();
    out.sinr = sinr.data();
    out.re_grid_indices0 = grid_idx0.data();
    out.re_grid_indices1 = grid_idx1.data();

    const auto backend = mmse::pdcch::make_backend_pdcch_td_equalized_indication(meta, out);
    expect_true(backend.x_hat_re.size() == 2U, "td backend dto size");
    expect_true(backend.re_grid_indices0[0] == 4U, "td backend dto first re0");
    expect_true(backend.re_grid_indices1[1] == 5U, "td backend dto second re1");
    expect_true(backend.x_hat_im[1] == 0.2F, "td backend dto imag");
    expect_true(backend.chain.request_id == meta.chain.request_id, "td backend dto chain");
}

void test_pdcch_td_normalization_restores_cce_re_order() {
    mmse::pdcch::BackendPdcchTdEqualizedIndication td_backend{};
    td_backend.sfn_subframe = 17U;
    td_backend.cell_id = 21U;
    td_backend.n_prb = 100U;
    td_backend.n_tx_ports = 2U;
    td_backend.n_rx_ant = 2U;
    td_backend.n_layers = 1U;
    td_backend.tx_mode = 2U;
    td_backend.control_symbol_count = 1U;
    td_backend.mod_order = 2U;
    td_backend.chain.request_id = 991U;
    td_backend.x_hat_re = {1.0F, 2.0F, 3.0F, 4.0F};
    td_backend.x_hat_im = {-1.0F, -2.0F, -3.0F, -4.0F};
    td_backend.sinr = {10.0F, 11.0F, 12.0F, 13.0F};
    td_backend.re_grid_indices0 = {0U, 0U, 2U, 2U};
    td_backend.re_grid_indices1 = {1U, 1U, 3U, 3U};

    mmse::pdcch::BackendPdcchEqualizedIndication normalized{};
    expect_true(mmse::pdcch::normalize_pdcch_td_cce_order(td_backend, normalized) ==
                    MmseStatus::kOk,
                "td normalization should accept paired output");
    expect_true(normalized.re_grid_indices == std::vector<std::uint16_t>{0U, 1U, 2U, 3U},
                "td normalization must restore source RE order");
    expect_true(normalized.x_hat_re == td_backend.x_hat_re &&
                    normalized.chain.request_id == td_backend.chain.request_id,
                "td normalization must preserve soft symbols and chain metadata");

    td_backend.re_grid_indices1[1] = 4U;
    expect_true(mmse::pdcch::normalize_pdcch_td_cce_order(td_backend, normalized) ==
                    MmseStatus::kInvalidArgument,
                "td normalization should reject inconsistent RE pair metadata");
}

void test_pdcch_td_normalization_accepts_empty_control_region() {
    mmse::pdcch::BackendPdcchTdEqualizedIndication td_backend{};
    td_backend.sfn_subframe = 3U;
    td_backend.cell_id = 1U;
    td_backend.n_prb = 6U;
    td_backend.n_tx_ports = 2U;
    td_backend.n_rx_ant = 2U;
    td_backend.n_layers = 1U;
    td_backend.tx_mode = 2U;
    td_backend.control_symbol_count = 1U;
    td_backend.mod_order = 2U;
    td_backend.chain.request_id = 991U;

    mmse::pdcch::BackendPdcchEqualizedIndication normalized{};
    expect_true(mmse::pdcch::normalize_pdcch_td_cce_order(td_backend, normalized) ==
                        MmseStatus::kOk &&
                    normalized.x_hat_re.empty() && normalized.re_grid_indices.empty() &&
                    normalized.chain.request_id == td_backend.chain.request_id &&
                    normalized.n_tx_ports == td_backend.n_tx_ports &&
                    normalized.tx_mode == td_backend.tx_mode,
                "td normalization must preserve valid empty control regions as empty backends");
}

std::vector<std::uint8_t> make_si_rnti_dci_format1a_bits(std::uint16_t n_prb = kLteNumPrb20MHz,
                                                         std::uint16_t start_prb = 10U,
                                                         std::uint16_t allocated_prb_count = 20U) {
    const mmse::pdcch::PdcchDciFormat1AConfig config{.n_prb = n_prb};
    expect_true(start_prb + allocated_prb_count <= n_prb,
                "SI-RNTI DCI allocation must fit the configured carrier bandwidth");
    const std::uint16_t riv = reference_type2_riv(config.n_prb, start_prb, allocated_prb_count);
    std::vector<std::uint8_t> payload{};
    append_bits(payload, 1U, 1U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, riv, mmse::pdcch::pdcch_dci_riv_bit_count(config.n_prb));
    append_bits(payload, 7U, 5U);
    append_bits(payload, 5U, 3U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, 2U, 2U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, 1U, 1U);
    while (payload.size() < mmse::pdcch::pdcch_dci_format1a_payload_bit_count(config)) {
        payload.push_back(0U);
    }
    return append_pdcch_crc_rnti_mask(payload, mmse::pdcch::kSiRnti);
}

void rewrite_pdcch_crc_rnti_mask(std::vector<std::uint8_t>& decoded_bits, std::uint16_t rnti) {
    expect_true(decoded_bits.size() > mmse::pdcch::kPdcchCrcBitCount,
                "rewrite CRC requires a DCI payload");
    decoded_bits.resize(decoded_bits.size() - mmse::pdcch::kPdcchCrcBitCount);
    const std::uint16_t masked_crc = static_cast<std::uint16_t>(
        mmse::pdcch::calculate_pdcch_crc16(decoded_bits.data(),
                                           static_cast<std::uint32_t>(decoded_bits.size())) ^
        rnti);
    append_bits(decoded_bits, masked_crc, mmse::pdcch::kPdcchCrcBitCount);
}

mmse::pdcch::BackendPdcchEqualizedIndication
make_native_si_rnti_backend(std::uint16_t first_cce, std::uint8_t aggregation_level,
                            const std::vector<std::uint8_t>& decoded_bits,
                            std::uint32_t cce_count = 8U);

std::vector<Complex32>
make_native_si_rnti_qpsk_symbols(std::uint32_t n_re, std::uint16_t first_cce,
                                 std::uint8_t aggregation_level,
                                 const std::vector<std::uint8_t>& decoded_bits,
                                 std::uint16_t cell_id, std::uint32_t sfn_subframe);

void test_pdcch_si_rnti_semantics_reject_invalid_grants() {
    std::vector<std::uint8_t> wrong_rnti_bits = make_si_rnti_dci_format1a_bits();
    rewrite_pdcch_crc_rnti_mask(wrong_rnti_bits, 0x1234U);
    auto backend = make_native_si_rnti_backend(4U, 4U, wrong_rnti_bits);
    mmse::pdcch::PdcchSiRntiSearchResult result{};
    expect_true(mmse::pdcch::decode_pdcch_si_rnti_dci_format1a(backend, {}, result) ==
                        MmseStatus::kOk &&
                    result.hits.empty(),
                "SI-RNTI search must treat another RNTI as a normal miss");

    std::vector<std::uint8_t> crc_flip_bits = make_si_rnti_dci_format1a_bits();
    crc_flip_bits.back() ^= 1U;
    backend = make_native_si_rnti_backend(4U, 4U, crc_flip_bits);
    expect_true(mmse::pdcch::decode_pdcch_si_rnti_dci_format1a(backend, {}, result) ==
                        MmseStatus::kOk &&
                    result.hits.empty(),
                "SI-RNTI search must treat CRC flips as normal misses");

    const mmse::pdcch::PdcchDciFormat1AConfig config{};
    std::vector<std::uint8_t> pdcch_order_payload{};
    append_bits(pdcch_order_payload, 1U, 1U);
    append_bits(pdcch_order_payload, 0U, 1U);
    append_bits(pdcch_order_payload,
                (1U << mmse::pdcch::pdcch_dci_riv_bit_count(config.n_prb)) - 1U,
                mmse::pdcch::pdcch_dci_riv_bit_count(config.n_prb));
    append_bits(pdcch_order_payload, 42U, 6U);
    append_bits(pdcch_order_payload, 9U, 4U);
    while (pdcch_order_payload.size() < mmse::pdcch::pdcch_dci_format1a_payload_bit_count(config)) {
        pdcch_order_payload.push_back(0U);
    }
    backend = make_native_si_rnti_backend(
        4U, 4U, append_pdcch_crc_rnti_mask(pdcch_order_payload, mmse::pdcch::kSiRnti));
    expect_true(mmse::pdcch::decode_pdcch_si_rnti_dci_format1a(backend, {}, result) ==
                        MmseStatus::kOk &&
                    result.hits.empty(),
                "SI-RNTI search must not accept PDCCH order as a SIB authorization");
}

void test_pdcch_dci_format1a_si_numeric_boundaries() {
    std::vector<std::uint8_t> decoded_bits = make_si_rnti_dci_format1a_bits();
    constexpr std::uint32_t kPayloadBitCount = 28U;
    decoded_bits[15U] = 1U;
    decoded_bits[16U] = 1U;
    decoded_bits[17U] = 1U;
    decoded_bits[18U] = 1U;
    decoded_bits[19U] = 1U;
    decoded_bits[24U] = 1U;
    decoded_bits[25U] = 1U;
    decoded_bits[kPayloadBitCount - 1U] = 0U;
    rewrite_pdcch_crc_rnti_mask(decoded_bits, mmse::pdcch::kSiRnti);

    mmse::pdcch::PdcchDciFormat1ADecodeResult result{};
    expect_true(mmse::pdcch::validate_and_parse_pdcch_dci_format1a(
                    decoded_bits.data(), static_cast<std::uint32_t>(decoded_bits.size()),
                    mmse::pdcch::kSiRnti, 0U, 0U, {}, {}, result) == MmseStatus::kOk &&
                    result.matched && result.dci.mcs_tbs_index == 31U &&
                    result.dci.redundancy_version == 3U && result.dci.n_prb_1a == 2U &&
                    !result.dci.n_prb_1a_is_three,
                "SI DCI 1A parser must expose MCS/RV and N_PRB^1A numeric boundaries");
}

void test_pdcch_fdd_dci_format1a_supports_standard_bandwidths() {
    constexpr std::array<std::uint16_t, 6> kBandwidths = {6U, 15U, 25U, 50U, 75U, 100U};
    constexpr std::array<std::uint32_t, 6> kPayloadBitCounts = {21U, 22U, 25U, 27U, 27U, 28U};
    constexpr std::array<std::uint8_t, 6> kRivBitCounts = {5U, 7U, 9U, 11U, 12U, 13U};

    for (std::size_t index = 0U; index < kBandwidths.size(); ++index) {
        const std::uint16_t n_prb = kBandwidths[index];
        const std::uint16_t start_prb = static_cast<std::uint16_t>(n_prb / 3U);
        const std::uint16_t allocated_prb_count =
            static_cast<std::uint16_t>(std::min<std::uint16_t>(3U, n_prb - start_prb));
        const mmse::pdcch::PdcchDciFormat1AConfig config{.n_prb = n_prb};
        const std::vector<std::uint8_t> decoded_bits =
            make_si_rnti_dci_format1a_bits(n_prb, start_prb, allocated_prb_count);

        expect_true(
            mmse::pdcch::pdcch_dci_riv_bit_count(n_prb) == kRivBitCounts[index] &&
                mmse::pdcch::pdcch_dci_format1a_payload_bit_count(config) ==
                    kPayloadBitCounts[index] &&
                decoded_bits.size() == kPayloadBitCounts[index] + mmse::pdcch::kPdcchCrcBitCount,
            "FDD DCI 1A must use the standard RIV and payload widths for each LTE bandwidth");

        const std::uint32_t unpadded_bit_count = kRivBitCounts[index] + 15U;
        for (std::uint32_t bit = unpadded_bit_count; bit < kPayloadBitCounts[index]; ++bit) {
            expect_true(decoded_bits[bit] == 0U,
                        "Format 0/1A alignment padding must be encoded as zero bits");
        }

        mmse::pdcch::PdcchDciFormat1ADecodeResult result{};
        expect_true(mmse::pdcch::validate_and_parse_pdcch_dci_format1a(
                        decoded_bits.data(), static_cast<std::uint32_t>(decoded_bits.size()),
                        mmse::pdcch::kSiRnti, 3U, 19U, {}, config, result) == MmseStatus::kOk &&
                        result.matched && result.dci.start_prb == start_prb &&
                        result.dci.n_prb == allocated_prb_count,
                    "FDD DCI 1A parser must decode grants at every standard LTE bandwidth");
    }

    expect_true(mmse::pdcch::pdcch_dci_format1a_payload_bit_count({.n_prb = 12U}) == 0U,
                "non-standard LTE bandwidths must remain unsupported");
}

void test_pdcch_control_geometry_supports_standard_bandwidths() {
    constexpr std::array<std::uint16_t, 6> kBandwidths = {6U, 15U, 25U, 50U, 75U, 100U};
    for (const std::uint16_t n_prb : kBandwidths) {
        const std::uint8_t control_symbol_count = n_prb == 6U ? 4U : 3U;
        const auto pcfich = mmse::pdcch::build_pcfich_reserved_control_re_list(19U, n_prb);
        const auto phich = mmse::pdcch::build_fdd_phich_reserved_control_re_list(
            19U, mmse::pdcch::PhichResource::kOne, n_prb);
        expect_true(pcfich.size() == 16U && !phich.empty(),
                    "every standard LTE bandwidth must expose PCFICH and PHICH reservations");
        for (const auto& re : pcfich) {
            expect_true(re.symbol == 0U && re.prb < n_prb &&
                            re.tone_in_prb < kLteNumSubcarriersPerPrb,
                        "PCFICH reservation must remain inside the configured carrier");
        }
        for (const auto& re : phich) {
            expect_true(re.symbol == 0U && re.prb < n_prb &&
                            re.tone_in_prb < kLteNumSubcarriersPerPrb,
                        "PHICH reservation must remain inside the configured carrier");
        }

        const auto desc = make_pdcch_desc(n_prb, control_symbol_count);
        mmse::detail::ReLayout layout{};
        const std::uint32_t n_re = mmse::detail::build_pdcch_re_layout(desc, layout);
        mmse::pdcch::PdcchControlRegion region{};
        expect_true(mmse::pdcch::build_pdcch_control_region(desc.cell_id, control_symbol_count,
                                                            layout.grid_indices.data(), n_re,
                                                            region, n_prb) == MmseStatus::kOk &&
                        !region.cces.empty(),
                    "PDCCH REG/CCE construction must succeed for every standard LTE bandwidth");
        for (std::uint32_t re = 0U; re < n_re; ++re) {
            const auto coordinate =
                mmse::pdcch::decode_re_grid_index(layout.grid_indices[re], n_prb);
            expect_true(coordinate.symbol < control_symbol_count && coordinate.prb < n_prb,
                        "PDCCH RE index must use the configured carrier stride");
        }
    }
}

void test_pdcch_cpu_si_rnti_search_supports_standard_bandwidths() {
    constexpr std::array<std::uint16_t, 6> kBandwidths = {6U, 15U, 25U, 50U, 75U, 100U};
    for (const std::uint16_t n_prb : kBandwidths) {
        MmseEqualizerCpuContext context;
        MmseEqualizerCpuConfig cpu_config{};
        cpu_config.worker_count = 1U;
        cpu_config.sigma2_min = 1.0e-5F;
        expect_true(context.init(cpu_config) == MmseStatus::kOk,
                    "standard-bandwidth SI-RNTI CPU context init");

        const std::uint8_t control_symbol_count = n_prb == 6U ? 4U : 3U;
        const auto desc = make_pdcch_desc(n_prb, control_symbol_count);
        GridBuffers buffers = make_zero_grid(lte_downlink_subcarrier_count(n_prb));
        fill_identity_channel(buffers, desc, 0.0F, 0.0F);
        mmse::detail::ReLayout layout{};
        const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
        const std::uint16_t start_prb = static_cast<std::uint16_t>(n_prb / 3U);
        const std::uint16_t allocated_prb_count =
            static_cast<std::uint16_t>(std::min<std::uint16_t>(3U, n_prb - start_prb));
        const std::uint16_t first_cce = n_prb == 6U ? 0U : 4U;
        const auto symbols = make_native_si_rnti_qpsk_symbols(
            n_source_re, first_cce, 4U,
            make_si_rnti_dci_format1a_bits(n_prb, start_prb, allocated_prb_count), desc.cell_id,
            desc.sfn_subframe);
        for (std::uint32_t re = 0U; re < n_source_re; ++re) {
            buffers.re[0][layout.grid_indices[re]] = symbols[re].re;
            buffers.im[0][layout.grid_indices[re]] = symbols[re].im;
        }

        PdcchMmseInput input{};
        input.grid = make_grid_view(buffers);
        input.sfn_subframe = desc.sfn_subframe;
        input.cell_id = desc.cell_id;
        input.n_tx_ports = desc.n_tx_ports;
        input.tx_mode = desc.tx_mode;
        input.control_symbol_count = control_symbol_count;
        input.n_prb = n_prb;
        input.prb_bitmap = desc.prb_bitmap;
        input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                                  .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                                  .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

        mmse::pdcch::PdcchSiRntiSearchResult result{};
        const MmseStatus status =
            mmse::pdcch::run_pdcch_cpu_si_rnti_search(context, input, {}, result);
        expect_true(status == MmseStatus::kOk && result.hits.size() == 1U &&
                        result.hits[0].first_cce == first_cce &&
                        result.hits[0].aggregation_level == 4U &&
                        result.hits[0].decoded.dci.start_prb == start_prb &&
                        result.hits[0].decoded.dci.n_prb == allocated_prb_count,
                    "CPU SI-RNTI search must decode every standard LTE bandwidth n_prb=" +
                        std::to_string(n_prb) +
                        " status=" + std::to_string(static_cast<std::uint32_t>(status)) +
                        " hits=" + std::to_string(result.hits.size()));
    }
}

void test_pdcch_ue_specific_search_supports_standard_bandwidths() {
    constexpr std::array<std::uint16_t, 6> kBandwidths = {6U, 15U, 25U, 50U, 75U, 100U};
    constexpr std::uint16_t kRnti = 0x1234U;
    for (const std::uint16_t n_prb : kBandwidths) {
        auto decoded_bits = make_si_rnti_dci_format1a_bits(n_prb, 0U, 1U);
        rewrite_pdcch_crc_rnti_mask(decoded_bits, kRnti);
        auto backend = make_native_si_rnti_backend(0U, 1U, decoded_bits, 1U);
        backend.n_prb = n_prb;

        mmse::pdcch::PdcchUeSpecificSearchConfig config{};
        config.rntis = {kRnti};
        config.aggregation_level_mask = mmse::pdcch::kPdcchAggregationLevelMaskL1;
        config.dci_format1a.n_prb = n_prb;
        mmse::pdcch::PdcchUeSpecificSearchResult result{};
        expect_true(mmse::pdcch::decode_pdcch_ue_specific_search_dci_format1a(
                        backend, config, result) == MmseStatus::kOk &&
                        result.hits.size() == 1U && result.hits[0].rnti == kRnti &&
                        result.hits[0].decoded.dci.start_prb == 0U &&
                        result.hits[0].decoded.dci.n_prb == 1U,
                    "UE-specific DCI 1A search must use the configured LTE bandwidth");
    }
}

void test_pdcch_cpu_si_rnti_search_two_tx_supports_selected_bandwidths() {
    constexpr std::array<std::uint16_t, 3> kBandwidths = {6U, 50U, 100U};
    for (const std::uint16_t n_prb : kBandwidths) {
        MmseEqualizerCpuContext context;
        MmseEqualizerCpuConfig cpu_config{};
        cpu_config.worker_count = 1U;
        cpu_config.sigma2_min = 1.0e-5F;
        expect_true(context.init(cpu_config) == MmseStatus::kOk,
                    "two-tx standard-bandwidth SI-RNTI CPU context init");

        const std::uint8_t control_symbol_count = n_prb == 6U ? 4U : 3U;
        auto desc = make_pdcch_desc(n_prb, control_symbol_count);
        desc.cell_id = 1U;
        desc.n_tx_ports = 2U;
        desc.tx_mode = 2U;
        GridBuffers buffers = make_zero_grid(lte_downlink_subcarrier_count(n_prb));
        fill_identity_channel(buffers, desc, 0.0F, 0.0F);
        mmse::detail::ReLayout layout{};
        const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
        const std::uint16_t first_cce = n_prb == 6U ? 0U : 4U;
        const std::uint16_t start_prb = static_cast<std::uint16_t>(n_prb / 3U);
        const auto symbols = make_native_si_rnti_qpsk_symbols(
            n_source_re, first_cce, 4U, make_si_rnti_dci_format1a_bits(n_prb, start_prb, 1U),
            desc.cell_id, desc.sfn_subframe);
        expect_true((n_source_re & 1U) == 0U,
                    "two-tx standard-bandwidth PDCCH source RE count must be even");
        bool found_nonadjacent_pair = false;
        const std::uint32_t n_subcarriers = lte_downlink_subcarrier_count(n_prb);
        for (std::uint32_t re = 0U; re < n_source_re; re += 2U) {
            const std::uint16_t grid0 = layout.grid_indices[re];
            const std::uint16_t grid1 = layout.grid_indices[re + 1U];
            expect_true(grid0 / n_subcarriers == grid1 / n_subcarriers,
                        "two-tx PDCCH source-order pair must remain in one OFDM symbol");
            found_nonadjacent_pair |= (grid1 % n_subcarriers) != (grid0 % n_subcarriers) + 1U;
        }
        expect_true(found_nonadjacent_pair,
                    "PCI 1 PDCCH layout must exercise non-adjacent physical source RE pairs");
        for (std::uint32_t re = 0U; re < n_source_re; re += 2U) {
            fill_pdcch_td_pair(buffers, desc, layout.grid_indices[re], layout.grid_indices[re + 1U],
                               {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F}, {1.0F, 0.0F}, symbols[re],
                               symbols[re + 1U]);
        }

        PdcchMmseInput input{};
        input.grid = make_grid_view(buffers);
        input.sfn_subframe = desc.sfn_subframe;
        input.cell_id = desc.cell_id;
        input.n_tx_ports = desc.n_tx_ports;
        input.tx_mode = desc.tx_mode;
        input.control_symbol_count = control_symbol_count;
        input.n_prb = n_prb;
        input.prb_bitmap = desc.prb_bitmap;
        input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                                  .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                                  .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

        mmse::pdcch::PdcchSiRntiSearchResult result{};
        expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_search(context, input, {}, result) ==
                            MmseStatus::kOk &&
                        result.hits.size() == 1U && result.hits[0].first_cce == first_cce &&
                        result.hits[0].decoded.dci.start_prb == start_prb &&
                        result.hits[0].decoded.dci.n_prb == 1U,
                    "two-tx CPU SI-RNTI search must decode selected LTE bandwidths");
    }
}

void test_pdcch_geometry_cache_tracks_bandwidth() {
    mmse::pdcch::PdcchSiRntiGeometrySearchRequest request{};
    request.cell_id = 19U;
    request.n_tx_ports = 1U;
    request.tx_mode = 1U;
    request.n_prb = 75U;
    request.control_subframe.kind = mmse::pdcch::LteControlSubframeKind::kRegular;

    mmse::pdcch::PdcchSiRntiGeometrySearchCache cache{};
    cache.locked = true;
    cache.cell_id = request.cell_id;
    cache.n_tx_ports = request.n_tx_ports;
    cache.tx_mode = request.tx_mode;
    cache.n_prb = 100U;
    cache.subframe_kind = request.control_subframe.kind;
    expect_true(!mmse::pdcch::detail::pdcch_si_rnti_geometry_cache_matches(cache, request),
                "SI-RNTI geometry cache must not cross carrier bandwidths");

    cache.n_prb = request.n_prb;
    expect_true(mmse::pdcch::detail::pdcch_si_rnti_geometry_cache_matches(cache, request),
                "SI-RNTI geometry cache must match an identical carrier bandwidth");
}

void test_pdcch_native_tail_biting_rate_recovery_l4_l8() {
    const std::vector<std::uint8_t> decoded_bits = make_si_rnti_dci_format1a_bits();
    const std::vector<float> convolutional_llrs = reference_llrs_from_encoded_bits(
        reference_pdcch_tail_biting_convolutional_encode(decoded_bits), 5.0F);
    for (const std::uint8_t aggregation_level : {1U, 2U, 4U, 8U}) {
        mmse::pdcch::PdcchCandidateLlr candidate{};
        candidate.chain.aggregation_level = aggregation_level;
        candidate.encoded_bit_count = 72U * aggregation_level;
        candidate.llrs = reference_pdcch_convolutional_rate_match(convolutional_llrs,
                                                                  candidate.encoded_bit_count);
        mmse::pdcch::PdcchRateRecoveredLlr recovered{};
        expect_true(mmse::pdcch::recover_pdcch_convolutional_rate_matched_llrs(
                        candidate,
                        mmse::pdcch::pdcch_dci_format1a_payload_bit_count(
                            mmse::pdcch::PdcchDciFormat1AConfig{}),
                        recovered) == MmseStatus::kOk,
                    "native tail-biting rate recovery must accept L4/L8 candidates");
        std::vector<std::uint8_t> decoded{};
        expect_true(mmse::pdcch::decode_pdcch_tail_biting_convolutional(recovered, decoded) ==
                            MmseStatus::kOk &&
                        decoded == decoded_bits,
                    "native tail-biting decoder must close rate recovery with repeats and dummies");
    }
}

mmse::pdcch::BackendPdcchEqualizedIndication
make_native_si_rnti_backend(std::uint16_t first_cce, std::uint8_t aggregation_level,
                            const std::vector<std::uint8_t>& decoded_bits,
                            std::uint32_t cce_count) {
    constexpr std::uint32_t kReCount = mmse::pdcch::kPdcchRegsPerCce * mmse::pdcch::kPdcchRePerReg;
    const std::uint32_t encoded_offset = static_cast<std::uint32_t>(first_cce) * 72U;
    const std::uint32_t encoded_count = 72U * aggregation_level;
    const std::vector<float> target_llrs = reference_pdcch_convolutional_rate_match(
        reference_llrs_from_encoded_bits(
            reference_pdcch_tail_biting_convolutional_encode(decoded_bits), 5.0F),
        encoded_count);
    expect_true(cce_count != 0U && encoded_offset + encoded_count <= cce_count * kReCount * 2U,
                "native SI test candidate must fit in control region");

    std::vector<float> descrambled_llrs(cce_count * kReCount * 2U, -1.0F);
    std::copy(target_llrs.begin(), target_llrs.end(), descrambled_llrs.begin() + encoded_offset);
    mmse::lte::descramble_llrs_inplace(descrambled_llrs.data(),
                                       static_cast<std::uint32_t>(descrambled_llrs.size()),
                                       mmse::lte::pdcch_c_init(19U, 7U));

    mmse::pdcch::BackendPdcchEqualizedIndication backend{};
    backend.sfn_subframe = 7U;
    backend.cell_id = 19U;
    backend.n_prb = kLteNumPrb20MHz;
    backend.n_tx_ports = 1U;
    backend.n_rx_ant = 2U;
    backend.n_layers = 1U;
    backend.tx_mode = 1U;
    backend.control_symbol_count = 1U;
    backend.mod_order = 2U;
    backend.chain.request_id = 451U;
    backend.x_hat_re.resize(cce_count * kReCount);
    backend.x_hat_im.resize(cce_count * kReCount);
    backend.sinr.assign(cce_count * kReCount, 1.0F);
    backend.re_grid_indices.resize(cce_count * kReCount);
    constexpr float kQpskAxis = 0.7071067811865475F;
    for (std::uint32_t re = 0U; re < backend.x_hat_re.size(); ++re) {
        backend.x_hat_re[re] = -descrambled_llrs[2U * re] * kQpskAxis / 2.0F;
        backend.x_hat_im[re] = -descrambled_llrs[2U * re + 1U] * kQpskAxis / 2.0F;
        backend.re_grid_indices[re] = static_cast<std::uint16_t>(re);
    }
    return backend;
}

std::vector<Complex32>
make_native_si_rnti_qpsk_symbols(std::uint32_t n_re, std::uint16_t first_cce,
                                 std::uint8_t aggregation_level,
                                 const std::vector<std::uint8_t>& decoded_bits,
                                 std::uint16_t cell_id, std::uint32_t sfn_subframe) {
    const std::uint32_t encoded_offset = static_cast<std::uint32_t>(first_cce) * 72U;
    const std::uint32_t encoded_count = 72U * aggregation_level;
    const std::vector<float> target_llrs = reference_pdcch_convolutional_rate_match(
        reference_llrs_from_encoded_bits(
            reference_pdcch_tail_biting_convolutional_encode(decoded_bits), 5.0F),
        encoded_count);
    expect_true(encoded_offset + encoded_count <= n_re * 2U,
                "native SI symbols must fit in PDCCH control region");

    std::vector<float> transmitted_llrs(n_re * 2U, -1.0F);
    std::copy(target_llrs.begin(), target_llrs.end(), transmitted_llrs.begin() + encoded_offset);
    mmse::lte::descramble_llrs_inplace(transmitted_llrs.data(),
                                       static_cast<std::uint32_t>(transmitted_llrs.size()),
                                       mmse::lte::pdcch_c_init(cell_id, sfn_subframe));

    constexpr float kQpskAxis = 0.7071067811865475F;
    std::vector<Complex32> symbols(n_re);
    for (std::uint32_t re = 0U; re < n_re; ++re) {
        symbols[re] = {
            .re = -transmitted_llrs[2U * re] * kQpskAxis / 2.0F,
            .im = -transmitted_llrs[2U * re + 1U] * kQpskAxis / 2.0F,
        };
    }
    return symbols;
}

struct GpuPdcchDecodeCase {
    GridBuffers buffers{};
    mmse::pdcch::PdcchGpuCommonSearchDecodeRequest request{};
    std::uint16_t expected_first_cce = 0U;
    std::uint8_t expected_aggregation_level = 0U;
    bool expected_hit = false;
};

GpuPdcchDecodeCase make_gpu_pdcch_decode_case(std::uint8_t control_symbol_count,
                                              std::uint8_t n_rx_ant, std::uint8_t n_tx_ports,
                                              std::uint8_t aggregation_level, std::uint16_t cell_id,
                                              std::uint32_t sfn_subframe, std::uint64_t request_id,
                                              bool hit, std::uint64_t generation) {
    auto desc = make_pdcch_desc(kLteNumPrb20MHz, control_symbol_count);
    desc.cell_id = cell_id;
    desc.sfn_subframe = sfn_subframe;
    desc.n_rx_ant = n_rx_ant;
    desc.n_tx_ports = n_tx_ports;
    desc.tx_mode = n_tx_ports == 1U ? 1U : 2U;

    GpuPdcchDecodeCase test_case{};
    test_case.buffers = make_zero_grid();
    fill_identity_channel(test_case.buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    mmse::pdcch::PdcchControlRegion control_region{};
    expect_true(mmse::pdcch::build_pdcch_control_region(desc.cell_id, desc.control_symbol_count,
                                                        layout.grid_indices.data(), n_source_re,
                                                        control_region) == MmseStatus::kOk,
                "GPU matrix control-region build");
    std::vector<mmse::pdcch::PdcchCommonSearchCandidate> candidates{};
    mmse::pdcch::build_pdcch_common_search_candidates(control_region, candidates);
    const auto candidate =
        std::find_if(candidates.begin(), candidates.end(), [aggregation_level](const auto& value) {
            return value.aggregation_level == aggregation_level;
        });
    expect_true(candidate != candidates.end(), "GPU matrix aggregation candidate must exist");

    if (hit) {
        const auto symbols = make_native_si_rnti_qpsk_symbols(
            n_source_re, candidate->first_cce, aggregation_level, make_si_rnti_dci_format1a_bits(),
            desc.cell_id, desc.sfn_subframe);
        if (n_tx_ports == 1U) {
            for (std::uint32_t re = 0U; re < n_source_re; ++re) {
                test_case.buffers.re[0][layout.grid_indices[re]] = symbols[re].re;
                test_case.buffers.im[0][layout.grid_indices[re]] = symbols[re].im;
                if (n_rx_ant == 2U) {
                    test_case.buffers.re[1][layout.grid_indices[re]] = symbols[re].re;
                    test_case.buffers.im[1][layout.grid_indices[re]] = symbols[re].im;
                }
            }
        } else {
            for (std::uint32_t re = 0U; re < n_source_re; re += 2U) {
                fill_td_pair(test_case.buffers, layout.grid_indices[re],
                             layout.grid_indices[re + 1U], {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                             {1.0F, 0.0F}, symbols[re], symbols[re + 1U]);
            }
        }
    }

    PdcchMmseInput input{};
    input.grid = n_rx_ant == 1U ? make_single_rx_grid_view(test_case.buffers)
                                : make_grid_view(test_case.buffers);
    input.grid.generation = generation;
    input.sfn_subframe = desc.sfn_subframe;
    input.cell_id = desc.cell_id;
    input.n_tx_ports = desc.n_tx_ports;
    input.tx_mode = desc.tx_mode;
    input.control_symbol_count = desc.control_symbol_count;
    input.n_prb = desc.n_prb;
    input.prb_bitmap = desc.prb_bitmap;
    input.control_re_exclusion_masks = desc.control_re_exclusion_masks;
    input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                              .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                              .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    input.chain.request_id = request_id;
    test_case.request.input = input;
    test_case.expected_first_cce = candidate->first_cce;
    test_case.expected_aggregation_level = aggregation_level;
    test_case.expected_hit = hit;
    return test_case;
}

void test_pdcch_native_si_rnti_search_returns_only_target_candidate() {
    const auto backend = make_native_si_rnti_backend(4U, 4U, make_si_rnti_dci_format1a_bits());
    mmse::pdcch::PdcchSiRntiSearchResult result{};
    expect_true(mmse::pdcch::decode_pdcch_si_rnti_dci_format1a(backend, {}, result) ==
                    MmseStatus::kOk,
                "native SI-RNTI search should complete");
    expect_true(result.candidate_count == 3U && result.hits.size() == 1U &&
                    result.hits[0].first_cce == 4U && result.hits[0].aggregation_level == 4U &&
                    result.hits[0].decoded.dci.start_prb == 10U &&
                    result.hits[0].decoded.dci.n_prb == 20U &&
                    result.hits[0].decoded.dci.mcs_tbs_index == 7U &&
                    result.hits[0].decoded.dci.redundancy_version == 2U &&
                    result.hits[0].decoded.dci.n_prb_1a == 3U,
                "native SI-RNTI search must expose target CCE and DCI grant fields");
}

void test_pdcch_ue_specific_search_decodes_l1_l2_candidates() {
    constexpr std::uint16_t kRnti = 0x1234U;
    std::vector<std::uint8_t> decoded_bits = make_si_rnti_dci_format1a_bits();
    rewrite_pdcch_crc_rnti_mask(decoded_bits, kRnti);
    for (const std::uint8_t aggregation_level : {1U, 2U}) {
        const auto backend =
            make_native_si_rnti_backend(0U, aggregation_level, decoded_bits, aggregation_level);
        mmse::pdcch::PdcchUeSpecificSearchConfig config{};
        config.rntis = {kRnti};
        config.aggregation_level_mask = aggregation_level == 1U
                                            ? mmse::pdcch::kPdcchAggregationLevelMaskL1
                                            : mmse::pdcch::kPdcchAggregationLevelMaskL2;
        mmse::pdcch::PdcchUeSpecificSearchResult result{};
        expect_true(mmse::pdcch::decode_pdcch_ue_specific_search_dci_format1a(
                        backend, config, result) == MmseStatus::kOk &&
                        result.candidate_count == 1U && result.decoded_candidate_count == 1U &&
                        result.crc_rnti_miss_count == 0U && result.semantic_reject_count == 0U &&
                        result.hits.size() == 1U && result.hits[0].rnti == kRnti &&
                        result.hits[0].first_cce == 0U &&
                        result.hits[0].aggregation_level == aggregation_level &&
                        result.hits[0].decoded.dci.start_prb == 10U &&
                        result.hits[0].decoded.dci.n_prb == 20U,
                    "pdcch UE-specific search must decode low aggregation candidates");

        FixedTailBitingDecoder fixed_decoder{.decoded_bits = decoded_bits};
        config.decoder = {.context = &fixed_decoder, .decode = fixed_tail_biting_decoder};
        result = {};
        expect_true(mmse::pdcch::decode_pdcch_ue_specific_search_dci_format1a(
                        backend, config, result) == MmseStatus::kOk &&
                        fixed_decoder.call_count == 1U && result.hits.size() == 1U &&
                        result.hits[0].aggregation_level == aggregation_level,
                    "pdcch UE-specific search must retain external decoder coverage");
    }
}

void test_pdcch_si_rnti_geometry_search_acquires_and_reprobes() {
    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(context.init(cpu_config) == MmseStatus::kOk,
                "PDCCH geometry search CPU context init");

    mmse::pdcch::PdcchSiRntiGeometrySearchRequest request{};
    request.sfn_subframe = 3U;
    request.cell_id = 0U;
    request.n_tx_ports = 1U;
    request.tx_mode = 1U;
    request.control_subframe = {
        .duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
        .subframe = 3U,
        .kind = mmse::pdcch::LteControlSubframeKind::kRegular,
    };
    const mmse::pdcch::PdcchControlGeometry target_geometry{
        .control_symbol_count = 1U,
        .phich_resource = mmse::pdcch::PhichResource::kOneSixth,
        .phich_duration = mmse::pdcch::PhichDuration::kNormal,
        .standard_reg_order = true,
    };

    auto desc = make_pdcch_desc();
    desc.sfn_subframe = request.sfn_subframe;
    desc.cell_id = request.cell_id;
    desc.n_tx_ports = request.n_tx_ports;
    desc.tx_mode = request.tx_mode;
    desc.control_symbol_count = target_geometry.control_symbol_count;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    request.grid = make_grid_view(buffers);
    mmse::PdcchMmseInput target_input{};
    expect_true(mmse::pdcch::detail::build_pdcch_si_rnti_geometry_input(
                    request, target_geometry, target_input) == MmseStatus::kOk,
                "PDCCH geometry search target input build");
    desc.control_re_exclusion_masks = target_input.control_re_exclusion_masks;
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    expect_true(n_source_re >= 144U, "PDCCH geometry search target must contain L4");
    for (std::uint32_t re = 0U; re < n_source_re; ++re) {
        buffers.re[0][layout.grid_indices[re]] = 0.125F + static_cast<float>(re) * 0.001F;
        buffers.im[0][layout.grid_indices[re]] = -0.375F + static_cast<float>(re) * 0.0005F;
    }
    request.grid = make_grid_view(buffers);

    constexpr std::uint32_t kMaxPdcchRe = kLteMaxControlSymbolsNormalCp * kLteNumSubcarriers20MHz;
    std::vector<float> x_hat_re(kMaxPdcchRe);
    std::vector<float> x_hat_im(kMaxPdcchRe);
    std::vector<float> sinr(kMaxPdcchRe);
    std::vector<std::uint16_t> re_grid_indices(kMaxPdcchRe);
    mmse::PdcchMmseOutputView output{
        .x_hat_re = x_hat_re.data(),
        .x_hat_im = x_hat_im.data(),
        .sinr = sinr.data(),
        .re_grid_indices = re_grid_indices.data(),
        .capacity_re_per_layer = kMaxPdcchRe,
        .capacity_re_metadata = kMaxPdcchRe,
    };
    mmse::PdcchMmseResult metadata{};
    expect_true(context.run_pdcch(target_input, output, metadata) == MmseStatus::kOk,
                "PDCCH geometry search target equalization");
    const auto target_backend =
        mmse::pdcch::make_backend_pdcch_equalized_indication(metadata, output);
    std::vector<mmse::pdcch::PdcchCandidateLlr> target_candidates{};
    expect_true(mmse::pdcch::build_pdcch_common_search_candidate_llrs(
                    target_backend, target_candidates) == MmseStatus::kOk,
                "PDCCH geometry search target candidates");
    const auto target_candidate =
        std::find_if(target_candidates.begin(), target_candidates.end(), [](const auto& candidate) {
            return candidate.chain.first_cce == 0U && candidate.chain.aggregation_level == 4U;
        });
    expect_true(target_candidate != target_candidates.end(),
                "PDCCH geometry search target L4 candidate");
    mmse::pdcch::PdcchRateRecoveredLlr target_recovered{};
    expect_true(mmse::pdcch::recover_pdcch_convolutional_rate_matched_llrs(
                    *target_candidate,
                    mmse::pdcch::pdcch_dci_format1a_payload_bit_count(
                        mmse::pdcch::PdcchDciFormat1AConfig{}),
                    target_recovered) == MmseStatus::kOk,
                "PDCCH geometry search target rate recovery");
    SelectiveTailBitingDecoder selective_decoder{
        .expected_llrs = target_recovered.convolutional_llrs,
        .matched_bits = make_si_rnti_dci_format1a_bits(),
        .miss_bits =
            [] {
                auto bits = make_si_rnti_dci_format1a_bits();
                bits.back() ^= 1U;
                return bits;
            }(),
    };
    const mmse::pdcch::PdcchSiRntiGeometrySearchConfig search_config{
        .decoder = {.context = &selective_decoder, .decode = selective_tail_biting_decoder},
    };

    mmse::pdcch::PdcchSiRntiGeometrySearchCache cache{};
    mmse::pdcch::PdcchSiRntiGeometrySearchResult result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(
                    context, request, search_config, cache, result) == MmseStatus::kOk,
                "PDCCH geometry search acquire status");
    expect_true(result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kAcquired,
                "PDCCH geometry search must acquire a unique SI-RNTI control geometry status=" +
                    std::to_string(static_cast<std::uint32_t>(result.status)) +
                    " attempts=" + std::to_string(result.geometry_attempt_count) +
                    " hits=" + std::to_string(result.decoded.hits.size()));
    expect_true(result.geometry_attempt_count == 16U && cache.locked &&
                    cache.geometry.control_symbol_count == target_geometry.control_symbol_count &&
                    cache.geometry.phich_resource == target_geometry.phich_resource &&
                    cache.geometry.phich_duration == target_geometry.phich_duration &&
                    result.decoded.hits.size() == 1U,
                "PDCCH geometry search acquisition metadata");

    auto changed_cell_request = request;
    changed_cell_request.cell_id = static_cast<std::uint16_t>(request.cell_id + 1U);
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(context, changed_cell_request,
                                                                   search_config, cache,
                                                                   result) == MmseStatus::kOk &&
                    result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kMiss &&
                    result.geometry_attempt_count == 16U && !cache.locked,
                "PDCCH geometry search must invalidate the cache when PCI changes");
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(
                    context, request, search_config, cache, result) == MmseStatus::kOk &&
                    result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kAcquired &&
                    cache.locked,
                "PDCCH geometry search must reacquire after PCI cache invalidation");

    buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    request.grid = make_grid_view(buffers);
    for (std::uint32_t miss = 1U; miss <= 4U; ++miss) {
        expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(
                        context, request, search_config, cache, result) == MmseStatus::kOk &&
                        result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kMiss &&
                        result.geometry_attempt_count == 1U && cache.locked &&
                        cache.consecutive_miss_count == miss,
                    "PDCCH geometry search must retain a locked geometry through four misses");
    }
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(
                    context, request, search_config, cache, result) == MmseStatus::kOk &&
                    result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kMiss &&
                    result.geometry_attempt_count == 16U && !cache.locked,
                "PDCCH geometry search must re-enumerate after the fifth miss");
}

void test_pdcch_si_rnti_geometry_search_two_tx_handles_empty_profiles() {
    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(context.init(cpu_config) == MmseStatus::kOk,
                "two-tx narrowband geometry CPU context init");

    auto desc = make_pdcch_desc(6U, 4U);
    desc.cell_id = 1U;
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    GridBuffers buffers = make_zero_grid(lte_downlink_subcarrier_count(desc.n_prb));
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);

    mmse::pdcch::PdcchSiRntiGeometrySearchRequest request{};
    request.grid = make_grid_view(buffers);
    request.sfn_subframe = desc.sfn_subframe;
    request.cell_id = desc.cell_id;
    request.n_tx_ports = desc.n_tx_ports;
    request.tx_mode = desc.tx_mode;
    request.n_prb = desc.n_prb;
    request.control_subframe = {
        .duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
        .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
        .kind = mmse::pdcch::LteControlSubframeKind::kRegular,
    };

    mmse::pdcch::PdcchSiRntiGeometrySearchCache cache{};
    mmse::pdcch::PdcchSiRntiGeometrySearchResult result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(context, request, {}, cache,
                                                                   result) == MmseStatus::kOk &&
                    result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kMiss &&
                    result.geometry_attempt_count == 24U && result.candidate_count > 0U,
                "two-tx 6 PRB geometry search must treat zero-CCE profiles as normal misses");

    cache = {
        .locked = true,
        .cell_id = request.cell_id,
        .n_tx_ports = request.n_tx_ports,
        .tx_mode = request.tx_mode,
        .n_prb = request.n_prb,
        .subframe_kind = request.control_subframe.kind,
        .geometry =
            {
                .control_symbol_count = 4U,
                .phich_resource = mmse::pdcch::PhichResource::kOne,
                .phich_duration = mmse::pdcch::PhichDuration::kNormal,
                .standard_reg_order = true,
            },
    };
    for (std::uint32_t miss = 1U; miss <= 4U; ++miss) {
        expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(context, request, {}, cache,
                                                                       result) == MmseStatus::kOk &&
                        result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kMiss &&
                        result.geometry_attempt_count == 1U && cache.locked &&
                        cache.consecutive_miss_count == miss,
                    "two-tx geometry cache must retain its lock for four misses");
    }
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(context, request, {}, cache,
                                                                   result) == MmseStatus::kOk &&
                    result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kMiss &&
                    result.geometry_attempt_count == 24U && !cache.locked,
                "two-tx geometry cache must reprobe after the fifth miss");
}

void test_pdcch_si_rnti_geometry_search_rejects_ambiguous_locks() {
    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    expect_true(context.init(cpu_config) == MmseStatus::kOk,
                "PDCCH ambiguous geometry CPU context init");

    auto desc = make_pdcch_desc();
    desc.control_symbol_count = 3U;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::pdcch::PdcchSiRntiGeometrySearchRequest request{};
    request.grid = make_grid_view(buffers);
    request.sfn_subframe = desc.sfn_subframe;
    request.cell_id = desc.cell_id;
    request.n_tx_ports = desc.n_tx_ports;
    request.tx_mode = desc.tx_mode;
    request.control_subframe = {
        .duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
        .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
        .kind = mmse::pdcch::LteControlSubframeKind::kRegular,
    };
    FixedTailBitingDecoder fixed_decoder{.decoded_bits = make_si_rnti_dci_format1a_bits()};
    mmse::pdcch::PdcchSiRntiGeometrySearchConfig config{
        .decoder = {.context = &fixed_decoder, .decode = fixed_tail_biting_decoder},
    };
    mmse::pdcch::PdcchSiRntiGeometrySearchCache cache{};
    mmse::pdcch::PdcchSiRntiGeometrySearchResult result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_geometry_search(context, request, config, cache,
                                                                   result) == MmseStatus::kOk &&
                    result.status == mmse::pdcch::PdcchSiRntiGeometrySearchStatus::kAmbiguous &&
                    result.geometry_attempt_count == 16U && !cache.locked &&
                    result.decoded.hits.empty() && fixed_decoder.call_count > 1U,
                "PDCCH geometry search must not lock an ambiguous SI-RNTI geometry");
}

void test_pdcch_cpu_si_rnti_search_runs_full_one_tx_chain() {
    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(context.init(cpu_config) == MmseStatus::kOk, "native SI one-tx cpu init");

    const auto desc = make_pdcch_desc();
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    const auto symbols = make_native_si_rnti_qpsk_symbols(
        n_source_re, 4U, 4U, make_si_rnti_dci_format1a_bits(), desc.cell_id, desc.sfn_subframe);
    for (std::uint32_t re = 0U; re < n_source_re; ++re) {
        buffers.re[0][layout.grid_indices[re]] = symbols[re].re;
        buffers.im[0][layout.grid_indices[re]] = symbols[re].im;
    }

    PdcchMmseInput input{};
    input.grid = make_grid_view(buffers);
    input.sfn_subframe = desc.sfn_subframe;
    input.cell_id = desc.cell_id;
    input.n_tx_ports = desc.n_tx_ports;
    input.tx_mode = desc.tx_mode;
    input.control_symbol_count = desc.control_symbol_count;
    input.n_prb = desc.n_prb;
    input.prb_bitmap = desc.prb_bitmap;
    input.control_re_exclusion_masks = desc.control_re_exclusion_masks;
    input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                              .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                              .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    input.chain.request_id = 90301U;

    mmse::pdcch::PdcchSiRntiSearchResult result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_search(context, input, {}, result) ==
                    MmseStatus::kOk,
                "native SI one-tx CPU entry should run full chain");
    expect_true(result.hits.size() == 1U && result.hits[0].first_cce == 4U &&
                    result.hits[0].decoded.dci.chain.request_id == input.chain.request_id &&
                    result.hits[0].decoded.dci.start_prb == 10U &&
                    result.hits[0].decoded.dci.n_prb == 20U &&
                    result.hits[0].decoded.dci.mcs_tbs_index == 7U &&
                    result.hits[0].decoded.dci.redundancy_version == 2U &&
                    result.hits[0].decoded.dci.n_prb_1a == 3U,
                "native SI one-tx entry must retain target CCE and DCI fields");

    mmse::pdcch::PdcchCommonSearchDecodeResult common_result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_common_search_decode(context, input, {},
                                                                common_result) == MmseStatus::kOk &&
                    common_result.hits.size() == 1U &&
                    common_result.hits[0].dci.chain.first_cce == 4U,
                "common-search CPU entry must default to native tail-biting Viterbi");
}

void test_pdcch_cpu_si_rnti_search_runs_full_two_tx_chain() {
    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(context.init(cpu_config) == MmseStatus::kOk, "native SI two-tx cpu init");

    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    const auto symbols = make_native_si_rnti_qpsk_symbols(
        n_source_re, 4U, 4U, make_si_rnti_dci_format1a_bits(), desc.cell_id, desc.sfn_subframe);
    expect_true((n_source_re & 1U) == 0U, "native SI two-tx source RE count must be even");
    for (std::uint32_t re = 0U; re < n_source_re; re += 2U) {
        fill_pdcch_td_pair(buffers, desc, layout.grid_indices[re], layout.grid_indices[re + 1U],
                           {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F}, {1.0F, 0.0F}, symbols[re],
                           symbols[re + 1U]);
    }

    PdcchMmseInput input{};
    input.grid = make_grid_view(buffers);
    input.sfn_subframe = desc.sfn_subframe;
    input.cell_id = desc.cell_id;
    input.n_tx_ports = desc.n_tx_ports;
    input.tx_mode = desc.tx_mode;
    input.control_symbol_count = desc.control_symbol_count;
    input.n_prb = desc.n_prb;
    input.prb_bitmap = desc.prb_bitmap;
    input.control_re_exclusion_masks = desc.control_re_exclusion_masks;
    input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                              .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                              .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    input.chain.request_id = 90302U;

    mmse::pdcch::PdcchSiRntiSearchResult result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_si_rnti_search(context, input, {}, result) ==
                    MmseStatus::kOk,
                "native SI two-tx CPU entry should run full chain");
    expect_true(result.hits.size() == 1U && result.hits[0].first_cce == 4U &&
                    result.hits[0].decoded.dci.chain.request_id == input.chain.request_id &&
                    result.hits[0].decoded.dci.start_prb == 10U &&
                    result.hits[0].decoded.dci.n_prb == 20U &&
                    result.hits[0].decoded.dci.mcs_tbs_index == 7U &&
                    result.hits[0].decoded.dci.redundancy_version == 2U &&
                    result.hits[0].decoded.dci.n_prb_1a == 3U,
                "native SI two-tx entry must normalize CCE order and retain DCI fields");
}

void test_pdcch_cpu_ue_specific_search_runs_full_one_and_two_tx_chains() {
    constexpr std::uint16_t kRnti = 0x1234U;
    for (const std::uint8_t n_tx_ports : {1U, 2U}) {
        MmseEqualizerCpuContext context;
        MmseEqualizerCpuConfig cpu_config{};
        cpu_config.worker_count = 1U;
        cpu_config.sigma2_min = 1.0e-5F;
        expect_true(context.init(cpu_config) == MmseStatus::kOk, "UE-specific full-chain CPU init");

        auto desc = make_pdcch_desc();
        desc.n_tx_ports = n_tx_ports;
        desc.tx_mode = n_tx_ports == 1U ? 1U : 2U;
        GridBuffers buffers = make_zero_grid();
        fill_identity_channel(buffers, desc, 0.0F, 0.0F);
        mmse::detail::ReLayout layout{};
        const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
        mmse::pdcch::PdcchControlRegion control_region{};
        expect_true(mmse::pdcch::build_pdcch_control_region(desc.cell_id, desc.control_symbol_count,
                                                            layout.grid_indices.data(), n_source_re,
                                                            control_region) == MmseStatus::kOk,
                    "UE-specific full-chain control-region build");

        const std::uint8_t aggregation_level = n_tx_ports == 1U ? 1U : 2U;
        mmse::pdcch::PdcchUeSpecificSearchConfig config{};
        config.rntis = {kRnti};
        config.aggregation_level_mask = aggregation_level == 1U
                                            ? mmse::pdcch::kPdcchAggregationLevelMaskL1
                                            : mmse::pdcch::kPdcchAggregationLevelMaskL2;
        std::vector<mmse::pdcch::PdcchUeSpecificSearchCandidate> candidates{};
        expect_true(mmse::pdcch::build_pdcch_ue_specific_search_candidates(
                        control_region, desc.sfn_subframe, config, candidates) == MmseStatus::kOk &&
                        !candidates.empty(),
                    "UE-specific full-chain candidate build");
        const std::uint16_t first_cce = candidates.front().search_candidate.first_cce;
        std::vector<std::uint8_t> decoded_bits = make_si_rnti_dci_format1a_bits();
        rewrite_pdcch_crc_rnti_mask(decoded_bits, kRnti);
        const auto symbols =
            make_native_si_rnti_qpsk_symbols(n_source_re, first_cce, aggregation_level,
                                             decoded_bits, desc.cell_id, desc.sfn_subframe);
        if (n_tx_ports == 1U) {
            for (std::uint32_t re = 0U; re < n_source_re; ++re) {
                buffers.re[0][layout.grid_indices[re]] = symbols[re].re;
                buffers.im[0][layout.grid_indices[re]] = symbols[re].im;
            }
        } else {
            expect_true((n_source_re & 1U) == 0U,
                        "UE-specific two-tx source RE count must be even");
            for (std::uint32_t re = 0U; re < n_source_re; re += 2U) {
                fill_pdcch_td_pair(buffers, desc, layout.grid_indices[re],
                                   layout.grid_indices[re + 1U], {1.0F, 0.0F}, {0.0F, 0.0F},
                                   {0.0F, 0.0F}, {1.0F, 0.0F}, symbols[re], symbols[re + 1U]);
            }
        }

        PdcchMmseInput input{};
        input.grid = make_grid_view(buffers);
        input.sfn_subframe = desc.sfn_subframe;
        input.cell_id = desc.cell_id;
        input.n_tx_ports = desc.n_tx_ports;
        input.tx_mode = desc.tx_mode;
        input.control_symbol_count = desc.control_symbol_count;
        input.n_prb = desc.n_prb;
        input.prb_bitmap = desc.prb_bitmap;
        input.control_re_exclusion_masks = desc.control_re_exclusion_masks;
        input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                                  .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                                  .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
        input.chain.request_id = 90400U + n_tx_ports;

        mmse::pdcch::PdcchUeSpecificSearchResult result{};
        expect_true(mmse::pdcch::run_pdcch_cpu_ue_specific_search(context, input, config, result) ==
                            MmseStatus::kOk &&
                        result.hits.size() == 1U && result.hits[0].rnti == kRnti &&
                        result.hits[0].first_cce == first_cce &&
                        result.hits[0].aggregation_level == aggregation_level &&
                        result.hits[0].decoded.dci.chain.request_id == input.chain.request_id &&
                        result.hits[0].decoded.dci.start_prb == 10U &&
                        result.hits[0].decoded.dci.n_prb == 20U,
                    "UE-specific CPU entry must decode the target CCE for both transmit modes");
    }
}

void test_pdcch_common_search_decode_returns_all_hits() {
    constexpr std::uint32_t kCceCount = 4U;
    constexpr std::uint32_t kReCount =
        kCceCount * mmse::pdcch::kPdcchRegsPerCce * mmse::pdcch::kPdcchRePerReg;
    mmse::pdcch::BackendPdcchEqualizedIndication backend{};
    backend.sfn_subframe = 7U;
    backend.cell_id = 19U;
    backend.n_tx_ports = 1U;
    backend.n_rx_ant = 2U;
    backend.n_layers = 1U;
    backend.tx_mode = 1U;
    backend.control_symbol_count = 1U;
    backend.mod_order = 2U;
    backend.chain.request_id = 880U;
    backend.x_hat_re.assign(kReCount, 0.5F);
    backend.x_hat_im.assign(kReCount, -0.5F);
    backend.sinr.assign(kReCount, 1.0F);
    backend.re_grid_indices.resize(kReCount);
    for (std::uint32_t re = 0U; re < kReCount; ++re) {
        backend.re_grid_indices[re] = static_cast<std::uint16_t>(re);
    }

    FixedTailBitingDecoder fixed_decoder{.decoded_bits = make_si_rnti_dci_format1a_bits()};
    mmse::pdcch::PdcchCommonSearchDecodeConfig config{};
    config.decoder = {.context = &fixed_decoder, .decode = fixed_tail_biting_decoder};
    mmse::pdcch::PdcchCommonSearchDecodeResult result{};
    expect_true(mmse::pdcch::decode_pdcch_common_search_dci_format1a(backend, config, result) ==
                    MmseStatus::kOk,
                "common-search decode should complete");
    expect_true(result.candidate_count == 1U && result.hits.size() == 1U &&
                    fixed_decoder.call_count == 1U,
                "common-search decode should retain every matching candidate");
    expect_true(result.hits[0].dci.chain.request_id == backend.chain.request_id &&
                    result.hits[0].dci.chain.first_cce == 0U &&
                    result.hits[0].dci.chain.aggregation_level == 4U &&
                    result.hits[0].dci.start_prb == 10U && result.hits[0].dci.n_prb == 20U,
                "common-search hit must preserve CCE metadata and parsed DCI");
}

void test_pdcch_cpu_common_search_decode_runs_full_one_tx_chain() {
    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(context.init(cpu_config) == MmseStatus::kOk, "cpu common-search init");

    const auto desc = make_pdcch_desc();
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 1.0F, 0.0F);
    PdcchMmseInput input{};
    input.grid = make_grid_view(buffers);
    input.sfn_subframe = desc.sfn_subframe;
    input.cell_id = desc.cell_id;
    input.n_tx_ports = desc.n_tx_ports;
    input.tx_mode = desc.tx_mode;
    input.control_symbol_count = desc.control_symbol_count;
    input.n_prb = desc.n_prb;
    input.prb_bitmap = desc.prb_bitmap;
    input.control_re_exclusion_masks = desc.control_re_exclusion_masks;
    input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                              .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                              .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    input.chain.request_id = 90210U;

    FixedTailBitingDecoder fixed_decoder{.decoded_bits = make_si_rnti_dci_format1a_bits()};
    mmse::pdcch::PdcchCommonSearchDecodeConfig config{};
    config.decoder = {.context = &fixed_decoder, .decode = fixed_tail_biting_decoder};
    mmse::pdcch::PdcchCommonSearchDecodeResult result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_common_search_decode(context, input, config, result) ==
                    MmseStatus::kOk,
                "cpu common-search entry should run the complete one-tx chain");
    expect_true(result.candidate_count == 6U && result.hits.size() == 6U &&
                    fixed_decoder.call_count == 6U,
                "cpu common-search entry should decode every public-search candidate");
    for (const auto& hit : result.hits) {
        expect_true(hit.dci.chain.request_id == input.chain.request_id &&
                        hit.dci.rnti == mmse::pdcch::kSiRnti && hit.dci.start_prb == 10U &&
                        hit.dci.n_prb == 20U,
                    "cpu common-search hit fields");
    }
}

void test_pdcch_cpu_common_search_decode_runs_full_two_tx_chain() {
    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(context.init(cpu_config) == MmseStatus::kOk, "cpu two-tx common-search init");

    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};
    fill_pdcch_td_layout(buffers, desc, layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                         {1.0F, 0.0F}, s0, s1);

    PdcchMmseInput input{};
    input.grid = make_grid_view(buffers);
    input.sfn_subframe = desc.sfn_subframe;
    input.cell_id = desc.cell_id;
    input.n_tx_ports = desc.n_tx_ports;
    input.tx_mode = desc.tx_mode;
    input.control_symbol_count = desc.control_symbol_count;
    input.n_prb = desc.n_prb;
    input.prb_bitmap = desc.prb_bitmap;
    input.control_re_exclusion_masks = desc.control_re_exclusion_masks;
    input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                              .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                              .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    input.chain.request_id = 90211U;

    FixedTailBitingDecoder fixed_decoder{.decoded_bits = make_si_rnti_dci_format1a_bits()};
    mmse::pdcch::PdcchCommonSearchDecodeConfig config{};
    config.decoder = {.context = &fixed_decoder, .decode = fixed_tail_biting_decoder};
    mmse::pdcch::PdcchCommonSearchDecodeResult result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_common_search_decode(context, input, config, result) ==
                    MmseStatus::kOk,
                "cpu common-search entry should run the complete two-tx chain");
    expect_true(n_source_re == 3168U && result.candidate_count == 6U && result.hits.size() == 6U &&
                    fixed_decoder.call_count == 6U,
                "two-tx entry should normalize all CCE-ordered candidates before decoding");
    for (const auto& hit : result.hits) {
        expect_true(hit.dci.chain.request_id == input.chain.request_id &&
                        hit.dci.rnti == mmse::pdcch::kSiRnti && hit.dci.start_prb == 10U &&
                        hit.dci.n_prb == 20U,
                    "two-tx common-search hit fields");
    }
}

void test_pdcch_gpu_common_search_decode_matches_cpu_and_rejects_callbacks() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "GPU common-search CPU reference init");

    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    gpu_config.stream_count = 2U;
    gpu_config.sigma2_min = cpu_config.sigma2_min;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "GPU common-search CUDA init");

    const auto desc = make_pdcch_desc();
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    const auto symbols = make_native_si_rnti_qpsk_symbols(
        n_source_re, 4U, 4U, make_si_rnti_dci_format1a_bits(), desc.cell_id, desc.sfn_subframe);
    for (std::uint32_t re = 0U; re < n_source_re; ++re) {
        buffers.re[0][layout.grid_indices[re]] = symbols[re].re;
        buffers.im[0][layout.grid_indices[re]] = symbols[re].im;
    }

    PdcchMmseInput input{};
    input.grid = make_grid_view(buffers);
    input.sfn_subframe = desc.sfn_subframe;
    input.cell_id = desc.cell_id;
    input.n_tx_ports = desc.n_tx_ports;
    input.tx_mode = desc.tx_mode;
    input.control_symbol_count = desc.control_symbol_count;
    input.n_prb = desc.n_prb;
    input.prb_bitmap = desc.prb_bitmap;
    input.control_re_exclusion_masks = desc.control_re_exclusion_masks;
    input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                              .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                              .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    input.chain.request_id = 90501U;

    mmse::pdcch::PdcchCommonSearchDecodeResult cpu_result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_common_search_decode(cpu, input, {}, cpu_result) ==
                    MmseStatus::kOk,
                "GPU common-search CPU reference decode");

    mmse::pdcch::PdcchGpuCommonSearchDecodeRequest request{};
    request.input = input;
    mmse::pdcch::PdcchGpuCommonSearchDecodeResult gpu_result{};
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(gpu, request, gpu_result) ==
                    MmseStatus::kOk,
                "GPU common-search device decode");
    expect_true(gpu_result.candidate_count == cpu_result.candidate_count &&
                    gpu_result.hits.size() == cpu_result.hits.size() &&
                    gpu_result.profile.crc_hit_count == gpu_result.hits.size() &&
                    gpu_result.profile.crc_miss_count + gpu_result.profile.crc_hit_count ==
                        gpu_result.candidate_count,
                "GPU common-search candidate and CRC accounting");
    expect_true(gpu_result.profile.d2h_bytes <
                    static_cast<std::uint64_t>(n_source_re) * 3U * sizeof(float),
                "GPU common-search must not transfer equalized soft information");
    const std::uint64_t expected_one_tx_h2d_bytes =
        2U * static_cast<std::uint64_t>(input.grid.n_rx_ant) * mmse::detail::kCudaMaxGridRe *
            sizeof(float) +
        offsetof(mmse::detail::CudaGridMeta, grid_indices) +
        static_cast<std::uint64_t>(n_source_re) * sizeof(std::uint16_t) + sizeof(float) +
        static_cast<std::uint64_t>(gpu_result.candidate_count) *
            sizeof(mmse::detail::CudaPdcchCandidateDescriptor) +
        static_cast<std::uint64_t>((n_source_re * 2U + 31U) / 32U) * sizeof(std::uint32_t);
    const std::uint64_t expected_d2h_bytes =
        sizeof(std::uint32_t) + static_cast<std::uint64_t>(mmse::detail::kCudaPdcchMaxCandidates) *
                                    sizeof(mmse::detail::CudaPdcchCandidateResult);
    expect_true(gpu_result.profile.h2d_bytes == expected_one_tx_h2d_bytes &&
                    gpu_result.profile.d2h_bytes == expected_d2h_bytes,
                "one-tx GPU common-search transfer accounting must match memcpy sizes");
    const std::uint64_t legacy_h2d_bytes =
        4U * kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz * sizeof(float) +
        sizeof(mmse::detail::CudaGridMeta) +
        static_cast<std::uint64_t>(gpu_result.candidate_count) *
            sizeof(mmse::detail::CudaPdcchCandidateDescriptor);
    expect_true(gpu_result.profile.h2d_bytes < legacy_h2d_bytes,
                "GPU common-search must upload only the PDCCH metadata layout slice");
    for (std::size_t index = 0U; index < cpu_result.hits.size(); ++index) {
        const auto& expected = cpu_result.hits[index];
        const auto& actual = gpu_result.hits[index];
        expect_true(actual.crc.transmitted_crc == expected.crc.transmitted_crc &&
                        actual.crc.calculated_crc == expected.crc.calculated_crc &&
                        actual.crc.unmasked_rnti == expected.crc.unmasked_rnti &&
                        actual.crc.matches_expected_rnti == expected.crc.matches_expected_rnti &&
                        actual.dci.chain.candidate_id == expected.dci.chain.candidate_id &&
                        actual.dci.chain.first_cce == expected.dci.chain.first_cce &&
                        actual.dci.chain.aggregation_level ==
                            expected.dci.chain.aggregation_level &&
                        actual.dci.raw_payload_bits == expected.dci.raw_payload_bits &&
                        actual.dci.start_prb == expected.dci.start_prb &&
                        actual.dci.n_prb == expected.dci.n_prb &&
                        actual.dci.mcs_tbs_index == expected.dci.mcs_tbs_index &&
                        actual.dci.redundancy_version == expected.dci.redundancy_version,
                    "GPU common-search DCI and CRC equivalence");
    }

    PdcchMmseInput single_rx_input = input;
    single_rx_input.grid.n_rx_ant = 1U;
    mmse::pdcch::PdcchCommonSearchDecodeResult single_rx_cpu_result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_common_search_decode(
                    cpu, single_rx_input, {}, single_rx_cpu_result) == MmseStatus::kOk,
                "single-RX CPU common-search reference decode");
    auto single_rx_request = request;
    single_rx_request.input = single_rx_input;
    mmse::pdcch::PdcchGpuCommonSearchDecodeResult single_rx_gpu_result{};
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(
                    gpu, single_rx_request, single_rx_gpu_result) == MmseStatus::kOk &&
                    single_rx_gpu_result.hits.size() == single_rx_cpu_result.hits.size() &&
                    !single_rx_gpu_result.hits.empty() &&
                    single_rx_gpu_result.hits[0].dci.raw_payload_bits ==
                        single_rx_cpu_result.hits[0].dci.raw_payload_bits &&
                    single_rx_gpu_result.profile.h2d_bytes < gpu_result.profile.h2d_bytes,
                "single-RX GPU common-search must preserve decoding while skipping RX1 upload");

    std::array<mmse::pdcch::PdcchGpuCommonSearchDecodeRequest, 2> batch_requests = {
        request,
        request,
    };
    batch_requests[1].input.sfn_subframe += 10U;
    batch_requests[1].input.control_subframe.subframe =
        static_cast<std::uint8_t>(batch_requests[1].input.sfn_subframe % 10U);
    batch_requests[1].input.chain.request_id += 1U;
    std::array<mmse::pdcch::PdcchGpuCommonSearchDecodeResult, 2> batch_results{};
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode_batch(
                    gpu, batch_requests, batch_results) == MmseStatus::kOk &&
                    batch_results[0].candidate_count == cpu_result.candidate_count &&
                    batch_results[1].candidate_count == cpu_result.candidate_count &&
                    batch_results[0].hits.size() == cpu_result.hits.size() &&
                    batch_results[1].hits.size() == cpu_result.hits.size() &&
                    batch_results[0].hits[0].dci.raw_payload_bits ==
                        cpu_result.hits[0].dci.raw_payload_bits &&
                    batch_results[1].hits[0].dci.raw_payload_bits ==
                        cpu_result.hits[0].dci.raw_payload_bits,
                "GPU common-search batch must preserve CPU-equivalent DCI bits");

    auto failing_batch = batch_requests;
    failing_batch[1].input.n_prb = kLteNumPrb20MHz - 1U;
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode_batch(
                    gpu, failing_batch, batch_results) == MmseStatus::kUnsupportedConfig,
                "GPU common-search batch must report a mid-submission failure");
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(gpu, request, gpu_result) ==
                        MmseStatus::kOk &&
                    !gpu_result.hits.empty(),
                "GPU common-search context must recover after a partial batch failure");

    for (std::uint32_t re = 0U; re < n_source_re; ++re) {
        buffers.re[0][layout.grid_indices[re]] = 0.0F;
        buffers.im[0][layout.grid_indices[re]] = 0.0F;
    }
    request.input.grid.generation = 1U;
    request.input.chain.request_id = 90510U;
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(gpu, request, gpu_result) ==
                        MmseStatus::kOk &&
                    gpu_result.hits.empty() && gpu_result.profile.crc_hit_count == 0U,
                "GPU common-search slot reuse must clear a previous hit");
    for (std::uint32_t re = 0U; re < n_source_re; ++re) {
        buffers.re[0][layout.grid_indices[re]] = symbols[re].re;
        buffers.im[0][layout.grid_indices[re]] = symbols[re].im;
    }
    request.input.grid.generation = 2U;
    request.input.chain.request_id = 90511U;
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(gpu, request, gpu_result) ==
                        MmseStatus::kOk &&
                    gpu_result.hits.size() == 1U &&
                    gpu_result.hits[0].dci.chain.request_id == request.input.chain.request_id,
                "GPU common-search slot reuse must return only the new generation hit");

    FixedTailBitingDecoder decoder{.decoded_bits = make_si_rnti_dci_format1a_bits()};
    request.config.decoder = {.context = &decoder, .decode = fixed_tail_biting_decoder};
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(gpu, request, gpu_result) ==
                    MmseStatus::kUnsupportedConfig,
                "GPU common-search must reject external decoder callback");

    request.config = {};
    auto unsupported_request = request;
    unsupported_request.input.control_subframe.duplex_mode = mmse::pdcch::PhichDuplexMode::kTdd;
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(
                    gpu, unsupported_request, gpu_result) == MmseStatus::kUnsupportedConfig,
                "GPU common-search must reject TDD");

    unsupported_request = request;
    unsupported_request.input.n_prb = kLteNumPrb20MHz - 1U;
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(
                    gpu, unsupported_request, gpu_result) == MmseStatus::kUnsupportedConfig,
                "GPU common-search must reject non-20MHz input");

    for (const auto [n_tx_ports, tx_mode] :
         {std::pair<std::uint8_t, std::uint8_t>{1U, 2U}, {2U, 1U}, {0U, 1U}, {3U, 2U}}) {
        unsupported_request = request;
        unsupported_request.input.n_tx_ports = n_tx_ports;
        unsupported_request.input.tx_mode = tx_mode;
        expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(
                        gpu, unsupported_request, gpu_result) == MmseStatus::kUnsupportedConfig,
                    "GPU common-search must reject unsupported port/mode pair");
    }

    unsupported_request = request;
    unsupported_request.input.grid.n_symbols = kLteNumSymbolsNormalCp - 2U;
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(
                    gpu, unsupported_request, gpu_result) == MmseStatus::kUnsupportedConfig,
                "GPU common-search must reject non-normal CP input");

    std::array<mmse::pdcch::PdcchGpuCommonSearchDecodeResult, 1> short_results{};
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode_batch(
                    gpu, batch_requests, short_results) == MmseStatus::kInvalidArgument,
                "GPU common-search batch must require equal request and result spans");
#endif
}

void test_pdcch_gpu_common_search_two_tx_matches_cpu() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "two-tx GPU reference init");

    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    gpu_config.sigma2_min = cpu_config.sigma2_min;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "two-tx GPU CUDA init");

    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    const auto symbols = make_native_si_rnti_qpsk_symbols(
        n_source_re, 4U, 4U, make_si_rnti_dci_format1a_bits(), desc.cell_id, desc.sfn_subframe);
    for (std::uint32_t pair = 0U; pair < n_source_re / 2U; ++pair) {
        fill_pdcch_td_pair(buffers, desc, layout.grid_indices[2U * pair],
                           layout.grid_indices[2U * pair + 1U], {1.0F, 0.0F}, {0.0F, 0.0F},
                           {0.0F, 0.0F}, {1.0F, 0.0F}, symbols[2U * pair], symbols[2U * pair + 1U]);
    }

    PdcchMmseInput input{};
    input.grid = make_grid_view(buffers);
    input.sfn_subframe = desc.sfn_subframe;
    input.cell_id = desc.cell_id;
    input.n_tx_ports = desc.n_tx_ports;
    input.tx_mode = desc.tx_mode;
    input.control_symbol_count = desc.control_symbol_count;
    input.n_prb = desc.n_prb;
    input.prb_bitmap = desc.prb_bitmap;
    input.control_re_exclusion_masks = desc.control_re_exclusion_masks;
    input.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                              .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                              .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    input.chain.request_id = 90503U;

    mmse::pdcch::PdcchCommonSearchDecodeResult cpu_result{};
    expect_true(mmse::pdcch::run_pdcch_cpu_common_search_decode(cpu, input, {}, cpu_result) ==
                    MmseStatus::kOk,
                "two-tx GPU common-search CPU reference decode");
    mmse::pdcch::PdcchGpuCommonSearchDecodeRequest request{};
    request.input = input;
    mmse::pdcch::PdcchGpuCommonSearchDecodeResult gpu_result{};
    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(gpu, request, gpu_result) ==
                    MmseStatus::kOk,
                "two-tx GPU common-search device decode");
    expect_true(gpu_result.candidate_count == cpu_result.candidate_count &&
                    gpu_result.hits.size() == cpu_result.hits.size() &&
                    gpu_result.hits.size() == 1U,
                "two-tx GPU common-search candidate and hit count equivalence");
    const auto& expected = cpu_result.hits[0];
    const auto& actual = gpu_result.hits[0];
    expect_true(actual.crc.transmitted_crc == expected.crc.transmitted_crc &&
                    actual.crc.calculated_crc == expected.crc.calculated_crc &&
                    actual.crc.unmasked_rnti == expected.crc.unmasked_rnti &&
                    actual.crc.matches_expected_rnti == expected.crc.matches_expected_rnti &&
                    actual.dci.chain.request_id == expected.dci.chain.request_id &&
                    actual.dci.chain.candidate_id == expected.dci.chain.candidate_id &&
                    actual.dci.chain.first_cce == expected.dci.chain.first_cce &&
                    actual.dci.chain.aggregation_level == expected.dci.chain.aggregation_level &&
                    actual.dci.raw_payload_bits == expected.dci.raw_payload_bits &&
                    actual.dci.start_prb == expected.dci.start_prb &&
                    actual.dci.n_prb == expected.dci.n_prb &&
                    actual.dci.mcs_tbs_index == expected.dci.mcs_tbs_index &&
                    actual.dci.redundancy_version == expected.dci.redundancy_version &&
                    actual.dci.n_prb_1a == expected.dci.n_prb_1a &&
                    actual.dci.n_prb_1a_is_three == expected.dci.n_prb_1a_is_three,
                "two-tx GPU common-search DCI and CRC equivalence");
    const std::uint64_t expected_h2d_bytes =
        2U * static_cast<std::uint64_t>(input.grid.n_rx_ant) * mmse::detail::kCudaMaxGridRe *
            sizeof(float) +
        sizeof(mmse::detail::CudaGridMeta) + sizeof(float) +
        static_cast<std::uint64_t>(gpu_result.candidate_count) *
            sizeof(mmse::detail::CudaPdcchCandidateDescriptor) +
        static_cast<std::uint64_t>((n_source_re * 2U + 31U) / 32U) * sizeof(std::uint32_t);
    const std::uint64_t expected_d2h_bytes =
        sizeof(std::uint32_t) + static_cast<std::uint64_t>(mmse::detail::kCudaPdcchMaxCandidates) *
                                    sizeof(mmse::detail::CudaPdcchCandidateResult);
    expect_true(gpu_result.profile.h2d_bytes == expected_h2d_bytes &&
                    gpu_result.profile.d2h_bytes == expected_d2h_bytes,
                "two-tx GPU common-search transfer accounting must match memcpy sizes");
#endif
}

void test_pdcch_gpu_common_search_cfi_rx_aggregation_matrix() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "GPU CFI/RX matrix CPU init");

    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    gpu_config.stream_count = 1U;
    gpu_config.sigma2_min = cpu_config.sigma2_min;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "GPU CFI/RX matrix CUDA init");

    std::uint64_t request_id = 90600U;
    for (const std::uint8_t control_symbol_count : {1U, 2U, 3U}) {
        for (const std::uint8_t n_rx_ant : {1U, 2U}) {
            for (const std::uint8_t aggregation_level : {4U, 8U}) {
                for (const std::uint8_t n_tx_ports : {1U, 2U}) {
                    auto test_case = make_gpu_pdcch_decode_case(
                        control_symbol_count, n_rx_ant, n_tx_ports, aggregation_level,
                        static_cast<std::uint16_t>(10U + request_id % 200U),
                        static_cast<std::uint32_t>(request_id), request_id, true, request_id);
                    mmse::pdcch::PdcchCommonSearchDecodeResult cpu_result{};
                    mmse::pdcch::PdcchGpuCommonSearchDecodeResult gpu_result{};
                    expect_true(mmse::pdcch::run_pdcch_cpu_common_search_decode(
                                    cpu, test_case.request.input, test_case.request.config,
                                    cpu_result) == MmseStatus::kOk,
                                "GPU CFI/RX matrix CPU decode");
                    expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode(
                                    gpu, test_case.request, gpu_result) == MmseStatus::kOk,
                                "GPU CFI/RX matrix CUDA decode");
                    expect_true(cpu_result.candidate_count == gpu_result.candidate_count &&
                                    !cpu_result.hits.empty() &&
                                    gpu_result.hits.size() == cpu_result.hits.size(),
                                "GPU CFI/RX matrix candidate and hit equivalence: cfi=" +
                                    std::to_string(control_symbol_count) + " rx=" +
                                    std::to_string(n_rx_ant) + " tx=" + std::to_string(n_tx_ports) +
                                    " aggregation=" + std::to_string(aggregation_level) +
                                    " cpu_hits=" + std::to_string(cpu_result.hits.size()) +
                                    " gpu_hits=" + std::to_string(gpu_result.hits.size()));
                    for (std::size_t hit = 0U; hit < cpu_result.hits.size(); ++hit) {
                        const auto& expected = cpu_result.hits[hit];
                        const auto& actual = gpu_result.hits[hit];
                        expect_true(
                            actual.crc.transmitted_crc == expected.crc.transmitted_crc &&
                                actual.crc.calculated_crc == expected.crc.calculated_crc &&
                                actual.crc.unmasked_rnti == expected.crc.unmasked_rnti &&
                                actual.dci.chain.request_id == request_id &&
                                actual.dci.chain.first_cce == expected.dci.chain.first_cce &&
                                actual.dci.chain.aggregation_level ==
                                    expected.dci.chain.aggregation_level &&
                                actual.dci.raw_payload_bits == expected.dci.raw_payload_bits &&
                                actual.dci.start_prb == expected.dci.start_prb &&
                                actual.dci.n_prb == expected.dci.n_prb &&
                                actual.dci.mcs_tbs_index == expected.dci.mcs_tbs_index &&
                                actual.dci.redundancy_version == expected.dci.redundancy_version,
                            "GPU CFI/RX matrix DCI field equivalence");
                    }
                    expect_true(std::any_of(gpu_result.hits.begin(), gpu_result.hits.end(),
                                            [&](const auto& hit) {
                                                return hit.dci.chain.first_cce ==
                                                           test_case.expected_first_cce &&
                                                       hit.dci.chain.aggregation_level ==
                                                           aggregation_level;
                                            }),
                                "GPU CFI/RX matrix target candidate must be present");
                    ++request_id;
                }
            }
        }
    }
#endif
}

void test_pdcch_gpu_common_search_mixed_concurrency_matrix() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    cpu_config.sigma2_min = 1.0e-5F;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "GPU mixed matrix CPU init");

    for (const std::uint32_t stream_count : {1U, 2U, 4U}) {
        MmseEqualizerGpuContext gpu;
        MmseEqualizerGpuConfig gpu_config{};
        gpu_config.backend = MmseGpuBackend::kCuda;
        gpu_config.stream_count = stream_count;
        gpu_config.sigma2_min = cpu_config.sigma2_min;
        expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "GPU mixed matrix CUDA init");

        for (const std::uint32_t batch_size : {1U, 2U, 4U, 8U, 16U}) {
            std::vector<GpuPdcchDecodeCase> cases{};
            cases.reserve(batch_size);
            for (std::uint32_t index = 0U; index < batch_size; ++index) {
                const std::uint64_t request_id =
                    91000U + stream_count * 1000U + batch_size * 20U + index;
                cases.push_back(make_gpu_pdcch_decode_case(
                    3U, static_cast<std::uint8_t>(1U + (index / 2U) % 2U),
                    static_cast<std::uint8_t>(1U + index % 2U), index % 2U == 0U ? 4U : 8U,
                    static_cast<std::uint16_t>(20U + index),
                    static_cast<std::uint32_t>(30U + index), request_id, index % 3U != 1U,
                    request_id));
            }

            std::vector<mmse::pdcch::PdcchGpuCommonSearchDecodeRequest> requests(batch_size);
            std::vector<mmse::pdcch::PdcchGpuCommonSearchDecodeResult> results(batch_size);
            for (std::size_t index = 0U; index < cases.size(); ++index) {
                requests[index] = cases[index].request;
            }
            expect_true(mmse::pdcch::run_pdcch_gpu_common_search_decode_batch(
                            gpu, requests, results) == MmseStatus::kOk,
                        "GPU mixed concurrency batch decode");

            for (std::size_t index = 0U; index < cases.size(); ++index) {
                mmse::pdcch::PdcchCommonSearchDecodeResult cpu_result{};
                expect_true(mmse::pdcch::run_pdcch_cpu_common_search_decode(
                                cpu, cases[index].request.input, cases[index].request.config,
                                cpu_result) == MmseStatus::kOk,
                            "GPU mixed matrix CPU oracle decode");
                const auto& gpu_result = results[index];
                expect_true(gpu_result.candidate_count == cpu_result.candidate_count &&
                                gpu_result.hits.size() == cpu_result.hits.size() &&
                                gpu_result.profile.crc_hit_count == gpu_result.hits.size() &&
                                gpu_result.profile.crc_miss_count +
                                        gpu_result.profile.crc_hit_count ==
                                    gpu_result.candidate_count,
                            "GPU mixed matrix candidate and CRC accounting");
                if (cases[index].expected_hit) {
                    expect_true(
                        !gpu_result.hits.empty() &&
                            std::any_of(
                                gpu_result.hits.begin(), gpu_result.hits.end(),
                                [&](const auto& hit) {
                                    return hit.dci.chain.request_id ==
                                               cases[index].request.input.chain.request_id &&
                                           hit.dci.chain.first_cce ==
                                               cases[index].expected_first_cce &&
                                           hit.dci.chain.aggregation_level ==
                                               cases[index].expected_aggregation_level;
                                }),
                        "GPU mixed matrix request isolation");
                } else {
                    expect_true(gpu_result.hits.empty() && gpu_result.profile.crc_hit_count == 0U,
                                "GPU mixed matrix miss must not retain an old hit");
                }
            }
        }
    }
#endif
}

void test_gpu_context_strict_cuda_init_succeeds_and_runs_via_fallback() {
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kCuda;
#if MMSE_CUDA_ENABLED
    expect_true(gpu.init(config) == MmseStatus::kOk, "gpu strict cuda init should succeed");

    GridBuffers buffers = make_zero_grid();
    auto desc = make_fullband_desc();
    desc.n_layers = 1;
    desc.n_tx_ports = 1;
    desc.tx_mode = 1;
    fill_identity_channel(buffers, desc, 0.75F, 0.0F);
    auto grid = make_grid_view(buffers);
    const std::uint32_t cap = 20000U;
    std::vector<float> xre(cap);
    std::vector<float> xim(cap);
    std::vector<float> sinr(cap);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};
    expect_true(gpu.run(grid, desc, out) == MmseStatus::kOk,
                "gpu strict cuda run should succeed via fallback");
    expect_true(out.n_layers == 1U, "gpu strict cuda fallback layer count");
    expect_near(out.x_hat_re[0], 0.75F, 1.0e-3F, "gpu strict cuda fallback xhat");
#else
    expect_true(gpu.init(config) == MmseStatus::kUnsupportedConfig,
                "gpu strict cuda init should be unsupported without cuda build");
#endif
}

void test_gpu_context_auto_td_fallback_matches_cpu_td() {
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kAuto;
    gpu_config.sigma2_min = 1.0e-5F;
    gpu_config.det_floor = 1.0e-6F;
    gpu_config.g_min = 1.0e-4F;
    gpu_config.gamma_max = 1.0e4F;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "gpu auto td init");

    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1;
    cpu_config.sigma2_min = gpu_config.sigma2_min;
    cpu_config.det_floor = gpu_config.det_floor;
    cpu_config.g_min = gpu_config.g_min;
    cpu_config.gamma_max = gpu_config.gamma_max;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "cpu td init");

    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};
    fill_pdcch_td_layout(buffers, desc, layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                         {1.0F, 0.0F}, s0, s1);

    PdcchMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.control_symbol_count = desc.control_symbol_count;
    in.n_prb = desc.n_prb;
    in.prb_bitmap = desc.prb_bitmap;
    in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                           .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                           .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

    std::vector<float> cpu_re(n_source_re), cpu_im(n_source_re), cpu_sinr(n_source_re);
    std::vector<float> gpu_re(n_source_re), gpu_im(n_source_re), gpu_sinr(n_source_re);
    std::vector<std::uint16_t> cpu_g0(n_source_re), cpu_g1(n_source_re);
    std::vector<std::uint16_t> gpu_g0(n_source_re), gpu_g1(n_source_re);
    PdcchTdMmseOutputView cpu_out{cpu_re.data(), cpu_im.data(), cpu_sinr.data(),
                                  cpu_g0.data(), cpu_g1.data(), n_source_re};
    PdcchTdMmseOutputView gpu_out{gpu_re.data(), gpu_im.data(), gpu_sinr.data(),
                                  gpu_g0.data(), gpu_g1.data(), n_source_re};
    PdcchTdMmseResult cpu_meta{}, gpu_meta{};

    expect_true(cpu.run_pdcch_td(in, cpu_out, cpu_meta) == MmseStatus::kOk, "cpu td run");
    expect_true(gpu.run_pdcch_td(in, gpu_out, gpu_meta) == MmseStatus::kOk, "gpu auto td run");
    expect_true(cpu_meta.n_symbols == gpu_meta.n_symbols, "gpu auto td symbol count");
    for (std::uint32_t i = 0; i < cpu_meta.n_symbols; ++i) {
        expect_near(gpu_re[i], cpu_re[i], 1.0e-5F, "gpu auto td xhat real");
        expect_near(gpu_im[i], cpu_im[i], 1.0e-5F, "gpu auto td xhat imag");
        expect_near(gpu_sinr[i], cpu_sinr[i], 1.0e-5F, "gpu auto td sinr");
        expect_true(gpu_g0[i] == cpu_g0[i] && gpu_g1[i] == cpu_g1[i], "gpu auto td re pairs");
    }
}

void test_gpu_context_cuda_td_matches_cpu_td() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    gpu_config.validation_policy = MmseGpuValidationPolicy::kTestDeepTrace;
    gpu_config.sigma2_min = 1.0e-5F;
    gpu_config.det_floor = 1.0e-6F;
    gpu_config.g_min = 1.0e-4F;
    gpu_config.gamma_max = 1.0e4F;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "gpu cuda td init");

    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1;
    cpu_config.sigma2_min = gpu_config.sigma2_min;
    cpu_config.det_floor = gpu_config.det_floor;
    cpu_config.g_min = gpu_config.g_min;
    cpu_config.gamma_max = gpu_config.gamma_max;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "cpu td init");

    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};
    fill_pdcch_td_layout(buffers, desc, layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                         {1.0F, 0.0F}, s0, s1);

    PdcchMmseInput in{};
    in.grid = make_grid_view(buffers);
    in.sfn_subframe = desc.sfn_subframe;
    in.cell_id = desc.cell_id;
    in.n_tx_ports = desc.n_tx_ports;
    in.tx_mode = desc.tx_mode;
    in.control_symbol_count = desc.control_symbol_count;
    in.n_prb = desc.n_prb;
    in.prb_bitmap = desc.prb_bitmap;
    in.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                           .subframe = static_cast<std::uint8_t>(desc.sfn_subframe % 10U),
                           .kind = mmse::pdcch::LteControlSubframeKind::kRegular};

    std::vector<float> cpu_re(n_source_re), cpu_im(n_source_re), cpu_sinr(n_source_re);
    std::vector<float> gpu_re(n_source_re), gpu_im(n_source_re), gpu_sinr(n_source_re);
    std::vector<std::uint16_t> cpu_g0(n_source_re), cpu_g1(n_source_re);
    std::vector<std::uint16_t> gpu_g0(n_source_re), gpu_g1(n_source_re);
    PdcchTdMmseOutputView cpu_out{cpu_re.data(), cpu_im.data(), cpu_sinr.data(),
                                  cpu_g0.data(), cpu_g1.data(), n_source_re};
    PdcchTdMmseOutputView gpu_out{gpu_re.data(), gpu_im.data(), gpu_sinr.data(),
                                  gpu_g0.data(), gpu_g1.data(), n_source_re};
    PdcchTdMmseResult cpu_meta{}, gpu_meta{};

    expect_true(cpu.run_pdcch_td(in, cpu_out, cpu_meta) == MmseStatus::kOk, "cpu td run");
    expect_true(gpu.run_pdcch_td(in, gpu_out, gpu_meta) == MmseStatus::kOk, "gpu cuda td run");
    expect_true(cpu_meta.n_symbols == gpu_meta.n_symbols, "gpu cuda td symbol count");
    for (std::uint32_t i = 0; i < cpu_meta.n_symbols; ++i) {
        expect_near(gpu_re[i], cpu_re[i], 1.0e-4F, "gpu cuda td xhat real");
        expect_near(gpu_im[i], cpu_im[i], 1.0e-4F, "gpu cuda td xhat imag");
        expect_near(gpu_sinr[i], cpu_sinr[i], 1.0e-4F, "gpu cuda td sinr i=" + std::to_string(i));
        expect_true(gpu_g0[i] == cpu_g0[i] && gpu_g1[i] == cpu_g1[i], "gpu cuda td re pairs");
    }
#endif
}

void test_gpu_context_cuda_td_backend_run_matches_cpu_run() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    gpu_config.sigma2_min = 1.0e-5F;
    gpu_config.det_floor = 1.0e-6F;
    gpu_config.g_min = 1.0e-4F;
    gpu_config.gamma_max = 1.0e4F;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "gpu cuda td backend init");

    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1;
    cpu_config.sigma2_min = gpu_config.sigma2_min;
    cpu_config.det_floor = gpu_config.det_floor;
    cpu_config.g_min = gpu_config.g_min;
    cpu_config.gamma_max = gpu_config.gamma_max;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "cpu td backend init");

    auto desc = make_pdcch_desc();
    desc.n_tx_ports = 2U;
    desc.tx_mode = 2U;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.0F, 0.0F);
    mmse::detail::ReLayout layout{};
    const std::uint32_t n_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    const Complex32 s0{0.70710678F, 0.70710678F};
    const Complex32 s1{-0.70710678F, 0.70710678F};
    fill_pdcch_td_layout(buffers, desc, layout, {1.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F},
                         {1.0F, 0.0F}, s0, s1);
    auto grid = make_grid_view(buffers);

    std::vector<float> cpu_re(n_re), cpu_im(n_re), cpu_sinr(n_re);
    std::vector<float> gpu_re(n_re), gpu_im(n_re), gpu_sinr(n_re);
    EqualizerOutputView cpu_out{cpu_re.data(), cpu_im.data(), cpu_sinr.data(), n_re};
    EqualizerOutputView gpu_out{gpu_re.data(), gpu_im.data(), gpu_sinr.data(), n_re};

    expect_true(cpu.run(grid, desc, cpu_out) == MmseStatus::kUnsupportedConfig,
                "cpu generic run must reject two-port pdcch transmit diversity");
    expect_true(gpu.run(grid, desc, gpu_out) == MmseStatus::kUnsupportedConfig,
                "gpu generic run must reject two-port pdcch transmit diversity");
#endif
}

void test_gpu_context_auto_fallback_matches_cpu_context() {
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kAuto;
    gpu_config.sigma2_min = 1.0e-3F;
    gpu_config.det_floor = 1.0e-6F;
    gpu_config.g_min = 1.0e-4F;
    gpu_config.gamma_max = 1.0e4F;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "gpu auto init should succeed");

    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1;
    cpu_config.sigma2_min = gpu_config.sigma2_min;
    cpu_config.det_floor = gpu_config.det_floor;
    cpu_config.g_min = gpu_config.g_min;
    cpu_config.gamma_max = gpu_config.gamma_max;
    cpu_config.backend = MmseCpuBackend::kAuto;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "cpu auto init should succeed");

    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    const TwoLayerCase c = make_two_layer_case();
    fill_constant_mimo_channel(buffers, desc, c.h00, c.h01, c.h10, c.h11, c.x0, c.x1);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> cpu_re(cap * 2U);
    std::vector<float> cpu_im(cap * 2U);
    std::vector<float> cpu_sinr(cap * 2U);
    std::vector<float> gpu_re(cap * 2U);
    std::vector<float> gpu_im(cap * 2U);
    std::vector<float> gpu_sinr(cap * 2U);
    EqualizerOutputView cpu_out{cpu_re.data(), cpu_im.data(), cpu_sinr.data(), cap};
    EqualizerOutputView gpu_out{gpu_re.data(), gpu_im.data(), gpu_sinr.data(), cap};

    expect_true(cpu.run(grid, desc, cpu_out) == MmseStatus::kOk, "cpu run");
    expect_true(gpu.run(grid, desc, gpu_out) == MmseStatus::kOk, "gpu fallback run");
    expect_true(cpu_out.n_re_per_layer == gpu_out.n_re_per_layer, "cpu/gpu fallback RE count");
    expect_true(cpu_out.n_layers == gpu_out.n_layers, "cpu/gpu fallback layer count");
    expect_true(cpu_out.mod_order == gpu_out.mod_order, "cpu/gpu fallback mod order");
    expect_gpu_matches_cpu_samples(desc, gpu_out, cpu_out, "gpu auto fallback");
}

void test_gpu_context_cuda_two_layer_matches_cpu_context_samples() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    gpu_config.sigma2_min = 1.0e-3F;
    gpu_config.det_floor = 1.0e-6F;
    gpu_config.g_min = 1.0e-4F;
    gpu_config.gamma_max = 1.0e4F;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "gpu strict cuda init should succeed");

    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1;
    cpu_config.sigma2_min = gpu_config.sigma2_min;
    cpu_config.det_floor = gpu_config.det_floor;
    cpu_config.g_min = gpu_config.g_min;
    cpu_config.gamma_max = gpu_config.gamma_max;
    cpu_config.backend = MmseCpuBackend::kScalar;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "cpu scalar init should succeed");

    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    const TwoLayerCase c = make_two_layer_case();
    fill_constant_mimo_channel(buffers, desc, c.h00, c.h01, c.h10, c.h11, c.x0, c.x1);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> cpu_re(cap * 2U);
    std::vector<float> cpu_im(cap * 2U);
    std::vector<float> cpu_sinr(cap * 2U);
    std::vector<float> gpu_re(cap * 2U);
    std::vector<float> gpu_im(cap * 2U);
    std::vector<float> gpu_sinr(cap * 2U);
    EqualizerOutputView cpu_out{cpu_re.data(), cpu_im.data(), cpu_sinr.data(), cap};
    EqualizerOutputView gpu_out{gpu_re.data(), gpu_im.data(), gpu_sinr.data(), cap};

    expect_true(cpu.run(grid, desc, cpu_out) == MmseStatus::kOk, "cpu scalar run");
    expect_true(gpu.run(grid, desc, gpu_out) == MmseStatus::kOk, "gpu strict cuda run");
    expect_true(cpu_out.n_re_per_layer == gpu_out.n_re_per_layer, "gpu strict RE count");
    expect_true(cpu_out.n_layers == gpu_out.n_layers, "gpu strict layer count");
    expect_true(cpu_out.mod_order == gpu_out.mod_order, "gpu strict mod order");
    expect_gpu_matches_cpu_samples(desc, gpu_out, cpu_out, "gpu strict cuda");
#endif
}

void test_gpu_context_cuda_two_layer_deep_trace_matches_cpu_context_samples() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    gpu_config.sigma2_min = 1.0e-3F;
    gpu_config.det_floor = 1.0e-6F;
    gpu_config.g_min = 1.0e-4F;
    gpu_config.gamma_max = 1.0e4F;
    gpu_config.validation_policy = MmseGpuValidationPolicy::kTestDeepTrace;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "gpu deep-trace cuda init should succeed");

    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1;
    cpu_config.sigma2_min = gpu_config.sigma2_min;
    cpu_config.det_floor = gpu_config.det_floor;
    cpu_config.g_min = gpu_config.g_min;
    cpu_config.gamma_max = gpu_config.gamma_max;
    cpu_config.backend = MmseCpuBackend::kScalar;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "cpu scalar init should succeed");

    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    const TwoLayerCase c = make_two_layer_case();
    fill_constant_mimo_channel(buffers, desc, c.h00, c.h01, c.h10, c.h11, c.x0, c.x1);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> cpu_re(cap * 2U);
    std::vector<float> cpu_im(cap * 2U);
    std::vector<float> cpu_sinr(cap * 2U);
    std::vector<float> gpu_re(cap * 2U);
    std::vector<float> gpu_im(cap * 2U);
    std::vector<float> gpu_sinr(cap * 2U);
    EqualizerOutputView cpu_out{cpu_re.data(), cpu_im.data(), cpu_sinr.data(), cap};
    EqualizerOutputView gpu_out{gpu_re.data(), gpu_im.data(), gpu_sinr.data(), cap};

    expect_true(cpu.run(grid, desc, cpu_out) == MmseStatus::kOk, "cpu scalar run");
    expect_true(gpu.run(grid, desc, gpu_out) == MmseStatus::kOk, "gpu deep-trace cuda run");
    expect_true(cpu_out.n_re_per_layer == gpu_out.n_re_per_layer, "gpu deep-trace RE count");
    expect_true(cpu_out.n_layers == gpu_out.n_layers, "gpu deep-trace layer count");
    expect_true(cpu_out.mod_order == gpu_out.mod_order, "gpu deep-trace mod order");
    expect_gpu_matches_cpu_samples(desc, gpu_out, cpu_out, "gpu deep-trace cuda");
#endif
}

void test_gpu_context_cuda_float_transport_preserves_small_signal() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig gpu_config{};
    gpu_config.backend = MmseGpuBackend::kCuda;
    expect_true(gpu.init(gpu_config) == MmseStatus::kOk, "gpu strict cuda init should succeed");

    MmseEqualizerCpuContext cpu;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1;
    expect_true(cpu.init(cpu_config) == MmseStatus::kOk, "cpu init should succeed");

    auto desc = make_fullband_desc();
    desc.n_layers = 1;
    desc.n_tx_ports = 1;
    desc.tx_mode = 1;
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 0.03125F, 0.0F);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> cpu_re(cap);
    std::vector<float> cpu_im(cap);
    std::vector<float> cpu_sinr(cap);
    std::vector<float> gpu_re(cap);
    std::vector<float> gpu_im(cap);
    std::vector<float> gpu_sinr(cap);
    EqualizerOutputView cpu_out{cpu_re.data(), cpu_im.data(), cpu_sinr.data(), cap};
    EqualizerOutputView gpu_out{gpu_re.data(), gpu_im.data(), gpu_sinr.data(), cap};

    expect_true(cpu.run(grid, desc, cpu_out) == MmseStatus::kOk, "cpu run");
    expect_true(gpu.run(grid, desc, gpu_out) == MmseStatus::kOk, "gpu run");
    expect_near(gpu_re[0], cpu_re[0], 1.0e-4F, "gpu float transport xhat real");
    expect_near(gpu_im[0], cpu_im[0], 1.0e-4F, "gpu float transport xhat imag");
#endif
}

void test_gpu_context_sigma2_state_persists() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kCuda;
    config.sigma2_iir_alpha = 0.5F;
    config.sigma2_min = 1.0e-6F;
    expect_true(gpu.init(config) == MmseStatus::kOk, "gpu init should succeed");

    auto desc = make_fullband_desc();
    GridBuffers clean = make_zero_grid();
    fill_identity_channel(clean, desc, 1.0F, 1.0F);
    auto clean_grid = make_grid_view(clean);

    GridBuffers noisy = clean;
    for (std::size_t i = 0; i < noisy.re[0].size(); i += 17U) {
        noisy.re[0][i] += 0.2F;
        noisy.im[1][i] -= 0.1F;
    }
    auto noisy_grid = make_grid_view(noisy);

    const std::uint32_t cap = 20000U;
    std::vector<float> xre(cap * 2U);
    std::vector<float> xim(cap * 2U);
    std::vector<float> sinr(cap * 2U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};

    expect_true(gpu.run(noisy_grid, desc, out) == MmseStatus::kOk, "gpu noisy run");
    const float sinr_noisy = out.sinr[0];
    expect_true(gpu.run(clean_grid, desc, out) == MmseStatus::kOk, "gpu clean run");
    const float sinr_after = out.sinr[0];
    (void)sinr_noisy;
    expect_true(sinr_after < 1.0e6F, "gpu clamped finite sinr");
    expect_true(sinr_after > 0.0F, "gpu positive sinr after state update");
#endif
}

void test_gpu_context_host_owned_and_device_owned_sigma2_match_samples() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuConfig host_cfg{};
    host_cfg.backend = MmseGpuBackend::kCuda;
    host_cfg.sigma2_min = 1.0e-3F;
    host_cfg.det_floor = 1.0e-6F;
    host_cfg.g_min = 1.0e-4F;
    host_cfg.gamma_max = 1.0e4F;
    host_cfg.sigma2_ownership = MmseGpuSigma2Ownership::kHostOwnedIir;
    host_cfg.validation_policy = MmseGpuValidationPolicy::kReleaseSanity;

    MmseEqualizerGpuConfig device_cfg = host_cfg;
    device_cfg.sigma2_ownership = MmseGpuSigma2Ownership::kDeviceOwnedState;

    MmseEqualizerGpuContext host_gpu;
    MmseEqualizerGpuContext device_gpu;
    expect_true(host_gpu.init(host_cfg) == MmseStatus::kOk, "host-owned gpu init");
    expect_true(device_gpu.init(device_cfg) == MmseStatus::kOk, "device-owned gpu init");

    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    const TwoLayerCase c = make_two_layer_case();
    fill_constant_mimo_channel(buffers, desc, c.h00, c.h01, c.h10, c.h11, c.x0, c.x1);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> host_re(cap * 2U);
    std::vector<float> host_im(cap * 2U);
    std::vector<float> host_sinr(cap * 2U);
    std::vector<float> device_re(cap * 2U);
    std::vector<float> device_im(cap * 2U);
    std::vector<float> device_sinr(cap * 2U);
    EqualizerOutputView host_out{host_re.data(), host_im.data(), host_sinr.data(), cap};
    EqualizerOutputView device_out{device_re.data(), device_im.data(), device_sinr.data(), cap};

    expect_true(host_gpu.run(grid, desc, host_out) == MmseStatus::kOk, "host-owned gpu run");
    expect_true(device_gpu.run(grid, desc, device_out) == MmseStatus::kOk, "device-owned gpu run");
    expect_true(host_out.n_re_per_layer == device_out.n_re_per_layer, "ownership RE count");
    expect_true(host_out.n_layers == device_out.n_layers, "ownership layer count");
    expect_gpu_matches_cpu_samples(desc, device_out, host_out, "sigma2 ownership compare");
#endif
}

void test_gpu_context_device_owned_sigma2_state_persists() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kCuda;
    config.sigma2_iir_alpha = 0.5F;
    config.sigma2_min = 1.0e-6F;
    config.sigma2_ownership = MmseGpuSigma2Ownership::kDeviceOwnedState;
    config.validation_policy = MmseGpuValidationPolicy::kReleaseSanity;
    expect_true(gpu.init(config) == MmseStatus::kOk, "device-owned sigma2 init should succeed");

    auto desc = make_fullband_desc();
    GridBuffers clean = make_zero_grid();
    fill_identity_channel(clean, desc, 1.0F, 1.0F);
    auto clean_grid = make_grid_view(clean);

    GridBuffers noisy = clean;
    for (std::size_t i = 0; i < noisy.re[0].size(); i += 17U) {
        noisy.re[0][i] += 0.2F;
        noisy.im[1][i] -= 0.1F;
    }
    auto noisy_grid = make_grid_view(noisy);

    const std::uint32_t cap = 20000U;
    std::vector<float> xre(cap * 2U);
    std::vector<float> xim(cap * 2U);
    std::vector<float> sinr(cap * 2U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};

    expect_true(gpu.run(noisy_grid, desc, out) == MmseStatus::kOk, "device-owned noisy run");
    const float sinr_noisy = out.sinr[0];
    expect_true(gpu.run(clean_grid, desc, out) == MmseStatus::kOk, "device-owned clean run");
    const float sinr_after = out.sinr[0];
    expect_true(sinr_after < 1.0e6F, "device-owned clamped finite sinr");
    expect_true(sinr_after > sinr_noisy, "device-owned cleaner second pass should improve sinr");
#endif
}

void test_gpu_context_device_owned_sigma2_tracks_cell_id() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuContext host_gpu;
    MmseEqualizerGpuContext device_gpu;

    MmseEqualizerGpuConfig host_cfg{};
    host_cfg.backend = MmseGpuBackend::kCuda;
    host_cfg.sigma2_iir_alpha = 0.5F;
    host_cfg.sigma2_min = 1.0e-6F;
    host_cfg.sigma2_ownership = MmseGpuSigma2Ownership::kHostOwnedIir;
    host_cfg.validation_policy = MmseGpuValidationPolicy::kReleaseSanity;

    MmseEqualizerGpuConfig device_cfg = host_cfg;
    device_cfg.sigma2_ownership = MmseGpuSigma2Ownership::kDeviceOwnedState;

    expect_true(host_gpu.init(host_cfg) == MmseStatus::kOk, "host-owned multi-cell init");
    expect_true(device_gpu.init(device_cfg) == MmseStatus::kOk, "device-owned multi-cell init");

    auto desc0 = make_fullband_desc();
    auto desc1 = make_fullband_desc();
    desc1.cell_id = 1U;

    GridBuffers clean0 = make_zero_grid();
    fill_identity_channel(clean0, desc0, 1.0F, 1.0F);
    auto clean_grid0 = make_grid_view(clean0);

    GridBuffers noisy0 = clean0;
    for (std::size_t i = 0; i < noisy0.re[0].size(); i += 17U) {
        noisy0.re[0][i] += 0.2F;
        noisy0.im[1][i] -= 0.1F;
    }
    auto noisy_grid0 = make_grid_view(noisy0);

    GridBuffers clean1 = make_zero_grid();
    fill_identity_channel(clean1, desc1, 1.0F, 1.0F);
    auto clean_grid1 = make_grid_view(clean1);

    const std::uint32_t cap = 20000U;
    std::vector<float> host_re(cap * 2U);
    std::vector<float> host_im(cap * 2U);
    std::vector<float> host_sinr(cap * 2U);
    std::vector<float> device_re(cap * 2U);
    std::vector<float> device_im(cap * 2U);
    std::vector<float> device_sinr(cap * 2U);
    EqualizerOutputView host_out{host_re.data(), host_im.data(), host_sinr.data(), cap};
    EqualizerOutputView device_out{device_re.data(), device_im.data(), device_sinr.data(), cap};

    expect_true(host_gpu.run(noisy_grid0, desc0, host_out) == MmseStatus::kOk,
                "host-owned noisy cell0");
    expect_true(device_gpu.run(noisy_grid0, desc0, device_out) == MmseStatus::kOk,
                "device-owned noisy cell0");

    expect_true(host_gpu.run(clean_grid1, desc1, host_out) == MmseStatus::kOk,
                "host-owned clean cell1");
    expect_true(device_gpu.run(clean_grid1, desc1, device_out) == MmseStatus::kOk,
                "device-owned clean cell1");
    expect_gpu_matches_cpu_samples(desc1, device_out, host_out, "device-owned cell1");

    expect_true(host_gpu.run(clean_grid0, desc0, host_out) == MmseStatus::kOk,
                "host-owned clean cell0");
    expect_true(device_gpu.run(clean_grid0, desc0, device_out) == MmseStatus::kOk,
                "device-owned clean cell0");
    expect_gpu_matches_cpu_samples(desc0, device_out, host_out, "device-owned cell0");
#endif
}

void test_gpu_context_destructor_releases_cuda_runtime_state() {
#if MMSE_CUDA_ENABLED
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kCuda;
    config.stream_count = 4U;

    {
        MmseEqualizerGpuContext warmup;
        expect_true(warmup.init(config) == MmseStatus::kOk, "gpu destructor warmup init");
    }

    std::size_t free_before = 0U;
    std::size_t total_before = 0U;
    expect_true(mmse::detail::cuda_query_current_memory_info(free_before, total_before) ==
                    MmseStatus::kOk,
                "gpu destructor memory baseline");
    for (std::uint32_t iteration = 0U; iteration < 4U; ++iteration) {
        MmseEqualizerGpuContext context;
        expect_true(context.init(config) == MmseStatus::kOk, "gpu destructor repeated init");
    }
    std::size_t free_after = 0U;
    std::size_t total_after = 0U;
    expect_true(mmse::detail::cuda_query_current_memory_info(free_after, total_after) ==
                    MmseStatus::kOk,
                "gpu destructor memory result");
    constexpr std::size_t kAllocatorToleranceBytes = 8U * 1024U * 1024U;
    expect_true(total_after == total_before && free_after + kAllocatorToleranceBytes >= free_before,
                "gpu destructor must release per-slot CUDA allocations");
#endif
}

void test_gpu_context_invalid_stream_count_is_rejected() {
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kAuto;
    config.stream_count = 0;
    expect_true(gpu.init(config) == MmseStatus::kInvalidArgument,
                "gpu init should reject zero stream count");
}

void test_lte_descrambling_cinit_helpers_match_expected_values() {
    expect_true(mmse::lte::subframe_from_sfn_subframe(47U) == 7U, "subframe helper");
    expect_true(mmse::lte::pbch_c_init(503U) == 503U, "pbch c_init");
    expect_true(mmse::lte::pdcch_c_init(42U, 13U) == 1578U, "pdcch c_init");
    expect_true(mmse::lte::pcfich_c_init(42U, 13U) == 174122U, "pcfich c_init");
    expect_true(mmse::lte::pdsch_c_init(42U, 0x1234U, 13U, 1U) == 76359210U, "pdsch c_init");
}

void test_lte_descrambling_gold_sequence_matches_known_vectors() {
    std::vector<std::uint8_t> seq0(32U, 0U);
    std::vector<std::uint8_t> seq1(32U, 0U);
    mmse::lte::generate_scrambling_bits(0U, seq0.data(), static_cast<std::uint32_t>(seq0.size()));
    mmse::lte::generate_scrambling_bits(1578U, seq1.data(),
                                        static_cast<std::uint32_t>(seq1.size()));
    expect_bits_match_binary_string(seq0, "00000010000110100001001001111010", "gold c_init=0");
    expect_bits_match_binary_string(seq1, "00111000010101010000100111100010", "gold c_init=1578");
}

void test_lte_descrambling_sequence_matches_reference_generator() {
    const std::uint32_t c_init = mmse::lte::pdsch_c_init(11U, 0x3456U, 28U, 1U);
    std::vector<std::uint8_t> actual(257U, 0U);
    mmse::lte::generate_scrambling_bits(c_init, actual.data(),
                                        static_cast<std::uint32_t>(actual.size()));
    const std::vector<std::uint8_t> expected = reference_gold_sequence(c_init, 257U);
    expect_true(actual == expected, "gold sequence should match reference generator");
}

void test_lte_descramble_bits_is_self_inverse() {
    std::vector<std::uint8_t> bits = {1U, 0U, 1U, 1U, 0U, 0U, 1U, 0U, 1U, 0U, 1U, 1U, 1U, 0U};
    const std::vector<std::uint8_t> original = bits;
    const std::uint32_t c_init = mmse::lte::pdcch_c_init(321U, 47U);
    mmse::lte::descramble_bits_inplace(bits.data(), static_cast<std::uint32_t>(bits.size()),
                                       c_init);
    mmse::lte::descramble_bits_inplace(bits.data(), static_cast<std::uint32_t>(bits.size()),
                                       c_init);
    expect_true(bits == original, "bit descrambling should be self-inverse");
}

void test_lte_descramble_llrs_flips_sign_on_scrambling_ones() {
    const std::uint32_t c_init = mmse::lte::pcfich_c_init(42U, 13U);
    std::vector<std::uint8_t> scramble(16U, 0U);
    mmse::lte::generate_scrambling_bits(c_init, scramble.data(),
                                        static_cast<std::uint32_t>(scramble.size()));

    std::vector<float> llrs = {0.25F, -0.50F, 0.75F, -1.00F, 1.25F, -1.50F, 1.75F, -2.00F,
                               2.25F, -2.50F, 2.75F, -3.00F, 3.25F, -3.50F, 3.75F, -4.00F};
    const std::vector<float> original = llrs;
    mmse::lte::descramble_llrs_inplace(llrs.data(), static_cast<std::uint32_t>(llrs.size()),
                                       c_init);

    for (std::size_t i = 0; i < llrs.size(); ++i) {
        const float expected = scramble[i] != 0U ? -original[i] : original[i];
        expect_near(llrs[i], expected, 0.0F, "llr sign flip");
    }
}

void test_lte_soft_demod_qpsk_matches_known_llrs() {
    const float inv_sqrt2 = 1.0F / std::sqrt(2.0F);
    std::array<float, 2> xre = {inv_sqrt2, -inv_sqrt2};
    std::array<float, 2> xim = {inv_sqrt2, inv_sqrt2};
    std::array<float, 2> sinr = {5.0F, 5.0F};
    std::vector<float> llrs{};

    expect_true(mmse::lte::build_max_log_llrs(xre.data(), xim.data(), sinr.data(), 2U, 2U, 1U, 2U,
                                              llrs) == MmseStatus::kOk,
                "qpsk demod should succeed");
    expect_true(llrs.size() == 4U, "qpsk llr count");
    expect_near(llrs[0], -10.0F, 1.0e-4F, "qpsk symbol0 i-bit");
    expect_near(llrs[1], -10.0F, 1.0e-4F, "qpsk symbol0 q-bit");
    expect_near(llrs[2], 10.0F, 1.0e-4F, "qpsk symbol1 i-bit");
    expect_near(llrs[3], -10.0F, 1.0e-4F, "qpsk symbol1 q-bit");
}

void test_lte_soft_demod_64qam_sign_pattern_matches_gray_mapping() {
    const float inv_sqrt42 = 1.0F / std::sqrt(42.0F);
    std::array<float, 1> xre = {7.0F * inv_sqrt42};
    std::array<float, 1> xim = {-1.0F * inv_sqrt42};
    std::array<float, 1> sinr = {2.0F};
    std::vector<float> llrs{};
    const std::vector<float> expected =
        reference_build_max_log_llrs(xre.data(), xim.data(), sinr.data(), 1U, 1U, 1U, 6U);

    expect_true(mmse::lte::build_max_log_llrs(xre.data(), xim.data(), sinr.data(), 1U, 1U, 1U, 6U,
                                              llrs) == MmseStatus::kOk,
                "64qam demod should succeed");
    expect_true(llrs.size() == 6U, "64qam llr count");

    for (std::size_t i = 0; i < llrs.size(); ++i) {
        const int expected_sign = expected[i] > 0.0F ? 1 : (expected[i] < 0.0F ? -1 : 0);
        expect_sign(llrs[i], expected_sign, "64qam sign matches reference");
    }
}

void test_lte_soft_demod_specialized_paths_match_reference() {
    std::array<float, 4> xre = {0.3F, -0.7F, 3.0F / std::sqrt(10.0F), 7.0F / std::sqrt(42.0F)};
    std::array<float, 4> xim = {-0.4F, 0.2F, -1.0F / std::sqrt(10.0F), -3.0F / std::sqrt(42.0F)};
    std::array<float, 4> sinr = {0.5F, 1.0F, 2.0F, 3.0F};

    for (const std::uint8_t mod_order : {2U, 4U, 6U}) {
        std::vector<float> actual{};
        expect_true(mmse::lte::build_max_log_llrs(xre.data(), xim.data(), sinr.data(), 4U, 4U, 1U,
                                                  mod_order, actual) == MmseStatus::kOk,
                    "specialized demod should succeed");
        const std::vector<float> expected = reference_build_max_log_llrs(
            xre.data(), xim.data(), sinr.data(), 4U, 4U, 1U, mod_order);
        expect_true(actual.size() == expected.size(), "specialized demod llr size");
        for (std::size_t i = 0; i < expected.size(); ++i) {
            expect_relative_near(actual[i], expected[i], 1.0e-5F, 1.0e-5F,
                                 "specialized demod matches reference");
        }
    }
}

void test_lte_soft_demod_descrambled_fused_matches_split_reference() {
    std::array<float, 3> xre = {0.3F, -3.0F / std::sqrt(10.0F), 7.0F / std::sqrt(42.0F)};
    std::array<float, 3> xim = {-0.4F, 1.0F / std::sqrt(10.0F), -3.0F / std::sqrt(42.0F)};
    std::array<float, 3> sinr = {0.5F, 2.0F, 3.0F};
    const std::uint32_t c_init = mmse::lte::pdsch_c_init(17U, 0x1234U, 8U, 0U);

    for (const std::uint8_t mod_order : {2U, 4U, 6U}) {
        std::vector<float> split = reference_build_max_log_llrs(xre.data(), xim.data(), sinr.data(),
                                                                3U, 3U, 1U, mod_order);
        mmse::lte::descramble_llrs_inplace(split.data(), static_cast<std::uint32_t>(split.size()),
                                           c_init);

        std::vector<float> fused{};
        expect_true(mmse::lte::build_max_log_descrambled_llrs(xre.data(), xim.data(), sinr.data(),
                                                              3U, 3U, 1U, mod_order, c_init,
                                                              fused) == MmseStatus::kOk,
                    "fused descrambled demod should succeed");
        expect_true(fused.size() == split.size(), "fused descrambled demod size");
        for (std::size_t i = 0; i < split.size(); ++i) {
            expect_relative_near(fused[i], split[i], 1.0e-5F, 1.0e-5F,
                                 "fused descrambled demod matches split reference");
        }
    }
}

void test_pbch_backend_descrambled_llr_indication() {
    const float inv_sqrt2 = 1.0F / std::sqrt(2.0F);
    mmse::pbch::BackendPbchEqualizedIndication backend{};
    backend.sfn_subframe = 0U;
    backend.cell_id = 37U;
    backend.start_prb = kLtePbchStartPrb;
    backend.n_prb = kLtePbchNumPrb;
    backend.start_symbol = kLtePbchStartSymbolNormalCp;
    backend.n_tx_ports = 1U;
    backend.n_rx_ant = 2U;
    backend.n_layers = 1U;
    backend.tx_mode = 1U;
    backend.mod_order = 2U;
    backend.sigma2 = 0.1F;
    backend.chain.request_id = 901U;
    backend.x_hat_re = {inv_sqrt2, -inv_sqrt2};
    backend.x_hat_im = {inv_sqrt2, -inv_sqrt2};
    backend.sinr = {4.0F, 4.0F};
    backend.re_grid_indices = {100U, 101U};

    const auto llr_backend = mmse::pbch::make_backend_pbch_descrambled_llr_indication(backend);
    std::vector<float> expected{};
    expect_true(mmse::lte::build_max_log_llrs(backend.x_hat_re.data(), backend.x_hat_im.data(),
                                              backend.sinr.data(), 2U, 2U, 1U, 2U,
                                              expected) == MmseStatus::kOk,
                "pbch raw llr demod");
    std::vector<std::uint8_t> scramble(expected.size(), 0U);
    mmse::lte::generate_scrambling_bits(mmse::lte::pbch_c_init(backend.cell_id), scramble.data(),
                                        static_cast<std::uint32_t>(scramble.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (scramble[i] != 0U) {
            expected[i] = -expected[i];
        }
    }

    expect_true(llr_backend.llrs.size() == expected.size(), "pbch descrambled llr size");
    expect_true(llr_backend.chain.request_id == backend.chain.request_id, "pbch llr chain");
    expect_true(llr_backend.re_grid_indices == backend.re_grid_indices, "pbch llr indices");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect_near(llr_backend.llrs[i], expected[i], 1.0e-4F, "pbch llr sample");
    }
}

void test_pcfich_backend_descrambled_llr_indication() {
    const float inv_sqrt2 = 1.0F / std::sqrt(2.0F);
    mmse::pcfich::BackendPcfichEqualizedIndication backend{};
    backend.sfn_subframe = 13U;
    backend.cell_id = 42U;
    backend.n_prb = kLteNumPrb20MHz;
    backend.start_symbol = 0U;
    backend.reg_count = static_cast<std::uint8_t>(kLtePcfichNumRegs);
    backend.n_tx_ports = 1U;
    backend.n_rx_ant = 2U;
    backend.n_layers = 1U;
    backend.tx_mode = 1U;
    backend.mod_order = 2U;
    backend.sigma2 = 0.2F;
    backend.chain.request_id = 902U;
    backend.x_hat_re = {inv_sqrt2};
    backend.x_hat_im = {-inv_sqrt2};
    backend.sinr = {3.0F};
    backend.re_grid_indices = {55U};

    const auto llr_backend = mmse::pcfich::make_backend_pcfich_descrambled_llr_indication(backend);
    std::vector<float> expected{};
    expect_true(mmse::lte::build_max_log_llrs(backend.x_hat_re.data(), backend.x_hat_im.data(),
                                              backend.sinr.data(), 1U, 1U, 1U, 2U,
                                              expected) == MmseStatus::kOk,
                "pcfich raw llr demod");
    std::vector<std::uint8_t> scramble(expected.size(), 0U);
    mmse::lte::generate_scrambling_bits(
        mmse::lte::pcfich_c_init(backend.cell_id, backend.sfn_subframe), scramble.data(),
        static_cast<std::uint32_t>(scramble.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (scramble[i] != 0U) {
            expected[i] = -expected[i];
        }
    }

    expect_true(llr_backend.llrs.size() == expected.size(), "pcfich descrambled llr size");
    expect_true(llr_backend.reg_count == backend.reg_count, "pcfich reg count passthrough");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect_near(llr_backend.llrs[i], expected[i], 1.0e-4F, "pcfich llr sample");
    }
}

void test_pdcch_backend_descrambled_llr_indication() {
    const float inv_sqrt2 = 1.0F / std::sqrt(2.0F);
    mmse::pdcch::BackendPdcchEqualizedIndication backend{};
    backend.sfn_subframe = 9U;
    backend.cell_id = 21U;
    backend.n_prb = kLteNumPrb20MHz;
    backend.n_tx_ports = 1U;
    backend.n_rx_ant = 2U;
    backend.n_layers = 1U;
    backend.tx_mode = 1U;
    backend.control_symbol_count = 3U;
    backend.mod_order = 2U;
    backend.sigma2 = 0.125F;
    backend.chain.request_id = 903U;
    backend.chain.candidate_id = 7U;
    backend.x_hat_re = {-inv_sqrt2};
    backend.x_hat_im = {inv_sqrt2};
    backend.sinr = {6.0F};
    backend.re_grid_indices = {66U};

    const auto llr_backend = mmse::pdcch::make_backend_pdcch_descrambled_llr_indication(backend);
    std::vector<float> expected{};
    expect_true(mmse::lte::build_max_log_llrs(backend.x_hat_re.data(), backend.x_hat_im.data(),
                                              backend.sinr.data(), 1U, 1U, 1U, 2U,
                                              expected) == MmseStatus::kOk,
                "pdcch raw llr demod");
    std::vector<std::uint8_t> scramble(expected.size(), 0U);
    mmse::lte::generate_scrambling_bits(
        mmse::lte::pdcch_c_init(backend.cell_id, backend.sfn_subframe), scramble.data(),
        static_cast<std::uint32_t>(scramble.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (scramble[i] != 0U) {
            expected[i] = -expected[i];
        }
    }

    expect_true(llr_backend.llrs.size() == expected.size(), "pdcch descrambled llr size");
    expect_true(llr_backend.chain.candidate_id == backend.chain.candidate_id, "pdcch llr chain");
    expect_true(llr_backend.re_grid_indices == backend.re_grid_indices, "pdcch llr indices");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect_near(llr_backend.llrs[i], expected[i], 1.0e-4F, "pdcch llr sample");
    }
}

void test_pdcch_td_backend_descrambled_llr_indication() {
    const float inv_sqrt2 = 1.0F / std::sqrt(2.0F);
    mmse::pdcch::BackendPdcchTdEqualizedIndication backend{};
    backend.sfn_subframe = 19U;
    backend.cell_id = 31U;
    backend.n_prb = kLteNumPrb20MHz;
    backend.n_tx_ports = 2U;
    backend.n_rx_ant = 2U;
    backend.n_layers = 1U;
    backend.tx_mode = 2U;
    backend.control_symbol_count = 3U;
    backend.mod_order = 2U;
    backend.sigma2 = 0.25F;
    backend.chain.request_id = 904U;
    backend.x_hat_re = {inv_sqrt2, -inv_sqrt2};
    backend.x_hat_im = {inv_sqrt2, inv_sqrt2};
    backend.sinr = {2.5F, 2.5F};
    backend.re_grid_indices0 = {70U, 72U};
    backend.re_grid_indices1 = {71U, 73U};

    const auto llr_backend = mmse::pdcch::make_backend_pdcch_td_descrambled_llr_indication(backend);
    std::vector<float> expected{};
    expect_true(mmse::lte::build_max_log_llrs(backend.x_hat_re.data(), backend.x_hat_im.data(),
                                              backend.sinr.data(), 2U, 2U, 1U, 2U,
                                              expected) == MmseStatus::kOk,
                "pdcch td raw llr demod");
    std::vector<std::uint8_t> scramble(expected.size(), 0U);
    mmse::lte::generate_scrambling_bits(
        mmse::lte::pdcch_c_init(backend.cell_id, backend.sfn_subframe), scramble.data(),
        static_cast<std::uint32_t>(scramble.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (scramble[i] != 0U) {
            expected[i] = -expected[i];
        }
    }

    expect_true(llr_backend.llrs.size() == expected.size(), "pdcch td descrambled llr size");
    expect_true(llr_backend.re_grid_indices0 == backend.re_grid_indices0,
                "pdcch td re0 passthrough");
    expect_true(llr_backend.re_grid_indices1 == backend.re_grid_indices1,
                "pdcch td re1 passthrough");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect_near(llr_backend.llrs[i], expected[i], 1.0e-4F, "pdcch td llr sample");
    }
}

void test_pdsch_backend_descrambled_llr_indication_16qam() {
    ExtractDescriptor desc{};
    desc.sfn_subframe = 12U;
    desc.cell_id = 17U;
    desc.n_tx_ports = 2U;
    desc.n_rx_ant = 2U;
    desc.n_layers = 1U;
    desc.tx_mode = 2U;
    desc.channel_type = MmseChannelType::kPdsch;
    desc.start_symbol = 3U;
    desc.mod_order = 4U;
    desc.n_prb = 50U;
    desc.prb_bitmap.fill(0U);
    desc.prb_bitmap[0] = 0x00FFU;
    desc.pmi = -1;

    const float inv_sqrt10 = 1.0F / std::sqrt(10.0F);
    std::array<float, 1> xre = {3.0F * inv_sqrt10};
    std::array<float, 1> xim = {-1.0F * inv_sqrt10};
    std::array<float, 1> sinr = {2.0F};
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), 1U, 1U, 1U, 4U};

    const std::uint16_t rnti = 0x1234U;
    const auto backend =
        mmse::pdsch::make_backend_pdsch_descrambled_llr_indication(desc, out, rnti, 0U);
    expect_true(backend.llrs.size() == 4U, "pdsch llr size");
    expect_true(backend.llr_count_per_layer == 4U, "pdsch llr per layer");
    expect_true(backend.rnti == rnti, "pdsch rnti passthrough");

    std::vector<float> expected =
        reference_build_max_log_llrs(xre.data(), xim.data(), sinr.data(), 1U, 1U, 1U, 4U);
    mmse::lte::descramble_llrs_inplace(
        expected.data(), static_cast<std::uint32_t>(expected.size()),
        mmse::lte::pdsch_c_init(desc.cell_id, rnti, desc.sfn_subframe, 0U));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect_relative_near(backend.llrs[i], expected[i], 1.0e-5F, 1.0e-5F,
                             "pdsch 16qam llr exact match");
    }
}

void test_pdsch_caller_owned_llr_output_matches_backend_builder() {
    ExtractDescriptor desc{};
    desc.sfn_subframe = 9U;
    desc.cell_id = 21U;
    desc.n_tx_ports = 2U;
    desc.n_rx_ant = 2U;
    desc.n_layers = 2U;
    desc.tx_mode = 2U;
    desc.channel_type = MmseChannelType::kPdsch;
    desc.start_symbol = 1U;
    desc.mod_order = 6U;
    desc.n_prb = 20U;
    desc.prb_bitmap.fill(0U);
    desc.prb_bitmap[0] = 0x0FFFU;
    desc.prb_bitmap[1] = 0x00FFU;
    desc.pmi = -1;

    std::array<float, 4> xre = {0.3F, -0.4F, 7.0F / std::sqrt(42.0F), -3.0F / std::sqrt(42.0F)};
    std::array<float, 4> xim = {-0.2F, 0.5F, -1.0F / std::sqrt(42.0F), 5.0F / std::sqrt(42.0F)};
    std::array<float, 4> sinr = {0.5F, 1.0F, 2.0F, 3.0F};
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), 2U, 2U, 2U, 6U};

    const std::uint16_t rnti = 0x2468U;
    const auto backend =
        mmse::pdsch::make_backend_pdsch_descrambled_llr_indication(desc, out, rnti);

    std::vector<float> llrs(backend.llrs.size(), 0.0F);
    mmse::pdsch::PdschDescrambledLlrOutputView llr_out{llrs.data(),
                                                       static_cast<std::uint32_t>(llrs.size())};
    mmse::pdsch::PdschDescrambledLlrResult result{};
    expect_true(mmse::pdsch::build_backend_pdsch_descrambled_llr_result(desc, out, rnti, llr_out,
                                                                        result) == MmseStatus::kOk,
                "pdsch caller-owned llr builder should succeed");
    expect_true(result.llr_count == backend.llrs.size(), "pdsch caller-owned llr count");
    expect_true(result.llr_count_per_layer == backend.llr_count_per_layer,
                "pdsch caller-owned llr count per layer");
    expect_true(result.n_layers == backend.n_layers, "pdsch caller-owned n_layers");
    for (std::size_t i = 0; i < backend.llrs.size(); ++i) {
        expect_relative_near(llrs[i], backend.llrs[i], 1.0e-5F, 1.0e-5F,
                             "pdsch caller-owned matches backend llrs");
    }
}

void test_pdsch_cached_builder_matches_uncached_builder() {
    ExtractDescriptor desc{};
    desc.sfn_subframe = 9U;
    desc.cell_id = 21U;
    desc.n_tx_ports = 2U;
    desc.n_rx_ant = 2U;
    desc.n_layers = 2U;
    desc.tx_mode = 2U;
    desc.channel_type = MmseChannelType::kPdsch;
    desc.start_symbol = 1U;
    desc.mod_order = 6U;
    desc.n_prb = 20U;
    desc.prb_bitmap.fill(0U);
    desc.prb_bitmap[0] = 0x0FFFU;
    desc.prb_bitmap[1] = 0x00FFU;
    desc.pmi = -1;

    std::array<float, 4> xre = {0.3F, -0.4F, 7.0F / std::sqrt(42.0F), -3.0F / std::sqrt(42.0F)};
    std::array<float, 4> xim = {-0.2F, 0.5F, -1.0F / std::sqrt(42.0F), 5.0F / std::sqrt(42.0F)};
    std::array<float, 4> sinr = {0.5F, 1.0F, 2.0F, 3.0F};
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), 2U, 2U, 2U, 6U};

    const std::uint16_t rnti = 0x2468U;
    const auto uncached =
        mmse::pdsch::make_backend_pdsch_descrambled_llr_indication(desc, out, rnti);
    mmse::pdsch::PdschDescramblingPlanCache cache{};
    const auto cached =
        mmse::pdsch::make_backend_pdsch_descrambled_llr_indication(desc, out, rnti, cache);
    expect_true(cache.valid, "pdsch scrambling cache should be populated");
    expect_true(cache.llr_count == uncached.llrs.size(), "pdsch scrambling cache llr_count");
    expect_true(cached.llrs.size() == uncached.llrs.size(), "pdsch cached llr size");
    for (std::size_t i = 0; i < uncached.llrs.size(); ++i) {
        expect_relative_near(cached.llrs[i], uncached.llrs[i], 1.0e-5F, 1.0e-5F,
                             "pdsch cached builder matches uncached");
    }
}

void test_pdsch_cached_caller_owned_output_matches_cached_builder() {
    ExtractDescriptor desc{};
    desc.sfn_subframe = 9U;
    desc.cell_id = 21U;
    desc.n_tx_ports = 2U;
    desc.n_rx_ant = 2U;
    desc.n_layers = 2U;
    desc.tx_mode = 2U;
    desc.channel_type = MmseChannelType::kPdsch;
    desc.start_symbol = 1U;
    desc.mod_order = 6U;
    desc.n_prb = 20U;
    desc.prb_bitmap.fill(0U);
    desc.prb_bitmap[0] = 0x0FFFU;
    desc.prb_bitmap[1] = 0x00FFU;
    desc.pmi = -1;

    std::array<float, 4> xre = {0.3F, -0.4F, 7.0F / std::sqrt(42.0F), -3.0F / std::sqrt(42.0F)};
    std::array<float, 4> xim = {-0.2F, 0.5F, -1.0F / std::sqrt(42.0F), 5.0F / std::sqrt(42.0F)};
    std::array<float, 4> sinr = {0.5F, 1.0F, 2.0F, 3.0F};
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), 2U, 2U, 2U, 6U};

    const std::uint16_t rnti = 0x2468U;
    mmse::pdsch::PdschDescramblingPlanCache cache{};
    const auto cached =
        mmse::pdsch::make_backend_pdsch_descrambled_llr_indication(desc, out, rnti, cache);

    std::vector<float> llrs(cached.llrs.size(), 0.0F);
    mmse::pdsch::PdschDescrambledLlrOutputView llr_out{llrs.data(),
                                                       static_cast<std::uint32_t>(llrs.size())};
    mmse::pdsch::PdschDescrambledLlrResult result{};
    expect_true(mmse::pdsch::build_backend_pdsch_descrambled_llr_result(
                    desc, out, rnti, cache, llr_out, result) == MmseStatus::kOk,
                "pdsch cached caller-owned llr builder should succeed");
    expect_true(result.llr_count == cached.llrs.size(), "pdsch cached caller-owned llr count");
    for (std::size_t i = 0; i < cached.llrs.size(); ++i) {
        expect_relative_near(llrs[i], cached.llrs[i], 1.0e-5F, 1.0e-5F,
                             "pdsch cached caller-owned matches cached builder");
    }
}

void test_pdsch_scrambling_plan_reuses_bits_for_same_key() {
    mmse::pdsch::PdschDescramblingPlanCache cache{};
    const std::uint32_t llr_count = 24U;
    expect_true(mmse::pdsch::prepare_pdsch_descrambling_plan(11U, 0x1234U, 8U, 0U, llr_count,
                                                             cache) == MmseStatus::kOk,
                "pdsch prepare scrambling plan");
    expect_true(cache.valid, "pdsch scrambling cache valid");
    expect_true(cache.bits.size() == llr_count, "pdsch scrambling cache size");
    auto* first_bits_ptr = cache.bits.data();
    std::vector<std::uint8_t> first_bits = cache.bits;
    expect_true(mmse::pdsch::prepare_pdsch_descrambling_plan(11U, 0x1234U, 8U, 0U, llr_count,
                                                             cache) == MmseStatus::kOk,
                "pdsch prepare scrambling plan reuse");
    expect_true(cache.bits.data() == first_bits_ptr, "pdsch scrambling cache should reuse storage");
    expect_true(cache.bits == first_bits, "pdsch scrambling cache bits should stay unchanged");
}

void test_pdsch_caller_owned_llr_output_rejects_small_buffer() {
    ExtractDescriptor desc{};
    desc.sfn_subframe = 0U;
    desc.cell_id = 1U;
    desc.n_tx_ports = 2U;
    desc.n_rx_ant = 2U;
    desc.n_layers = 1U;
    desc.tx_mode = 2U;
    desc.channel_type = MmseChannelType::kPdsch;
    desc.start_symbol = 1U;
    desc.mod_order = 6U;
    desc.n_prb = 4U;
    desc.prb_bitmap.fill(0U);
    desc.prb_bitmap[0] = 0x000FU;
    desc.pmi = -1;

    std::array<float, 1> xre = {0.3F};
    std::array<float, 1> xim = {-0.2F};
    std::array<float, 1> sinr = {1.0F};
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), 1U, 1U, 1U, 6U};

    std::array<float, 5> small_buffer{};
    mmse::pdsch::PdschDescrambledLlrOutputView llr_out{small_buffer.data(), 5U};
    mmse::pdsch::PdschDescrambledLlrResult result{};
    expect_true(mmse::pdsch::build_backend_pdsch_descrambled_llr_result(
                    desc, out, 0x1111U, llr_out, result) == MmseStatus::kBufferTooSmall,
                "pdsch caller-owned llr builder should reject small buffer");
}

} // namespace

int main() {
    const std::array tests = {
        std::pair{"lte_descrambling_cinit_helpers_match_expected_values",
                  &test_lte_descrambling_cinit_helpers_match_expected_values},
        std::pair{"lte_descrambling_gold_sequence_matches_known_vectors",
                  &test_lte_descrambling_gold_sequence_matches_known_vectors},
        std::pair{"lte_descrambling_sequence_matches_reference_generator",
                  &test_lte_descrambling_sequence_matches_reference_generator},
        std::pair{"lte_descramble_bits_is_self_inverse", &test_lte_descramble_bits_is_self_inverse},
        std::pair{"lte_descramble_llrs_flips_sign_on_scrambling_ones",
                  &test_lte_descramble_llrs_flips_sign_on_scrambling_ones},
        std::pair{"lte_soft_demod_qpsk_matches_known_llrs",
                  &test_lte_soft_demod_qpsk_matches_known_llrs},
        std::pair{"lte_soft_demod_64qam_sign_pattern_matches_gray_mapping",
                  &test_lte_soft_demod_64qam_sign_pattern_matches_gray_mapping},
        std::pair{"lte_soft_demod_specialized_paths_match_reference",
                  &test_lte_soft_demod_specialized_paths_match_reference},
        std::pair{"lte_soft_demod_descrambled_fused_matches_split_reference",
                  &test_lte_soft_demod_descrambled_fused_matches_split_reference},
        std::pair{"pbch_backend_descrambled_llr_indication",
                  &test_pbch_backend_descrambled_llr_indication},
        std::pair{"pcfich_backend_descrambled_llr_indication",
                  &test_pcfich_backend_descrambled_llr_indication},
        std::pair{"pdcch_backend_descrambled_llr_indication",
                  &test_pdcch_backend_descrambled_llr_indication},
        std::pair{"pdcch_td_backend_descrambled_llr_indication",
                  &test_pdcch_td_backend_descrambled_llr_indication},
        std::pair{"pdsch_backend_descrambled_llr_indication_16qam",
                  &test_pdsch_backend_descrambled_llr_indication_16qam},
        std::pair{"pdsch_caller_owned_llr_output_matches_backend_builder",
                  &test_pdsch_caller_owned_llr_output_matches_backend_builder},
        std::pair{"pdsch_cached_builder_matches_uncached_builder",
                  &test_pdsch_cached_builder_matches_uncached_builder},
        std::pair{"pdsch_cached_caller_owned_output_matches_cached_builder",
                  &test_pdsch_cached_caller_owned_output_matches_cached_builder},
        std::pair{"pdsch_scrambling_plan_reuses_bits_for_same_key",
                  &test_pdsch_scrambling_plan_reuses_bits_for_same_key},
        std::pair{"pdsch_caller_owned_llr_output_rejects_small_buffer",
                  &test_pdsch_caller_owned_llr_output_rejects_small_buffer},
        std::pair{"reject_invalid_descriptor", &test_reject_invalid_descriptor},
        std::pair{"buffer_too_small", &test_buffer_too_small},
        std::pair{"single_layer_identity_channel_equalization",
                  &test_single_layer_identity_channel_equalization},
        std::pair{"single_rx_single_layer_equalization_and_pdcch",
                  &test_single_rx_single_layer_equalization_and_pdcch},
        std::pair{"sigma2_state_persists", &test_sigma2_state_persists},
        std::pair{"single_layer_path", &test_single_layer_path},
        std::pair{"pdcch_layout_excludes_crs_and_reserved_res",
                  &test_pdcch_layout_excludes_crs_and_reserved_res},
        std::pair{"pbch_layout_matches_lte_center_72_subcarriers",
                  &test_pbch_layout_matches_lte_center_72_subcarriers},
        std::pair{"pcfich_layout_matches_four_regs_without_crs",
                  &test_pcfich_layout_matches_four_regs_without_crs},
        std::pair{"single_port_lte_control_reference_vector",
                  &test_single_port_lte_control_reference_vector},
        std::pair{"pbch_frontend_dto_builds_mmse_input", &test_pbch_frontend_dto_builds_mmse_input},
        std::pair{"pcfich_frontend_dto_builds_mmse_input",
                  &test_pcfich_frontend_dto_builds_mmse_input},
        std::pair{"pbch_backend_dto_packs_equalized_output",
                  &test_pbch_backend_dto_packs_equalized_output},
        std::pair{"pcfich_backend_dto_packs_equalized_output",
                  &test_pcfich_backend_dto_packs_equalized_output},
        std::pair{"pbch_cpu_run_returns_equalized_pbch_surface",
                  &test_pbch_cpu_run_returns_equalized_pbch_surface},
        std::pair{"pcfich_cpu_run_returns_equalized_pcfich_surface",
                  &test_pcfich_cpu_run_returns_equalized_pcfich_surface},
        std::pair{"crs_values_match_independent_reference",
                  &test_crs_values_match_independent_reference},
        std::pair{"channel_interpolation_extrapolates_linearly",
                  &test_channel_interpolation_extrapolates_linearly},
        std::pair{"noise_estimate_has_two_thirds_calibration",
                  &test_noise_estimate_has_two_thirds_calibration},
        std::pair{"generation_invalidates_cpu_subframe_cache",
                  &test_generation_invalidates_cpu_subframe_cache},
        std::pair{"pbch_and_pcfich_td_cpu_golden_and_dto",
                  &test_pbch_and_pcfich_td_cpu_golden_and_dto},
        std::pair{"gpu_cuda_pbch_and_pcfich_td_match_cpu",
                  &test_gpu_cuda_pbch_and_pcfich_td_match_cpu},
        std::pair{"cpu_shared_estimate_reuses_once_per_subframe",
                  &test_cpu_shared_estimate_reuses_once_per_subframe},
        std::pair{"gpu_shared_estimate_skips_second_estimate_path",
                  &test_gpu_shared_estimate_skips_second_estimate_path},
        std::pair{"pdcch_cpu_run_supports_single_port_and_generic_two_port_layout",
                  &test_pdcch_cpu_run_supports_single_port_and_generic_two_port_layout},
        std::pair{"pdcch_td_scalar_demap_recovers_qpsk_symbols",
                  &test_pdcch_td_scalar_demap_recovers_qpsk_symbols},
        std::pair{"pdcch_td_cpu_run_recovers_qpsk_symbols",
                  &test_pdcch_td_cpu_run_recovers_qpsk_symbols},
        std::pair{"pdcch_run_rejects_two_port_contract", &test_pdcch_run_rejects_two_port_contract},
        std::pair{"pdcch_module_api_returns_chain_metadata_and_re_indices",
                  &test_pdcch_module_api_returns_chain_metadata_and_re_indices},
        std::pair{"pdcch_mmse_input_validator", &test_pdcch_mmse_input_validator},
        std::pair{"pdcch_run_rejects_inconsistent_control_subframe",
                  &test_pdcch_run_rejects_inconsistent_control_subframe},
        std::pair{"pdcch_helper_mask_and_grid_index_decode",
                  &test_pdcch_helper_mask_and_grid_index_decode},
        std::pair{"pdcch_helper_apply_reserved_re_list", &test_pdcch_helper_apply_reserved_re_list},
        std::pair{"pdcch_control_region_builds_regs_and_cces",
                  &test_pdcch_control_region_builds_regs_and_cces},
        std::pair{"pdcch_control_region_rejects_invalid_re_indices",
                  &test_pdcch_control_region_rejects_invalid_re_indices},
        std::pair{"pdcch_common_search_candidates_follow_lte_levels",
                  &test_pdcch_common_search_candidates_follow_lte_levels},
        std::pair{"pdcch_ue_specific_search_candidates_follow_lte_levels",
                  &test_pdcch_ue_specific_search_candidates_follow_lte_levels},
        std::pair{"pdcch_common_search_candidate_llrs_follow_cce_offsets",
                  &test_pdcch_common_search_candidate_llrs_follow_cce_offsets},
        std::pair{"pdcch_common_search_candidate_llrs_reject_invalid_backend",
                  &test_pdcch_common_search_candidate_llrs_reject_invalid_backend},
        std::pair{"pdcch_common_search_decode_returns_all_hits",
                  &test_pdcch_common_search_decode_returns_all_hits},
        std::pair{"pdcch_native_si_rnti_search_returns_only_target_candidate",
                  &test_pdcch_native_si_rnti_search_returns_only_target_candidate},
        std::pair{"pdcch_ue_specific_search_decodes_l1_l2_candidates",
                  &test_pdcch_ue_specific_search_decodes_l1_l2_candidates},
        std::pair{"pdcch_si_rnti_geometry_search_acquires_and_reprobes",
                  &test_pdcch_si_rnti_geometry_search_acquires_and_reprobes},
        std::pair{"pdcch_si_rnti_geometry_search_two_tx_handles_empty_profiles",
                  &test_pdcch_si_rnti_geometry_search_two_tx_handles_empty_profiles},
        std::pair{"pdcch_si_rnti_geometry_search_rejects_ambiguous_locks",
                  &test_pdcch_si_rnti_geometry_search_rejects_ambiguous_locks},
        std::pair{"pdcch_si_rnti_semantics_reject_invalid_grants",
                  &test_pdcch_si_rnti_semantics_reject_invalid_grants},
        std::pair{"pdcch_dci_format1a_si_numeric_boundaries",
                  &test_pdcch_dci_format1a_si_numeric_boundaries},
        std::pair{"pdcch_fdd_dci_format1a_supports_standard_bandwidths",
                  &test_pdcch_fdd_dci_format1a_supports_standard_bandwidths},
        std::pair{"pdcch_control_geometry_supports_standard_bandwidths",
                  &test_pdcch_control_geometry_supports_standard_bandwidths},
        std::pair{"pdcch_cpu_si_rnti_search_supports_standard_bandwidths",
                  &test_pdcch_cpu_si_rnti_search_supports_standard_bandwidths},
        std::pair{"pdcch_ue_specific_search_supports_standard_bandwidths",
                  &test_pdcch_ue_specific_search_supports_standard_bandwidths},
        std::pair{"pdcch_cpu_si_rnti_search_two_tx_supports_selected_bandwidths",
                  &test_pdcch_cpu_si_rnti_search_two_tx_supports_selected_bandwidths},
        std::pair{"pdcch_geometry_cache_tracks_bandwidth",
                  &test_pdcch_geometry_cache_tracks_bandwidth},
        std::pair{"pdcch_cpu_si_rnti_search_runs_full_one_tx_chain",
                  &test_pdcch_cpu_si_rnti_search_runs_full_one_tx_chain},
        std::pair{"pdcch_cpu_si_rnti_search_runs_full_two_tx_chain",
                  &test_pdcch_cpu_si_rnti_search_runs_full_two_tx_chain},
        std::pair{"pdcch_cpu_ue_specific_search_runs_full_one_and_two_tx_chains",
                  &test_pdcch_cpu_ue_specific_search_runs_full_one_and_two_tx_chains},
        std::pair{"pdcch_convolutional_rate_recovery_combines_repetitions",
                  &test_pdcch_convolutional_rate_recovery_combines_repetitions},
        std::pair{"pdcch_convolutional_rate_recovery_handles_dummy_bits",
                  &test_pdcch_convolutional_rate_recovery_handles_dummy_bits},
        std::pair{"pdcch_convolutional_rate_recovery_rejects_invalid_candidate",
                  &test_pdcch_convolutional_rate_recovery_rejects_invalid_candidate},
        std::pair{"pdcch_tail_biting_decoder_adapter_contract",
                  &test_pdcch_tail_biting_decoder_adapter_contract},
        std::pair{"pdcch_native_tail_biting_decoder_all_initial_states",
                  &test_pdcch_native_tail_biting_decoder_all_initial_states},
        std::pair{"pdcch_native_tail_biting_decoder_length_boundaries",
                  &test_pdcch_native_tail_biting_decoder_length_boundaries},
        std::pair{"pdcch_native_tail_biting_decoder_llr_and_invalid_inputs",
                  &test_pdcch_native_tail_biting_decoder_llr_and_invalid_inputs},
        std::pair{"pdcch_native_tail_biting_decoder_boundaries_tie_break_and_failures",
                  &test_pdcch_native_tail_biting_decoder_boundaries_tie_break_and_failures},
        std::pair{"pdcch_native_tail_biting_rate_recovery_l4_l8",
                  &test_pdcch_native_tail_biting_rate_recovery_l4_l8},
        std::pair{"pdcch_crc16_and_rnti_mask_check", &test_pdcch_crc16_and_rnti_mask_check},
        std::pair{"pdcch_dci_format1a_fdd_si_rnti_parse",
                  &test_pdcch_dci_format1a_fdd_si_rnti_parse},
        std::pair{"pdcch_dci_format1a_tdd_distributed_parse",
                  &test_pdcch_dci_format1a_tdd_distributed_parse},
        std::pair{"pdcch_dci_format1a_pdcch_order_and_crc_miss",
                  &test_pdcch_dci_format1a_pdcch_order_and_crc_miss},
        std::pair{"pdcch_dci_format1a_adapter_end_to_end",
                  &test_pdcch_dci_format1a_adapter_end_to_end},
        std::pair{"pdcch_type2_riv_decodes_long_allocation",
                  &test_pdcch_type2_riv_decodes_long_allocation},
        std::pair{"pdcch_helper_builds_pcfich_reserved_res",
                  &test_pdcch_helper_builds_pcfich_reserved_res},
        std::pair{"pdcch_helper_builds_fdd_phich_reserved_res_properties",
                  &test_pdcch_helper_builds_fdd_phich_reserved_res_properties},
        std::pair{"pdcch_helper_phich_config_contract", &test_pdcch_helper_phich_config_contract},
        std::pair{"pdcch_helper_extended_phich_special_case_distribution",
                  &test_pdcch_helper_extended_phich_special_case_distribution},
        std::pair{"pdcch_helper_extended_tdd_sf1_automatic_special_case",
                  &test_pdcch_helper_extended_tdd_sf1_automatic_special_case},
        std::pair{"pdcch_helper_tdd_phich_affects_layout",
                  &test_pdcch_helper_tdd_phich_affects_layout},
        std::pair{"pdcch_frontend_helper_auto_reserved_res_reduce_layout",
                  &test_pdcch_frontend_helper_auto_reserved_res_reduce_layout},
        std::pair{"pdcch_frontend_dto_builds_mmse_input",
                  &test_pdcch_frontend_dto_builds_mmse_input},
        std::pair{"pdcch_frontend_control_subframe_drives_helpers",
                  &test_pdcch_frontend_control_subframe_drives_helpers},
        std::pair{"pdcch_backend_dto_packs_equalized_output",
                  &test_pdcch_backend_dto_packs_equalized_output},
        std::pair{"pdcch_td_backend_dto_packs_equalized_output",
                  &test_pdcch_td_backend_dto_packs_equalized_output},
        std::pair{"pdcch_td_normalization_accepts_empty_control_region",
                  &test_pdcch_td_normalization_accepts_empty_control_region},
        std::pair{"pdcch_td_normalization_restores_cce_re_order",
                  &test_pdcch_td_normalization_restores_cce_re_order},
        std::pair{"pdcch_cpu_common_search_decode_runs_full_one_tx_chain",
                  &test_pdcch_cpu_common_search_decode_runs_full_one_tx_chain},
        std::pair{"pdcch_cpu_common_search_decode_runs_full_two_tx_chain",
                  &test_pdcch_cpu_common_search_decode_runs_full_two_tx_chain},
        std::pair{"pdcch_gpu_common_search_decode_matches_cpu_and_rejects_callbacks",
                  &test_pdcch_gpu_common_search_decode_matches_cpu_and_rejects_callbacks},
        std::pair{"pdcch_gpu_common_search_two_tx_matches_cpu",
                  &test_pdcch_gpu_common_search_two_tx_matches_cpu},
        std::pair{"pdcch_gpu_common_search_cfi_rx_aggregation_matrix",
                  &test_pdcch_gpu_common_search_cfi_rx_aggregation_matrix},
        std::pair{"pdcch_gpu_common_search_mixed_concurrency_matrix",
                  &test_pdcch_gpu_common_search_mixed_concurrency_matrix},
        std::pair{"two_layer_scalar_golden_matches", &test_two_layer_scalar_golden_matches},
        std::pair{"two_layer_avx2_matches_scalar_kernel",
                  &test_two_layer_avx2_matches_scalar_kernel},
        std::pair{"two_layer_avx2_context_matches_scalar_context",
                  &test_two_layer_avx2_context_matches_scalar_context},
        std::pair{"two_layer_constant_channel_matches_golden",
                  &test_two_layer_constant_channel_matches_golden},
        std::pair{"two_layer_repeated_runs_are_stable", &test_two_layer_repeated_runs_are_stable},
        std::pair{"gpu_context_strict_cuda_init_succeeds_and_runs_via_fallback",
                  &test_gpu_context_strict_cuda_init_succeeds_and_runs_via_fallback},
        std::pair{"gpu_context_auto_td_fallback_matches_cpu_td",
                  &test_gpu_context_auto_td_fallback_matches_cpu_td},
        std::pair{"gpu_context_cuda_td_backend_run_matches_cpu_run",
                  &test_gpu_context_cuda_td_backend_run_matches_cpu_run},
        std::pair{"gpu_context_cuda_td_matches_cpu_td", &test_gpu_context_cuda_td_matches_cpu_td},
        std::pair{"gpu_context_auto_fallback_matches_cpu_context",
                  &test_gpu_context_auto_fallback_matches_cpu_context},
        std::pair{"gpu_context_cuda_two_layer_matches_cpu_context_samples",
                  &test_gpu_context_cuda_two_layer_matches_cpu_context_samples},
        std::pair{"gpu_context_cuda_two_layer_deep_trace_matches_cpu_context_samples",
                  &test_gpu_context_cuda_two_layer_deep_trace_matches_cpu_context_samples},
        std::pair{"gpu_context_cuda_float_transport_preserves_small_signal",
                  &test_gpu_context_cuda_float_transport_preserves_small_signal},
        std::pair{"gpu_context_sigma2_state_persists", &test_gpu_context_sigma2_state_persists},
        std::pair{"gpu_context_host_owned_and_device_owned_sigma2_match_samples",
                  &test_gpu_context_host_owned_and_device_owned_sigma2_match_samples},
        std::pair{"gpu_context_destructor_releases_cuda_runtime_state",
                  &test_gpu_context_destructor_releases_cuda_runtime_state},
        std::pair{"gpu_context_device_owned_sigma2_state_persists",
                  &test_gpu_context_device_owned_sigma2_state_persists},
        std::pair{"gpu_context_device_owned_sigma2_tracks_cell_id",
                  &test_gpu_context_device_owned_sigma2_tracks_cell_id},
        std::pair{"gpu_context_invalid_stream_count_is_rejected",
                  &test_gpu_context_invalid_stream_count_is_rejected},
    };

    for (const auto& [name, fn] : tests) {
        try {
            fn();
            std::cout << "[PASS] " << name << '\n';
        } catch (const TestFailure& failure) {
            std::cerr << "[FAIL] " << name << ": " << failure.message << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
