#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "internal/mmse_internal.h"
#include "mmse/lte_descrambling.h"
#include "mmse/pdcch_chain_sdk.h"

namespace mmse::benchmark::pdcch_fixture {

constexpr std::uint32_t kDefaultCellId = 1U;
constexpr float kQpskAxis = 0.7071067811865475F;

enum class Workload {
    kRandom,
    kMixed,
};

inline std::string_view workload_name(Workload workload) noexcept {
    return workload == Workload::kMixed ? "mixed" : "random";
}

inline bool parse_workload(std::string_view text, Workload& workload) noexcept {
    if (text == "random") {
        workload = Workload::kRandom;
        return true;
    }
    if (text == "mixed") {
        workload = Workload::kMixed;
        return true;
    }
    return false;
}

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

struct ExpectedDci {
    std::uint16_t first_cce = 0U;
    std::uint8_t aggregation_level = 0U;
    std::uint16_t start_prb = 0U;
    std::uint16_t n_prb = 0U;
    std::uint8_t mcs_tbs_index = 0U;
    std::uint8_t redundancy_version = 0U;
    std::uint8_t n_prb_1a = 0U;
};

struct Case {
    GridBuffers buffers{};
    pdcch::PdcchGpuCommonSearchDecodeRequest request{};
    std::vector<ExpectedDci> expected_hits{};
};

struct HitDistribution {
    std::uint32_t subframes = 0U;
    std::uint32_t zero_hit_subframes = 0U;
    std::uint32_t one_hit_subframes = 0U;
    std::uint32_t multi_hit_subframes = 0U;
    std::uint32_t l4_hits = 0U;
    std::uint32_t l8_hits = 0U;
    std::uint32_t crc_hits = 0U;
    std::uint32_t crc_misses = 0U;

    void add(std::uint32_t candidate_count,
             const std::vector<pdcch::PdcchDciFormat1ADecodeResult>& hits) noexcept {
        ++subframes;
        crc_hits += static_cast<std::uint32_t>(hits.size());
        crc_misses += candidate_count - static_cast<std::uint32_t>(hits.size());
        if (hits.empty()) {
            ++zero_hit_subframes;
        } else if (hits.size() == 1U) {
            ++one_hit_subframes;
        } else {
            ++multi_hit_subframes;
        }
        for (const pdcch::PdcchDciFormat1ADecodeResult& hit : hits) {
            if (hit.dci.chain.aggregation_level == 4U) {
                ++l4_hits;
            } else if (hit.dci.chain.aggregation_level == 8U) {
                ++l8_hits;
            }
        }
    }
};

struct MixedWorkload {
    std::vector<Case> cases{};
};

inline PlanarGridViewF32 make_grid_view(const GridBuffers& buffers, std::uint32_t n_rx_ant,
                                        std::uint64_t generation) noexcept {
    return {
        .re = {buffers.re[0].data(), buffers.re[1].data()},
        .im = {buffers.im[0].data(), buffers.im[1].data()},
        .n_rx_ant = n_rx_ant,
        .n_symbols = kLteNumSymbolsNormalCp,
        .n_subcarriers = static_cast<std::uint32_t>(buffers.re[0].size() / kLteNumSymbolsNormalCp),
        .generation = generation,
    };
}

inline void append_bits(std::vector<std::uint8_t>& bits, std::uint32_t value, std::uint8_t width) {
    for (std::uint8_t bit = 0U; bit < width; ++bit) {
        const std::uint8_t shift = static_cast<std::uint8_t>(width - 1U - bit);
        bits.push_back(static_cast<std::uint8_t>((value >> shift) & 1U));
    }
}

inline std::uint16_t type2_riv(std::uint16_t n_prb, std::uint16_t start_prb,
                               std::uint16_t allocated_prb_count) noexcept {
    if (allocated_prb_count - 1U <= n_prb / 2U) {
        return static_cast<std::uint16_t>(n_prb * (allocated_prb_count - 1U) + start_prb);
    }
    return static_cast<std::uint16_t>(n_prb * (n_prb - allocated_prb_count + 1U) + n_prb - 1U -
                                      start_prb);
}

inline std::vector<std::uint8_t>
make_dci_bits(std::uint16_t n_prb, std::uint16_t start_prb, std::uint16_t allocated_prb_count,
              std::uint8_t mcs_tbs_index, std::uint8_t redundancy_version, std::uint8_t n_prb_1a) {
    const pdcch::PdcchDciFormat1AConfig config{.n_prb = n_prb};
    std::vector<std::uint8_t> payload{};
    append_bits(payload, 1U, 1U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, type2_riv(n_prb, start_prb, allocated_prb_count),
                pdcch::pdcch_dci_riv_bit_count(n_prb));
    append_bits(payload, mcs_tbs_index, 5U);
    append_bits(payload, 5U, 3U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, redundancy_version, 2U);
    append_bits(payload, 0U, 1U);
    append_bits(payload, n_prb_1a == 3U ? 1U : 0U, 1U);
    while (payload.size() < pdcch::pdcch_dci_format1a_payload_bit_count(config)) {
        payload.push_back(0U);
    }

    const std::uint16_t masked_crc = static_cast<std::uint16_t>(
        pdcch::calculate_pdcch_crc16(payload.data(), static_cast<std::uint32_t>(payload.size())) ^
        pdcch::kSiRnti);
    append_bits(payload, masked_crc, pdcch::kPdcchCrcBitCount);
    return payload;
}

inline std::vector<std::uint8_t>
tail_biting_convolutional_encode(const std::vector<std::uint8_t>& bits) {
    constexpr std::array<std::uint8_t, 3> kGenerators = {0133U, 0171U, 0165U};
    std::uint32_t state = 0U;
    for (std::uint32_t delay = 0U; delay < 6U; ++delay) {
        state |= static_cast<std::uint32_t>(bits[bits.size() - 1U - delay]) << delay;
    }

    std::vector<std::uint8_t> encoded{};
    encoded.reserve(bits.size() * kGenerators.size());
    for (const std::uint8_t bit : bits) {
        const std::uint32_t shift_register = (state << 1U) | bit;
        for (const std::uint8_t generator : kGenerators) {
            encoded.push_back(
                static_cast<std::uint8_t>(std::popcount(shift_register & generator) & 1U));
        }
        state = shift_register & 0x3FU;
    }
    return encoded;
}

inline std::vector<float> rate_match(const std::vector<std::uint8_t>& encoded,
                                     std::uint32_t encoded_bit_count) {
    constexpr std::uint32_t kColumns = 32U;
    constexpr std::array<std::uint8_t, kColumns> kPermutation = {
        1U, 17U, 9U, 25U, 5U, 21U, 13U, 29U, 3U, 19U, 11U, 27U, 7U, 23U, 15U, 31U,
        0U, 16U, 8U, 24U, 4U, 20U, 12U, 28U, 2U, 18U, 10U, 26U, 6U, 22U, 14U, 30U,
    };
    constexpr std::uint32_t kRate = pdcch::kPdcchConvolutionalCodeRate;
    const std::uint32_t codeword_bit_count = static_cast<std::uint32_t>(encoded.size() / kRate);
    const std::uint32_t row_count = (codeword_bit_count + kColumns - 1U) / kColumns;
    const std::uint32_t interleaver_size = row_count * kColumns;
    const std::uint32_t dummy_bit_count = interleaver_size - codeword_bit_count;
    std::vector<float> collected(kRate * interleaver_size, 0.0F);
    std::vector<std::uint8_t> present(collected.size(), 0U);

    std::uint32_t collection_index = 0U;
    for (std::uint32_t stream = 0U; stream < kRate; ++stream) {
        for (std::uint32_t column = 0U; column < kColumns; ++column) {
            for (std::uint32_t row = 0U; row < row_count; ++row) {
                const std::uint32_t padded_bit = row * kColumns + kPermutation[column];
                if (padded_bit >= dummy_bit_count) {
                    const std::uint8_t bit =
                        encoded[(padded_bit - dummy_bit_count) * kRate + stream];
                    collected[collection_index] = bit != 0U ? 5.0F : -5.0F;
                    present[collection_index] = 1U;
                }
                ++collection_index;
            }
        }
    }

    std::vector<float> output{};
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

inline std::vector<mmse::detail::Complex32>
make_qpsk_symbols(std::uint32_t n_re, std::uint16_t first_cce, std::uint8_t aggregation_level,
                  const std::vector<std::uint8_t>& decoded_bits, std::uint16_t cell_id,
                  std::uint32_t sfn_subframe) {
    const std::uint32_t encoded_offset = static_cast<std::uint32_t>(first_cce) * 72U;
    const std::uint32_t encoded_count = 72U * aggregation_level;
    const std::vector<float> target_llrs =
        rate_match(tail_biting_convolutional_encode(decoded_bits), encoded_count);
    std::vector<float> transmitted_llrs(n_re * 2U, -1.0F);
    std::copy(target_llrs.begin(), target_llrs.end(), transmitted_llrs.begin() + encoded_offset);
    lte::descramble_llrs_inplace(transmitted_llrs.data(),
                                 static_cast<std::uint32_t>(transmitted_llrs.size()),
                                 lte::pdcch_c_init(cell_id, sfn_subframe));

    std::vector<mmse::detail::Complex32> symbols(n_re);
    for (std::uint32_t re = 0U; re < n_re; ++re) {
        symbols[re] = {
            .re = -transmitted_llrs[2U * re] * kQpskAxis / 2.0F,
            .im = -transmitted_llrs[2U * re + 1U] * kQpskAxis / 2.0F,
        };
    }
    return symbols;
}

inline void fill_identity_channel(GridBuffers& buffers, const ExtractDescriptor& desc) {
    mmse::detail::ensure_crs_tables();
    const std::uint32_t subframe = mmse::detail::subframe_from_descriptor(desc);
    const std::uint32_t n_subcarriers =
        static_cast<std::uint32_t>(buffers.re[0].size() / kLteNumSymbolsNormalCp);
    for (std::uint32_t symbol = 0U; symbol < kLteNumSymbolsNormalCp; ++symbol) {
        for (std::uint32_t sc = 0U; sc < n_subcarriers; ++sc) {
            const std::size_t index = static_cast<std::size_t>(symbol) * n_subcarriers + sc;
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
                buffers.re[0][index] = crs.re;
                buffers.im[0][index] = crs.im;
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
                buffers.re[1][index] = crs.re;
                buffers.im[1][index] = crs.im;
            }
        }
    }
}

inline void fill_td_pair(GridBuffers& buffers, std::uint16_t grid0, std::uint16_t grid1,
                         mmse::detail::Complex32 s0, mmse::detail::Complex32 s1) noexcept {
    buffers.re[0][grid0] = s0.re;
    buffers.im[0][grid0] = s0.im;
    buffers.re[1][grid0] = s1.re;
    buffers.im[1][grid0] = s1.im;
    const mmse::detail::Complex32 conjugate_s1 = mmse::detail::cconj(s1);
    const mmse::detail::Complex32 conjugate_s0 = mmse::detail::cconj(s0);
    buffers.re[0][grid1] = conjugate_s1.re;
    buffers.im[0][grid1] = conjugate_s1.im;
    buffers.re[1][grid1] = -conjugate_s0.re;
    buffers.im[1][grid1] = -conjugate_s0.im;
}

inline pdcch::PdcchGpuCommonSearchDecodeRequest
make_request(const PlanarGridViewF32& grid, std::uint32_t sfn_subframe, std::uint16_t cell_id,
             std::uint16_t n_prb, std::uint8_t n_tx_ports, std::uint64_t request_id) {
    pdcch::FrontendPdcchIndication frontend{};
    frontend.sfn_subframe = sfn_subframe;
    frontend.cell_id = cell_id;
    frontend.n_tx_ports = n_tx_ports;
    frontend.tx_mode = n_tx_ports == 1U ? 1U : 2U;
    frontend.control_symbol_count = n_prb == 6U ? 4U : 3U;
    frontend.n_prb = n_prb;
    frontend.prb_bitmap.fill(0U);
    for (std::uint32_t prb = 0U; prb < n_prb; ++prb) {
        frontend.prb_bitmap[prb / 16U] |= static_cast<std::uint16_t>(1U << (prb % 16U));
    }
    frontend.control_subframe = {.duplex_mode = pdcch::PhichDuplexMode::kFdd,
                                 .subframe = static_cast<std::uint8_t>(sfn_subframe % 10U),
                                 .kind = pdcch::LteControlSubframeKind::kRegular};
    pdcch::append_pcfich_reserved_control_re_list(frontend);
    (void)pdcch::append_phich_reserved_control_re_list(frontend,
                                                       {.resource = pdcch::PhichResource::kOne,
                                                        .duration = pdcch::PhichDuration::kNormal,
                                                        .mi = 1U,
                                                        .subframe_ctx = frontend.control_subframe});

    pdcch::PdcchGpuCommonSearchDecodeRequest request{};
    request.input = pdcch::make_pdcch_mmse_input(grid, frontend);
    request.input.chain.request_id = request_id;
    request.config.dci_format1a.n_prb = n_prb;
    return request;
}

inline ExtractDescriptor make_descriptor(const PdcchMmseInput& input) {
    ExtractDescriptor desc{};
    desc.sfn_subframe = input.sfn_subframe;
    desc.cell_id = input.cell_id;
    desc.n_tx_ports = input.n_tx_ports;
    desc.n_rx_ant = static_cast<std::uint8_t>(input.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = input.tx_mode;
    desc.channel_type = MmseChannelType::kPdcch;
    desc.start_symbol = 0U;
    desc.control_symbol_count = input.control_symbol_count;
    desc.mod_order = 2U;
    desc.n_prb = input.n_prb;
    desc.prb_bitmap = input.prb_bitmap;
    desc.control_re_exclusion_masks = input.control_re_exclusion_masks;
    return desc;
}

inline bool add_symbols(Case& benchmark_case, const pdcch::PdcchCommonSearchCandidate& candidate,
                        std::uint16_t start_prb, std::uint16_t allocated_prb_count,
                        std::uint8_t mcs_tbs_index, std::uint8_t redundancy_version,
                        std::uint8_t n_prb_1a, std::vector<float>& transmitted_llrs) {
    const PdcchMmseInput& input = benchmark_case.request.input;
    const std::uint32_t encoded_offset = static_cast<std::uint32_t>(candidate.first_cce) * 72U;
    const std::vector<float> target_llrs =
        rate_match(tail_biting_convolutional_encode(
                       make_dci_bits(input.n_prb, start_prb, allocated_prb_count, mcs_tbs_index,
                                     redundancy_version, n_prb_1a)),
                   candidate.encoded_bit_count);
    if (encoded_offset > transmitted_llrs.size() ||
        target_llrs.size() > transmitted_llrs.size() - encoded_offset) {
        return false;
    }
    std::copy(target_llrs.begin(), target_llrs.end(), transmitted_llrs.begin() + encoded_offset);
    benchmark_case.expected_hits.push_back({
        .first_cce = candidate.first_cce,
        .aggregation_level = candidate.aggregation_level,
        .start_prb = start_prb,
        .n_prb = allocated_prb_count,
        .mcs_tbs_index = mcs_tbs_index,
        .redundancy_version = redundancy_version,
        .n_prb_1a = n_prb_1a,
    });
    if (candidate.aggregation_level == 8U) {
        benchmark_case.expected_hits.push_back({
            .first_cce = candidate.first_cce,
            .aggregation_level = 4U,
            .start_prb = start_prb,
            .n_prb = allocated_prb_count,
            .mcs_tbs_index = mcs_tbs_index,
            .redundancy_version = redundancy_version,
            .n_prb_1a = n_prb_1a,
        });
    }
    return true;
}

inline bool make_case(std::uint16_t n_prb, std::uint8_t n_tx_ports, std::uint32_t sfn_subframe,
                      std::uint64_t generation,
                      const std::vector<std::pair<std::uint8_t, std::uint16_t>>& candidates_to_fill,
                      Case& benchmark_case) {
    benchmark_case = {};
    const std::uint32_t n_subcarriers = lte_downlink_subcarrier_count(n_prb);
    for (std::uint32_t rx = 0U; rx < kMmseV1MaxNumRxAntennas; ++rx) {
        benchmark_case.buffers.re[rx].assign(kLteNumSymbolsNormalCp * n_subcarriers, 0.0F);
        benchmark_case.buffers.im[rx].assign(kLteNumSymbolsNormalCp * n_subcarriers, 0.0F);
    }
    const PlanarGridViewF32 grid =
        make_grid_view(benchmark_case.buffers, kMmseV1MaxNumRxAntennas, generation);
    benchmark_case.request =
        make_request(grid, sfn_subframe, kDefaultCellId, n_prb, n_tx_ports, generation);
    const PdcchMmseInput& input = benchmark_case.request.input;
    const ExtractDescriptor desc = make_descriptor(input);
    fill_identity_channel(benchmark_case.buffers, desc);

    mmse::detail::ReLayout layout{};
    const std::uint32_t n_source_re = mmse::detail::build_pdcch_re_layout(desc, layout);
    pdcch::PdcchControlRegion region{};
    if (n_source_re == 0U ||
        pdcch::build_pdcch_control_region(desc.cell_id, desc.control_symbol_count,
                                          layout.grid_indices.data(), n_source_re, region,
                                          n_prb) != MmseStatus::kOk) {
        return candidates_to_fill.empty();
    }
    std::vector<pdcch::PdcchCommonSearchCandidate> candidates{};
    pdcch::build_pdcch_common_search_candidates(region, candidates);
    std::vector<float> transmitted_llrs(layout.n_re * 2U, -1.0F);
    for (const auto [aggregation_level, first_cce] : candidates_to_fill) {
        const auto candidate = std::find_if(
            candidates.begin(), candidates.end(),
            [aggregation_level, first_cce](const pdcch::PdcchCommonSearchCandidate& value) {
                return value.aggregation_level == aggregation_level && value.first_cce == first_cce;
            });
        if (candidate == candidates.end()) {
            return false;
        }
        const bool is_l8 = aggregation_level == 8U;
        const std::uint16_t start_prb = static_cast<std::uint16_t>(is_l8 ? n_prb / 2U : n_prb / 3U);
        const std::uint16_t allocated_prb_count = static_cast<std::uint16_t>(
            std::min<std::uint16_t>(3U, static_cast<std::uint16_t>(n_prb - start_prb)));
        if (!add_symbols(benchmark_case, *candidate, start_prb, allocated_prb_count,
                         is_l8 ? 12U : 7U, is_l8 ? 3U : 2U, is_l8 ? 2U : 3U, transmitted_llrs)) {
            return false;
        }
    }
    lte::descramble_llrs_inplace(transmitted_llrs.data(),
                                 static_cast<std::uint32_t>(transmitted_llrs.size()),
                                 lte::pdcch_c_init(input.cell_id, input.sfn_subframe));
    std::vector<mmse::detail::Complex32> symbols(layout.n_re);
    for (std::uint32_t re = 0U; re < layout.n_re; ++re) {
        symbols[re] = {
            .re = -transmitted_llrs[2U * re] * kQpskAxis / 2.0F,
            .im = -transmitted_llrs[2U * re + 1U] * kQpskAxis / 2.0F,
        };
    }
    if (input.n_tx_ports == 1U) {
        for (std::uint32_t re = 0U; re < layout.n_re; ++re) {
            benchmark_case.buffers.re[0][layout.grid_indices[re]] = symbols[re].re;
            benchmark_case.buffers.im[0][layout.grid_indices[re]] = symbols[re].im;
            benchmark_case.buffers.re[1][layout.grid_indices[re]] = symbols[re].re;
            benchmark_case.buffers.im[1][layout.grid_indices[re]] = symbols[re].im;
        }
    } else {
        if ((layout.n_re & 1U) != 0U) {
            return false;
        }
        for (std::uint32_t re = 0U; re < layout.n_re; re += 2U) {
            fill_td_pair(benchmark_case.buffers, layout.grid_indices[re],
                         layout.grid_indices[re + 1U], symbols[re], symbols[re + 1U]);
        }
    }
    return true;
}

inline bool make_mixed_workload(std::uint16_t n_prb, std::uint8_t n_tx_ports,
                                MixedWorkload& workload) {
    workload = {};
    workload.cases.reserve(4U);
    const bool supports_l8 = n_prb != 6U;
    const std::array<std::vector<std::pair<std::uint8_t, std::uint16_t>>, 4> case_candidates = {
        std::vector<std::pair<std::uint8_t, std::uint16_t>>{},
        supports_l8 ? std::vector<std::pair<std::uint8_t, std::uint16_t>>{{4U, 4U}}
                    : std::vector<std::pair<std::uint8_t, std::uint16_t>>{{4U, 0U}},
        supports_l8 ? std::vector<std::pair<std::uint8_t, std::uint16_t>>{{8U, 8U}}
                    : std::vector<std::pair<std::uint8_t, std::uint16_t>>{{4U, 0U}},
        supports_l8 ? std::vector<std::pair<std::uint8_t, std::uint16_t>>{{4U, 4U}, {8U, 8U}}
                    : std::vector<std::pair<std::uint8_t, std::uint16_t>>{{4U, 0U}},
    };
    for (std::size_t index = 0U; index < case_candidates.size(); ++index) {
        Case benchmark_case{};
        if (!make_case(n_prb, n_tx_ports, static_cast<std::uint32_t>(index), index + 1U,
                       case_candidates[index], benchmark_case)) {
            workload = {};
            return false;
        }
        workload.cases.push_back(std::move(benchmark_case));
    }
    return true;
}

inline bool matches_expected_dci(const pdcch::PdcchDciFormat1ADecodeResult& decoded,
                                 const ExpectedDci& expected) noexcept {
    return decoded.matched && decoded.dci.chain.first_cce == expected.first_cce &&
           decoded.dci.chain.aggregation_level == expected.aggregation_level &&
           decoded.dci.start_prb == expected.start_prb && decoded.dci.n_prb == expected.n_prb &&
           decoded.dci.mcs_tbs_index == expected.mcs_tbs_index &&
           decoded.dci.redundancy_version == expected.redundancy_version &&
           decoded.dci.n_prb_1a == expected.n_prb_1a;
}

inline bool validate_result(const pdcch::PdcchCommonSearchDecodeResult& result,
                            const Case& benchmark_case) noexcept {
    if (result.hits.size() != benchmark_case.expected_hits.size()) {
        return false;
    }
    for (const ExpectedDci& expected : benchmark_case.expected_hits) {
        const auto found =
            std::find_if(result.hits.begin(), result.hits.end(),
                         [&expected](const pdcch::PdcchDciFormat1ADecodeResult& decoded) {
                             return matches_expected_dci(decoded, expected);
                         });
        if (found == result.hits.end()) {
            return false;
        }
    }
    return true;
}

inline bool validate_result(const pdcch::PdcchGpuCommonSearchDecodeResult& result,
                            const Case& benchmark_case) noexcept {
    pdcch::PdcchCommonSearchDecodeResult cpu_shape{};
    cpu_shape.candidate_count = result.candidate_count;
    cpu_shape.hits = result.hits;
    return validate_result(cpu_shape, benchmark_case);
}

} // namespace mmse::benchmark::pdcch_fixture