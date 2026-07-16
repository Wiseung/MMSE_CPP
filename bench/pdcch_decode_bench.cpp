#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

#include "pdcch_benchmark_fixture.h"
#include "mmse/pdcch_chain_sdk.h"

using namespace mmse;
namespace fixture = mmse::benchmark::pdcch_fixture;

namespace {

constexpr std::uint32_t kPdcchCapacity =
    kLteMaxPdcchControlSymbolsNormalCp * kLteNumSubcarriers20MHz;
constexpr std::uint32_t kWarmupSubframes = 10U;
constexpr std::uint32_t kMeasuredSubframes = 100U;
constexpr std::uint32_t kWindowSubframes = 10U;

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

struct PdcchBuffers {
    std::vector<float> x_hat_re = std::vector<float>(kPdcchCapacity);
    std::vector<float> x_hat_im = std::vector<float>(kPdcchCapacity);
    std::vector<float> sinr = std::vector<float>(kPdcchCapacity);
    std::vector<std::uint16_t> re_grid_indices = std::vector<std::uint16_t>(kPdcchCapacity);
    std::vector<std::uint16_t> td_re_grid_indices0 = std::vector<std::uint16_t>(kPdcchCapacity);
    std::vector<std::uint16_t> td_re_grid_indices1 = std::vector<std::uint16_t>(kPdcchCapacity);
    std::vector<std::uint16_t> td4_re_grid_indices0 = std::vector<std::uint16_t>(kPdcchCapacity);
    std::vector<std::uint16_t> td4_re_grid_indices1 = std::vector<std::uint16_t>(kPdcchCapacity);
    std::vector<std::uint16_t> td4_re_grid_indices2 = std::vector<std::uint16_t>(kPdcchCapacity);
    std::vector<std::uint16_t> td4_re_grid_indices3 = std::vector<std::uint16_t>(kPdcchCapacity);
    PdcchMmseOutputView output{
        .x_hat_re = x_hat_re.data(),
        .x_hat_im = x_hat_im.data(),
        .sinr = sinr.data(),
        .re_grid_indices = re_grid_indices.data(),
        .capacity_re_per_layer = kPdcchCapacity,
        .capacity_re_metadata = kPdcchCapacity,
    };
    PdcchMmseResult metadata{};
    PdcchTdMmseOutputView td_output{
        .x_hat_re = x_hat_re.data(),
        .x_hat_im = x_hat_im.data(),
        .sinr = sinr.data(),
        .re_grid_indices0 = td_re_grid_indices0.data(),
        .re_grid_indices1 = td_re_grid_indices1.data(),
        .capacity_symbols = kPdcchCapacity,
    };
    PdcchTdMmseResult td_metadata{};
    PdcchTd4MmseOutputView td4_output{
        .x_hat_re = x_hat_re.data(),
        .x_hat_im = x_hat_im.data(),
        .sinr = sinr.data(),
        .re_grid_indices0 = td4_re_grid_indices0.data(),
        .re_grid_indices1 = td4_re_grid_indices1.data(),
        .re_grid_indices2 = td4_re_grid_indices2.data(),
        .re_grid_indices3 = td4_re_grid_indices3.data(),
        .capacity_symbols = kPdcchCapacity,
    };
    PdcchTd4MmseResult td4_metadata{};
};

struct StageTimes {
    double ce_mmse_us = 0.0;
    double reg_cce_us = 0.0;
    double llr_us = 0.0;
    double candidate_gather_us = 0.0;
    double rate_recovery_us = 0.0;
    double tail_biting_viterbi_us = 0.0;
    double crc_dci_us = 0.0;
    double total_us = 0.0;
    std::uint32_t candidate_count = 0U;
    std::uint32_t crc_hit_count = 0U;
    std::uint32_t crc_miss_count = 0U;
    std::uint32_t zero_hit_subframes = 0U;
    std::uint32_t one_hit_subframes = 0U;
    std::uint32_t multi_hit_subframes = 0U;
    std::uint32_t l4_hit_count = 0U;
    std::uint32_t l8_hit_count = 0U;
};

GridBuffers make_random_grid(std::uint32_t seed, std::uint16_t n_prb) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    GridBuffers buffers{};
    for (std::uint32_t rx = 0U; rx < kMmseV1MaxNumRxAntennas; ++rx) {
        buffers.re[rx].resize(kLteNumSymbolsNormalCp * lte_downlink_subcarrier_count(n_prb));
        buffers.im[rx].resize(kLteNumSymbolsNormalCp * lte_downlink_subcarrier_count(n_prb));
        for (std::size_t sample = 0U; sample < buffers.re[rx].size(); ++sample) {
            buffers.re[rx][sample] = distribution(generator);
            buffers.im[rx][sample] = distribution(generator);
        }
    }
    return buffers;
}

PlanarGridViewF32 make_grid_view(const GridBuffers& buffers, std::uint32_t n_rx_ant) {
    return {
        .re = {buffers.re[0].data(), n_rx_ant > 1U ? buffers.re[1].data() : nullptr},
        .im = {buffers.im[0].data(), n_rx_ant > 1U ? buffers.im[1].data() : nullptr},
        .n_rx_ant = n_rx_ant,
        .n_symbols = kLteNumSymbolsNormalCp,
        .n_subcarriers = static_cast<std::uint32_t>(buffers.re[0].size() / kLteNumSymbolsNormalCp),
    };
}

PdcchMmseInput make_pdcch_input(const PlanarGridViewF32& grid, std::uint16_t n_prb,
                                std::uint8_t n_tx_ports) {
    pdcch::FrontendPdcchIndication frontend{};
    frontend.cell_id = 1U;
    frontend.n_tx_ports = n_tx_ports;
    frontend.tx_mode = n_tx_ports == 1U ? 1U : 2U;
    frontend.control_symbol_count = n_prb == 6U ? 4U : 3U;
    frontend.n_prb = n_prb;
    frontend.prb_bitmap.fill(0U);
    for (std::uint32_t prb = 0U; prb < n_prb; ++prb) {
        frontend.prb_bitmap[prb / 16U] |= static_cast<std::uint16_t>(1U << (prb % 16U));
    }
    frontend.control_subframe = {.duplex_mode = pdcch::PhichDuplexMode::kFdd,
                                 .subframe = 0U,
                                 .ul_dl_config = 0U,
                                 .kind = pdcch::LteControlSubframeKind::kRegular};
    pdcch::append_pcfich_reserved_control_re_list(frontend);
    (void)pdcch::append_phich_reserved_control_re_list(frontend,
                                                       {.resource = pdcch::PhichResource::kOne,
                                                        .duration = pdcch::PhichDuration::kNormal,
                                                        .mi = 1U,
                                                        .subframe_ctx = frontend.control_subframe});
    return pdcch::make_pdcch_mmse_input(grid, frontend);
}

bool gather_common_search_candidates(const pdcch::BackendPdcchEqualizedIndication& backend,
                                     const pdcch::PdcchControlRegion& control_region,
                                     const std::vector<float>& full_llrs,
                                     std::vector<pdcch::PdcchCandidateLlr>& candidates,
                                     fixture::Workload workload) {
    candidates.clear();
    std::vector<pdcch::PdcchCommonSearchCandidate> descriptors{};
    pdcch::build_pdcch_common_search_candidates(control_region, descriptors);
    if (workload == fixture::Workload::kFixed) {
        const auto fixed =
            std::find_if(descriptors.begin(), descriptors.end(), [](const auto& value) {
                return value.aggregation_level == fixture::kFixedAggregationLevel &&
                       value.first_cce == fixture::kFixedFirstCce;
            });
        if (fixed == descriptors.end()) {
            return false;
        }
        descriptors = {*fixed};
    }
    candidates.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        const std::uint32_t llr_offset = static_cast<std::uint32_t>(descriptor.first_cce) * 72U;
        if (llr_offset > full_llrs.size() ||
            descriptor.encoded_bit_count > full_llrs.size() - llr_offset) {
            candidates.clear();
            return false;
        }
        pdcch::PdcchCandidateLlr candidate{};
        candidate.sfn_subframe = backend.sfn_subframe;
        candidate.cell_id = backend.cell_id;
        candidate.chain = backend.chain;
        candidate.chain.candidate_id = descriptor.candidate_id;
        candidate.chain.first_cce = descriptor.first_cce;
        candidate.chain.aggregation_level = descriptor.aggregation_level;
        candidate.encoded_bit_count = descriptor.encoded_bit_count;
        candidate.llrs.assign(full_llrs.begin() + llr_offset,
                              full_llrs.begin() + llr_offset + descriptor.encoded_bit_count);
        candidates.push_back(std::move(candidate));
    }
    return true;
}

bool run_subframe(MmseEqualizerCpuContext& context, PdcchMmseInput input, PdcchBuffers& buffers,
                  StageTimes& times, std::uint32_t subframe_index, fixture::Workload workload,
                  const fixture::Case* expected_case = nullptr) {
    using Clock = std::chrono::steady_clock;
    if (expected_case == nullptr) {
        input.sfn_subframe = subframe_index;
        input.control_subframe.subframe = static_cast<std::uint8_t>(subframe_index % 10U);
    }

    double fixed_e2e_us = 0.0;
    if (workload == fixture::Workload::kFixed) {
        if (expected_case == nullptr) {
            return false;
        }
        const auto fixed_request = fixture::make_fixed_request(*expected_case);
        pdcch::PdcchDciFormat1ADecodeResult fixed_result{};
        const auto fixed_start = Clock::now();
        const MmseStatus fixed_status = pdcch::run_pdcch_cpu_fixed_candidate_decode(
            context, fixed_request.input, fixed_request.config, fixed_result);
        fixed_e2e_us =
            std::chrono::duration<double, std::micro>(Clock::now() - fixed_start).count();
        if (fixed_status != MmseStatus::kOk ||
            !fixture::validate_fixed_result(fixed_result, *expected_case)) {
            return false;
        }
    }

    const auto total_start = Clock::now();
    auto start = Clock::now();
    pdcch::BackendPdcchEqualizedIndication backend{};
    MmseStatus ce_mmse_status = MmseStatus::kUnsupportedConfig;
    if (input.n_tx_ports == 1U) {
        ce_mmse_status = context.run_pdcch(input, buffers.output, buffers.metadata);
    } else if (input.n_tx_ports == 2U) {
        ce_mmse_status = context.run_pdcch_td(input, buffers.td_output, buffers.td_metadata);
    } else if (input.n_tx_ports == 4U) {
        ce_mmse_status = context.run_pdcch_td4(input, buffers.td4_output, buffers.td4_metadata);
    }
    auto end = Clock::now();
    times.ce_mmse_us = std::chrono::duration<double, std::micro>(end - start).count();
    if (ce_mmse_status != MmseStatus::kOk) {
        return false;
    }
    if (input.n_tx_ports == 1U) {
        backend = pdcch::make_backend_pdcch_equalized_indication(buffers.metadata, buffers.output);
    } else if (input.n_tx_ports == 2U) {
        const pdcch::BackendPdcchTdEqualizedIndication td_backend =
            pdcch::make_backend_pdcch_td_equalized_indication(buffers.td_metadata,
                                                              buffers.td_output);
        if (pdcch::normalize_pdcch_td_cce_order(td_backend, backend) != MmseStatus::kOk) {
            return false;
        }
    } else {
        const pdcch::BackendPdcchTd4EqualizedIndication td4_backend =
            pdcch::make_backend_pdcch_td4_equalized_indication(buffers.td4_metadata,
                                                               buffers.td4_output);
        if (pdcch::normalize_pdcch_td4_cce_order(td4_backend, backend) != MmseStatus::kOk) {
            return false;
        }
    }

    pdcch::PdcchControlRegion control_region{};
    start = Clock::now();
    if (pdcch::build_pdcch_control_region(
            backend.cell_id, backend.control_symbol_count, backend.re_grid_indices.data(),
            static_cast<std::uint32_t>(backend.re_grid_indices.size()), control_region,
            backend.n_prb) != MmseStatus::kOk) {
        return false;
    }
    end = Clock::now();
    times.reg_cce_us = std::chrono::duration<double, std::micro>(end - start).count();

    std::vector<float> full_llrs{};
    start = Clock::now();
    if (lte::build_max_log_descrambled_llrs(
            backend.x_hat_re.data(), backend.x_hat_im.data(), backend.sinr.data(),
            static_cast<std::uint32_t>(backend.x_hat_re.size()),
            static_cast<std::uint32_t>(backend.x_hat_re.size()), backend.n_layers,
            backend.mod_order, lte::pdcch_c_init(backend.cell_id, backend.sfn_subframe),
            full_llrs) != MmseStatus::kOk) {
        return false;
    }
    end = Clock::now();
    times.llr_us = std::chrono::duration<double, std::micro>(end - start).count();

    std::vector<pdcch::PdcchCandidateLlr> candidates{};
    start = Clock::now();
    if (!gather_common_search_candidates(backend, control_region, full_llrs, candidates,
                                         workload)) {
        return false;
    }
    end = Clock::now();
    times.candidate_gather_us = std::chrono::duration<double, std::micro>(end - start).count();

    const pdcch::PdcchDciFormat1AConfig dci_config{.n_prb = backend.n_prb};
    const std::uint32_t payload_bit_count = pdcch::pdcch_dci_format1a_payload_bit_count(dci_config);
    std::vector<pdcch::PdcchRateRecoveredLlr> recovered(candidates.size());
    start = Clock::now();
    for (std::size_t candidate = 0U; candidate < candidates.size(); ++candidate) {
        if (pdcch::recover_pdcch_convolutional_rate_matched_llrs(
                candidates[candidate], payload_bit_count, recovered[candidate]) !=
            MmseStatus::kOk) {
            return false;
        }
    }
    end = Clock::now();
    times.rate_recovery_us = std::chrono::duration<double, std::micro>(end - start).count();

    std::vector<std::vector<std::uint8_t>> decoded_bits(recovered.size());
    start = Clock::now();
    for (std::size_t candidate = 0U; candidate < recovered.size(); ++candidate) {
        if (pdcch::decode_pdcch_tail_biting_convolutional(
                recovered[candidate], decoded_bits[candidate]) != MmseStatus::kOk) {
            return false;
        }
    }
    end = Clock::now();
    times.tail_biting_viterbi_us = std::chrono::duration<double, std::micro>(end - start).count();

    start = Clock::now();
    std::uint32_t hit_count = 0U;
    std::vector<pdcch::PdcchDciFormat1ADecodeResult> hits{};
    for (std::size_t candidate = 0U; candidate < recovered.size(); ++candidate) {
        pdcch::PdcchDciFormat1ADecodeResult result{};
        if (pdcch::validate_and_parse_pdcch_dci_format1a(
                decoded_bits[candidate].data(),
                static_cast<std::uint32_t>(decoded_bits[candidate].size()), pdcch::kSiRnti,
                recovered[candidate].sfn_subframe, recovered[candidate].cell_id,
                recovered[candidate].chain, dci_config, result) != MmseStatus::kOk) {
            return false;
        }
        hit_count += result.matched ? 1U : 0U;
        if (result.matched) {
            hits.push_back(std::move(result));
        }
    }
    end = Clock::now();
    times.crc_dci_us = std::chrono::duration<double, std::micro>(end - start).count();
    times.candidate_count = static_cast<std::uint32_t>(candidates.size());
    times.crc_hit_count = hit_count;
    times.crc_miss_count = times.candidate_count - hit_count;
    if (hit_count == 0U) {
        times.zero_hit_subframes = 1U;
    } else if (hit_count == 1U) {
        times.one_hit_subframes = 1U;
    } else {
        times.multi_hit_subframes = 1U;
    }
    for (const auto& hit : hits) {
        if (hit.dci.chain.aggregation_level == 4U) {
            ++times.l4_hit_count;
        } else if (hit.dci.chain.aggregation_level == 8U) {
            ++times.l8_hit_count;
        }
    }
    if (expected_case != nullptr) {
        if (workload == fixture::Workload::kFixed) {
            pdcch::PdcchDciFormat1ADecodeResult result{};
            if (!hits.empty()) {
                result = std::move(hits.front());
            }
            if (!fixture::validate_fixed_result(result, *expected_case)) {
                return false;
            }
        } else {
            pdcch::PdcchCommonSearchDecodeResult result{};
            result.candidate_count = times.candidate_count;
            result.hits = std::move(hits);
            if (!fixture::validate_result(result, *expected_case)) {
                return false;
            }
        }
    }
    times.total_us =
        workload == fixture::Workload::kFixed
            ? fixed_e2e_us
            : std::chrono::duration<double, std::micro>(Clock::now() - total_start).count();
    return true;
}

StageTimes sum_window(const std::vector<StageTimes>& values, std::size_t first, std::size_t count) {
    StageTimes sum{};
    for (std::size_t index = first; index < first + count; ++index) {
        sum.ce_mmse_us += values[index].ce_mmse_us;
        sum.reg_cce_us += values[index].reg_cce_us;
        sum.llr_us += values[index].llr_us;
        sum.candidate_gather_us += values[index].candidate_gather_us;
        sum.rate_recovery_us += values[index].rate_recovery_us;
        sum.tail_biting_viterbi_us += values[index].tail_biting_viterbi_us;
        sum.crc_dci_us += values[index].crc_dci_us;
        sum.total_us += values[index].total_us;
        sum.candidate_count += values[index].candidate_count;
        sum.crc_hit_count += values[index].crc_hit_count;
        sum.crc_miss_count += values[index].crc_miss_count;
        sum.zero_hit_subframes += values[index].zero_hit_subframes;
        sum.one_hit_subframes += values[index].one_hit_subframes;
        sum.multi_hit_subframes += values[index].multi_hit_subframes;
        sum.l4_hit_count += values[index].l4_hit_count;
        sum.l8_hit_count += values[index].l8_hit_count;
    }
    return sum;
}

template <typename Projection>
void print_summary(std::string_view scope, std::string_view stage,
                   const std::vector<StageTimes>& values, Projection projection) {
    std::vector<double> samples{};
    samples.reserve(values.size());
    for (const StageTimes& value : values) {
        samples.push_back(projection(value));
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double percentile_value) {
        const std::size_t index =
            static_cast<std::size_t>(percentile_value * static_cast<double>(samples.size() - 1U));
        return samples[index];
    };
    const double average =
        std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    std::cout << scope << '.' << stage << ".avg_us=" << average << '\n';
    std::cout << scope << '.' << stage << ".p50_us=" << percentile(0.50) << '\n';
    std::cout << scope << '.' << stage << ".p95_us=" << percentile(0.95) << '\n';
}

template <typename Projection>
void print_count_summary(std::string_view scope, std::string_view metric,
                         const std::vector<StageTimes>& values, Projection projection) {
    std::vector<double> samples{};
    samples.reserve(values.size());
    for (const StageTimes& value : values) {
        samples.push_back(projection(value));
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double percentile_value) {
        const std::size_t index =
            static_cast<std::size_t>(percentile_value * static_cast<double>(samples.size() - 1U));
        return samples[index];
    };
    const double average =
        std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    std::cout << scope << '.' << metric << ".avg_count=" << average << '\n';
    std::cout << scope << '.' << metric << ".p50_count=" << percentile(0.50) << '\n';
    std::cout << scope << '.' << metric << ".p95_count=" << percentile(0.95) << '\n';
}

void print_summaries(std::string_view scope, const std::vector<StageTimes>& values) {
    print_summary(scope, "ce_mmse", values,
                  [](const StageTimes& times) { return times.ce_mmse_us; });
    print_summary(scope, "reg_cce", values,
                  [](const StageTimes& times) { return times.reg_cce_us; });
    print_summary(scope, "llr", values, [](const StageTimes& times) { return times.llr_us; });
    print_summary(scope, "candidate_gather", values,
                  [](const StageTimes& times) { return times.candidate_gather_us; });
    print_summary(scope, "rate_recovery", values,
                  [](const StageTimes& times) { return times.rate_recovery_us; });
    print_summary(scope, "tail_biting_viterbi", values,
                  [](const StageTimes& times) { return times.tail_biting_viterbi_us; });
    print_summary(scope, "crc_dci", values,
                  [](const StageTimes& times) { return times.crc_dci_us; });
    print_count_summary(scope, "candidate", values, [](const StageTimes& times) {
        return static_cast<double>(times.candidate_count);
    });
    print_count_summary(scope, "crc_hit", values, [](const StageTimes& times) {
        return static_cast<double>(times.crc_hit_count);
    });
    print_count_summary(scope, "crc_miss", values, [](const StageTimes& times) {
        return static_cast<double>(times.crc_miss_count);
    });
    print_count_summary(scope, "zero_hit_subframes", values, [](const StageTimes& times) {
        return static_cast<double>(times.zero_hit_subframes);
    });
    print_count_summary(scope, "one_hit_subframes", values, [](const StageTimes& times) {
        return static_cast<double>(times.one_hit_subframes);
    });
    print_count_summary(scope, "multi_hit_subframes", values, [](const StageTimes& times) {
        return static_cast<double>(times.multi_hit_subframes);
    });
    print_count_summary(scope, "l4_hit", values, [](const StageTimes& times) {
        return static_cast<double>(times.l4_hit_count);
    });
    print_count_summary(scope, "l8_hit", values, [](const StageTimes& times) {
        return static_cast<double>(times.l8_hit_count);
    });
    print_summary(scope, "e2e_host_wall", values,
                  [](const StageTimes& times) { return times.total_us; });
}

} // namespace

bool parse_uint_argument(const char* value_text, unsigned int& value) {
    const auto [end, error] =
        std::from_chars(value_text, value_text + std::char_traits<char>::length(value_text), value);
    return error == std::errc{} && *end == '\0';
}

bool parse_arguments(int argc, char** argv, std::uint16_t& n_prb, std::uint8_t& n_tx_ports,
                     std::uint8_t& n_rx_ant, fixture::Workload& workload) {
    bool n_rx_ant_explicit = false;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            return false;
        }
        const std::string_view option(argv[index]);
        if (option == "--workload") {
            if (!fixture::parse_workload(argv[index + 1], workload)) {
                return false;
            }
            continue;
        }
        unsigned int value = 0U;
        if (!parse_uint_argument(argv[index + 1], value)) {
            return false;
        }
        if (option == "--n-prb") {
            if (value > std::numeric_limits<std::uint16_t>::max()) {
                return false;
            }
            n_prb = static_cast<std::uint16_t>(value);
            continue;
        }
        if (option == "--n-tx-ports") {
            if (value > std::numeric_limits<std::uint8_t>::max()) {
                return false;
            }
            n_tx_ports = static_cast<std::uint8_t>(value);
            continue;
        }
        if (option == "--n-rx-ant") {
            if (value > std::numeric_limits<std::uint8_t>::max()) {
                return false;
            }
            n_rx_ant = static_cast<std::uint8_t>(value);
            n_rx_ant_explicit = true;
            continue;
        }
        return false;
    }
    if (!n_rx_ant_explicit) {
        n_rx_ant = workload == fixture::Workload::kFixed ? 2U : 1U;
    }
    return is_supported_lte_downlink_bandwidth(n_prb) &&
           (n_tx_ports == 1U || n_tx_ports == 2U || n_tx_ports == 4U) &&
           (n_rx_ant == 1U || n_rx_ant == 2U) && (n_tx_ports != 4U || n_rx_ant == 1U) &&
           (workload != fixture::Workload::kFixed ||
            (n_prb == kLteNumPrb20MHz && n_tx_ports == 1U && n_rx_ant == 2U));
}

void print_usage() {
    std::cerr << "usage: pdcch_decode_bench [--n-prb 6|15|25|50|75|100] "
                 "[--n-tx-ports 1|2|4] [--n-rx-ant 1|2] "
                 "[--workload fixed|mixed|random]\n"
                 "       defaults to 1Rx; fixed defaults to its historical 1Tx x 2Rx scope; "
                 "4Tx requires 1Rx\n";
}

int main(int argc, char** argv) {
    std::uint16_t n_prb = kLteNumPrb20MHz;
    std::uint8_t n_tx_ports = 1U;
    std::uint8_t n_rx_ant = 1U;
    fixture::Workload workload_kind = fixture::Workload::kMixed;
    if (!parse_arguments(argc, argv, n_prb, n_tx_ports, n_rx_ant, workload_kind)) {
        print_usage();
        return 2;
    }

    GridBuffers grid_buffers = make_random_grid(42U, n_prb);
    const PlanarGridViewF32 grid = make_grid_view(grid_buffers, n_rx_ant);
    const PdcchMmseInput input = make_pdcch_input(grid, n_prb, n_tx_ports);

    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    if (context.init(cpu_config) != MmseStatus::kOk) {
        std::cerr << "failed to initialize CPU context\n";
        return 1;
    }

    fixture::MixedWorkload mixed_workload{};
    if (workload_kind != fixture::Workload::kRandom) {
        if (!fixture::make_mixed_workload(n_prb, n_tx_ports, n_rx_ant, mixed_workload)) {
            std::cerr << "failed to construct deterministic workload\n";
            return 1;
        }
        for (std::size_t case_index = 0U; case_index < mixed_workload.cases.size(); ++case_index) {
            const fixture::Case& benchmark_case = mixed_workload.cases[case_index];
            if (workload_kind == fixture::Workload::kFixed) {
                const auto request = fixture::make_fixed_request(benchmark_case);
                pdcch::PdcchDciFormat1ADecodeResult smoke_result{};
                if (pdcch::run_pdcch_cpu_fixed_candidate_decode(
                        context, request.input, request.config, smoke_result) != MmseStatus::kOk ||
                    !fixture::validate_fixed_result(smoke_result, benchmark_case)) {
                    std::cerr << "fixed workload smoke check failed case=" << case_index << '\n';
                    return 1;
                }
            } else {
                pdcch::PdcchCommonSearchDecodeResult smoke_result{};
                if (pdcch::run_pdcch_cpu_common_search_decode(context, benchmark_case.request.input,
                                                              benchmark_case.request.config,
                                                              smoke_result) != MmseStatus::kOk ||
                    !fixture::validate_result(smoke_result, benchmark_case)) {
                    std::cerr << "mixed workload smoke check failed case=" << case_index << '\n';
                    return 1;
                }
            }
        }
    }

    const auto select_input = [&](std::uint32_t iteration,
                                  const fixture::Case*& expected_case) -> PdcchMmseInput {
        expected_case = nullptr;
        if (workload_kind != fixture::Workload::kRandom) {
            expected_case = &mixed_workload.cases[iteration % mixed_workload.cases.size()];
            return expected_case->request.input;
        }
        return input;
    };

    PdcchBuffers buffers{};
    StageTimes ignored{};
    for (std::uint32_t subframe = 0U; subframe < kWarmupSubframes; ++subframe) {
        const fixture::Case* expected_case = nullptr;
        const PdcchMmseInput selected_input = select_input(subframe, expected_case);
        if (!run_subframe(context, selected_input, buffers, ignored, subframe, workload_kind,
                          expected_case)) {
            std::cerr << "warmup failed\n";
            return 1;
        }
    }

    std::vector<StageTimes> per_subframe{};
    per_subframe.reserve(kMeasuredSubframes);
    for (std::uint32_t subframe = 0U; subframe < kMeasuredSubframes; ++subframe) {
        StageTimes times{};
        const fixture::Case* expected_case = nullptr;
        const PdcchMmseInput selected_input =
            select_input(kWarmupSubframes + subframe, expected_case);
        if (!run_subframe(context, selected_input, buffers, times, kWarmupSubframes + subframe,
                          workload_kind, expected_case)) {
            std::cerr << "measurement failed\n";
            return 1;
        }
        per_subframe.push_back(times);
    }

    std::vector<StageTimes> per_window{};
    per_window.reserve(per_subframe.size() / kWindowSubframes);
    for (std::size_t first = 0U; first + kWindowSubframes <= per_subframe.size();
         first += kWindowSubframes) {
        per_window.push_back(sum_window(per_subframe, first, kWindowSubframes));
    }

    std::cout << "pdcch_decode_bench.subframe_count=" << per_subframe.size() << '\n';
    std::cout << "pdcch_decode_bench.window_count=" << per_window.size() << '\n';
    std::cout << "pdcch_decode_bench.n_prb=" << n_prb << '\n';
    std::cout << "pdcch_decode_bench.n_tx_ports=" << static_cast<unsigned>(input.n_tx_ports)
              << '\n';
    std::cout << "pdcch_decode_bench.n_rx_ant=" << input.grid.n_rx_ant << '\n';
    std::cout << "pdcch_decode_bench.tx_mode=" << static_cast<unsigned>(input.tx_mode) << '\n';
    std::cout << "pdcch_decode_bench.decoder=native_tail_biting_viterbi\n";
    std::cout << "pdcch_decode_bench.schema=mmse.pdcch.benchmark."
              << (workload_kind == fixture::Workload::kFixed ? "v3" : "v2") << '\n';
    std::cout << "pdcch_decode_bench.measurement_source=native\n";
    std::cout << "pdcch_decode_bench.workload=" << fixture::workload_name(workload_kind) << '\n';
    std::cout << "pdcch_decode_bench.e2e.clock_domain=host_steady\n";
    std::cout << "pdcch_decode_bench.e2e.scope=single_request_wall\n";
    std::cout << "pdcch_decode_bench.stage.clock_domain=host_steady\n";
    std::cout << "pdcch_decode_bench.stage.composable=false\n";
    if (workload_kind == fixture::Workload::kFixed) {
        std::cout << "pdcch_decode_bench.stage.scope=diagnostic_replay_after_e2e\n";
        std::cout << "pdcch_decode_bench.fixed.aggregation_level="
                  << static_cast<unsigned>(fixture::kFixedAggregationLevel) << '\n';
        std::cout << "pdcch_decode_bench.fixed.first_cce=" << fixture::kFixedFirstCce << '\n';
        std::cout << "pdcch_decode_bench.fixed.expected_rnti=" << pdcch::kSiRnti << '\n';
        std::cout << "pdcch_decode_bench.preflight.fixed_crc_dci_chain_validated=1\n";
    }
    print_summaries("subframe_1ms", per_subframe);
    print_summaries("window_10ms", per_window);

    if (n_tx_ports == 4U || workload_kind == fixture::Workload::kFixed) {
        std::cout << "pdcch_decode_bench.geometry.supported=0\n";
        return 0;
    }

    pdcch::PdcchSiRntiGeometrySearchRequest geometry_request{};
    const PdcchMmseInput geometry_input = workload_kind != fixture::Workload::kRandom
                                              ? mixed_workload.cases.front().request.input
                                              : input;
    geometry_request.grid = geometry_input.grid;
    geometry_request.sfn_subframe = geometry_input.sfn_subframe;
    geometry_request.cell_id = geometry_input.cell_id;
    geometry_request.n_tx_ports = geometry_input.n_tx_ports;
    geometry_request.tx_mode = geometry_input.tx_mode;
    geometry_request.n_prb = geometry_input.n_prb;
    geometry_request.control_subframe = geometry_input.control_subframe;
    geometry_request.chain = geometry_input.chain;
    const auto measure_geometry_search = [&](pdcch::PdcchSiRntiGeometrySearchCache& cache,
                                             pdcch::PdcchSiRntiGeometrySearchResult& result) {
        const auto start = std::chrono::steady_clock::now();
        const MmseStatus status = pdcch::run_pdcch_cpu_si_rnti_geometry_search(
            context, geometry_request, {}, cache, result);
        const auto end = std::chrono::steady_clock::now();
        if (status != MmseStatus::kOk) {
            return -1.0;
        }
        return std::chrono::duration<double, std::micro>(end - start).count();
    };

    pdcch::PdcchSiRntiGeometrySearchCache cold_cache{};
    pdcch::PdcchSiRntiGeometrySearchResult cold_result{};
    const double cold_geometry_us = measure_geometry_search(cold_cache, cold_result);

    pdcch::PdcchSiRntiGeometrySearchCache locked_cache{
        .locked = true,
        .cell_id = geometry_request.cell_id,
        .n_tx_ports = geometry_request.n_tx_ports,
        .tx_mode = geometry_request.tx_mode,
        .n_prb = geometry_request.n_prb,
        .subframe_kind = geometry_request.control_subframe.kind,
        .geometry =
            {
                .control_symbol_count = geometry_input.control_symbol_count,
                .phich_resource = pdcch::PhichResource::kOne,
                .phich_duration = pdcch::PhichDuration::kNormal,
                .standard_reg_order = true,
            },
    };
    pdcch::PdcchSiRntiGeometrySearchResult locked_result{};
    const double locked_geometry_us = measure_geometry_search(locked_cache, locked_result);
    locked_cache.consecutive_miss_count = 4U;
    pdcch::PdcchSiRntiGeometrySearchResult reprobe_result{};
    const double reprobe_geometry_us = measure_geometry_search(locked_cache, reprobe_result);
    if (cold_geometry_us < 0.0 || locked_geometry_us < 0.0 || reprobe_geometry_us < 0.0) {
        std::cerr << "geometry search benchmark failed\n";
        return 1;
    }
    std::cout << "pdcch_decode_bench.geometry.cold_us=" << cold_geometry_us << '\n';
    std::cout << "pdcch_decode_bench.geometry.cold_attempt_count="
              << cold_result.geometry_attempt_count << '\n';
    std::cout << "pdcch_decode_bench.geometry.locked_us=" << locked_geometry_us << '\n';
    std::cout << "pdcch_decode_bench.geometry.locked_attempt_count="
              << locked_result.geometry_attempt_count << '\n';
    std::cout << "pdcch_decode_bench.geometry.reprobe_us=" << reprobe_geometry_us << '\n';
    std::cout << "pdcch_decode_bench.geometry.reprobe_attempt_count="
              << reprobe_result.geometry_attempt_count << '\n';
    return 0;
}