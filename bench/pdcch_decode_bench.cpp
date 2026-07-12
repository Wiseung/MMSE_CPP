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

#include "mmse/pdcch_chain_sdk.h"

using namespace mmse;

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
    PdcchMmseOutputView output{
        .x_hat_re = x_hat_re.data(),
        .x_hat_im = x_hat_im.data(),
        .sinr = sinr.data(),
        .re_grid_indices = re_grid_indices.data(),
        .capacity_re_per_layer = kPdcchCapacity,
        .capacity_re_metadata = kPdcchCapacity,
    };
    PdcchMmseResult metadata{};
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

PlanarGridViewF32 make_grid_view(const GridBuffers& buffers) {
    return {
        .re = {buffers.re[0].data(), buffers.re[1].data()},
        .im = {buffers.im[0].data(), buffers.im[1].data()},
        .n_rx_ant = kMmseV1MaxNumRxAntennas,
        .n_symbols = kLteNumSymbolsNormalCp,
        .n_subcarriers = static_cast<std::uint32_t>(buffers.re[0].size() / kLteNumSymbolsNormalCp),
    };
}

PdcchMmseInput make_pdcch_input(const PlanarGridViewF32& grid, std::uint16_t n_prb) {
    pdcch::FrontendPdcchIndication frontend{};
    frontend.cell_id = 1U;
    frontend.n_tx_ports = 1U;
    frontend.tx_mode = 1U;
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
                                     std::vector<pdcch::PdcchCandidateLlr>& candidates) {
    candidates.clear();
    std::vector<pdcch::PdcchCommonSearchCandidate> descriptors{};
    pdcch::build_pdcch_common_search_candidates(control_region, descriptors);
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
                  StageTimes& times, std::uint32_t subframe_index) {
    using Clock = std::chrono::steady_clock;
    input.sfn_subframe = subframe_index;
    input.control_subframe.subframe = static_cast<std::uint8_t>(subframe_index % 10U);

    const auto total_start = Clock::now();
    auto start = Clock::now();
    if (context.run_pdcch(input, buffers.output, buffers.metadata) != MmseStatus::kOk) {
        return false;
    }
    auto end = Clock::now();
    times.ce_mmse_us = std::chrono::duration<double, std::micro>(end - start).count();

    const auto backend =
        pdcch::make_backend_pdcch_equalized_indication(buffers.metadata, buffers.output);
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
    if (!gather_common_search_candidates(backend, control_region, full_llrs, candidates)) {
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
    }
    end = Clock::now();
    times.crc_dci_us = std::chrono::duration<double, std::micro>(end - start).count();
    times.candidate_count = static_cast<std::uint32_t>(candidates.size());
    times.crc_hit_count = hit_count;
    times.crc_miss_count = times.candidate_count - hit_count;
    times.total_us = std::chrono::duration<double, std::micro>(Clock::now() - total_start).count();
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
    print_summary(scope, "candidate_count", values, [](const StageTimes& times) {
        return static_cast<double>(times.candidate_count);
    });
    print_summary(scope, "crc_hit", values,
                  [](const StageTimes& times) { return static_cast<double>(times.crc_hit_count); });
    print_summary(scope, "crc_miss", values, [](const StageTimes& times) {
        return static_cast<double>(times.crc_miss_count);
    });
    print_summary(scope, "total", values, [](const StageTimes& times) { return times.total_us; });
}

} // namespace

bool parse_bandwidth(int argc, char** argv, std::uint16_t& n_prb) {
    if (argc == 1) {
        return true;
    }
    if (argc != 3 || std::string_view(argv[1]) != "--n-prb") {
        return false;
    }
    unsigned int value = 0U;
    const auto [end, error] =
        std::from_chars(argv[2], argv[2] + std::char_traits<char>::length(argv[2]), value);
    if (error != std::errc{} || *end != '\0' || value > std::numeric_limits<std::uint16_t>::max()) {
        return false;
    }
    n_prb = static_cast<std::uint16_t>(value);
    return is_supported_lte_downlink_bandwidth(n_prb);
}

int main(int argc, char** argv) {
    std::uint16_t n_prb = kLteNumPrb20MHz;
    if (!parse_bandwidth(argc, argv, n_prb)) {
        std::cerr << "usage: pdcch_decode_bench [--n-prb 6|15|25|50|75|100]\n";
        return 2;
    }

    GridBuffers grid_buffers = make_random_grid(42U, n_prb);
    const PlanarGridViewF32 grid = make_grid_view(grid_buffers);
    const PdcchMmseInput input = make_pdcch_input(grid, n_prb);

    MmseEqualizerCpuContext context;
    MmseEqualizerCpuConfig cpu_config{};
    cpu_config.worker_count = 1U;
    if (context.init(cpu_config) != MmseStatus::kOk) {
        std::cerr << "failed to initialize CPU context\n";
        return 1;
    }

    PdcchBuffers buffers{};
    StageTimes ignored{};
    for (std::uint32_t subframe = 0U; subframe < kWarmupSubframes; ++subframe) {
        if (!run_subframe(context, input, buffers, ignored, subframe)) {
            std::cerr << "warmup failed\n";
            return 1;
        }
    }

    std::vector<StageTimes> per_subframe{};
    per_subframe.reserve(kMeasuredSubframes);
    for (std::uint32_t subframe = 0U; subframe < kMeasuredSubframes; ++subframe) {
        StageTimes times{};
        if (!run_subframe(context, input, buffers, times, kWarmupSubframes + subframe)) {
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
    std::cout << "pdcch_decode_bench.decoder=native_tail_biting_viterbi\n";
    print_summaries("subframe_1ms", per_subframe);
    print_summaries("window_10ms", per_window);

    pdcch::PdcchSiRntiGeometrySearchRequest geometry_request{};
    geometry_request.grid = grid;
    geometry_request.sfn_subframe = input.sfn_subframe;
    geometry_request.cell_id = input.cell_id;
    geometry_request.n_tx_ports = input.n_tx_ports;
    geometry_request.tx_mode = input.tx_mode;
    geometry_request.n_prb = input.n_prb;
    geometry_request.control_subframe = input.control_subframe;
    geometry_request.chain = input.chain;
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
                .control_symbol_count = input.control_symbol_count,
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