#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include "pdcch_benchmark_fixture.h"
#include "internal/mmse_cuda_runtime.h"
#include "mmse/pdcch_chain_sdk.h"

using namespace mmse;
namespace fixture = mmse::benchmark::pdcch_fixture;

namespace {

constexpr std::uint32_t kWarmupIterations = 4U;
constexpr std::uint32_t kMeasuredIterations = 20U;
constexpr std::uint32_t kMemoryReportStreamCount = 2U;
constexpr std::uint32_t kMemoryReportBatchSize = 2U;

struct DeviceMemoryAccounting {
    std::size_t grid_bytes_per_slot = 0U;
    std::size_t metadata_bytes_per_slot = 0U;
    std::size_t channel_estimate_bytes_per_slot = 0U;
    std::size_t equalizer_output_bytes_per_slot = 0U;
    std::size_t control_bytes_per_slot = 0U;
    std::size_t pdcch_llr_bytes_per_slot = 0U;
    std::size_t pdcch_gold_bytes_per_slot = 0U;
    std::size_t pdcch_candidate_bytes_per_slot = 0U;
    std::size_t pdcch_rate_recovery_bytes_per_slot = 0U;
    std::size_t pdcch_viterbi_bytes_per_slot = 0U;
    std::size_t pdcch_result_bytes_per_slot = 0U;
    std::size_t total_bytes_per_slot = 0U;
    std::size_t total_bytes = 0U;
};

DeviceMemoryAccounting build_device_memory_accounting(std::uint32_t stream_count) {
    DeviceMemoryAccounting accounting{};
    accounting.grid_bytes_per_slot = 4U * detail::kCudaMaxGridRe * sizeof(float);
    accounting.metadata_bytes_per_slot = sizeof(detail::CudaGridMeta);
    accounting.channel_estimate_bytes_per_slot =
        detail::kCudaEstimateStubFloatCount * sizeof(float);
    accounting.equalizer_output_bytes_per_slot = 3U * 2U * detail::kCudaMaxDataRe * sizeof(float);
    accounting.control_bytes_per_slot = sizeof(float) +
                                        detail::kCudaScratchHeaderFloatCount * sizeof(float) +
                                        sizeof(std::uint32_t);
    accounting.pdcch_llr_bytes_per_slot = 2U * detail::kCudaMaxDataRe * sizeof(float);
    accounting.pdcch_gold_bytes_per_slot = detail::kCudaPdcchGoldWordCount * sizeof(std::uint32_t);
    accounting.pdcch_candidate_bytes_per_slot =
        detail::kCudaPdcchMaxCandidates * sizeof(detail::CudaPdcchCandidateDescriptor);
    accounting.pdcch_rate_recovery_bytes_per_slot =
        detail::kCudaPdcchMaxCandidates * detail::kCudaPdcchRecoveredLlrCount * sizeof(float);
    accounting.pdcch_viterbi_bytes_per_slot =
        static_cast<std::size_t>(detail::kCudaPdcchMaxCandidates) * 64U *
        (detail::kCudaPdcchCodewordBitCount * sizeof(std::uint64_t) + sizeof(double));
    accounting.pdcch_result_bytes_per_slot =
        2U * detail::kCudaPdcchMaxCandidates * sizeof(detail::CudaPdcchCandidateResult) +
        sizeof(std::uint32_t);
    accounting.total_bytes_per_slot =
        accounting.grid_bytes_per_slot + accounting.metadata_bytes_per_slot +
        accounting.channel_estimate_bytes_per_slot + accounting.equalizer_output_bytes_per_slot +
        accounting.control_bytes_per_slot + accounting.pdcch_llr_bytes_per_slot +
        accounting.pdcch_gold_bytes_per_slot + accounting.pdcch_candidate_bytes_per_slot +
        accounting.pdcch_rate_recovery_bytes_per_slot + accounting.pdcch_viterbi_bytes_per_slot +
        accounting.pdcch_result_bytes_per_slot;
    accounting.total_bytes = accounting.total_bytes_per_slot * stream_count;
    return accounting;
}

std::int64_t memory_delta(std::size_t before_free_bytes, std::size_t after_free_bytes) {
    if (before_free_bytes >= after_free_bytes) {
        return static_cast<std::int64_t>(before_free_bytes - after_free_bytes);
    }
    return -static_cast<std::int64_t>(after_free_bytes - before_free_bytes);
}

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

GridBuffers make_random_grid(std::uint32_t seed) {
    GridBuffers buffers{};
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    for (std::uint32_t rx = 0U; rx < kMmseV1MaxNumRxAntennas; ++rx) {
        buffers.re[rx].resize(kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz);
        buffers.im[rx].resize(kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz);
        for (std::size_t index = 0U; index < buffers.re[rx].size(); ++index) {
            buffers.re[rx][index] = distribution(generator);
            buffers.im[rx][index] = distribution(generator);
        }
    }
    return buffers;
}

PlanarGridViewF32 make_grid_view(const GridBuffers& buffers, std::uint32_t n_rx_ant) {
    return {
        .re = {buffers.re[0].data(), buffers.re[1].data()},
        .im = {buffers.im[0].data(), buffers.im[1].data()},
        .n_rx_ant = n_rx_ant,
        .n_symbols = kLteNumSymbolsNormalCp,
        .n_subcarriers = kLteNumSubcarriers20MHz,
    };
}

pdcch::PdcchGpuCommonSearchDecodeRequest
make_request(const PlanarGridViewF32& grid, std::uint32_t sfn_subframe, std::uint16_t cell_id,
             std::uint8_t n_tx_ports, std::uint8_t tx_mode) {
    pdcch::FrontendPdcchIndication frontend{};
    frontend.sfn_subframe = sfn_subframe;
    frontend.cell_id = cell_id;
    frontend.n_tx_ports = n_tx_ports;
    frontend.tx_mode = tx_mode;
    frontend.control_symbol_count = 3U;
    frontend.n_prb = kLteNumPrb20MHz;
    frontend.prb_bitmap.fill(0xFFFFU);
    frontend.prb_bitmap.back() = 0x000FU;
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
    request.input.chain.request_id = sfn_subframe;
    return request;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const std::size_t index =
        static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1U));
    return values[index];
}

void run_batch(MmseEqualizerGpuContext& context, const PlanarGridViewF32& grid,
               const fixture::MixedWorkload* mixed_workload, fixture::Workload workload,
               std::uint32_t stream_count, std::uint32_t batch_size, std::uint8_t n_tx_ports,
               std::uint8_t tx_mode) {
    std::vector<pdcch::PdcchGpuCommonSearchDecodeRequest> requests(batch_size);
    std::vector<pdcch::PdcchGpuCommonSearchDecodeResult> results(batch_size);
    const auto populate_requests = [&](std::uint32_t phase) {
        for (std::uint32_t index = 0U; index < batch_size; ++index) {
            if (workload == fixture::Workload::kMixed) {
                const fixture::Case& benchmark_case =
                    mixed_workload->cases[(static_cast<std::size_t>(phase) * batch_size + index) %
                                          mixed_workload->cases.size()];
                requests[index] = benchmark_case.request;
                requests[index].input.chain.request_id =
                    static_cast<std::uint64_t>(phase) * batch_size + index;
            } else {
                requests[index] = make_request(grid, phase * batch_size + index,
                                               static_cast<std::uint16_t>(1U + (index % 32U)),
                                               n_tx_ports, tx_mode);
            }
        }
    };
    for (std::uint32_t warmup = 0U; warmup < kWarmupIterations; ++warmup) {
        populate_requests(warmup);
        if (pdcch::run_pdcch_gpu_common_search_decode_batch(context, requests, results) !=
            MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench warmup failed\n";
            std::exit(1);
        }
    }

    std::vector<double> latencies_us{};
    double total_h2d_bytes = 0.0;
    double total_d2h_bytes = 0.0;
    double total_h2d_gpu_us = 0.0;
    double total_ce_mmse_gpu_us = 0.0;
    double total_llr_gpu_us = 0.0;
    double total_rate_recovery_gpu_us = 0.0;
    double total_viterbi_us = 0.0;
    double total_crc_gpu_us = 0.0;
    double total_d2h_gpu_us = 0.0;
    double total_host_submit_us = 0.0;
    double total_host_collect_us = 0.0;
    std::uint32_t total_candidates = 0U;
    std::uint32_t total_hits = 0U;
    std::uint32_t total_misses = 0U;
    fixture::HitDistribution hit_distribution{};
    for (std::uint32_t iteration = 0U; iteration < kMeasuredIterations; ++iteration) {
        populate_requests(1000U + iteration);
        const auto start = std::chrono::steady_clock::now();
        const MmseStatus status =
            pdcch::run_pdcch_gpu_common_search_decode_batch(context, requests, results);
        const auto end = std::chrono::steady_clock::now();
        if (status != MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench measurement failed: " << to_string(status) << '\n';
            std::exit(1);
        }
        latencies_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        for (const auto& result : results) {
            total_h2d_bytes += static_cast<double>(result.profile.h2d_bytes);
            total_d2h_bytes += static_cast<double>(result.profile.d2h_bytes);
            total_h2d_gpu_us += result.profile.h2d_us;
            total_ce_mmse_gpu_us += result.profile.ce_mmse_gpu_us;
            total_llr_gpu_us += result.profile.llr_gpu_us;
            total_rate_recovery_gpu_us += result.profile.rate_recovery_gpu_us;
            total_viterbi_us += result.profile.viterbi_gpu_us;
            total_crc_gpu_us += result.profile.crc_gpu_us;
            total_d2h_gpu_us += result.profile.d2h_us;
            total_host_submit_us += result.profile.host_submit_us;
            total_host_collect_us += result.profile.host_collect_us;
            total_candidates += result.candidate_count;
            total_hits += result.profile.crc_hit_count;
            total_misses += result.profile.crc_miss_count;
            hit_distribution.add(result.candidate_count, result.hits);
        }
    }

    const double average_us =
        std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
    const double subframes_per_second = static_cast<double>(batch_size) * 1.0e6 / average_us;
    const double sample_count = static_cast<double>(batch_size) * kMeasuredIterations;
    std::cout << "pdcch_gpu_decode_bench.n_tx_ports=" << static_cast<unsigned>(n_tx_ports) << '\n';
    std::cout << "pdcch_gpu_decode_bench.tx_mode=" << static_cast<unsigned>(tx_mode) << '\n';
    std::cout << "pdcch_gpu_decode_bench.n_rx_ant=" << grid.n_rx_ant << '\n';
    std::cout << "pdcch_gpu_decode_bench.stream_count=" << stream_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.batch=" << batch_size << '\n';
    std::cout << "pdcch_gpu_decode_bench.schema=mmse.pdcch.benchmark.v2\n";
    std::cout << "pdcch_gpu_decode_bench.measurement_source=native\n";
    std::cout << "pdcch_gpu_decode_bench.workload=" << fixture::workload_name(workload) << '\n';
    std::cout << "pdcch_gpu_decode_bench.e2e.clock_domain=host_steady\n";
    std::cout << "pdcch_gpu_decode_bench.e2e.scope=batch_wall\n";
    std::cout << "pdcch_gpu_decode_bench.device_interval.clock_domain=cuda_event\n";
    std::cout << "pdcch_gpu_decode_bench.device_interval.composable=false\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "pdcch_gpu_decode_bench.batch_wall_avg_us=" << average_us << '\n';
    std::cout << "pdcch_gpu_decode_bench.batch_wall_p50_us=" << percentile(latencies_us, 0.50)
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.batch_wall_p95_us=" << percentile(latencies_us, 0.95)
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.throughput_equivalent_us=" << average_us / batch_size
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.subframes_per_second=" << subframes_per_second << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.h2d_per_request_mean_us="
              << total_h2d_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.ce_mmse_per_request_mean_us="
              << total_ce_mmse_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.llr_per_request_mean_us="
              << total_llr_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.rate_recovery_per_request_mean_us="
              << total_rate_recovery_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.viterbi_per_request_mean_us="
              << total_viterbi_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.crc_per_request_mean_us="
              << total_crc_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.d2h_per_request_mean_us="
              << total_d2h_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.host_steady.submit_per_request_mean_us="
              << total_host_submit_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.host_steady.collect_per_request_mean_us="
              << total_host_collect_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.h2d_bytes_per_subframe=" << total_h2d_bytes / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.d2h_bytes_per_subframe=" << total_d2h_bytes / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.candidates_per_subframe="
              << static_cast<double>(total_candidates) / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.crc_hits=" << total_hits << '\n';
    std::cout << "pdcch_gpu_decode_bench.crc_misses=" << total_misses << '\n';
    std::cout << "pdcch_gpu_decode_bench.zero_hit_subframes=" << hit_distribution.zero_hit_subframes
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.one_hit_subframes=" << hit_distribution.one_hit_subframes
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.multi_hit_subframes="
              << hit_distribution.multi_hit_subframes << '\n';
    std::cout << "pdcch_gpu_decode_bench.l4_hits=" << hit_distribution.l4_hits << '\n';
    std::cout << "pdcch_gpu_decode_bench.l8_hits=" << hit_distribution.l8_hits << '\n';
}

void run_fixed_batch(MmseEqualizerGpuContext& context, const fixture::MixedWorkload& mixed_workload,
                     std::uint32_t stream_count, std::uint32_t batch_size) {
    std::vector<pdcch::PdcchGpuFixedCandidateDecodeRequest> requests(batch_size);
    std::vector<pdcch::PdcchGpuFixedCandidateDecodeResult> results(batch_size);
    const auto populate_requests = [&](std::uint32_t phase) {
        for (std::uint32_t index = 0U; index < batch_size; ++index) {
            const fixture::Case& benchmark_case =
                mixed_workload.cases[(static_cast<std::size_t>(phase) * batch_size + index) %
                                     mixed_workload.cases.size()];
            requests[index] = fixture::make_fixed_request(benchmark_case);
            requests[index].input.chain.request_id =
                static_cast<std::uint64_t>(phase) * batch_size + index;
        }
    };
    for (std::uint32_t warmup = 0U; warmup < kWarmupIterations; ++warmup) {
        populate_requests(warmup);
        if (pdcch::run_pdcch_gpu_fixed_candidate_decode_batch(context, requests, results) !=
            MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench fixed warmup failed\n";
            std::exit(1);
        }
    }

    std::vector<double> latencies_us{};
    double total_h2d_bytes = 0.0;
    double total_d2h_bytes = 0.0;
    double total_h2d_gpu_us = 0.0;
    double total_ce_mmse_gpu_us = 0.0;
    double total_llr_gpu_us = 0.0;
    double total_rate_recovery_gpu_us = 0.0;
    double total_viterbi_us = 0.0;
    double total_crc_gpu_us = 0.0;
    double total_d2h_gpu_us = 0.0;
    double total_host_submit_us = 0.0;
    double total_host_collect_us = 0.0;
    std::uint32_t total_hits = 0U;
    std::uint32_t total_misses = 0U;
    for (std::uint32_t iteration = 0U; iteration < kMeasuredIterations; ++iteration) {
        const std::uint32_t phase = 1000U + iteration;
        populate_requests(phase);
        const auto start = std::chrono::steady_clock::now();
        const MmseStatus status =
            pdcch::run_pdcch_gpu_fixed_candidate_decode_batch(context, requests, results);
        const auto end = std::chrono::steady_clock::now();
        if (status != MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench fixed measurement failed: " << to_string(status)
                      << '\n';
            std::exit(1);
        }
        latencies_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        for (std::uint32_t index = 0U; index < batch_size; ++index) {
            const auto& result = results[index];
            const fixture::Case& benchmark_case =
                mixed_workload.cases[(static_cast<std::size_t>(phase) * batch_size + index) %
                                     mixed_workload.cases.size()];
            if (!fixture::validate_fixed_result(result.decoded, benchmark_case) ||
                result.profile.d2h_bytes != 24U) {
                std::cerr << "pdcch_gpu_decode_bench fixed result or D2H validation failed\n";
                std::exit(1);
            }
            total_h2d_bytes += static_cast<double>(result.profile.h2d_bytes);
            total_d2h_bytes += static_cast<double>(result.profile.d2h_bytes);
            total_h2d_gpu_us += result.profile.h2d_us;
            total_ce_mmse_gpu_us += result.profile.ce_mmse_gpu_us;
            total_llr_gpu_us += result.profile.llr_gpu_us;
            total_rate_recovery_gpu_us += result.profile.rate_recovery_gpu_us;
            total_viterbi_us += result.profile.viterbi_gpu_us;
            total_crc_gpu_us += result.profile.crc_gpu_us;
            total_d2h_gpu_us += result.profile.d2h_us;
            total_host_submit_us += result.profile.host_submit_us;
            total_host_collect_us += result.profile.host_collect_us;
            total_hits += result.profile.crc_hit_count;
            total_misses += result.profile.crc_miss_count;
        }
    }

    const double average_us =
        std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
    const double sample_count = static_cast<double>(batch_size) * kMeasuredIterations;
    const double subframes_per_second = static_cast<double>(batch_size) * 1.0e6 / average_us;
    const bool single_request_latency = stream_count == 1U && batch_size == 1U;
    std::cout << "pdcch_gpu_decode_bench.n_tx_ports=1\n";
    std::cout << "pdcch_gpu_decode_bench.tx_mode=1\n";
    std::cout << "pdcch_gpu_decode_bench.n_rx_ant=2\n";
    std::cout << "pdcch_gpu_decode_bench.stream_count=" << stream_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.batch=" << batch_size << '\n';
    std::cout << "pdcch_gpu_decode_bench.schema=mmse.pdcch.benchmark.v3\n";
    std::cout << "pdcch_gpu_decode_bench.measurement_source=native\n";
    std::cout << "pdcch_gpu_decode_bench.workload=fixed\n";
    std::cout << "pdcch_gpu_decode_bench.fixed.aggregation_level="
              << static_cast<unsigned>(fixture::kFixedAggregationLevel) << '\n';
    std::cout << "pdcch_gpu_decode_bench.fixed.first_cce=" << fixture::kFixedFirstCce << '\n';
    std::cout << "pdcch_gpu_decode_bench.fixed.expected_rnti=" << pdcch::kSiRnti << '\n';
    std::cout << "pdcch_gpu_decode_bench.e2e.clock_domain=host_steady\n";
    std::cout << "pdcch_gpu_decode_bench.e2e.scope="
              << (single_request_latency ? "single_request_wall" : "batch_wall") << '\n';
    std::cout << "pdcch_gpu_decode_bench.single_request_latency_valid="
              << (single_request_latency ? 1 : 0) << '\n';
    std::cout << "pdcch_gpu_decode_bench.device_interval.clock_domain=cuda_event\n";
    std::cout << "pdcch_gpu_decode_bench.device_interval.composable=false\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "pdcch_gpu_decode_bench.batch_wall_avg_us=" << average_us << '\n';
    std::cout << "pdcch_gpu_decode_bench.batch_wall_p50_us=" << percentile(latencies_us, 0.50)
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.batch_wall_p95_us=" << percentile(latencies_us, 0.95)
              << '\n';
    if (single_request_latency) {
        std::cout << "pdcch_gpu_decode_bench.single_request_host_wall_avg_us=" << average_us
                  << '\n';
        std::cout << "pdcch_gpu_decode_bench.single_request_host_wall_p50_us="
                  << percentile(latencies_us, 0.50) << '\n';
        std::cout << "pdcch_gpu_decode_bench.single_request_host_wall_p95_us="
                  << percentile(latencies_us, 0.95) << '\n';
    }
    std::cout << "pdcch_gpu_decode_bench.throughput_equivalent_us=" << average_us / batch_size
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.subframes_per_second=" << subframes_per_second << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.h2d_per_request_mean_us="
              << total_h2d_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.ce_mmse_per_request_mean_us="
              << total_ce_mmse_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.llr_per_request_mean_us="
              << total_llr_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.rate_recovery_per_request_mean_us="
              << total_rate_recovery_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.viterbi_per_request_mean_us="
              << total_viterbi_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.crc_per_request_mean_us="
              << total_crc_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.device.cuda_event.d2h_per_request_mean_us="
              << total_d2h_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.host_steady.submit_per_request_mean_us="
              << total_host_submit_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.host_steady.collect_per_request_mean_us="
              << total_host_collect_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.h2d_bytes_per_subframe=" << total_h2d_bytes / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.d2h_bytes_per_subframe=" << total_d2h_bytes / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.candidates_per_subframe=1.00\n";
    std::cout << "pdcch_gpu_decode_bench.crc_hits=" << total_hits << '\n';
    std::cout << "pdcch_gpu_decode_bench.crc_misses=" << total_misses << '\n';
}

bool query_memory(std::size_t& free_bytes, std::size_t& total_bytes) {
    if (detail::cuda_query_current_memory_info(free_bytes, total_bytes) != MmseStatus::kOk) {
        std::cerr << "pdcch_gpu_decode_bench CUDA memory query failed\n";
        return false;
    }
    return true;
}

bool run_memory_workload(MmseEqualizerGpuContext& context,
                         const fixture::MixedWorkload& mixed_workload,
                         std::size_t& free_after_warmup_bytes,
                         std::size_t& min_observed_free_bytes) {
    std::vector<pdcch::PdcchGpuCommonSearchDecodeRequest> requests(kMemoryReportBatchSize);
    std::vector<pdcch::PdcchGpuCommonSearchDecodeResult> results(kMemoryReportBatchSize);
    const auto run_phase = [&](std::uint32_t phase) {
        for (std::uint32_t index = 0U; index < kMemoryReportBatchSize; ++index) {
            const fixture::Case& benchmark_case =
                mixed_workload
                    .cases[(static_cast<std::size_t>(phase) * kMemoryReportBatchSize + index) %
                           mixed_workload.cases.size()];
            requests[index] = benchmark_case.request;
            requests[index].input.chain.request_id =
                static_cast<std::uint64_t>(phase) * kMemoryReportBatchSize + index;
        }
        return pdcch::run_pdcch_gpu_common_search_decode_batch(context, requests, results);
    };

    for (std::uint32_t warmup = 0U; warmup < kWarmupIterations; ++warmup) {
        if (run_phase(warmup) != MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench memory warmup failed\n";
            return false;
        }
    }
    std::size_t total_bytes = 0U;
    if (!query_memory(free_after_warmup_bytes, total_bytes)) {
        return false;
    }
    min_observed_free_bytes = free_after_warmup_bytes;
    for (std::uint32_t iteration = 0U; iteration < kMeasuredIterations; ++iteration) {
        if (run_phase(1000U + iteration) != MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench memory measurement failed\n";
            return false;
        }
        std::size_t current_free_bytes = 0U;
        if (!query_memory(current_free_bytes, total_bytes)) {
            return false;
        }
        min_observed_free_bytes = std::min(min_observed_free_bytes, current_free_bytes);
    }
    return true;
}

bool run_memory_report_case(std::uint8_t n_tx_ports, std::uint8_t tx_mode) {
    std::size_t free_before_init_bytes = 0U;
    std::size_t total_bytes = 0U;
    if (!query_memory(free_before_init_bytes, total_bytes)) {
        return false;
    }

    std::size_t free_after_init_bytes = 0U;
    std::size_t free_after_warmup_bytes = 0U;
    std::size_t min_observed_free_bytes = std::numeric_limits<std::size_t>::max();
    {
        MmseEqualizerGpuContext context;
        MmseEqualizerGpuConfig config{};
        config.backend = MmseGpuBackend::kCuda;
        config.stream_count = kMemoryReportStreamCount;
        if (context.init(config) != MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench memory context initialization failed\n";
            return false;
        }
        std::size_t total_after_init_bytes = 0U;
        if (!query_memory(free_after_init_bytes, total_after_init_bytes) ||
            total_after_init_bytes != total_bytes) {
            return false;
        }
        fixture::MixedWorkload mixed_workload{};
        if (!fixture::make_mixed_workload(kLteNumPrb20MHz, n_tx_ports, mixed_workload)) {
            std::cerr << "pdcch_gpu_decode_bench memory workload construction failed\n";
            return false;
        }
        if (!run_memory_workload(context, mixed_workload, free_after_warmup_bytes,
                                 min_observed_free_bytes)) {
            return false;
        }
    }

    std::size_t free_after_release_bytes = 0U;
    std::size_t total_after_release_bytes = 0U;
    if (!query_memory(free_after_release_bytes, total_after_release_bytes) ||
        total_after_release_bytes != total_bytes) {
        return false;
    }

    const DeviceMemoryAccounting accounting =
        build_device_memory_accounting(kMemoryReportStreamCount);
    std::cout << "pdcch_gpu_memory.schema=mmse.pdcch.memory.v1\n";
    std::cout << "pdcch_gpu_memory.workload=mixed\n";
    std::cout << "pdcch_gpu_memory.n_tx_ports=" << static_cast<unsigned>(n_tx_ports) << '\n';
    std::cout << "pdcch_gpu_memory.tx_mode=" << static_cast<unsigned>(tx_mode) << '\n';
    std::cout << "pdcch_gpu_memory.stream_count=" << kMemoryReportStreamCount << '\n';
    std::cout << "pdcch_gpu_memory.batch=" << kMemoryReportBatchSize << '\n';
    std::cout << "pdcch_gpu_memory.observed.baseline_scope=post_cuda_runtime_prime\n";
    std::cout << "pdcch_gpu_memory.accounting.grid_bytes_per_slot="
              << accounting.grid_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.metadata_bytes_per_slot="
              << accounting.metadata_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.channel_estimate_bytes_per_slot="
              << accounting.channel_estimate_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.equalizer_output_bytes_per_slot="
              << accounting.equalizer_output_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.control_bytes_per_slot="
              << accounting.control_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.pdcch_llr_bytes_per_slot="
              << accounting.pdcch_llr_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.pdcch_gold_bytes_per_slot="
              << accounting.pdcch_gold_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.pdcch_candidate_bytes_per_slot="
              << accounting.pdcch_candidate_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.pdcch_rate_recovery_bytes_per_slot="
              << accounting.pdcch_rate_recovery_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.pdcch_viterbi_bytes_per_slot="
              << accounting.pdcch_viterbi_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.pdcch_result_bytes_per_slot="
              << accounting.pdcch_result_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.total_bytes_per_slot="
              << accounting.total_bytes_per_slot << '\n';
    std::cout << "pdcch_gpu_memory.accounting.total_bytes=" << accounting.total_bytes << '\n';
    std::cout << "pdcch_gpu_memory.observed.total_bytes=" << total_bytes << '\n';
    std::cout << "pdcch_gpu_memory.observed.free_before_init_bytes=" << free_before_init_bytes
              << '\n';
    std::cout << "pdcch_gpu_memory.observed.free_after_init_bytes=" << free_after_init_bytes
              << '\n';
    std::cout << "pdcch_gpu_memory.observed.free_after_warmup_bytes=" << free_after_warmup_bytes
              << '\n';
    std::cout << "pdcch_gpu_memory.observed.min_free_bytes=" << min_observed_free_bytes << '\n';
    std::cout << "pdcch_gpu_memory.observed.free_after_release_bytes=" << free_after_release_bytes
              << '\n';
    std::cout << "pdcch_gpu_memory.observed.init_delta_bytes="
              << memory_delta(free_before_init_bytes, free_after_init_bytes) << '\n';
    std::cout << "pdcch_gpu_memory.observed.run_peak_delta_after_init_bytes="
              << memory_delta(free_after_init_bytes, min_observed_free_bytes) << '\n';
    std::cout << "pdcch_gpu_memory.observed.peak_delta_from_baseline_bytes="
              << memory_delta(free_before_init_bytes, min_observed_free_bytes) << '\n';
    std::cout << "pdcch_gpu_memory.observed.unreleased_delta_bytes="
              << memory_delta(free_before_init_bytes, free_after_release_bytes) << '\n';
    return true;
}

int run_memory_report() {
    bool runtime_available = false;
    if (detail::cuda_select_device(0U, runtime_available) != MmseStatus::kOk ||
        !runtime_available) {
        std::cerr << "pdcch_gpu_decode_bench failed to select CUDA device 0\n";
        return 1;
    }
    {
        MmseEqualizerGpuContext context;
        MmseEqualizerGpuConfig config{};
        config.backend = MmseGpuBackend::kCuda;
        config.stream_count = kMemoryReportStreamCount;
        fixture::MixedWorkload mixed_workload{};
        std::size_t free_after_warmup_bytes = 0U;
        std::size_t min_observed_free_bytes = 0U;
        if (context.init(config) != MmseStatus::kOk ||
            !fixture::make_mixed_workload(kLteNumPrb20MHz, 1U, mixed_workload) ||
            !run_memory_workload(context, mixed_workload, free_after_warmup_bytes,
                                 min_observed_free_bytes)) {
            std::cerr << "pdcch_gpu_decode_bench CUDA runtime prime failed\n";
            return 1;
        }
    }
    if (!run_memory_report_case(1U, 1U) || !run_memory_report_case(2U, 2U)) {
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--memory-report") {
        return run_memory_report();
    }
    fixture::Workload workload = fixture::Workload::kMixed;
    if (argc == 3 && std::string_view(argv[1]) == "--workload" &&
        fixture::parse_workload(argv[2], workload)) {
    } else if (argc != 1) {
        std::cerr << "usage: pdcch_gpu_decode_bench [--workload fixed|mixed|random] | "
                     "--memory-report\n";
        return 2;
    }
    const GridBuffers grid_buffers = make_random_grid(42U);
    fixture::MixedWorkload fixed_workload{};
    if (workload == fixture::Workload::kFixed &&
        !fixture::make_mixed_workload(kLteNumPrb20MHz, 1U, fixed_workload)) {
        std::cerr << "pdcch_gpu_decode_bench fixed workload construction failed\n";
        return 1;
    }
    for (const std::uint32_t stream_count : {1U, 2U, 4U}) {
        MmseEqualizerGpuContext context;
        MmseEqualizerGpuConfig config{};
        config.backend = MmseGpuBackend::kCuda;
        config.stream_count = stream_count;
        if (context.init(config) != MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench CUDA initialization failed\n";
            return 1;
        }
        if (workload == fixture::Workload::kFixed) {
            MmseEqualizerCpuContext cpu;
            MmseEqualizerCpuConfig cpu_config{};
            cpu_config.worker_count = 1U;
            if (cpu.init(cpu_config) != MmseStatus::kOk) {
                std::cerr << "pdcch_gpu_decode_bench fixed CPU initialization failed\n";
                return 1;
            }
            for (const fixture::Case& benchmark_case : fixed_workload.cases) {
                const auto request = fixture::make_fixed_request(benchmark_case);
                pdcch::PdcchDciFormat1ADecodeResult cpu_result{};
                pdcch::PdcchGpuFixedCandidateDecodeResult gpu_result{};
                if (pdcch::run_pdcch_cpu_fixed_candidate_decode(cpu, request.input, request.config,
                                                                cpu_result) != MmseStatus::kOk ||
                    pdcch::run_pdcch_gpu_fixed_candidate_decode(context, request, gpu_result) !=
                        MmseStatus::kOk ||
                    !fixture::validate_fixed_result(cpu_result, benchmark_case) ||
                    !fixture::validate_fixed_result(gpu_result.decoded, benchmark_case) ||
                    !fixture::fixed_results_equal(gpu_result.decoded, cpu_result) ||
                    gpu_result.profile.d2h_bytes != 24U) {
                    std::cerr << "pdcch_gpu_decode_bench fixed CPU/GPU preflight failed\n";
                    return 1;
                }
            }
            std::cout << "pdcch_gpu_decode_bench.preflight.stream_count=" << stream_count << '\n';
            std::cout << "pdcch_gpu_decode_bench.preflight.cpu_gpu_crc_dci_chain_equivalent=1\n";
            std::cout << "pdcch_gpu_decode_bench.preflight.fixed_d2h_bytes=24\n";
            for (const std::uint32_t batch_size : {1U, 2U, 4U, 8U, 16U}) {
                run_fixed_batch(context, fixed_workload, stream_count, batch_size);
            }
            continue;
        }
        for (const auto [n_tx_ports, tx_mode] :
             {std::pair<std::uint8_t, std::uint8_t>{1U, 1U}, {2U, 2U}, {4U, 2U}}) {
            const PlanarGridViewF32 grid =
                make_grid_view(grid_buffers, n_tx_ports == 4U ? 1U : kMmseV1MaxNumRxAntennas);
            fixture::MixedWorkload mixed_workload{};
            if (workload == fixture::Workload::kMixed &&
                !fixture::make_mixed_workload(kLteNumPrb20MHz, n_tx_ports, mixed_workload)) {
                std::cerr << "pdcch_gpu_decode_bench mixed workload construction failed\n";
                return 1;
            }
            MmseEqualizerCpuContext cpu;
            MmseEqualizerCpuConfig cpu_config{};
            cpu_config.worker_count = 1U;
            if (cpu.init(cpu_config) != MmseStatus::kOk) {
                std::cerr << "pdcch_gpu_decode_bench CPU initialization failed\n";
                return 1;
            }
            if (workload == fixture::Workload::kMixed) {
                for (const fixture::Case& benchmark_case : mixed_workload.cases) {
                    pdcch::PdcchCommonSearchDecodeResult cpu_result{};
                    pdcch::PdcchGpuCommonSearchDecodeResult gpu_result{};
                    if (pdcch::run_pdcch_cpu_common_search_decode(cpu, benchmark_case.request.input,
                                                                  benchmark_case.request.config,
                                                                  cpu_result) != MmseStatus::kOk ||
                        pdcch::run_pdcch_gpu_common_search_decode(context, benchmark_case.request,
                                                                  gpu_result) != MmseStatus::kOk ||
                        !fixture::validate_result(cpu_result, benchmark_case) ||
                        !fixture::validate_result(gpu_result, benchmark_case) ||
                        cpu_result.candidate_count != gpu_result.candidate_count ||
                        cpu_result.hits.size() != gpu_result.hits.size()) {
                        std::cerr << "pdcch_gpu_decode_bench mixed CPU/GPU smoke check failed\n";
                        return 1;
                    }
                }
            } else {
                const auto smoke_request = make_request(grid, 0U, 1U, n_tx_ports, tx_mode);
                pdcch::PdcchCommonSearchDecodeResult cpu_result{};
                pdcch::PdcchGpuCommonSearchDecodeResult gpu_result{};
                if (pdcch::run_pdcch_cpu_common_search_decode(cpu, smoke_request.input,
                                                              smoke_request.config,
                                                              cpu_result) != MmseStatus::kOk ||
                    pdcch::run_pdcch_gpu_common_search_decode(context, smoke_request, gpu_result) !=
                        MmseStatus::kOk ||
                    cpu_result.candidate_count != gpu_result.candidate_count ||
                    cpu_result.hits.size() != gpu_result.hits.size()) {
                    std::cerr << "pdcch_gpu_decode_bench CPU/GPU smoke check failed\n";
                    return 1;
                }
            }
            for (const std::uint32_t batch_size : {1U, 2U, 4U, 8U, 16U}) {
                run_batch(context, grid,
                          workload == fixture::Workload::kMixed ? &mixed_workload : nullptr,
                          workload, stream_count, batch_size, n_tx_ports, tx_mode);
            }
        }
    }
    return 0;
}
