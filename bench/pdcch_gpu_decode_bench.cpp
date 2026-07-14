#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "mmse/pdcch_chain_sdk.h"

using namespace mmse;

namespace {

constexpr std::uint32_t kWarmupIterations = 4U;
constexpr std::uint32_t kMeasuredIterations = 20U;

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
               std::uint32_t stream_count, std::uint32_t batch_size, std::uint8_t n_tx_ports,
               std::uint8_t tx_mode) {
    std::vector<pdcch::PdcchGpuCommonSearchDecodeRequest> requests(batch_size);
    std::vector<pdcch::PdcchGpuCommonSearchDecodeResult> results(batch_size);
    for (std::uint32_t warmup = 0U; warmup < kWarmupIterations; ++warmup) {
        for (std::uint32_t index = 0U; index < batch_size; ++index) {
            requests[index] =
                make_request(grid, warmup * batch_size + index, 1U + index, n_tx_ports, tx_mode);
        }
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
    for (std::uint32_t iteration = 0U; iteration < kMeasuredIterations; ++iteration) {
        for (std::uint32_t index = 0U; index < batch_size; ++index) {
            requests[index] =
                make_request(grid, 1000U + iteration * batch_size + index,
                             static_cast<std::uint16_t>(1U + (index % 32U)), n_tx_ports, tx_mode);
        }
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
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "pdcch_gpu_decode_bench.avg_us=" << average_us << '\n';
    std::cout << "pdcch_gpu_decode_bench.p50_us=" << percentile(latencies_us, 0.50) << '\n';
    std::cout << "pdcch_gpu_decode_bench.p95_us=" << percentile(latencies_us, 0.95) << '\n';
    std::cout << "pdcch_gpu_decode_bench.subframes_per_second=" << subframes_per_second << '\n';
    std::cout << "pdcch_gpu_decode_bench.h2d_gpu_avg_us=" << total_h2d_gpu_us / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.ce_mmse_gpu_avg_us=" << total_ce_mmse_gpu_us / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.llr_gpu_avg_us=" << total_llr_gpu_us / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.rate_recovery_gpu_avg_us="
              << total_rate_recovery_gpu_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.viterbi_gpu_avg_us=" << total_viterbi_us / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.crc_gpu_avg_us=" << total_crc_gpu_us / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.d2h_gpu_avg_us=" << total_d2h_gpu_us / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.host_submit_avg_us=" << total_host_submit_us / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.host_collect_avg_us="
              << total_host_collect_us / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.h2d_bytes_per_subframe=" << total_h2d_bytes / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.d2h_bytes_per_subframe=" << total_d2h_bytes / sample_count
              << '\n';
    std::cout << "pdcch_gpu_decode_bench.candidates_per_subframe="
              << static_cast<double>(total_candidates) / sample_count << '\n';
    std::cout << "pdcch_gpu_decode_bench.crc_hits=" << total_hits << '\n';
    std::cout << "pdcch_gpu_decode_bench.crc_misses=" << total_misses << '\n';
}

} // namespace

int main() {
    const GridBuffers grid_buffers = make_random_grid(42U);
    const PlanarGridViewF32 grid = make_grid_view(grid_buffers, kMmseV1MaxNumRxAntennas);
    for (const std::uint32_t stream_count : {1U, 2U, 4U}) {
        MmseEqualizerGpuContext context;
        MmseEqualizerGpuConfig config{};
        config.backend = MmseGpuBackend::kCuda;
        config.stream_count = stream_count;
        if (context.init(config) != MmseStatus::kOk) {
            std::cerr << "pdcch_gpu_decode_bench CUDA initialization failed\n";
            return 1;
        }
        for (const auto [n_tx_ports, tx_mode] :
             {std::pair<std::uint8_t, std::uint8_t>{1U, 1U}, {2U, 2U}}) {
            MmseEqualizerCpuContext cpu;
            MmseEqualizerCpuConfig cpu_config{};
            cpu_config.worker_count = 1U;
            if (cpu.init(cpu_config) != MmseStatus::kOk) {
                std::cerr << "pdcch_gpu_decode_bench CPU initialization failed\n";
                return 1;
            }
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
            for (const std::uint32_t batch_size : {1U, 2U, 4U, 8U, 16U}) {
                run_batch(context, grid, stream_count, batch_size, n_tx_ports, tx_mode);
            }
        }
    }
    return 0;
}
