#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

#include "internal/mmse_cuda_runtime.h"
#include "internal/mmse_internal.h"
#include "mmse/mmse_equalizer.h"

using namespace mmse;

namespace {

constexpr std::uint32_t kSubframesPer10msFrame = 10U;
constexpr std::uint32_t kSubframesPer100msWindow = 100U;
constexpr std::uint32_t kDefaultWarmupSubframes = 20U;
constexpr std::uint32_t kDefault10msIterations = 50U;
constexpr std::uint32_t kDefault100msIterations = 20U;
constexpr std::uint32_t kDatasetSubframes = kSubframesPer100msWindow;
constexpr std::uint32_t kOutputCapacityPerLayer = 20000U;
constexpr std::int32_t kEqualizeBlockSize = 256;
constexpr std::uint32_t kPinnedRingSlotCount = 3U;
constexpr std::uint32_t kOutputPlaneFloatCount = detail::kCudaMaxDataRe * 2U;

struct Options {
    std::uint32_t warmup_subframes = kDefaultWarmupSubframes;
    std::uint32_t iterations_10ms = kDefault10msIterations;
    std::uint32_t iterations_100ms = kDefault100msIterations;
};

struct ProcessMemorySample {
    std::uint64_t working_set_bytes = 0;
    std::uint64_t private_bytes = 0;
};

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

struct SubframeData {
    GridBuffers grid;
    ExtractDescriptor desc{};
};

struct RunBuffers {
    std::vector<float> xhat_re;
    std::vector<float> xhat_im;
    std::vector<float> sinr;
};

struct TimingSummary {
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double max_ms = 0.0;
    double avg_ms = 0.0;
};

struct TimedRun {
    double elapsed_ms = 0.0;
    double checksum = 0.0;
};

struct HostPhaseSummary {
    double quantize_us = 0.0;
    double layout_build_us = 0.0;
    double grid_meta_pack_us = 0.0;
    double grid_h2d_us = 0.0;
    double estimate_launch_us = 0.0;
    double sigma2_d2h_us = 0.0;
    double sigma2_mid_sync_us = 0.0;
    double sigma2_host_update_us = 0.0;
    double sigma2_h2d_us = 0.0;
    double equalize_launch_us = 0.0;
    double outputs_d2h_us = 0.0;
    double scratch_d2h_us = 0.0;
    double completion_d2h_us = 0.0;
    double final_sync_us = 0.0;
    double output_stage_us = 0.0;
    double total_host_us = 0.0;
    double estimate_gpu_us = 0.0;
    double estimate_residual_gpu_us = 0.0;
    double estimate_channel_gpu_us = 0.0;
    double equalize_gpu_us = 0.0;
    double stream_gpu_us = 0.0;
};

struct BufferAccounting {
    std::size_t logical_input_bytes_per_subframe = 0;
    std::size_t logical_output_bytes_per_subframe = 0;
    std::size_t logical_input_bytes_per_10ms = 0;
    std::size_t logical_output_bytes_per_10ms = 0;
    std::size_t logical_input_bytes_per_100ms = 0;
    std::size_t logical_output_bytes_per_100ms = 0;
    std::size_t dataset_input_bytes = 0;
    std::size_t host_pageable_bytes_per_slot = 0;
    std::size_t host_pinned_bytes_per_slot = 0;
    std::size_t host_pageable_bytes_total = 0;
    std::size_t host_pinned_bytes_total = 0;
    std::size_t device_bytes_per_slot = 0;
    std::size_t device_bytes_total = 0;
    std::size_t device_scratch_bytes = 0;
};

ProcessMemorySample capture_process_memory() {
    ProcessMemorySample sample{};
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters)) != 0) {
        sample.working_set_bytes = static_cast<std::uint64_t>(counters.WorkingSetSize);
        sample.private_bytes = static_cast<std::uint64_t>(counters.PrivateUsage);
    }
#endif
    return sample;
}

PlanarGridViewF32 make_grid_view(const GridBuffers& buffers) {
    PlanarGridViewF32 view{};
    view.re = {buffers.re[0].data(), buffers.re[1].data()};
    view.im = {buffers.im[0].data(), buffers.im[1].data()};
    view.n_rx_ant = kMmseV1MaxNumRxAntennas;
    view.n_symbols = kLteNumSymbolsNormalCp;
    view.n_subcarriers = kLteNumSubcarriers20MHz;
    return view;
}

EqualizerOutputView make_output_view(RunBuffers& buffers) {
    EqualizerOutputView out{};
    out.x_hat_re = buffers.xhat_re.data();
    out.x_hat_im = buffers.xhat_im.data();
    out.sinr = buffers.sinr.data();
    out.capacity_re_per_layer = kOutputCapacityPerLayer;
    return out;
}

detail::Complex32 random_channel_coeff(std::mt19937& rng, float mean_re, float mean_im,
                                       float sigma) {
    std::normal_distribution<float> noise(0.0F, sigma);
    return {mean_re + noise(rng), mean_im + noise(rng)};
}

float random_qam_axis(std::mt19937& rng, std::uint8_t mod_order) {
    switch (mod_order) {
    case 2: {
        static constexpr std::array<float, 2> kLevels = {-0.70710678F, 0.70710678F};
        std::uniform_int_distribution<int> pick(0, 1);
        return kLevels[static_cast<std::size_t>(pick(rng))];
    }
    case 4: {
        static constexpr std::array<float, 4> kLevels = {-0.94868330F, -0.31622777F, 0.31622777F,
                                                         0.94868330F};
        std::uniform_int_distribution<int> pick(0, 3);
        return kLevels[static_cast<std::size_t>(pick(rng))];
    }
    case 6: {
        static constexpr std::array<float, 8> kLevels = {-1.08012345F, -0.77151674F, -0.46291006F,
                                                         -0.15430335F, 0.15430335F,  0.46291006F,
                                                         0.77151674F,  1.08012345F};
        std::uniform_int_distribution<int> pick(0, 7);
        return kLevels[static_cast<std::size_t>(pick(rng))];
    }
    case 8: {
        static constexpr std::array<float, 16> kLevels = {
            -1.15044749F, -0.99705446F, -0.84366149F, -0.69026852F, -0.53687549F, -0.38348249F,
            -0.23008950F, -0.07669650F, 0.07669650F,  0.23008950F,  0.38348249F,  0.53687549F,
            0.69026852F,  0.84366149F,  0.99705446F,  1.15044749F};
        std::uniform_int_distribution<int> pick(0, 15);
        return kLevels[static_cast<std::size_t>(pick(rng))];
    }
    default:
        return 0.70710678F;
    }
}

detail::Complex32 random_qam_symbol(std::mt19937& rng, std::uint8_t mod_order) {
    return {random_qam_axis(rng, mod_order), random_qam_axis(rng, mod_order)};
}

ExtractDescriptor make_fullband_desc(std::uint32_t subframe_index) {
    ExtractDescriptor desc{};
    desc.sfn_subframe = subframe_index;
    desc.cell_id = 11U;
    desc.n_tx_ports = kMmseV1MaxNumCrsTxPorts;
    desc.n_rx_ant = kMmseV1MaxNumRxAntennas;
    desc.n_layers = 2U;
    desc.tx_mode = 2U;
    desc.start_symbol = 1U;
    desc.mod_order = 6U;
    desc.n_prb = kLteNumPrb20MHz;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    desc.pmi = -1;
    return desc;
}

GridBuffers make_empty_grid() {
    GridBuffers buffers;
    for (std::uint32_t rx = 0; rx < kMmseV1MaxNumRxAntennas; ++rx) {
        buffers.re[rx].assign(detail::kMaxGridRe, 0.0F);
        buffers.im[rx].assign(detail::kMaxGridRe, 0.0F);
    }
    return buffers;
}

void fill_lte_subframe(GridBuffers& buffers, const ExtractDescriptor& desc, detail::Complex32 h00,
                       detail::Complex32 h01, detail::Complex32 h10, detail::Complex32 h11,
                       std::mt19937& rng) {
    std::normal_distribution<float> noise(0.0F, 0.01F);
    detail::ensure_crs_tables();
    const std::uint32_t subframe = detail::subframe_from_descriptor(desc);

    for (std::uint32_t symbol = 0; symbol < kLteNumSymbolsNormalCp; ++symbol) {
        for (std::uint32_t sc = 0; sc < kLteNumSubcarriers20MHz; ++sc) {
            const std::size_t idx = static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz + sc;
            const bool is_port0_crs =
                detail::is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc) &&
                sc % 6U == detail::crs_frequency_offset(desc.cell_id, 0U,
                                                        static_cast<std::uint8_t>(symbol));
            const bool is_port1_crs =
                detail::is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc) &&
                sc % 6U == detail::crs_frequency_offset(desc.cell_id, 1U,
                                                        static_cast<std::uint8_t>(symbol));

            detail::Complex32 y0{};
            detail::Complex32 y1{};
            if (is_port0_crs || is_port1_crs) {
                const std::uint8_t tx = static_cast<std::uint8_t>(is_port1_crs ? 1U : 0U);
                const std::uint32_t offset = detail::crs_frequency_offset(
                    desc.cell_id, tx, static_cast<std::uint8_t>(symbol));
                const std::uint32_t pilot = (sc - offset) / 6U;
                const auto& crs = detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(subframe),
                     .port = tx,
                     .crs_symbol_index = static_cast<std::uint8_t>(
                         std::find(detail::kCrsSymbols.begin(), detail::kCrsSymbols.end(), symbol) -
                         detail::kCrsSymbols.begin())},
                    pilot);
                if (tx == 0U) {
                    y0 = detail::cmul(h00, crs);
                    y1 = detail::cmul(h10, crs);
                } else {
                    y0 = detail::cmul(h01, crs);
                    y1 = detail::cmul(h11, crs);
                }
            } else if (symbol >= desc.start_symbol) {
                const detail::Complex32 x0 = random_qam_symbol(rng, desc.mod_order);
                const detail::Complex32 x1 = random_qam_symbol(rng, desc.mod_order);
                y0 = detail::cadd(detail::cmul(h00, x0), detail::cmul(h01, x1));
                y1 = detail::cadd(detail::cmul(h10, x0), detail::cmul(h11, x1));
            }

            buffers.re[0][idx] = y0.re + noise(rng);
            buffers.im[0][idx] = y0.im + noise(rng);
            buffers.re[1][idx] = y1.re + noise(rng);
            buffers.im[1][idx] = y1.im + noise(rng);
        }
    }
}

std::vector<SubframeData> make_dataset() {
    std::mt19937 rng(42U);
    std::vector<SubframeData> dataset;
    dataset.reserve(kDatasetSubframes);

    for (std::uint32_t sf = 0; sf < kDatasetSubframes; ++sf) {
        SubframeData item{};
        item.desc = make_fullband_desc(sf % 10U);
        item.grid = make_empty_grid();

        const float phase = static_cast<float>(sf) * 0.071F;
        const float cross = static_cast<float>(sf) * 0.053F;
        const detail::Complex32 h00 = random_channel_coeff(rng, 0.90F + 0.05F * std::cos(phase),
                                                           0.10F * std::sin(phase), 0.02F);
        const detail::Complex32 h01 = random_channel_coeff(rng, 0.24F * std::cos(cross),
                                                           -0.30F + 0.05F * std::sin(cross), 0.02F);
        const detail::Complex32 h10 = random_channel_coeff(rng, 0.08F * std::sin(cross),
                                                           0.34F + 0.03F * std::cos(cross), 0.02F);
        const detail::Complex32 h11 = random_channel_coeff(rng, 1.00F + 0.04F * std::cos(phase),
                                                           -0.18F * std::sin(phase), 0.02F);
        fill_lte_subframe(item.grid, item.desc, h00, h01, h10, h11, rng);
        dataset.push_back(std::move(item));
    }
    return dataset;
}

TimingSummary summarize_ms(const std::vector<double>& samples_ms) {
    TimingSummary summary{};
    if (samples_ms.empty()) {
        return summary;
    }

    std::vector<double> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&](double q) {
        const std::size_t idx =
            static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1U));
        return sorted[idx];
    };

    summary.min_ms = sorted.front();
    summary.p50_ms = pct(0.50);
    summary.p95_ms = pct(0.95);
    summary.max_ms = sorted.back();
    summary.avg_ms =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    return summary;
}

TimedRun run_subframe_window(MmseEqualizerGpuContext& ctx, const std::vector<SubframeData>& dataset,
                             std::uint32_t begin_subframe, std::uint32_t subframe_count,
                             RunBuffers& run_buffers, HostPhaseSummary* host_summary = nullptr) {
    EqualizerOutputView out = make_output_view(run_buffers);
    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < subframe_count; ++i) {
        const SubframeData& subframe = dataset[(begin_subframe + i) % dataset.size()];
        const PlanarGridViewF32 grid = make_grid_view(subframe.grid);
        out.n_re_per_layer = 0U;
        out.n_layers = 0U;
        out.mod_order = 0U;
        const MmseStatus status = ctx.run(grid, subframe.desc, out);
        if (status != MmseStatus::kOk) {
            std::cerr << "benchmark run failed at subframe " << i << " status=" << to_string(status)
                      << '\n';
            std::exit(1);
        }
        checksum += static_cast<double>(out.x_hat_re[0]);
        checksum += static_cast<double>(out.x_hat_im[0]);
        checksum += static_cast<double>(out.sinr[0]);
        checksum += static_cast<double>(out.n_re_per_layer);
        if (host_summary != nullptr) {
            const MmseGpuHostProfileSnapshot snapshot = ctx.last_host_profile();
            host_summary->quantize_us += snapshot.quantize_us;
            host_summary->layout_build_us += snapshot.layout_build_us;
            host_summary->grid_meta_pack_us += snapshot.grid_meta_pack_us;
            host_summary->grid_h2d_us += snapshot.grid_h2d_us;
            host_summary->estimate_launch_us += snapshot.estimate_launch_us;
            host_summary->sigma2_d2h_us += snapshot.sigma2_d2h_us;
            host_summary->sigma2_mid_sync_us += snapshot.sigma2_mid_sync_us;
            host_summary->sigma2_host_update_us += snapshot.sigma2_host_update_us;
            host_summary->sigma2_h2d_us += snapshot.sigma2_h2d_us;
            host_summary->equalize_launch_us += snapshot.equalize_launch_us;
            host_summary->outputs_d2h_us += snapshot.outputs_d2h_us;
            host_summary->scratch_d2h_us += snapshot.scratch_d2h_us;
            host_summary->completion_d2h_us += snapshot.completion_d2h_us;
            host_summary->final_sync_us += snapshot.final_sync_us;
            host_summary->output_stage_us += snapshot.output_stage_us;
            host_summary->total_host_us += snapshot.total_host_us;
            host_summary->estimate_gpu_us += snapshot.estimate_gpu_us;
            host_summary->estimate_residual_gpu_us += snapshot.estimate_residual_gpu_us;
            host_summary->estimate_channel_gpu_us += snapshot.estimate_channel_gpu_us;
            host_summary->equalize_gpu_us += snapshot.equalize_gpu_us;
            host_summary->stream_gpu_us += snapshot.stream_gpu_us;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    return {.elapsed_ms = elapsed_ms, .checksum = checksum};
}

void warmup(MmseEqualizerGpuContext& ctx, const std::vector<SubframeData>& dataset,
            std::uint32_t warmup_subframes, RunBuffers& run_buffers) {
    if (warmup_subframes == 0U) {
        return;
    }
    (void)run_subframe_window(ctx, dataset, 0U, warmup_subframes, run_buffers);
}

BufferAccounting build_buffer_accounting(const ExtractDescriptor& desc,
                                         MmseGpuValidationPolicy validation_policy) {
    BufferAccounting accounting{};
    detail::ReLayout layout{};
    const std::uint32_t valid_re = detail::build_data_re_layout(desc, layout);

    const std::size_t grid_float_plane_bytes = detail::kCudaMaxGridRe * sizeof(float);
    const std::size_t output_plane_bytes = kOutputPlaneFloatCount * sizeof(float);
    const std::size_t estimate_bytes = detail::kCudaEstimateStubFloatCount * sizeof(float);
    const std::size_t host_scratch_bytes = detail::kCudaScratchTraceFloatCount * sizeof(float);
    const std::size_t device_scratch_bytes =
        (validation_policy == MmseGpuValidationPolicy::kTestDeepTrace
             ? detail::kCudaScratchTraceFloatCount
             : detail::kCudaScratchHeaderFloatCount) *
        sizeof(float);
    const std::size_t logical_output_plane_bytes =
        static_cast<std::size_t>(valid_re) * desc.n_layers * sizeof(float);

    accounting.logical_input_bytes_per_subframe =
        static_cast<std::size_t>(kMmseV1MaxNumRxAntennas) * 2U * detail::kCudaMaxGridRe *
        sizeof(float);
    accounting.logical_output_bytes_per_subframe = logical_output_plane_bytes * 3U;
    accounting.logical_input_bytes_per_10ms =
        accounting.logical_input_bytes_per_subframe * kSubframesPer10msFrame;
    accounting.logical_output_bytes_per_10ms =
        accounting.logical_output_bytes_per_subframe * kSubframesPer10msFrame;
    accounting.logical_input_bytes_per_100ms =
        accounting.logical_input_bytes_per_subframe * kSubframesPer100msWindow;
    accounting.logical_output_bytes_per_100ms =
        accounting.logical_output_bytes_per_subframe * kSubframesPer100msWindow;
    accounting.dataset_input_bytes =
        accounting.logical_input_bytes_per_subframe * kDatasetSubframes;

    accounting.host_pageable_bytes_per_slot =
        host_scratch_bytes + sizeof(detail::CudaGridMeta) + sizeof(detail::ReLayout);
    accounting.host_pinned_bytes_per_slot = 4U * grid_float_plane_bytes + 3U * output_plane_bytes;
    if (validation_policy == MmseGpuValidationPolicy::kTestDeepTrace) {
        accounting.host_pinned_bytes_per_slot += estimate_bytes;
    }
    accounting.host_pageable_bytes_total =
        accounting.host_pageable_bytes_per_slot * kPinnedRingSlotCount;
    accounting.host_pinned_bytes_total =
        accounting.host_pinned_bytes_per_slot * kPinnedRingSlotCount;

    accounting.device_scratch_bytes = device_scratch_bytes;
    accounting.device_bytes_per_slot = 4U * grid_float_plane_bytes + sizeof(detail::CudaGridMeta) +
                                       sizeof(float) + device_scratch_bytes + estimate_bytes +
                                       3U * output_plane_bytes + sizeof(std::uint32_t);
    accounting.device_bytes_total = accounting.device_bytes_per_slot * kPinnedRingSlotCount;
    return accounting;
}

std::string bytes_to_mib_string(std::size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << static_cast<double>(bytes) / (1024.0 * 1024.0);
    return oss.str();
}

std::string current_profile_date() {
    const std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#if defined(_WIN32)
    if (localtime_s(&local_tm, &now) != 0) {
        return "unknown";
    }
#else
    if (localtime_r(&now, &local_tm) == nullptr) {
        return "unknown";
    }
#endif

    char buffer[11] = {};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", &local_tm) == 0U) {
        return "unknown";
    }
    return buffer;
}

void print_process_memory(const char* label, const ProcessMemorySample& sample) {
    std::cout << label << ".working_set_bytes=" << sample.working_set_bytes << '\n';
    std::cout << label << ".private_bytes=" << sample.private_bytes << '\n';
}

void print_kernel_info(const char* prefix, const detail::CudaKernelResourceInfo& info) {
    std::cout << prefix << ".name=" << info.name.data() << '\n';
    std::cout << prefix << ".block_size=" << info.block_size << '\n';
    std::cout << prefix << ".max_threads_per_block=" << info.max_threads_per_block << '\n';
    std::cout << prefix << ".registers_per_thread=" << info.num_regs << '\n';
    std::cout << prefix << ".static_shared_bytes=" << info.static_shared_bytes << '\n';
    std::cout << prefix << ".local_bytes_per_thread=" << info.local_bytes << '\n';
    std::cout << prefix << ".const_bytes=" << info.const_bytes << '\n';
    std::cout << prefix << ".binary_version=" << info.binary_version << '\n';
    std::cout << prefix << ".ptx_version=" << info.ptx_version << '\n';
    std::cout << prefix << ".max_blocks_per_sm=" << info.max_blocks_per_multiprocessor << '\n';
    std::cout << prefix << ".theoretical_occupancy=" << info.theoretical_occupancy << '\n';
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
    if (text.empty()) {
        return false;
    }
    std::uint64_t parsed = 0;
    for (char c : text) {
        if (c < '0' || c > '9') {
            return false;
        }
        parsed = parsed * 10U + static_cast<std::uint64_t>(c - '0');
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
    }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

Options parse_options(int argc, char** argv) {
    Options options{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--warmup-subframes" && i + 1 < argc) {
            if (!parse_u32(argv[++i], options.warmup_subframes)) {
                std::cerr << "invalid --warmup-subframes value\n";
                std::exit(1);
            }
            continue;
        }
        if (arg == "--iters-10ms" && i + 1 < argc) {
            if (!parse_u32(argv[++i], options.iterations_10ms)) {
                std::cerr << "invalid --iters-10ms value\n";
                std::exit(1);
            }
            continue;
        }
        if (arg == "--iters-100ms" && i + 1 < argc) {
            if (!parse_u32(argv[++i], options.iterations_100ms)) {
                std::cerr << "invalid --iters-100ms value\n";
                std::exit(1);
            }
            continue;
        }

        std::cerr << "unknown argument: " << arg << '\n';
        std::exit(1);
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
#if !MMSE_CUDA_ENABLED
    std::cerr << "CUDA benchmark requires MMSE_CUDA_ENABLED=1\n";
    return 1;
#else
    const Options options = parse_options(argc, argv);
    const std::string profile_date = current_profile_date();
    if (!detail::cuda_backend_compiled()) {
        std::cerr << "CUDA backend is not compiled\n";
        return 1;
    }

    const ProcessMemorySample process_before_dataset = capture_process_memory();
    std::vector<SubframeData> dataset = make_dataset();
    const ProcessMemorySample process_after_dataset = capture_process_memory();

    bool runtime_available = false;
    if (detail::cuda_select_device(0U, runtime_available) != MmseStatus::kOk ||
        !runtime_available) {
        std::cerr << "failed to select CUDA device 0\n";
        return 1;
    }

    detail::CudaDeviceInfo device_info{};
    if (detail::cuda_query_device_info(0U, device_info) != MmseStatus::kOk) {
        std::cerr << "failed to query CUDA device info\n";
        return 1;
    }

    std::size_t gpu_free_before_init = 0;
    std::size_t gpu_total_before_init = 0;
    if (detail::cuda_query_current_memory_info(gpu_free_before_init, gpu_total_before_init) !=
        MmseStatus::kOk) {
        std::cerr << "failed to query initial CUDA memory info\n";
        return 1;
    }

    MmseEqualizerGpuContext ctx;
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kCuda;
    config.stream_count = 1U;
    config.sigma2_ownership = MmseGpuSigma2Ownership::kDeviceOwnedState;
    config.validation_policy = MmseGpuValidationPolicy::kReleaseSanity;

    const auto init_start = std::chrono::steady_clock::now();
    const MmseStatus init_status = ctx.init(config);
    const auto init_end = std::chrono::steady_clock::now();
    if (init_status != MmseStatus::kOk) {
        std::cerr << "GPU init failed: " << to_string(init_status) << '\n';
        return 1;
    }
    const double init_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(init_end - init_start)
            .count();

    const ProcessMemorySample process_after_init = capture_process_memory();
    std::size_t gpu_free_after_init = 0;
    std::size_t gpu_total_after_init = 0;
    if (detail::cuda_query_current_memory_info(gpu_free_after_init, gpu_total_after_init) !=
        MmseStatus::kOk) {
        std::cerr << "failed to query post-init CUDA memory info\n";
        return 1;
    }

    detail::CudaKernelResourceInfo estimate_kernel{};
    detail::CudaKernelResourceInfo equalize_kernel{};
    if (detail::cuda_query_estimate_kernel_resources(estimate_kernel) != MmseStatus::kOk ||
        detail::cuda_query_equalize_kernel_resources(kEqualizeBlockSize, equalize_kernel) !=
            MmseStatus::kOk) {
        std::cerr << "failed to query kernel resource info\n";
        return 1;
    }

    RunBuffers run_buffers{};
    run_buffers.xhat_re.assign(kOutputCapacityPerLayer * 2U, 0.0F);
    run_buffers.xhat_im.assign(kOutputCapacityPerLayer * 2U, 0.0F);
    run_buffers.sinr.assign(kOutputCapacityPerLayer * 2U, 0.0F);

    const TimedRun cold_10ms =
        run_subframe_window(ctx, dataset, 0U, kSubframesPer10msFrame, run_buffers);
    warmup(ctx, dataset, options.warmup_subframes, run_buffers);
    const TimedRun warm_10ms =
        run_subframe_window(ctx, dataset, 0U, kSubframesPer10msFrame, run_buffers);

    std::vector<double> times_10ms_ms;
    std::vector<double> times_100ms_ms;
    double checksum_10ms = 0.0;
    double checksum_100ms = 0.0;
    HostPhaseSummary host_phase_10ms{};

    times_10ms_ms.reserve(options.iterations_10ms);
    for (std::uint32_t iter = 0; iter < options.iterations_10ms; ++iter) {
        const TimedRun timed =
            run_subframe_window(ctx, dataset, (iter * kSubframesPer10msFrame) % dataset.size(),
                                kSubframesPer10msFrame, run_buffers, &host_phase_10ms);
        times_10ms_ms.push_back(timed.elapsed_ms);
        checksum_10ms += timed.checksum;
    }

    times_100ms_ms.reserve(options.iterations_100ms);
    for (std::uint32_t iter = 0; iter < options.iterations_100ms; ++iter) {
        const TimedRun timed = run_subframe_window(ctx, dataset, iter % dataset.size(),
                                                   kSubframesPer100msWindow, run_buffers);
        times_100ms_ms.push_back(timed.elapsed_ms);
        checksum_100ms += timed.checksum;
    }

    const ProcessMemorySample process_after_runs = capture_process_memory();
    std::size_t gpu_free_after_runs = 0;
    std::size_t gpu_total_after_runs = 0;
    if (detail::cuda_query_current_memory_info(gpu_free_after_runs, gpu_total_after_runs) !=
        MmseStatus::kOk) {
        std::cerr << "failed to query post-run CUDA memory info\n";
        return 1;
    }

    const TimingSummary summary_10ms = summarize_ms(times_10ms_ms);
    const TimingSummary summary_100ms = summarize_ms(times_100ms_ms);
    const BufferAccounting accounting =
        build_buffer_accounting(dataset.front().desc, config.validation_policy);

    detail::ReLayout layout{};
    const std::uint32_t valid_re = detail::build_data_re_layout(dataset.front().desc, layout);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[profile]\n";
    std::cout << "profile.date=" << profile_date << '\n';
    std::cout << "profile.mode=gpu_cuda_release_sanity\n";
    std::cout << "profile.init_ms=" << init_ms << '\n';
    std::cout << "profile.warmup_subframes=" << options.warmup_subframes << '\n';
    std::cout << "profile.iterations_10ms=" << options.iterations_10ms << '\n';
    std::cout << "profile.iterations_100ms=" << options.iterations_100ms << '\n';

    std::cout << "[lte]\n";
    std::cout << "lte.bandwidth_mhz=20\n";
    std::cout << "lte.subcarriers=" << kLteNumSubcarriers20MHz << '\n';
    std::cout << "lte.prb=" << kLteNumPrb20MHz << '\n';
    std::cout << "lte.symbols_per_subframe=" << kLteNumSymbolsNormalCp << '\n';
    std::cout << "lte.subframes_per_10ms=" << kSubframesPer10msFrame << '\n';
    std::cout << "lte.subframes_per_100ms=" << kSubframesPer100msWindow << '\n';
    std::cout << "lte.n_layers=" << static_cast<std::uint32_t>(dataset.front().desc.n_layers)
              << '\n';
    std::cout << "lte.mod_order_bits=" << static_cast<std::uint32_t>(dataset.front().desc.mod_order)
              << '\n';
    std::cout << "lte.valid_re_per_subframe=" << valid_re << '\n';

    std::cout << "[device]\n";
    std::cout << "device.name=" << device_info.name.data() << '\n';
    std::cout << "device.compute_capability=" << device_info.major << '.' << device_info.minor
              << '\n';
    std::cout << "device.sm_count=" << device_info.multi_processor_count << '\n';
    std::cout << "device.warp_size=" << device_info.warp_size << '\n';
    std::cout << "device.max_threads_per_block=" << device_info.max_threads_per_block << '\n';
    std::cout << "device.max_threads_per_sm=" << device_info.max_threads_per_multiprocessor << '\n';
    std::cout << "device.regs_per_block=" << device_info.regs_per_block << '\n';
    std::cout << "device.regs_per_sm=" << device_info.regs_per_multiprocessor << '\n';
    std::cout << "device.shared_mem_per_block_bytes=" << device_info.shared_mem_per_block << '\n';
    std::cout << "device.shared_mem_per_sm_bytes=" << device_info.shared_mem_per_multiprocessor
              << '\n';
    std::cout << "device.total_global_mem_bytes=" << device_info.total_global_mem << '\n';
    std::cout << "device.l2_cache_bytes=" << device_info.l2_cache_size << '\n';

    std::cout << "[data]\n";
    std::cout << "data.logical_input_bytes_per_subframe="
              << accounting.logical_input_bytes_per_subframe << '\n';
    std::cout << "data.logical_output_bytes_per_subframe="
              << accounting.logical_output_bytes_per_subframe << '\n';
    std::cout << "data.logical_input_bytes_per_10ms=" << accounting.logical_input_bytes_per_10ms
              << '\n';
    std::cout << "data.logical_output_bytes_per_10ms=" << accounting.logical_output_bytes_per_10ms
              << '\n';
    std::cout << "data.logical_input_bytes_per_100ms=" << accounting.logical_input_bytes_per_100ms
              << '\n';
    std::cout << "data.logical_output_bytes_per_100ms=" << accounting.logical_output_bytes_per_100ms
              << '\n';
    std::cout << "data.dataset_input_bytes=" << accounting.dataset_input_bytes << '\n';

    std::cout << "[buffers]\n";
    std::cout << "buffers.host_pageable_bytes_per_slot=" << accounting.host_pageable_bytes_per_slot
              << '\n';
    std::cout << "buffers.host_pinned_bytes_per_slot=" << accounting.host_pinned_bytes_per_slot
              << '\n';
    std::cout << "buffers.host_pageable_bytes_total=" << accounting.host_pageable_bytes_total
              << '\n';
    std::cout << "buffers.host_pinned_bytes_total=" << accounting.host_pinned_bytes_total << '\n';
    std::cout << "buffers.device_scratch_bytes_per_slot=" << accounting.device_scratch_bytes
              << '\n';
    std::cout << "buffers.device_bytes_per_slot=" << accounting.device_bytes_per_slot << '\n';
    std::cout << "buffers.device_bytes_total=" << accounting.device_bytes_total << '\n';

    std::cout << "[memory]\n";
    std::cout << "memory.gpu_free_before_init_bytes=" << gpu_free_before_init << '\n';
    std::cout << "memory.gpu_free_after_init_bytes=" << gpu_free_after_init << '\n';
    std::cout << "memory.gpu_free_after_runs_bytes=" << gpu_free_after_runs << '\n';
    std::cout << "memory.gpu_total_bytes=" << gpu_total_before_init << '\n';
    std::cout << "memory.gpu_allocated_delta_init_bytes="
              << (gpu_free_before_init - gpu_free_after_init) << '\n';
    std::cout << "memory.gpu_allocated_delta_run_bytes="
              << (gpu_free_after_init - gpu_free_after_runs) << '\n';
    print_process_memory("memory.process_before_dataset", process_before_dataset);
    print_process_memory("memory.process_after_dataset", process_after_dataset);
    print_process_memory("memory.process_after_init", process_after_init);
    print_process_memory("memory.process_after_runs", process_after_runs);

    std::cout << "[kernels]\n";
    print_kernel_info("kernel.estimate", estimate_kernel);
    print_kernel_info("kernel.equalize", equalize_kernel);

    std::cout << "[timing]\n";
    std::cout << "timing.cold_10ms_ms=" << cold_10ms.elapsed_ms << '\n';
    std::cout << "timing.cold_10ms_per_subframe_ms="
              << cold_10ms.elapsed_ms / static_cast<double>(kSubframesPer10msFrame) << '\n';
    std::cout << "timing.warm_10ms_ms=" << warm_10ms.elapsed_ms << '\n';
    std::cout << "timing.warm_10ms_per_subframe_ms="
              << warm_10ms.elapsed_ms / static_cast<double>(kSubframesPer10msFrame) << '\n';
    std::cout << "timing.avg_10ms_ms=" << summary_10ms.avg_ms << '\n';
    std::cout << "timing.p50_10ms_ms=" << summary_10ms.p50_ms << '\n';
    std::cout << "timing.p95_10ms_ms=" << summary_10ms.p95_ms << '\n';
    std::cout << "timing.max_10ms_ms=" << summary_10ms.max_ms << '\n';
    std::cout << "timing.min_10ms_ms=" << summary_10ms.min_ms << '\n';
    std::cout << "timing.avg_10ms_per_subframe_ms="
              << summary_10ms.avg_ms / static_cast<double>(kSubframesPer10msFrame) << '\n';
    std::cout << "timing.avg_100ms_ms=" << summary_100ms.avg_ms << '\n';
    std::cout << "timing.p50_100ms_ms=" << summary_100ms.p50_ms << '\n';
    std::cout << "timing.p95_100ms_ms=" << summary_100ms.p95_ms << '\n';
    std::cout << "timing.max_100ms_ms=" << summary_100ms.max_ms << '\n';
    std::cout << "timing.min_100ms_ms=" << summary_100ms.min_ms << '\n';
    std::cout << "timing.avg_100ms_per_subframe_ms="
              << summary_100ms.avg_ms / static_cast<double>(kSubframesPer100msWindow) << '\n';
    std::cout << "timing.checksum_10ms=" << checksum_10ms << '\n';
    std::cout << "timing.checksum_100ms=" << checksum_100ms << '\n';

    const double host_divisor =
        options.iterations_10ms == 0U
            ? 1.0
            : static_cast<double>(options.iterations_10ms * kSubframesPer10msFrame);
    std::cout << "[host]\n";
    std::cout << "host.phase.quantize_us=" << (host_phase_10ms.quantize_us / host_divisor) << '\n';
    std::cout << "host.phase.layout_build_us=" << (host_phase_10ms.layout_build_us / host_divisor)
              << '\n';
    std::cout << "host.phase.grid_meta_pack_us="
              << (host_phase_10ms.grid_meta_pack_us / host_divisor) << '\n';
    std::cout << "host.phase.grid_h2d_us=" << (host_phase_10ms.grid_h2d_us / host_divisor) << '\n';
    std::cout << "host.phase.estimate_launch_us="
              << (host_phase_10ms.estimate_launch_us / host_divisor) << '\n';
    std::cout << "host.phase.sigma2_d2h_us=" << (host_phase_10ms.sigma2_d2h_us / host_divisor)
              << '\n';
    std::cout << "host.phase.sigma2_mid_sync_us="
              << (host_phase_10ms.sigma2_mid_sync_us / host_divisor) << '\n';
    std::cout << "host.phase.sigma2_host_update_us="
              << (host_phase_10ms.sigma2_host_update_us / host_divisor) << '\n';
    std::cout << "host.phase.sigma2_h2d_us=" << (host_phase_10ms.sigma2_h2d_us / host_divisor)
              << '\n';
    std::cout << "host.phase.equalize_launch_us="
              << (host_phase_10ms.equalize_launch_us / host_divisor) << '\n';
    std::cout << "host.phase.outputs_d2h_us=" << (host_phase_10ms.outputs_d2h_us / host_divisor)
              << '\n';
    std::cout << "host.phase.scratch_d2h_us=" << (host_phase_10ms.scratch_d2h_us / host_divisor)
              << '\n';
    std::cout << "host.phase.completion_d2h_us="
              << (host_phase_10ms.completion_d2h_us / host_divisor) << '\n';
    std::cout << "host.phase.final_sync_us=" << (host_phase_10ms.final_sync_us / host_divisor)
              << '\n';
    std::cout << "host.phase.output_stage_us=" << (host_phase_10ms.output_stage_us / host_divisor)
              << '\n';
    std::cout << "host.phase.total_host_us=" << (host_phase_10ms.total_host_us / host_divisor)
              << '\n';
    std::cout << "host.phase.estimate_gpu_us=" << (host_phase_10ms.estimate_gpu_us / host_divisor)
              << '\n';
    std::cout << "host.phase.estimate_residual_gpu_us="
              << (host_phase_10ms.estimate_residual_gpu_us / host_divisor) << '\n';
    std::cout << "host.phase.estimate_channel_gpu_us="
              << (host_phase_10ms.estimate_channel_gpu_us / host_divisor) << '\n';
    std::cout << "host.phase.equalize_gpu_us=" << (host_phase_10ms.equalize_gpu_us / host_divisor)
              << '\n';
    std::cout << "host.phase.stream_gpu_us=" << (host_phase_10ms.stream_gpu_us / host_divisor)
              << '\n';

    std::cout << "[derived]\n";
    std::cout << "derived.dataset_input_mib=" << bytes_to_mib_string(accounting.dataset_input_bytes)
              << '\n';
    std::cout << "derived.device_bytes_total_mib="
              << bytes_to_mib_string(accounting.device_bytes_total) << '\n';
    std::cout << "derived.host_pinned_bytes_total_mib="
              << bytes_to_mib_string(accounting.host_pinned_bytes_total) << '\n';
    std::cout << "derived.host_pageable_bytes_total_mib="
              << bytes_to_mib_string(accounting.host_pageable_bytes_total) << '\n';
    return 0;
#endif
}
