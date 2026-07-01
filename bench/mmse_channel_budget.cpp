#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "mmse/lte_chain_sdk.h"

using namespace mmse;

namespace {

constexpr std::uint32_t kPbchCapacity = 256U;
constexpr std::uint32_t kPcfichCapacity = 32U;
constexpr std::uint32_t kPdcchCapacity = 4000U;
constexpr std::uint32_t kPdschCapacityPerLayer = 20000U;
constexpr std::uint32_t kWarmupIters = 8U;
constexpr std::uint32_t kMeasureIters = 64U;

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

struct ChannelBuffers {
    std::vector<float> pdsch_re;
    std::vector<float> pdsch_im;
    std::vector<float> pdsch_sinr;
    EqualizerOutputView pdsch_out{};

    std::vector<float> pbch_re;
    std::vector<float> pbch_im;
    std::vector<float> pbch_sinr;
    std::vector<std::uint16_t> pbch_idx;
    PbchMmseOutputView pbch_out{};
    PbchMmseResult pbch_meta{};

    std::vector<float> pcfich_re;
    std::vector<float> pcfich_im;
    std::vector<float> pcfich_sinr;
    std::vector<std::uint16_t> pcfich_idx;
    PcfichMmseOutputView pcfich_out{};
    PcfichMmseResult pcfich_meta{};

    std::vector<float> pdcch_re;
    std::vector<float> pdcch_im;
    std::vector<float> pdcch_sinr;
    std::vector<std::uint16_t> pdcch_idx;
    PdcchMmseOutputView pdcch_out{};
    PdcchMmseResult pdcch_meta{};
};

GridBuffers make_random_grid(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    GridBuffers grid;
    for (std::uint32_t rx = 0; rx < kLteNumRxAntV1; ++rx) {
        grid.re[rx].resize(kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz);
        grid.im[rx].resize(kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz);
        for (std::size_t i = 0; i < grid.re[rx].size(); ++i) {
            grid.re[rx][i] = dist(rng);
            grid.im[rx][i] = dist(rng);
        }
    }
    return grid;
}

PlanarGridViewF32 make_view(const GridBuffers& buffers) {
    PlanarGridViewF32 view{};
    view.re = {buffers.re[0].data(), buffers.re[1].data()};
    view.im = {buffers.im[0].data(), buffers.im[1].data()};
    view.n_rx_ant = kLteNumRxAntV1;
    view.n_symbols = kLteNumSymbolsNormalCp;
    view.n_subcarriers = kLteNumSubcarriers20MHz;
    return view;
}

ExtractDescriptor make_pdsch_desc() {
    ExtractDescriptor desc{};
    desc.sfn_subframe = 0U;
    desc.cell_id = 1U;
    desc.n_tx_ports = 2U;
    desc.n_rx_ant = 2U;
    desc.n_layers = 2U;
    desc.tx_mode = 2U;
    desc.channel_type = MmseChannelType::kPdsch;
    desc.start_symbol = 1U;
    desc.control_symbol_count = 0U;
    desc.mod_order = 6U;
    desc.n_prb = 100U;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    desc.pmi = -1;
    return desc;
}

PdcchMmseInput make_pdcch_input(const PlanarGridViewF32& grid) {
    pdcch::FrontendPdcchIndication frontend{};
    frontend.sfn_subframe = 0U;
    frontend.cell_id = 1U;
    frontend.n_tx_ports = 1U;
    frontend.tx_mode = 1U;
    frontend.control_symbol_count = 3U;
    frontend.n_prb = 100U;
    frontend.prb_bitmap.fill(0xFFFFU);
    frontend.prb_bitmap.back() = 0x000FU;
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

PbchMmseInput make_pbch_input(const PlanarGridViewF32& grid) {
    pbch::FrontendPbchIndication frontend{};
    frontend.sfn_subframe = 0U;
    frontend.cell_id = 1U;
    frontend.n_tx_ports = 1U;
    frontend.tx_mode = 1U;
    return pbch::make_pbch_mmse_input(grid, frontend);
}

PcfichMmseInput make_pcfich_input(const PlanarGridViewF32& grid) {
    pcfich::FrontendPcfichIndication frontend{};
    frontend.sfn_subframe = 0U;
    frontend.cell_id = 1U;
    frontend.n_tx_ports = 1U;
    frontend.tx_mode = 1U;
    return pcfich::make_pcfich_mmse_input(grid, frontend);
}

ChannelBuffers make_channel_buffers() {
    ChannelBuffers b{};

    b.pdsch_re.resize(kPdschCapacityPerLayer * 2U);
    b.pdsch_im.resize(kPdschCapacityPerLayer * 2U);
    b.pdsch_sinr.resize(kPdschCapacityPerLayer * 2U);
    b.pdsch_out = {b.pdsch_re.data(), b.pdsch_im.data(), b.pdsch_sinr.data(),
                   kPdschCapacityPerLayer};

    b.pbch_re.resize(kPbchCapacity);
    b.pbch_im.resize(kPbchCapacity);
    b.pbch_sinr.resize(kPbchCapacity);
    b.pbch_idx.resize(kPbchCapacity);
    b.pbch_out.x_hat_re = b.pbch_re.data();
    b.pbch_out.x_hat_im = b.pbch_im.data();
    b.pbch_out.sinr = b.pbch_sinr.data();
    b.pbch_out.re_grid_indices = b.pbch_idx.data();
    b.pbch_out.capacity_re_per_layer = kPbchCapacity;
    b.pbch_out.capacity_re_metadata = kPbchCapacity;

    b.pcfich_re.resize(kPcfichCapacity);
    b.pcfich_im.resize(kPcfichCapacity);
    b.pcfich_sinr.resize(kPcfichCapacity);
    b.pcfich_idx.resize(kPcfichCapacity);
    b.pcfich_out.x_hat_re = b.pcfich_re.data();
    b.pcfich_out.x_hat_im = b.pcfich_im.data();
    b.pcfich_out.sinr = b.pcfich_sinr.data();
    b.pcfich_out.re_grid_indices = b.pcfich_idx.data();
    b.pcfich_out.capacity_re_per_layer = kPcfichCapacity;
    b.pcfich_out.capacity_re_metadata = kPcfichCapacity;

    b.pdcch_re.resize(kPdcchCapacity);
    b.pdcch_im.resize(kPdcchCapacity);
    b.pdcch_sinr.resize(kPdcchCapacity);
    b.pdcch_idx.resize(kPdcchCapacity);
    b.pdcch_out.x_hat_re = b.pdcch_re.data();
    b.pdcch_out.x_hat_im = b.pdcch_im.data();
    b.pdcch_out.sinr = b.pdcch_sinr.data();
    b.pdcch_out.re_grid_indices = b.pdcch_idx.data();
    b.pdcch_out.capacity_re_per_layer = kPdcchCapacity;
    b.pdcch_out.capacity_re_metadata = kPdcchCapacity;

    return b;
}

template <typename T> void summarize(std::string_view label, const std::vector<T>& values) {
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&](double p) {
        const std::size_t idx =
            static_cast<std::size_t>(p * static_cast<double>(sorted.size() - 1U));
        return sorted[idx];
    };
    double sum = 0.0;
    for (const double v : sorted) {
        sum += v;
    }
    std::cout << label << ".avg_us=" << (sum / static_cast<double>(sorted.size())) << '\n';
    std::cout << label << ".p50_us=" << pct(0.50) << '\n';
    std::cout << label << ".p95_us=" << pct(0.95) << '\n';
    std::cout << label << ".p99_us=" << pct(0.99) << '\n';
}

template <typename RunPdschFn, typename RunPbchFn, typename RunPcfichFn, typename RunPdcchFn>
bool measure_path(std::string_view prefix, RunPdschFn&& run_pdsch, RunPbchFn&& run_pbch,
                  RunPcfichFn&& run_pcfich, RunPdcchFn&& run_pdcch, const ChannelBuffers& buffers) {
    std::vector<double> pdsch_us;
    std::vector<double> pbch_us;
    std::vector<double> pcfich_us;
    std::vector<double> pdcch_us;
    std::vector<double> aggregate6_us;
    pdsch_us.reserve(kMeasureIters);
    pbch_us.reserve(kMeasureIters);
    pcfich_us.reserve(kMeasureIters);
    pdcch_us.reserve(kMeasureIters);
    aggregate6_us.reserve(kMeasureIters);

    for (std::uint32_t i = 0; i < kMeasureIters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        if (!run_pdsch()) {
            return false;
        }
        auto end = std::chrono::high_resolution_clock::now();
        pdsch_us.push_back(
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start)
                .count());

        start = std::chrono::high_resolution_clock::now();
        if (!run_pbch()) {
            return false;
        }
        end = std::chrono::high_resolution_clock::now();
        pbch_us.push_back(
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start)
                .count());

        start = std::chrono::high_resolution_clock::now();
        if (!run_pcfich()) {
            return false;
        }
        end = std::chrono::high_resolution_clock::now();
        pcfich_us.push_back(
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start)
                .count());

        start = std::chrono::high_resolution_clock::now();
        if (!run_pdcch()) {
            return false;
        }
        end = std::chrono::high_resolution_clock::now();
        pdcch_us.push_back(
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start)
                .count());

        start = std::chrono::high_resolution_clock::now();
        if (!run_pbch() || !run_pcfich() || !run_pdcch() || !run_pdsch() || !run_pdcch() ||
            !run_pdsch()) {
            return false;
        }
        end = std::chrono::high_resolution_clock::now();
        aggregate6_us.push_back(
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start)
                .count());
    }

    summarize(std::string(prefix) + ".pbch", pbch_us);
    summarize(std::string(prefix) + ".pcfich", pcfich_us);
    summarize(std::string(prefix) + ".pdcch", pdcch_us);
    summarize(std::string(prefix) + ".pdsch", pdsch_us);
    summarize(std::string(prefix) + ".six_calls", aggregate6_us);
    std::cout << prefix << ".pbch_n_re=" << buffers.pbch_meta.n_re << '\n';
    std::cout << prefix << ".pcfich_n_re=" << buffers.pcfich_meta.n_re << '\n';
    std::cout << prefix << ".pdcch_n_re=" << buffers.pdcch_meta.n_re << '\n';
    std::cout << prefix << ".pdsch_n_re_per_layer=" << buffers.pdsch_out.n_re_per_layer << '\n';
    return true;
}

} // namespace

int main() {
    GridBuffers grid_buffers = make_random_grid(42U);
    const PlanarGridViewF32 grid = make_view(grid_buffers);
    const ExtractDescriptor pdsch_desc = make_pdsch_desc();
    const PdcchMmseInput pdcch_in = make_pdcch_input(grid);
    const PbchMmseInput pbch_in = make_pbch_input(grid);
    const PcfichMmseInput pcfich_in = make_pcfich_input(grid);

    MmseEqualizerCpuContext cpu_ctx;
    MmseEqualizerCpuConfig cpu_cfg{};
    cpu_cfg.worker_count = std::max(1U, std::thread::hardware_concurrency());
    if (cpu_ctx.init(cpu_cfg) != MmseStatus::kOk) {
        std::cerr << "failed to init cpu context\n";
        return 1;
    }

    MmseEqualizerGpuContext gpu_ctx;
    MmseEqualizerGpuConfig gpu_cfg{};
    gpu_cfg.backend = MmseGpuBackend::kCuda;
    if (gpu_ctx.init(gpu_cfg) != MmseStatus::kOk) {
        std::cerr << "failed to init gpu context\n";
        return 1;
    }

    ChannelBuffers cpu_buffers = make_channel_buffers();
    ChannelBuffers gpu_buffers = make_channel_buffers();

    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (cpu_ctx.run(grid, pdsch_desc, cpu_buffers.pdsch_out) != MmseStatus::kOk ||
            cpu_ctx.run_pbch(pbch_in, cpu_buffers.pbch_out, cpu_buffers.pbch_meta) !=
                MmseStatus::kOk ||
            cpu_ctx.run_pcfich(pcfich_in, cpu_buffers.pcfich_out, cpu_buffers.pcfich_meta) !=
                MmseStatus::kOk ||
            cpu_ctx.run_pdcch(pdcch_in, cpu_buffers.pdcch_out, cpu_buffers.pdcch_meta) !=
                MmseStatus::kOk) {
            std::cerr << "cpu warmup failed\n";
            return 1;
        }
        if (gpu_ctx.run(grid, pdsch_desc, gpu_buffers.pdsch_out) != MmseStatus::kOk ||
            gpu_ctx.run_pbch(pbch_in, gpu_buffers.pbch_out, gpu_buffers.pbch_meta) !=
                MmseStatus::kOk ||
            gpu_ctx.run_pcfich(pcfich_in, gpu_buffers.pcfich_out, gpu_buffers.pcfich_meta) !=
                MmseStatus::kOk ||
            gpu_ctx.run_pdcch(pdcch_in, gpu_buffers.pdcch_out, gpu_buffers.pdcch_meta) !=
                MmseStatus::kOk) {
            std::cerr << "gpu warmup failed\n";
            return 1;
        }
    }

    if (!measure_path(
            "cpu",
            [&] { return cpu_ctx.run(grid, pdsch_desc, cpu_buffers.pdsch_out) == MmseStatus::kOk; },
            [&] {
                return cpu_ctx.run_pbch(pbch_in, cpu_buffers.pbch_out, cpu_buffers.pbch_meta) ==
                       MmseStatus::kOk;
            },
            [&] {
                return cpu_ctx.run_pcfich(pcfich_in, cpu_buffers.pcfich_out,
                                          cpu_buffers.pcfich_meta) == MmseStatus::kOk;
            },
            [&] {
                return cpu_ctx.run_pdcch(pdcch_in, cpu_buffers.pdcch_out, cpu_buffers.pdcch_meta) ==
                       MmseStatus::kOk;
            },
            cpu_buffers)) {
        std::cerr << "cpu measurement failed\n";
        return 1;
    }

    if (!measure_path(
            "gpu",
            [&] { return gpu_ctx.run(grid, pdsch_desc, gpu_buffers.pdsch_out) == MmseStatus::kOk; },
            [&] {
                return gpu_ctx.run_pbch(pbch_in, gpu_buffers.pbch_out, gpu_buffers.pbch_meta) ==
                       MmseStatus::kOk;
            },
            [&] {
                return gpu_ctx.run_pcfich(pcfich_in, gpu_buffers.pcfich_out,
                                          gpu_buffers.pcfich_meta) == MmseStatus::kOk;
            },
            [&] {
                return gpu_ctx.run_pdcch(pdcch_in, gpu_buffers.pdcch_out, gpu_buffers.pdcch_meta) ==
                       MmseStatus::kOk;
            },
            gpu_buffers)) {
        std::cerr << "gpu measurement failed\n";
        return 1;
    }

    return 0;
}
