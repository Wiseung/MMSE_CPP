#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "mmse/mmse_equalizer.h"

using namespace mmse;

namespace {

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

GridBuffers make_random_grid(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    GridBuffers grid;
    for (std::uint32_t rx = 0; rx < 2; ++rx) {
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
    view.n_rx_ant = 2;
    view.n_symbols = kLteNumSymbolsNormalCp;
    view.n_subcarriers = kLteNumSubcarriers20MHz;
    return view;
}

ExtractDescriptor make_desc() {
    ExtractDescriptor desc{};
    desc.cell_id = 1;
    desc.n_tx_ports = 2;
    desc.n_rx_ant = 2;
    desc.n_layers = 2;
    desc.tx_mode = 2;
    desc.start_symbol = 1;
    desc.mod_order = 6;
    desc.n_prb = 100;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    desc.pmi = -1;
    return desc;
}

template <typename T>
void summarize(const std::vector<T>& values, const char* name) {
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) {
        const std::size_t idx = static_cast<std::size_t>(p * static_cast<double>(sorted.size() - 1U));
        return sorted[idx];
    };
    std::cout << name << " p50=" << pct(0.50) << "us"
              << " p95=" << pct(0.95) << "us"
              << " p99=" << pct(0.99) << "us\n";
}

}  // namespace

int main() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = std::max(1U, std::thread::hardware_concurrency());
    if (ctx.init(config) != MmseStatus::kOk) {
        std::cerr << "failed to init context\n";
        return 1;
    }

    GridBuffers buffers = make_random_grid(42U);
    auto grid = make_view(buffers);
    auto desc = make_desc();

    constexpr std::uint32_t cap = 20000U;
    std::vector<float> xre(cap * 2U);
    std::vector<float> xim(cap * 2U);
    std::vector<float> sinr(cap * 2U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};

    constexpr std::uint32_t warmup = 8U;
    constexpr std::uint32_t iters = 64U;

    for (std::uint32_t i = 0; i < warmup; ++i) {
        if (ctx.run(grid, desc, out) != MmseStatus::kOk) {
            std::cerr << "warmup failed\n";
            return 1;
        }
    }

    std::vector<double> times_us;
    times_us.reserve(iters);
    for (std::uint32_t i = 0; i < iters; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        if (ctx.run(grid, desc, out) != MmseStatus::kOk) {
            std::cerr << "benchmark iteration failed\n";
            return 1;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count();
        times_us.push_back(us);
    }

    summarize(times_us, "mmse_cpu");
    std::cout << "n_re_per_layer=" << out.n_re_per_layer << '\n';
    return 0;
}
