#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "mmse/mmse_equalizer.h"
#include "internal/mmse_internal.h"

using mmse::EqualizerOutputView;
using mmse::ExtractDescriptor;
using mmse::MmseEqualizerCpuConfig;
using mmse::MmseEqualizerCpuContext;
using mmse::MmseStatus;
using mmse::PlanarGridViewF32;
using namespace mmse;

namespace {

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

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

GridBuffers make_zero_grid() {
    GridBuffers buffers;
    for (std::uint32_t rx = 0; rx < 2; ++rx) {
        buffers.re[rx].assign(kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz, 0.0F);
        buffers.im[rx].assign(kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz, 0.0F);
    }
    return buffers;
}

PlanarGridViewF32 make_grid_view(const GridBuffers& buffers) {
    PlanarGridViewF32 grid{};
    grid.re = {buffers.re[0].data(), buffers.re[1].data()};
    grid.im = {buffers.im[0].data(), buffers.im[1].data()};
    grid.n_rx_ant = 2;
    grid.n_symbols = kLteNumSymbolsNormalCp;
    grid.n_subcarriers = kLteNumSubcarriers20MHz;
    return grid;
}

ExtractDescriptor make_fullband_desc() {
    ExtractDescriptor desc{};
    desc.cell_id = 0;
    desc.n_tx_ports = 2;
    desc.n_rx_ant = 2;
    desc.n_layers = 2;
    desc.start_symbol = 1;
    desc.mod_order = 2;
    desc.n_prb = 100;
    desc.tx_mode = 2;
    desc.pmi = -1;
    desc.sfn_subframe = 3;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    return desc;
}

void fill_identity_channel(GridBuffers& buffers, const ExtractDescriptor& desc, float data0, float data1) {
    mmse::detail::ensure_crs_tables();
    const std::uint32_t subframe = mmse::detail::subframe_from_descriptor(desc);
    for (std::uint32_t symbol = 0; symbol < kLteNumSymbolsNormalCp; ++symbol) {
        for (std::uint32_t sc = 0; sc < kLteNumSubcarriers20MHz; ++sc) {
            const std::size_t idx = static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz + sc;
            const bool is_data_symbol = symbol >= desc.start_symbol;
            const bool is_port0_crs =
                mmse::detail::is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc) &&
                sc % 6U == mmse::detail::crs_frequency_offset(desc.cell_id, 0U, static_cast<std::uint8_t>(symbol));
            const bool is_port1_crs =
                mmse::detail::is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc) &&
                sc % 6U == mmse::detail::crs_frequency_offset(desc.cell_id, 1U, static_cast<std::uint8_t>(symbol));
            if (is_port0_crs) {
                const std::uint32_t pilot =
                    (sc - mmse::detail::crs_frequency_offset(desc.cell_id, 0U, static_cast<std::uint8_t>(symbol))) / 6U;
                const auto& crs = mmse::detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(subframe),
                     .port = 0U,
                     .crs_symbol_index = static_cast<std::uint8_t>(
                         std::find(mmse::detail::kCrsSymbols.begin(), mmse::detail::kCrsSymbols.end(), symbol) -
                         mmse::detail::kCrsSymbols.begin())},
                    pilot);
                buffers.re[0][idx] = crs.re;
                buffers.im[0][idx] = crs.im;
            } else if (is_port1_crs) {
                const std::uint32_t pilot =
                    (sc - mmse::detail::crs_frequency_offset(desc.cell_id, 1U, static_cast<std::uint8_t>(symbol))) / 6U;
                const auto& crs = mmse::detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(subframe),
                     .port = 1U,
                     .crs_symbol_index = static_cast<std::uint8_t>(
                         std::find(mmse::detail::kCrsSymbols.begin(), mmse::detail::kCrsSymbols.end(), symbol) -
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
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kUnsupportedConfig, "unsupported descriptor should fail");
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

void test_two_layer_outputs_are_finite() {
    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig config{};
    config.worker_count = 1;
    expect_true(ctx.init(config) == MmseStatus::kOk, "init must succeed");

    auto desc = make_fullband_desc();
    GridBuffers buffers = make_zero_grid();
    fill_identity_channel(buffers, desc, 1.0F, 2.0F);
    auto grid = make_grid_view(buffers);

    const std::uint32_t cap = 20000U;
    std::vector<float> xre(cap * 2U);
    std::vector<float> xim(cap * 2U);
    std::vector<float> sinr(cap * 2U);
    EqualizerOutputView out{xre.data(), xim.data(), sinr.data(), cap};
    expect_true(ctx.run(grid, desc, out) == MmseStatus::kOk, "two-layer run");
    expect_true(std::isfinite(out.x_hat_re[0]), "finite layer0");
    expect_true(std::isfinite(out.x_hat_re[out.capacity_re_per_layer]), "finite layer1");
    expect_true(std::isfinite(out.sinr[0]), "finite sinr0");
    expect_true(std::isfinite(out.sinr[out.capacity_re_per_layer]), "finite sinr1");
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

}  // namespace

int main() {
    const std::array tests = {
        std::pair{"reject_invalid_descriptor", &test_reject_invalid_descriptor},
        std::pair{"buffer_too_small", &test_buffer_too_small},
        std::pair{"single_layer_identity_channel_equalization", &test_single_layer_identity_channel_equalization},
        std::pair{"sigma2_state_persists", &test_sigma2_state_persists},
        std::pair{"single_layer_path", &test_single_layer_path},
        std::pair{"two_layer_outputs_are_finite", &test_two_layer_outputs_are_finite},
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
