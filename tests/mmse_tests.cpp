#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "mmse/mmse_equalizer.h"
#include "mmse/pdcch_chain_dto.h"
#include "mmse/pdcch_module_api.h"
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

ExtractDescriptor make_pdcch_desc() {
    ExtractDescriptor desc{};
    desc.cell_id = 0;
    desc.n_tx_ports = 1;
    desc.n_rx_ant = 2;
    desc.n_layers = 1;
    desc.tx_mode = 1;
    desc.channel_type = MmseChannelType::kPdcch;
    desc.start_symbol = 0;
    desc.control_symbol_count = 3;
    desc.mod_order = 2;
    desc.n_prb = 100;
    desc.sfn_subframe = 3;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    desc.control_re_exclusion_masks.fill(0U);
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

void fill_identity_channel(GridBuffers& buffers, const ExtractDescriptor& desc, float data0,
                           float data1) {
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

void fill_pdcch_td_pair(GridBuffers& buffers, const ExtractDescriptor& desc, std::uint16_t grid0,
                        std::uint16_t grid1, Complex32 h00, Complex32 h01, Complex32 h10,
                        Complex32 h11, Complex32 s0, Complex32 s1) {
    const std::uint32_t symbol0 = grid0 / kLteNumSubcarriers20MHz;
    const std::uint32_t sc0 = grid0 % kLteNumSubcarriers20MHz;
    const std::uint32_t symbol1 = grid1 / kLteNumSubcarriers20MHz;
    const std::uint32_t sc1 = grid1 % kLteNumSubcarriers20MHz;
    expect_true(symbol0 == symbol1, "td pair must stay in one symbol");
    expect_true(sc1 == sc0 + 1U, "td pair must be adjacent");

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
    expect_true(n_re == 3400U, "pdcch RE count without extra reserved REs");
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
    expect_true(n_re_reserved == n_re - 4U, "reserved control REs must be excluded");
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
    expect_true(out.n_re_per_layer == 3400U, "single-port pdcch RE count");
    expect_true(out.mod_order == 2U, "pdcch mod order must stay qpsk");

    auto two_port = desc;
    two_port.n_tx_ports = 2U;
    two_port.tx_mode = 2U;
    expect_true(ctx.run(grid, two_port, out) == MmseStatus::kOk,
                "generic run should allow two-port pdcch backend path");
    expect_true(out.n_re_per_layer == 3200U, "two-port generic pdcch RE count");
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
    if (n_source_re != 3200U) {
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
    expect_true(meta.n_re == 3400U, "pdcch module api RE count");
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
    expect_true(reserved_n_re == base_n_re - 172U, "auto reserved RE helper should reduce layout");
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
        expect_near(gpu_sinr[i], cpu_sinr[i], 1.0e-4F, "gpu cuda td sinr");
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

    expect_true(cpu.run(grid, desc, cpu_out) == MmseStatus::kOk, "cpu td backend run");
    expect_true(gpu.run(grid, desc, gpu_out) == MmseStatus::kOk, "gpu td backend run");
    expect_true(cpu_out.n_re_per_layer == gpu_out.n_re_per_layer, "td backend re count");
    for (std::uint32_t i = 0; i < cpu_out.n_re_per_layer; ++i) {
        expect_near(gpu_re[i], cpu_re[i], 1.0e-4F, "gpu td backend xhat real");
        expect_near(gpu_im[i], cpu_im[i], 1.0e-4F, "gpu td backend xhat imag");
        expect_near(gpu_sinr[i], cpu_sinr[i], 1.0e-4F, "gpu td backend sinr");
    }
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

void test_gpu_context_invalid_stream_count_is_rejected() {
    MmseEqualizerGpuContext gpu;
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kAuto;
    config.stream_count = 0;
    expect_true(gpu.init(config) == MmseStatus::kInvalidArgument,
                "gpu init should reject zero stream count");
}

} // namespace

int main() {
    const std::array tests = {
        std::pair{"reject_invalid_descriptor", &test_reject_invalid_descriptor},
        std::pair{"buffer_too_small", &test_buffer_too_small},
        std::pair{"single_layer_identity_channel_equalization",
                  &test_single_layer_identity_channel_equalization},
        std::pair{"sigma2_state_persists", &test_sigma2_state_persists},
        std::pair{"single_layer_path", &test_single_layer_path},
        std::pair{"pdcch_layout_excludes_crs_and_reserved_res",
                  &test_pdcch_layout_excludes_crs_and_reserved_res},
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
