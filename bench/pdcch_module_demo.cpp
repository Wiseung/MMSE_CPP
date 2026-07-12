#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "mmse/pdcch_chain_sdk.h"

using namespace mmse;

namespace {

struct GridBuffers {
    std::array<std::vector<float>, 2> re;
    std::array<std::vector<float>, 2> im;
};

GridBuffers make_zero_grid() {
    GridBuffers buffers;
    for (std::uint32_t rx = 0; rx < kMmseV1MaxNumRxAntennas; ++rx) {
        buffers.re[rx].assign(kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz, 0.0F);
        buffers.im[rx].assign(kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz, 0.0F);
    }
    return buffers;
}

PlanarGridViewF32 make_grid_view(const GridBuffers& buffers) {
    PlanarGridViewF32 grid{};
    grid.re = {buffers.re[0].data(), buffers.re[1].data()};
    grid.im = {buffers.im[0].data(), buffers.im[1].data()};
    grid.n_rx_ant = kMmseV1MaxNumRxAntennas;
    grid.n_symbols = kLteNumSymbolsNormalCp;
    grid.n_subcarriers = kLteNumSubcarriers20MHz;
    return grid;
}

void fill_demo_fft_grid(GridBuffers& buffers, std::uint32_t control_symbol_count) {
    for (std::uint32_t symbol = 0; symbol < control_symbol_count; ++symbol) {
        for (std::uint32_t sc = 0; sc < kLteNumSubcarriers20MHz; ++sc) {
            const std::size_t idx = static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz + sc;
            buffers.re[0][idx] = 0.5F;
            buffers.im[0][idx] = 0.1F;
            buffers.re[1][idx] = 0.45F;
            buffers.im[1][idx] = -0.05F;
        }
    }

    // Give CRS-bearing symbols deterministic non-zero pilots so the demo run has a non-zero
    // channel estimate path without depending on any external waveform generator.
    for (std::uint32_t sc = 0; sc < kLteNumSubcarriers20MHz; sc += 6U) {
        for (std::uint32_t symbol : {0U, 4U, 7U, 11U}) {
            const std::size_t idx = static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz + sc;
            buffers.re[0][idx] = 1.0F;
            buffers.im[0][idx] = 0.0F;
            buffers.re[1][idx] = 1.0F;
            buffers.im[1][idx] = 0.0F;
        }
    }
}

mmse::pdcch::FrontendPdcchIndication make_mock_frontend_indication() {
    mmse::pdcch::FrontendPdcchIndication ind{};
    ind.sfn_subframe = 123U;
    ind.cell_id = 11U;
    ind.n_tx_ports = 1U;
    ind.tx_mode = 1U;
    ind.control_symbol_count = 3U;
    ind.n_prb = kLteNumPrb20MHz;
    ind.prb_bitmap.fill(0xFFFFU);
    ind.prb_bitmap.back() = 0x000FU;
    ind.chain.request_id = 1001U;
    ind.chain.candidate_id = 7U;
    ind.chain.first_cce = 4U;
    ind.chain.aggregation_level = 8U;

    // Mock upstream policy: use helper-generated non-PDCCH control RE reservations.
    ind.control_subframe = {.duplex_mode = mmse::pdcch::PhichDuplexMode::kFdd,
                            .subframe = static_cast<std::uint8_t>(ind.sfn_subframe % 10U),
                            .ul_dl_config = 0U,
                            .kind = mmse::pdcch::LteControlSubframeKind::kRegular};
    mmse::pdcch::append_pcfich_reserved_control_re_list(ind);
    (void)mmse::pdcch::append_phich_reserved_control_re_list(
        ind, {.resource = mmse::pdcch::PhichResource::kOne,
              .duration = mmse::pdcch::PhichDuration::kNormal,
              .mi = 1U,
              .subframe_ctx = ind.control_subframe});
    return ind;
}

void fill_demo_input(PdcchMmseInput& in, const PlanarGridViewF32& grid,
                     const mmse::pdcch::FrontendPdcchIndication& frontend) {
    in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);
}

void print_result_sample(const PdcchMmseResult& meta, const PdcchMmseOutputView& out,
                         std::uint32_t sample_count) {
    const mmse::pdcch::BackendPdcchEqualizedIndication backend =
        pdcch::make_backend_pdcch_equalized_indication(meta, out);
    const mmse::pdcch::BackendPdcchDescrambledLlrIndication llr_backend =
        pdcch::make_backend_pdcch_descrambled_llr_indication(backend);
    const std::uint32_t count =
        std::min(static_cast<std::uint32_t>(backend.re_grid_indices.size()), sample_count);
    std::cout << "meta.n_re=" << backend.re_grid_indices.size() << '\n';
    std::cout << "meta.control_symbol_count="
              << static_cast<std::uint32_t>(backend.control_symbol_count) << '\n';
    std::cout << "meta.sigma2=" << std::setprecision(6) << backend.sigma2 << '\n';
    std::cout << "meta.chain.request_id=" << backend.chain.request_id << '\n';
    std::cout << "meta.chain.candidate_id=" << backend.chain.candidate_id << '\n';
    std::cout << "meta.chain.first_cce=" << backend.chain.first_cce << '\n';
    std::cout << "meta.chain.aggregation_level="
              << static_cast<std::uint32_t>(backend.chain.aggregation_level) << '\n';

    for (std::uint32_t i = 0; i < count; ++i) {
        const auto coord = pdcch::decode_re_grid_index(backend.re_grid_indices[i]);
        std::cout << "re[" << i << "] "
                  << "symbol=" << coord.symbol << " "
                  << "subcarrier=" << coord.subcarrier << " "
                  << "prb=" << coord.prb << " "
                  << "tone=" << coord.tone_in_prb << " "
                  << "xhat=(" << backend.x_hat_re[i] << "," << backend.x_hat_im[i] << ") "
                  << "sinr=" << backend.sinr[i] << '\n';
    }

    const std::uint32_t llr_count =
        std::min(static_cast<std::uint32_t>(llr_backend.llrs.size()), sample_count * 2U);
    for (std::uint32_t i = 0; i < llr_count; ++i) {
        std::cout << "llr[" << i << "]=" << llr_backend.llrs[i] << '\n';
    }
}

} // namespace

int main() {
    GridBuffers buffers = make_zero_grid();
    fill_demo_fft_grid(buffers, 3U);
    const PlanarGridViewF32 grid = make_grid_view(buffers);
    const mmse::pdcch::FrontendPdcchIndication frontend = make_mock_frontend_indication();

    PdcchMmseInput in{};
    fill_demo_input(in, grid, frontend);
    const MmseStatus validate_status = mmse::pdcch::validate_pdcch_mmse_input(in);
    if (validate_status != MmseStatus::kOk) {
        std::cerr << "validate_pdcch_mmse_input failed: " << to_string(validate_status) << '\n';
        return 1;
    }

    constexpr std::uint32_t kCapacity = 4000U;
    std::vector<float> xhat_re(kCapacity);
    std::vector<float> xhat_im(kCapacity);
    std::vector<float> sinr(kCapacity);
    std::vector<std::uint16_t> re_grid_indices(kCapacity);

    PdcchMmseOutputView out{};
    out.x_hat_re = xhat_re.data();
    out.x_hat_im = xhat_im.data();
    out.sinr = sinr.data();
    out.re_grid_indices = re_grid_indices.data();
    out.capacity_re_per_layer = kCapacity;
    out.capacity_re_metadata = kCapacity;

    PdcchMmseResult meta{};

    MmseEqualizerCpuContext ctx;
    MmseEqualizerCpuConfig cfg{};
    cfg.worker_count = 1U;
    const MmseStatus init_status = ctx.init(cfg);
    if (init_status != MmseStatus::kOk) {
        std::cerr << "init failed: " << to_string(init_status) << '\n';
        return 1;
    }

    const MmseStatus run_status = ctx.run_pdcch(in, out, meta);
    if (run_status != MmseStatus::kOk) {
        std::cerr << "run_pdcch failed: " << to_string(run_status) << '\n';
        return 1;
    }

    print_result_sample(meta, out, 8U);
    return 0;
}
