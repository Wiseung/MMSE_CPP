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
constexpr std::uint16_t kPdschRnti = 0x1234U;

volatile float g_llr_sink = 0.0F;

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

struct PdschEqualizedIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::array<std::uint16_t, 7> prb_bitmap{};
    std::uint32_t n_re_per_layer = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t mod_order = 0;
    std::int8_t pmi = -1;
    std::vector<float> x_hat_re{};
    std::vector<float> x_hat_im{};
    std::vector<float> sinr{};
};

struct StageBuffers {
    std::vector<float> pdsch_llrs;
    std::vector<float> pbch_llrs;
    std::vector<float> pcfich_llrs;
    std::vector<float> pdcch_llrs;
    pdsch::PdschDescrambledLlrResult pdsch_llr_result{};
    pdsch::PdschDescramblingPlanCache pdsch_plan{};
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

StageBuffers make_stage_buffers() {
    return StageBuffers{};
}

template <typename BackendEqualizedIndication>
void consume_equalized_result(const BackendEqualizedIndication& backend) {
    if (!backend.x_hat_re.empty()) {
        g_llr_sink += backend.x_hat_re.front();
        g_llr_sink += static_cast<float>(backend.x_hat_re.size());
    }
    if (!backend.sinr.empty()) {
        g_llr_sink += backend.sinr.front();
    }
}

template <typename BackendLlrIndication>
void consume_llr_result(const BackendLlrIndication& llr_backend) {
    if (!llr_backend.llrs.empty()) {
        g_llr_sink += llr_backend.llrs.front();
        g_llr_sink += static_cast<float>(llr_backend.llrs.size());
    }
}

void consume_llr_values(const std::vector<float>& llrs) {
    if (!llrs.empty()) {
        g_llr_sink += llrs.front();
        g_llr_sink += static_cast<float>(llrs.size());
    }
}

PdschEqualizedIndication make_pdsch_equalized_indication(const ExtractDescriptor& desc,
                                                         const EqualizerOutputView& out) {
    PdschEqualizedIndication backend{};
    backend.sfn_subframe = desc.sfn_subframe;
    backend.cell_id = desc.cell_id;
    backend.n_prb = desc.n_prb;
    backend.prb_bitmap = desc.prb_bitmap;
    backend.n_re_per_layer = out.n_re_per_layer;
    backend.n_tx_ports = desc.n_tx_ports;
    backend.n_rx_ant = desc.n_rx_ant;
    backend.n_layers = out.n_layers;
    backend.tx_mode = desc.tx_mode;
    backend.start_symbol = desc.start_symbol;
    backend.mod_order = out.mod_order;
    backend.pmi = desc.pmi;

    const std::size_t total_re =
        static_cast<std::size_t>(out.n_layers) * static_cast<std::size_t>(out.n_re_per_layer);
    backend.x_hat_re.assign(out.x_hat_re, out.x_hat_re + total_re);
    backend.x_hat_im.assign(out.x_hat_im, out.x_hat_im + total_re);
    backend.sinr.assign(out.sinr, out.sinr + total_re);
    return backend;
}

bool pack_pdsch_equalized_indication(const ExtractDescriptor& desc, const ChannelBuffers& buffers) {
    const auto backend = make_pdsch_equalized_indication(desc, buffers.pdsch_out);
    const std::size_t expected =
        static_cast<std::size_t>(buffers.pdsch_out.n_layers) * buffers.pdsch_out.n_re_per_layer;
    if (backend.x_hat_re.size() != expected || backend.x_hat_im.size() != expected ||
        backend.sinr.size() != expected) {
        return false;
    }
    consume_equalized_result(backend);
    return true;
}

bool pack_pbch_equalized_indication(const ChannelBuffers& buffers) {
    auto backend =
        pbch::make_backend_pbch_equalized_indication(buffers.pbch_meta, buffers.pbch_out);
    const std::size_t expected = static_cast<std::size_t>(buffers.pbch_meta.n_re);
    if (backend.x_hat_re.size() != expected || backend.x_hat_im.size() != expected ||
        backend.sinr.size() != expected) {
        return false;
    }
    consume_equalized_result(backend);
    return true;
}

bool pack_pcfich_equalized_indication(const ChannelBuffers& buffers) {
    auto backend =
        pcfich::make_backend_pcfich_equalized_indication(buffers.pcfich_meta, buffers.pcfich_out);
    const std::size_t expected = static_cast<std::size_t>(buffers.pcfich_meta.n_re);
    if (backend.x_hat_re.size() != expected || backend.x_hat_im.size() != expected ||
        backend.sinr.size() != expected) {
        return false;
    }
    consume_equalized_result(backend);
    return true;
}

bool pack_pdcch_equalized_indication(const ChannelBuffers& buffers) {
    auto backend =
        pdcch::make_backend_pdcch_equalized_indication(buffers.pdcch_meta, buffers.pdcch_out);
    const std::size_t expected = static_cast<std::size_t>(buffers.pdcch_meta.n_re);
    if (backend.x_hat_re.size() != expected || backend.x_hat_im.size() != expected ||
        backend.sinr.size() != expected) {
        return false;
    }
    consume_equalized_result(backend);
    return true;
}

bool build_pdsch_raw_llrs(const ChannelBuffers& buffers, StageBuffers& stage_buffers) {
    if (lte::build_max_log_llrs(buffers.pdsch_out.x_hat_re, buffers.pdsch_out.x_hat_im,
                                buffers.pdsch_out.sinr, buffers.pdsch_out.capacity_re_per_layer,
                                buffers.pdsch_out.n_re_per_layer, buffers.pdsch_out.n_layers,
                                buffers.pdsch_out.mod_order,
                                stage_buffers.pdsch_llrs) != MmseStatus::kOk) {
        return false;
    }
    const std::size_t expected =
        static_cast<std::size_t>(buffers.pdsch_out.n_layers) *
        lte::llr_count_per_layer(buffers.pdsch_out.n_re_per_layer, buffers.pdsch_out.mod_order);
    if (stage_buffers.pdsch_llrs.size() != expected) {
        return false;
    }
    consume_llr_values(stage_buffers.pdsch_llrs);
    return true;
}

bool build_pdsch_fused_llr_indication(const ExtractDescriptor& desc,
                                      const ChannelBuffers& buffers) {
    auto backend =
        pdsch::make_backend_pdsch_descrambled_llr_indication(desc, buffers.pdsch_out, kPdschRnti);
    const std::size_t expected =
        static_cast<std::size_t>(buffers.pdsch_out.n_layers) *
        lte::llr_count_per_layer(buffers.pdsch_out.n_re_per_layer, buffers.pdsch_out.mod_order);
    if (backend.llrs.size() != expected) {
        return false;
    }
    consume_llr_result(backend);
    return true;
}

bool build_pdsch_caller_owned_llr_output(const ExtractDescriptor& desc,
                                         const ChannelBuffers& buffers,
                                         StageBuffers& stage_buffers) {
    const std::uint32_t expected = lte::total_llr_count(
        buffers.pdsch_out.n_re_per_layer, buffers.pdsch_out.n_layers, buffers.pdsch_out.mod_order);
    if (stage_buffers.pdsch_llrs.size() != expected) {
        stage_buffers.pdsch_llrs.resize(expected);
    }
    pdsch::PdschDescrambledLlrOutputView llr_out{
        stage_buffers.pdsch_llrs.data(),
        static_cast<std::uint32_t>(stage_buffers.pdsch_llrs.size())};
    if (pdsch::build_backend_pdsch_descrambled_llr_result(
            desc, buffers.pdsch_out, kPdschRnti, llr_out, stage_buffers.pdsch_llr_result) !=
        MmseStatus::kOk) {
        return false;
    }
    if (stage_buffers.pdsch_llr_result.llr_count != expected) {
        return false;
    }
    consume_llr_values(stage_buffers.pdsch_llrs);
    return true;
}

bool build_pdsch_cached_fused_llr_indication(const ExtractDescriptor& desc,
                                             const ChannelBuffers& buffers,
                                             StageBuffers& stage_buffers) {
    auto backend = pdsch::make_backend_pdsch_descrambled_llr_indication(
        desc, buffers.pdsch_out, kPdschRnti, stage_buffers.pdsch_plan);
    const std::size_t expected =
        static_cast<std::size_t>(buffers.pdsch_out.n_layers) *
        lte::llr_count_per_layer(buffers.pdsch_out.n_re_per_layer, buffers.pdsch_out.mod_order);
    if (backend.llrs.size() != expected) {
        return false;
    }
    consume_llr_result(backend);
    return true;
}

bool build_pdsch_cached_caller_owned_llr_output(const ExtractDescriptor& desc,
                                                const ChannelBuffers& buffers,
                                                StageBuffers& stage_buffers) {
    const std::uint32_t expected = lte::total_llr_count(
        buffers.pdsch_out.n_re_per_layer, buffers.pdsch_out.n_layers, buffers.pdsch_out.mod_order);
    if (stage_buffers.pdsch_llrs.size() != expected) {
        stage_buffers.pdsch_llrs.resize(expected);
    }
    pdsch::PdschDescrambledLlrOutputView llr_out{
        stage_buffers.pdsch_llrs.data(),
        static_cast<std::uint32_t>(stage_buffers.pdsch_llrs.size())};
    if (pdsch::build_backend_pdsch_descrambled_llr_result(
            desc, buffers.pdsch_out, kPdschRnti, stage_buffers.pdsch_plan, llr_out,
            stage_buffers.pdsch_llr_result) != MmseStatus::kOk) {
        return false;
    }
    if (stage_buffers.pdsch_llr_result.llr_count != expected) {
        return false;
    }
    consume_llr_values(stage_buffers.pdsch_llrs);
    return true;
}

bool build_pbch_raw_llrs(const ChannelBuffers& buffers, StageBuffers& stage_buffers) {
    if (lte::build_max_log_llrs(buffers.pbch_out.x_hat_re, buffers.pbch_out.x_hat_im,
                                buffers.pbch_out.sinr, buffers.pbch_out.capacity_re_per_layer,
                                buffers.pbch_meta.n_re, buffers.pbch_meta.n_layers,
                                buffers.pbch_meta.mod_order,
                                stage_buffers.pbch_llrs) != MmseStatus::kOk) {
        return false;
    }
    const std::size_t expected = static_cast<std::size_t>(buffers.pbch_meta.n_re) *
                                 static_cast<std::size_t>(buffers.pbch_meta.mod_order);
    if (stage_buffers.pbch_llrs.size() != expected) {
        return false;
    }
    consume_llr_values(stage_buffers.pbch_llrs);
    return true;
}

bool build_pcfich_raw_llrs(const ChannelBuffers& buffers, StageBuffers& stage_buffers) {
    if (lte::build_max_log_llrs(buffers.pcfich_out.x_hat_re, buffers.pcfich_out.x_hat_im,
                                buffers.pcfich_out.sinr, buffers.pcfich_out.capacity_re_per_layer,
                                buffers.pcfich_meta.n_re, buffers.pcfich_meta.n_layers,
                                buffers.pcfich_meta.mod_order,
                                stage_buffers.pcfich_llrs) != MmseStatus::kOk) {
        return false;
    }
    const std::size_t expected = static_cast<std::size_t>(buffers.pcfich_meta.n_re) *
                                 static_cast<std::size_t>(buffers.pcfich_meta.mod_order);
    if (stage_buffers.pcfich_llrs.size() != expected) {
        return false;
    }
    consume_llr_values(stage_buffers.pcfich_llrs);
    return true;
}

bool build_pdcch_raw_llrs(const ChannelBuffers& buffers, StageBuffers& stage_buffers) {
    if (lte::build_max_log_llrs(buffers.pdcch_out.x_hat_re, buffers.pdcch_out.x_hat_im,
                                buffers.pdcch_out.sinr, buffers.pdcch_out.capacity_re_per_layer,
                                buffers.pdcch_meta.n_re, buffers.pdcch_meta.n_layers,
                                buffers.pdcch_meta.mod_order,
                                stage_buffers.pdcch_llrs) != MmseStatus::kOk) {
        return false;
    }
    const std::size_t expected = static_cast<std::size_t>(buffers.pdcch_meta.n_re) *
                                 static_cast<std::size_t>(buffers.pdcch_meta.mod_order);
    if (stage_buffers.pdcch_llrs.size() != expected) {
        return false;
    }
    consume_llr_values(stage_buffers.pdcch_llrs);
    return true;
}

bool descramble_pdsch_llrs(StageBuffers& stage_buffers, const ExtractDescriptor& desc) {
    lte::descramble_llrs_inplace(stage_buffers.pdsch_llrs.data(),
                                 static_cast<std::uint32_t>(stage_buffers.pdsch_llrs.size()),
                                 lte::pdsch_c_init(desc.cell_id, kPdschRnti, desc.sfn_subframe));
    consume_llr_values(stage_buffers.pdsch_llrs);
    return true;
}

bool descramble_pbch_llrs(StageBuffers& stage_buffers, const ChannelBuffers& buffers) {
    lte::descramble_llrs_inplace(stage_buffers.pbch_llrs.data(),
                                 static_cast<std::uint32_t>(stage_buffers.pbch_llrs.size()),
                                 lte::pbch_c_init(buffers.pbch_meta.cell_id));
    consume_llr_values(stage_buffers.pbch_llrs);
    return true;
}

bool descramble_pcfich_llrs(StageBuffers& stage_buffers, const ChannelBuffers& buffers) {
    lte::descramble_llrs_inplace(
        stage_buffers.pcfich_llrs.data(),
        static_cast<std::uint32_t>(stage_buffers.pcfich_llrs.size()),
        lte::pcfich_c_init(buffers.pcfich_meta.cell_id, buffers.pcfich_meta.sfn_subframe));
    consume_llr_values(stage_buffers.pcfich_llrs);
    return true;
}

bool descramble_pdcch_llrs(StageBuffers& stage_buffers, const ChannelBuffers& buffers) {
    lte::descramble_llrs_inplace(
        stage_buffers.pdcch_llrs.data(),
        static_cast<std::uint32_t>(stage_buffers.pdcch_llrs.size()),
        lte::pdcch_c_init(buffers.pdcch_meta.cell_id, buffers.pdcch_meta.sfn_subframe));
    consume_llr_values(stage_buffers.pdcch_llrs);
    return true;
}

bool prepare_raw_llrs(const ExtractDescriptor& pdsch_desc, const ChannelBuffers& buffers,
                      StageBuffers& stage_buffers) {
    (void)pdsch_desc;
    return build_pdsch_raw_llrs(buffers, stage_buffers) &&
           build_pbch_raw_llrs(buffers, stage_buffers) &&
           build_pcfich_raw_llrs(buffers, stage_buffers) &&
           build_pdcch_raw_llrs(buffers, stage_buffers);
}

template <typename T> void summarize(std::string_view label, const std::vector<T>& values);

template <typename RunPdschFn, typename RunPbchFn, typename RunPcfichFn, typename RunPdcchFn>
bool measure_stage_path(std::string_view prefix, RunPdschFn&& run_pdsch, RunPbchFn&& run_pbch,
                        RunPcfichFn&& run_pcfich, RunPdcchFn&& run_pdcch,
                        const ChannelBuffers& buffers) {
    return measure_path(prefix, std::forward<RunPdschFn>(run_pdsch),
                        std::forward<RunPbchFn>(run_pbch), std::forward<RunPcfichFn>(run_pcfich),
                        std::forward<RunPdcchFn>(run_pdcch), buffers);
}

template <typename RunFn> bool measure_single_path(std::string_view prefix, RunFn&& run_fn) {
    std::vector<double> timings_us;
    timings_us.reserve(kMeasureIters);
    for (std::uint32_t i = 0; i < kMeasureIters; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        if (!run_fn()) {
            return false;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        timings_us.push_back(
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start)
                .count());
    }
    summarize(std::string(prefix), timings_us);
    return true;
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
    StageBuffers cpu_stage_buffers = make_stage_buffers();
    StageBuffers gpu_stage_buffers = make_stage_buffers();

    const auto refresh_cpu_outputs = [&] {
        return cpu_ctx.run(grid, pdsch_desc, cpu_buffers.pdsch_out) == MmseStatus::kOk &&
               cpu_ctx.run_pbch(pbch_in, cpu_buffers.pbch_out, cpu_buffers.pbch_meta) ==
                   MmseStatus::kOk &&
               cpu_ctx.run_pcfich(pcfich_in, cpu_buffers.pcfich_out, cpu_buffers.pcfich_meta) ==
                   MmseStatus::kOk &&
               cpu_ctx.run_pdcch(pdcch_in, cpu_buffers.pdcch_out, cpu_buffers.pdcch_meta) ==
                   MmseStatus::kOk;
    };
    const auto refresh_gpu_outputs = [&] {
        return gpu_ctx.run(grid, pdsch_desc, gpu_buffers.pdsch_out) == MmseStatus::kOk &&
               gpu_ctx.run_pbch(pbch_in, gpu_buffers.pbch_out, gpu_buffers.pbch_meta) ==
                   MmseStatus::kOk &&
               gpu_ctx.run_pcfich(pcfich_in, gpu_buffers.pcfich_out, gpu_buffers.pcfich_meta) ==
                   MmseStatus::kOk &&
               gpu_ctx.run_pdcch(pdcch_in, gpu_buffers.pdcch_out, gpu_buffers.pdcch_meta) ==
                   MmseStatus::kOk;
    };

    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!refresh_cpu_outputs()) {
            std::cerr << "cpu warmup failed\n";
            return 1;
        }
        if (!refresh_gpu_outputs()) {
            std::cerr << "gpu warmup failed\n";
            return 1;
        }
    }

    if (!measure_path(
            "cpu.eq_only",
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

    if (!refresh_cpu_outputs()) {
        std::cerr << "cpu refresh before pack_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!pack_pdsch_equalized_indication(pdsch_desc, cpu_buffers) ||
            !pack_pbch_equalized_indication(cpu_buffers) ||
            !pack_pcfich_equalized_indication(cpu_buffers) ||
            !pack_pdcch_equalized_indication(cpu_buffers)) {
            std::cerr << "cpu pack_only warmup failed\n";
            return 1;
        }
    }

    if (!measure_stage_path(
            "cpu.pack_only",
            [&] { return pack_pdsch_equalized_indication(pdsch_desc, cpu_buffers); },
            [&] { return pack_pbch_equalized_indication(cpu_buffers); },
            [&] { return pack_pcfich_equalized_indication(cpu_buffers); },
            [&] { return pack_pdcch_equalized_indication(cpu_buffers); }, cpu_buffers)) {
        std::cerr << "cpu pack_only measurement failed\n";
        return 1;
    }

    if (!refresh_cpu_outputs()) {
        std::cerr << "cpu refresh before soft_demod_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_raw_llrs(cpu_buffers, cpu_stage_buffers) ||
            !build_pbch_raw_llrs(cpu_buffers, cpu_stage_buffers) ||
            !build_pcfich_raw_llrs(cpu_buffers, cpu_stage_buffers) ||
            !build_pdcch_raw_llrs(cpu_buffers, cpu_stage_buffers)) {
            std::cerr << "cpu soft_demod_only warmup failed\n";
            return 1;
        }
    }

    if (!measure_stage_path(
            "cpu.soft_demod_only",
            [&] { return build_pdsch_raw_llrs(cpu_buffers, cpu_stage_buffers); },
            [&] { return build_pbch_raw_llrs(cpu_buffers, cpu_stage_buffers); },
            [&] { return build_pcfich_raw_llrs(cpu_buffers, cpu_stage_buffers); },
            [&] { return build_pdcch_raw_llrs(cpu_buffers, cpu_stage_buffers); }, cpu_buffers)) {
        std::cerr << "cpu soft_demod_only measurement failed\n";
        return 1;
    }

    if (!refresh_cpu_outputs() || !prepare_raw_llrs(pdsch_desc, cpu_buffers, cpu_stage_buffers)) {
        std::cerr << "cpu prepare descramble_only inputs failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!descramble_pdsch_llrs(cpu_stage_buffers, pdsch_desc) ||
            !descramble_pbch_llrs(cpu_stage_buffers, cpu_buffers) ||
            !descramble_pcfich_llrs(cpu_stage_buffers, cpu_buffers) ||
            !descramble_pdcch_llrs(cpu_stage_buffers, cpu_buffers)) {
            std::cerr << "cpu descramble_only warmup failed\n";
            return 1;
        }
    }
    if (!prepare_raw_llrs(pdsch_desc, cpu_buffers, cpu_stage_buffers)) {
        std::cerr << "cpu refresh descramble_only inputs failed\n";
        return 1;
    }

    if (!measure_stage_path(
            "cpu.descramble_only",
            [&] { return descramble_pdsch_llrs(cpu_stage_buffers, pdsch_desc); },
            [&] { return descramble_pbch_llrs(cpu_stage_buffers, cpu_buffers); },
            [&] { return descramble_pcfich_llrs(cpu_stage_buffers, cpu_buffers); },
            [&] { return descramble_pdcch_llrs(cpu_stage_buffers, cpu_buffers); }, cpu_buffers)) {
        std::cerr << "cpu descramble_only measurement failed\n";
        return 1;
    }

    if (!refresh_cpu_outputs()) {
        std::cerr << "cpu refresh before pdsch_fused_llr_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_fused_llr_indication(pdsch_desc, cpu_buffers)) {
            std::cerr << "cpu pdsch_fused_llr_only warmup failed\n";
            return 1;
        }
    }
    if (!measure_single_path("cpu.pdsch_fused_llr_only", [&] {
            return build_pdsch_fused_llr_indication(pdsch_desc, cpu_buffers);
        })) {
        std::cerr << "cpu pdsch_fused_llr_only measurement failed\n";
        return 1;
    }

    if (!refresh_cpu_outputs()) {
        std::cerr << "cpu refresh before pdsch_caller_owned_llr_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_caller_owned_llr_output(pdsch_desc, cpu_buffers, cpu_stage_buffers)) {
            std::cerr << "cpu pdsch_caller_owned_llr_only warmup failed\n";
            return 1;
        }
    }
    if (!measure_single_path("cpu.pdsch_caller_owned_llr_only", [&] {
            return build_pdsch_caller_owned_llr_output(pdsch_desc, cpu_buffers, cpu_stage_buffers);
        })) {
        std::cerr << "cpu pdsch_caller_owned_llr_only measurement failed\n";
        return 1;
    }

    if (!refresh_cpu_outputs()) {
        std::cerr << "cpu refresh before pdsch_cached_fused_llr_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_cached_fused_llr_indication(pdsch_desc, cpu_buffers, cpu_stage_buffers)) {
            std::cerr << "cpu pdsch_cached_fused_llr_only warmup failed\n";
            return 1;
        }
    }
    if (!measure_single_path("cpu.pdsch_cached_fused_llr_only", [&] {
            return build_pdsch_cached_fused_llr_indication(pdsch_desc, cpu_buffers,
                                                           cpu_stage_buffers);
        })) {
        std::cerr << "cpu pdsch_cached_fused_llr_only measurement failed\n";
        return 1;
    }

    if (!refresh_cpu_outputs()) {
        std::cerr << "cpu refresh before pdsch_cached_caller_owned_llr_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_cached_caller_owned_llr_output(pdsch_desc, cpu_buffers,
                                                        cpu_stage_buffers)) {
            std::cerr << "cpu pdsch_cached_caller_owned_llr_only warmup failed\n";
            return 1;
        }
    }
    if (!measure_single_path("cpu.pdsch_cached_caller_owned_llr_only", [&] {
            return build_pdsch_cached_caller_owned_llr_output(pdsch_desc, cpu_buffers,
                                                              cpu_stage_buffers);
        })) {
        std::cerr << "cpu pdsch_cached_caller_owned_llr_only measurement failed\n";
        return 1;
    }

    if (!measure_path(
            "gpu.eq_only",
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

    if (!refresh_gpu_outputs()) {
        std::cerr << "gpu refresh before pack_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!pack_pdsch_equalized_indication(pdsch_desc, gpu_buffers) ||
            !pack_pbch_equalized_indication(gpu_buffers) ||
            !pack_pcfich_equalized_indication(gpu_buffers) ||
            !pack_pdcch_equalized_indication(gpu_buffers)) {
            std::cerr << "gpu pack_only warmup failed\n";
            return 1;
        }
    }

    if (!measure_stage_path(
            "gpu.pack_only",
            [&] { return pack_pdsch_equalized_indication(pdsch_desc, gpu_buffers); },
            [&] { return pack_pbch_equalized_indication(gpu_buffers); },
            [&] { return pack_pcfich_equalized_indication(gpu_buffers); },
            [&] { return pack_pdcch_equalized_indication(gpu_buffers); }, gpu_buffers)) {
        std::cerr << "gpu pack_only measurement failed\n";
        return 1;
    }

    if (!refresh_gpu_outputs()) {
        std::cerr << "gpu refresh before soft_demod_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_raw_llrs(gpu_buffers, gpu_stage_buffers) ||
            !build_pbch_raw_llrs(gpu_buffers, gpu_stage_buffers) ||
            !build_pcfich_raw_llrs(gpu_buffers, gpu_stage_buffers) ||
            !build_pdcch_raw_llrs(gpu_buffers, gpu_stage_buffers)) {
            std::cerr << "gpu soft_demod_only warmup failed\n";
            return 1;
        }
    }

    if (!measure_stage_path(
            "gpu.soft_demod_only",
            [&] { return build_pdsch_raw_llrs(gpu_buffers, gpu_stage_buffers); },
            [&] { return build_pbch_raw_llrs(gpu_buffers, gpu_stage_buffers); },
            [&] { return build_pcfich_raw_llrs(gpu_buffers, gpu_stage_buffers); },
            [&] { return build_pdcch_raw_llrs(gpu_buffers, gpu_stage_buffers); }, gpu_buffers)) {
        std::cerr << "gpu soft_demod_only measurement failed\n";
        return 1;
    }

    if (!refresh_gpu_outputs() || !prepare_raw_llrs(pdsch_desc, gpu_buffers, gpu_stage_buffers)) {
        std::cerr << "gpu prepare descramble_only inputs failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!descramble_pdsch_llrs(gpu_stage_buffers, pdsch_desc) ||
            !descramble_pbch_llrs(gpu_stage_buffers, gpu_buffers) ||
            !descramble_pcfich_llrs(gpu_stage_buffers, gpu_buffers) ||
            !descramble_pdcch_llrs(gpu_stage_buffers, gpu_buffers)) {
            std::cerr << "gpu descramble_only warmup failed\n";
            return 1;
        }
    }
    if (!prepare_raw_llrs(pdsch_desc, gpu_buffers, gpu_stage_buffers)) {
        std::cerr << "gpu refresh descramble_only inputs failed\n";
        return 1;
    }

    if (!measure_stage_path(
            "gpu.descramble_only",
            [&] { return descramble_pdsch_llrs(gpu_stage_buffers, pdsch_desc); },
            [&] { return descramble_pbch_llrs(gpu_stage_buffers, gpu_buffers); },
            [&] { return descramble_pcfich_llrs(gpu_stage_buffers, gpu_buffers); },
            [&] { return descramble_pdcch_llrs(gpu_stage_buffers, gpu_buffers); }, gpu_buffers)) {
        std::cerr << "gpu descramble_only measurement failed\n";
        return 1;
    }

    if (!refresh_gpu_outputs()) {
        std::cerr << "gpu refresh before pdsch_fused_llr_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_fused_llr_indication(pdsch_desc, gpu_buffers)) {
            std::cerr << "gpu pdsch_fused_llr_only warmup failed\n";
            return 1;
        }
    }
    if (!measure_single_path("gpu.pdsch_fused_llr_only", [&] {
            return build_pdsch_fused_llr_indication(pdsch_desc, gpu_buffers);
        })) {
        std::cerr << "gpu pdsch_fused_llr_only measurement failed\n";
        return 1;
    }

    if (!refresh_gpu_outputs()) {
        std::cerr << "gpu refresh before pdsch_caller_owned_llr_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_caller_owned_llr_output(pdsch_desc, gpu_buffers, gpu_stage_buffers)) {
            std::cerr << "gpu pdsch_caller_owned_llr_only warmup failed\n";
            return 1;
        }
    }
    if (!measure_single_path("gpu.pdsch_caller_owned_llr_only", [&] {
            return build_pdsch_caller_owned_llr_output(pdsch_desc, gpu_buffers, gpu_stage_buffers);
        })) {
        std::cerr << "gpu pdsch_caller_owned_llr_only measurement failed\n";
        return 1;
    }

    if (!refresh_gpu_outputs()) {
        std::cerr << "gpu refresh before pdsch_cached_fused_llr_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_cached_fused_llr_indication(pdsch_desc, gpu_buffers, gpu_stage_buffers)) {
            std::cerr << "gpu pdsch_cached_fused_llr_only warmup failed\n";
            return 1;
        }
    }
    if (!measure_single_path("gpu.pdsch_cached_fused_llr_only", [&] {
            return build_pdsch_cached_fused_llr_indication(pdsch_desc, gpu_buffers,
                                                           gpu_stage_buffers);
        })) {
        std::cerr << "gpu pdsch_cached_fused_llr_only measurement failed\n";
        return 1;
    }

    if (!refresh_gpu_outputs()) {
        std::cerr << "gpu refresh before pdsch_cached_caller_owned_llr_only failed\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < kWarmupIters; ++i) {
        if (!build_pdsch_cached_caller_owned_llr_output(pdsch_desc, gpu_buffers,
                                                        gpu_stage_buffers)) {
            std::cerr << "gpu pdsch_cached_caller_owned_llr_only warmup failed\n";
            return 1;
        }
    }
    if (!measure_single_path("gpu.pdsch_cached_caller_owned_llr_only", [&] {
            return build_pdsch_cached_caller_owned_llr_output(pdsch_desc, gpu_buffers,
                                                              gpu_stage_buffers);
        })) {
        std::cerr << "gpu pdsch_cached_caller_owned_llr_only measurement failed\n";
        return 1;
    }

    return 0;
}
