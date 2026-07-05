#pragma once

#include "mmse/lte_descrambling.h"
#include "mmse/lte_soft_demod.h"
#include "mmse/pdsch_chain_dto.h"
#include "mmse/types.h"

namespace mmse::pdsch {

inline MmseStatus prepare_pdsch_descrambling_plan(std::uint16_t cell_id, std::uint16_t rnti,
                                                  std::uint32_t sfn_subframe, std::uint8_t codeword,
                                                  std::uint32_t llr_count,
                                                  PdschDescramblingPlanCache& plan) {
    const std::uint32_t c_init = mmse::lte::pdsch_c_init(cell_id, rnti, sfn_subframe, codeword);
    if (plan.valid && plan.c_init == c_init && plan.llr_count == llr_count) {
        return MmseStatus::kOk;
    }
    if (plan.bits.size() != llr_count) {
        plan.bits.resize(llr_count);
    }
    mmse::lte::generate_scrambling_bits(c_init, plan.bits.data(), llr_count);
    plan.c_init = c_init;
    plan.llr_count = llr_count;
    plan.valid = true;
    return MmseStatus::kOk;
}

inline MmseStatus build_backend_pdsch_descrambled_llr_result(const ExtractDescriptor& desc,
                                                             const EqualizerOutputView& out,
                                                             std::uint16_t rnti,
                                                             PdschDescrambledLlrOutputView& llr_out,
                                                             PdschDescrambledLlrResult& result,
                                                             std::uint8_t codeword = 0U) {
    result = {};
    result.sfn_subframe = desc.sfn_subframe;
    result.cell_id = desc.cell_id;
    result.n_prb = desc.n_prb;
    result.rnti = rnti;
    result.prb_bitmap = desc.prb_bitmap;
    result.n_re_per_layer = out.n_re_per_layer;
    result.llr_count_per_layer = mmse::lte::llr_count_per_layer(out.n_re_per_layer, out.mod_order);
    result.llr_count = mmse::lte::total_llr_count(out.n_re_per_layer, out.n_layers, out.mod_order);
    result.n_tx_ports = desc.n_tx_ports;
    result.n_rx_ant = desc.n_rx_ant;
    result.n_layers = out.n_layers;
    result.tx_mode = desc.tx_mode;
    result.start_symbol = desc.start_symbol;
    result.mod_order = out.mod_order;
    result.codeword = codeword;
    result.pmi = desc.pmi;

    return mmse::lte::build_max_log_descrambled_llrs(
        out.x_hat_re, out.x_hat_im, out.sinr, out.capacity_re_per_layer, out.n_re_per_layer,
        out.n_layers, out.mod_order,
        mmse::lte::pdsch_c_init(desc.cell_id, rnti, desc.sfn_subframe, codeword), llr_out.llrs,
        llr_out.capacity_llrs);
}

inline MmseStatus build_backend_pdsch_descrambled_llr_result(
    const ExtractDescriptor& desc, const EqualizerOutputView& out, std::uint16_t rnti,
    PdschDescramblingPlanCache& plan, PdschDescrambledLlrOutputView& llr_out,
    PdschDescrambledLlrResult& result, std::uint8_t codeword = 0U) {
    result = {};
    result.sfn_subframe = desc.sfn_subframe;
    result.cell_id = desc.cell_id;
    result.n_prb = desc.n_prb;
    result.rnti = rnti;
    result.prb_bitmap = desc.prb_bitmap;
    result.n_re_per_layer = out.n_re_per_layer;
    result.llr_count_per_layer = mmse::lte::llr_count_per_layer(out.n_re_per_layer, out.mod_order);
    result.llr_count = mmse::lte::total_llr_count(out.n_re_per_layer, out.n_layers, out.mod_order);
    result.n_tx_ports = desc.n_tx_ports;
    result.n_rx_ant = desc.n_rx_ant;
    result.n_layers = out.n_layers;
    result.tx_mode = desc.tx_mode;
    result.start_symbol = desc.start_symbol;
    result.mod_order = out.mod_order;
    result.codeword = codeword;
    result.pmi = desc.pmi;

    if (const MmseStatus status = prepare_pdsch_descrambling_plan(
            desc.cell_id, rnti, desc.sfn_subframe, codeword, result.llr_count, plan);
        status != MmseStatus::kOk) {
        return status;
    }

    return mmse::lte::build_max_log_descrambled_llrs(
        out.x_hat_re, out.x_hat_im, out.sinr, out.capacity_re_per_layer, out.n_re_per_layer,
        out.n_layers, out.mod_order, plan.bits.data(), static_cast<std::uint32_t>(plan.bits.size()),
        llr_out.llrs, llr_out.capacity_llrs);
}

inline BackendPdschDescrambledLlrIndication
make_backend_pdsch_descrambled_llr_indication(const ExtractDescriptor& desc,
                                              const EqualizerOutputView& out, std::uint16_t rnti,
                                              std::uint8_t codeword = 0U) {
    BackendPdschDescrambledLlrIndication backend{};
    backend.sfn_subframe = desc.sfn_subframe;
    backend.cell_id = desc.cell_id;
    backend.n_prb = desc.n_prb;
    backend.rnti = rnti;
    backend.prb_bitmap = desc.prb_bitmap;
    backend.n_re_per_layer = out.n_re_per_layer;
    backend.llr_count_per_layer = mmse::lte::llr_count_per_layer(out.n_re_per_layer, out.mod_order);
    backend.n_tx_ports = desc.n_tx_ports;
    backend.n_rx_ant = desc.n_rx_ant;
    backend.n_layers = out.n_layers;
    backend.tx_mode = desc.tx_mode;
    backend.start_symbol = desc.start_symbol;
    backend.mod_order = out.mod_order;
    backend.codeword = codeword;
    backend.pmi = desc.pmi;

    backend.llrs.resize(
        mmse::lte::total_llr_count(out.n_re_per_layer, out.n_layers, out.mod_order));
    PdschDescrambledLlrOutputView llr_out{backend.llrs.data(),
                                          static_cast<std::uint32_t>(backend.llrs.size())};
    PdschDescrambledLlrResult result{};
    (void)build_backend_pdsch_descrambled_llr_result(desc, out, rnti, llr_out, result, codeword);
    return backend;
}

inline BackendPdschDescrambledLlrIndication make_backend_pdsch_descrambled_llr_indication(
    const ExtractDescriptor& desc, const EqualizerOutputView& out, std::uint16_t rnti,
    PdschDescramblingPlanCache& plan, std::uint8_t codeword = 0U) {
    BackendPdschDescrambledLlrIndication backend{};
    backend.sfn_subframe = desc.sfn_subframe;
    backend.cell_id = desc.cell_id;
    backend.n_prb = desc.n_prb;
    backend.rnti = rnti;
    backend.prb_bitmap = desc.prb_bitmap;
    backend.n_re_per_layer = out.n_re_per_layer;
    backend.llr_count_per_layer = mmse::lte::llr_count_per_layer(out.n_re_per_layer, out.mod_order);
    backend.n_tx_ports = desc.n_tx_ports;
    backend.n_rx_ant = desc.n_rx_ant;
    backend.n_layers = out.n_layers;
    backend.tx_mode = desc.tx_mode;
    backend.start_symbol = desc.start_symbol;
    backend.mod_order = out.mod_order;
    backend.codeword = codeword;
    backend.pmi = desc.pmi;

    backend.llrs.resize(
        mmse::lte::total_llr_count(out.n_re_per_layer, out.n_layers, out.mod_order));
    PdschDescrambledLlrOutputView llr_out{backend.llrs.data(),
                                          static_cast<std::uint32_t>(backend.llrs.size())};
    PdschDescrambledLlrResult result{};
    (void)build_backend_pdsch_descrambled_llr_result(desc, out, rnti, plan, llr_out, result,
                                                     codeword);
    return backend;
}

} // namespace mmse::pdsch
