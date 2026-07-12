#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

#include "mmse/constants.h"
#include "mmse/lte_descrambling.h"
#include "mmse/lte_soft_demod.h"
#include "mmse/pdcch_chain_dto.h"
#include "mmse/types.h"

namespace mmse::pdcch {

MmseStatus decode_pdcch_tail_biting_convolutional(const PdcchRateRecoveredLlr& recovered,
                                                  std::vector<std::uint8_t>& decoded_bits);

using PhichSubframeContext = LteControlSubframeContext;

struct PhichReservationConfig {
    PhichResource resource = PhichResource::kOne;
    PhichDuration duration = PhichDuration::kNormal;
    std::uint8_t mi = 1;
    LteControlSubframeContext subframe_ctx{};
};

struct ReCoord {
    std::uint32_t symbol = 0;
    std::uint32_t subcarrier = 0;
    std::uint32_t prb = 0;
    std::uint32_t tone_in_prb = 0;
};

namespace detail {

struct ControlRegCoord {
    std::uint32_t symbol = 0;
    std::uint32_t prb = 0;
    std::uint32_t reg_in_symbol_prb = 0;
};

inline bool same_reserved_control_re(const ReservedControlRe& lhs, const ReservedControlRe& rhs) {
    return lhs.symbol == rhs.symbol && lhs.prb == rhs.prb && lhs.tone_in_prb == rhs.tone_in_prb;
}

inline void append_reserved_control_re_unique(std::vector<ReservedControlRe>& reserved_control_res,
                                              const ReservedControlRe& re) {
    for (const auto& existing : reserved_control_res) {
        if (same_reserved_control_re(existing, re)) {
            return;
        }
    }
    reserved_control_res.push_back(re);
}

inline void append_reg_reserved_res(std::vector<ReservedControlRe>& reserved_control_res,
                                    std::uint16_t cell_id, const ControlRegCoord& reg) {
    if (reg.symbol == 0U) {
        const std::uint32_t vo = cell_id % 3U;
        const std::uint32_t tone_base = reg.reg_in_symbol_prb * 6U;
        for (std::uint32_t local_tone = 0; local_tone < 6U; ++local_tone) {
            if (local_tone == vo || local_tone == vo + 3U) {
                continue;
            }
            append_reserved_control_re_unique(
                reserved_control_res,
                {.symbol = reg.symbol, .prb = reg.prb, .tone_in_prb = tone_base + local_tone});
        }
        return;
    }

    const std::uint32_t tone_base = reg.reg_in_symbol_prb * 4U;
    for (std::uint32_t local_tone = 0; local_tone < 4U; ++local_tone) {
        append_reserved_control_re_unique(
            reserved_control_res,
            {.symbol = reg.symbol, .prb = reg.prb, .tone_in_prb = tone_base + local_tone});
    }
}

inline std::array<ControlRegCoord, 4> pcfich_reg_coords(std::uint16_t cell_id) {
    std::array<ControlRegCoord, 4> regs{};
    const std::uint32_t bandwidth_sc = kLteNumPrb20MHz * kLteNumSubcarriersPerPrb;
    const std::uint32_t k_hat = 6U * (cell_id % (2U * kLteNumPrb20MHz));
    for (std::uint32_t i = 0; i < regs.size(); ++i) {
        const std::uint32_t k =
            (k_hat + ((i * kLteNumPrb20MHz) / 2U) * (kLteNumSubcarriersPerPrb / 2U)) % bandwidth_sc;
        regs[i] = {
            .symbol = 0U,
            .prb = k / kLteNumSubcarriersPerPrb,
            .reg_in_symbol_prb = (k % kLteNumSubcarriersPerPrb) / (kLteNumSubcarriersPerPrb / 2U),
        };
    }
    return regs;
}

inline bool is_pcfich_reg(const ControlRegCoord& reg,
                          const std::array<ControlRegCoord, 4>& pcfich_regs) {
    for (const auto& pcfich_reg : pcfich_regs) {
        if (pcfich_reg.symbol == reg.symbol && pcfich_reg.prb == reg.prb &&
            pcfich_reg.reg_in_symbol_prb == reg.reg_in_symbol_prb) {
            return true;
        }
    }
    return false;
}

inline std::vector<ControlRegCoord> available_symbol0_control_regs(std::uint16_t cell_id) {
    std::vector<ControlRegCoord> regs{};
    regs.reserve(kLteNumPrb20MHz * 2U);

    const auto pcfich_regs = pcfich_reg_coords(cell_id);
    for (std::uint32_t prb = 0; prb < kLteNumPrb20MHz; ++prb) {
        for (std::uint32_t reg_in_symbol_prb = 0; reg_in_symbol_prb < 2U; ++reg_in_symbol_prb) {
            const ControlRegCoord reg{
                .symbol = 0U,
                .prb = prb,
                .reg_in_symbol_prb = reg_in_symbol_prb,
            };
            if (!is_pcfich_reg(reg, pcfich_regs)) {
                regs.push_back(reg);
            }
        }
    }
    return regs;
}

inline std::array<std::vector<ControlRegCoord>, 3>
available_extended_phich_control_regs(std::uint16_t cell_id) {
    std::array<std::vector<ControlRegCoord>, 3> regs_by_symbol{};
    regs_by_symbol[0] = available_symbol0_control_regs(cell_id);
    for (std::uint32_t symbol = 1; symbol < regs_by_symbol.size(); ++symbol) {
        regs_by_symbol[symbol].reserve(kLteNumPrb20MHz * 3U);
        for (std::uint32_t prb = 0; prb < kLteNumPrb20MHz; ++prb) {
            for (std::uint32_t reg_in_symbol_prb = 0; reg_in_symbol_prb < 3U; ++reg_in_symbol_prb) {
                regs_by_symbol[symbol].push_back(
                    {.symbol = symbol, .prb = prb, .reg_in_symbol_prb = reg_in_symbol_prb});
            }
        }
    }
    return regs_by_symbol;
}

inline std::uint32_t phich_base_group_count(PhichResource resource) {
    switch (resource) {
    case PhichResource::kOneSixth:
        return (kLteNumPrb20MHz + 47U) / 48U;
    case PhichResource::kHalf:
        return (kLteNumPrb20MHz + 15U) / 16U;
    case PhichResource::kOne:
        return (kLteNumPrb20MHz + 7U) / 8U;
    case PhichResource::kTwo:
        return (kLteNumPrb20MHz + 3U) / 4U;
    }
    return 0U;
}

inline constexpr std::array<std::array<std::uint8_t, 10>, 7> kTddPhichMiByUlDlConfig = {{
    {{2U, 1U, 0U, 0U, 0U, 2U, 1U, 0U, 0U, 0U}},
    {{0U, 1U, 0U, 0U, 1U, 0U, 1U, 0U, 0U, 1U}},
    {{0U, 0U, 0U, 1U, 0U, 0U, 0U, 0U, 1U, 0U}},
    {{1U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U, 1U}},
    {{0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U, 1U}},
    {{0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U, 0U}},
    {{1U, 1U, 0U, 0U, 0U, 1U, 1U, 0U, 0U, 1U}},
}};

inline std::uint8_t expected_tdd_phich_mi(std::uint8_t ul_dl_config, std::uint8_t subframe) {
    return kTddPhichMiByUlDlConfig[ul_dl_config][subframe];
}

inline bool uses_extended_mbsfn_special_case(const PhichReservationConfig& config) {
    return config.duration == PhichDuration::kExtended &&
           config.subframe_ctx.kind == LteControlSubframeKind::kMbsfn;
}

inline bool uses_extended_tdd_sf1_sf6_special_case(const PhichReservationConfig& config) {
    return config.duration == PhichDuration::kExtended &&
           config.subframe_ctx.duplex_mode == PhichDuplexMode::kTdd &&
           (config.subframe_ctx.subframe == 1U || config.subframe_ctx.subframe == 6U);
}

inline bool uses_extended_special_case(const PhichReservationConfig& config) {
    return uses_extended_mbsfn_special_case(config) ||
           uses_extended_tdd_sf1_sf6_special_case(config);
}

inline MmseStatus validate_phich_reservation_config(const PhichReservationConfig& config) {
    if (config.subframe_ctx.subframe >= 10U) {
        return MmseStatus::kInvalidArgument;
    }
    if (config.subframe_ctx.duplex_mode == PhichDuplexMode::kTdd) {
        if (config.subframe_ctx.ul_dl_config > 6U || config.mi > 2U) {
            return MmseStatus::kInvalidArgument;
        }
        if (config.mi !=
            expected_tdd_phich_mi(config.subframe_ctx.ul_dl_config, config.subframe_ctx.subframe)) {
            return MmseStatus::kInvalidArgument;
        }
    } else if (config.mi != 1U) {
        return MmseStatus::kInvalidArgument;
    }
    return MmseStatus::kOk;
}

inline std::uint32_t phich_group_count(const PhichReservationConfig& config) {
    const std::uint32_t base_group_count = phich_base_group_count(config.resource);
    if (config.subframe_ctx.duplex_mode == PhichDuplexMode::kTdd) {
        return base_group_count * config.mi;
    }
    return base_group_count;
}

} // namespace detail

inline constexpr std::size_t control_mask_index(std::uint32_t symbol, std::uint32_t prb) {
    return static_cast<std::size_t>(symbol) * kLteNumPrb20MHz + static_cast<std::size_t>(prb);
}

inline void clear_control_re_exclusion_masks(PdcchMmseInput& in) {
    in.control_re_exclusion_masks.fill(0U);
}

inline void exclude_control_re(PdcchMmseInput& in, std::uint32_t symbol, std::uint32_t prb,
                               std::uint32_t tone_in_prb) {
    if (symbol >= kLteMaxControlSymbolsNormalCp || prb >= kLteNumPrb20MHz ||
        tone_in_prb >= kLteNumSubcarriersPerPrb) {
        return;
    }
    in.control_re_exclusion_masks[control_mask_index(symbol, prb)] |=
        static_cast<std::uint16_t>(1U << tone_in_prb);
}

inline bool is_control_re_excluded(const PdcchMmseInput& in, std::uint32_t symbol,
                                   std::uint32_t prb, std::uint32_t tone_in_prb) {
    if (symbol >= kLteMaxControlSymbolsNormalCp || prb >= kLteNumPrb20MHz ||
        tone_in_prb >= kLteNumSubcarriersPerPrb) {
        return false;
    }
    return (in.control_re_exclusion_masks[control_mask_index(symbol, prb)] &
            static_cast<std::uint16_t>(1U << tone_in_prb)) != 0U;
}

inline std::vector<ReservedControlRe> build_pcfich_reserved_control_re_list(std::uint16_t cell_id) {
    std::vector<ReservedControlRe> reserved_control_res;
    reserved_control_res.reserve(16U);
    for (const auto& reg : detail::pcfich_reg_coords(cell_id)) {
        detail::append_reg_reserved_res(reserved_control_res, cell_id, reg);
    }
    return reserved_control_res;
}

inline std::vector<ReservedControlRe>
build_pcfich_reserved_control_re_list(std::uint16_t cell_id,
                                      const LteControlSubframeContext& subframe_ctx) {
    (void)subframe_ctx;
    return build_pcfich_reserved_control_re_list(cell_id);
}

inline std::vector<ReservedControlRe>
build_fdd_phich_reserved_control_re_list(std::uint16_t cell_id, PhichResource resource) {
    std::vector<ReservedControlRe> reserved_control_res;
    const std::vector<detail::ControlRegCoord> regs =
        detail::available_symbol0_control_regs(cell_id);
    const std::uint32_t ngroups =
        detail::phich_group_count({.resource = resource,
                                   .duration = PhichDuration::kNormal,
                                   .subframe_ctx = {.duplex_mode = PhichDuplexMode::kFdd}});

    if (regs.empty() || ngroups == 0U) {
        return reserved_control_res;
    }

    const std::uint32_t symbol0_reg_count = 2U * kLteNumPrb20MHz - 4U;
    for (std::uint32_t group = 0; group < ngroups; ++group) {
        for (std::uint32_t reg_idx = 0; reg_idx < 3U; ++reg_idx) {
            const std::uint32_t reg_slot =
                (cell_id + group + (reg_idx * symbol0_reg_count) / 3U) % symbol0_reg_count;
            detail::append_reg_reserved_res(reserved_control_res, cell_id, regs[reg_slot]);
        }
    }

    return reserved_control_res;
}

inline MmseStatus
append_phich_reserved_control_re_list(std::vector<ReservedControlRe>& reserved_control_res,
                                      std::uint16_t cell_id, const PhichReservationConfig& config) {
    if (const MmseStatus status = detail::validate_phich_reservation_config(config);
        status != MmseStatus::kOk) {
        return status;
    }

    const std::vector<detail::ControlRegCoord> regs =
        detail::available_symbol0_control_regs(cell_id);
    const std::uint32_t ngroups = detail::phich_group_count(config);
    if (regs.empty() || ngroups == 0U) {
        return MmseStatus::kOk;
    }

    if (config.duration == PhichDuration::kNormal) {
        const std::uint32_t symbol0_reg_count = 2U * kLteNumPrb20MHz - 4U;
        for (std::uint32_t group = 0; group < ngroups; ++group) {
            for (std::uint32_t reg_idx = 0; reg_idx < 3U; ++reg_idx) {
                const std::uint32_t reg_slot =
                    (cell_id + group + (reg_idx * symbol0_reg_count) / 3U) % symbol0_reg_count;
                const auto& reg = regs[reg_slot];
                detail::append_reg_reserved_res(reserved_control_res, cell_id, reg);
            }
        }
        return MmseStatus::kOk;
    }

    const auto regs_by_symbol = detail::available_extended_phich_control_regs(cell_id);
    const std::uint32_t n0 = static_cast<std::uint32_t>(regs_by_symbol[0].size());
    const std::uint32_t n1 = static_cast<std::uint32_t>(regs_by_symbol[1].size());
    const std::uint32_t n2 = static_cast<std::uint32_t>(regs_by_symbol[2].size());
    if (n0 == 0U || n1 == 0U || n2 == 0U) {
        return MmseStatus::kOk;
    }

    for (std::uint32_t group = 0; group < ngroups; ++group) {
        for (std::uint32_t reg_idx = 0; reg_idx < 3U; ++reg_idx) {
            std::uint32_t symbol = reg_idx;
            std::uint32_t ni = 0U;
            if (detail::uses_extended_special_case(config)) {
                symbol = (group / 2U + reg_idx + 1U) % 2U;
                const std::uint32_t n_symbol =
                    static_cast<std::uint32_t>(regs_by_symbol[symbol].size());
                ni = ((cell_id * n_symbol) / n1 + group + (reg_idx * n_symbol) / 3U) % n_symbol;
            } else {
                const std::uint32_t n_symbol =
                    static_cast<std::uint32_t>(regs_by_symbol[symbol].size());
                ni = ((cell_id * n_symbol) / n0 + group + (reg_idx * n_symbol) / 3U) % n_symbol;
            }

            const auto& reg = regs_by_symbol[symbol][ni];
            detail::append_reg_reserved_res(reserved_control_res, cell_id, reg);
        }
    }
    return MmseStatus::kOk;
}

inline void
append_pcfich_reserved_control_re_list(std::vector<ReservedControlRe>& reserved_control_res,
                                       std::uint16_t cell_id) {
    for (const auto& re : build_pcfich_reserved_control_re_list(cell_id)) {
        detail::append_reserved_control_re_unique(reserved_control_res, re);
    }
}

inline void
append_pcfich_reserved_control_re_list(std::vector<ReservedControlRe>& reserved_control_res,
                                       std::uint16_t cell_id,
                                       const LteControlSubframeContext& subframe_ctx) {
    for (const auto& re : build_pcfich_reserved_control_re_list(cell_id, subframe_ctx)) {
        detail::append_reserved_control_re_unique(reserved_control_res, re);
    }
}

inline void
append_fdd_phich_reserved_control_re_list(std::vector<ReservedControlRe>& reserved_control_res,
                                          std::uint16_t cell_id, PhichResource resource) {
    (void)append_phich_reserved_control_re_list(
        reserved_control_res, cell_id,
        {.resource = resource,
         .duration = PhichDuration::kNormal,
         .subframe_ctx = {.duplex_mode = PhichDuplexMode::kFdd}});
}

inline void append_pcfich_reserved_control_re_list(FrontendPdcchIndication& frontend) {
    append_pcfich_reserved_control_re_list(frontend.reserved_control_res, frontend.cell_id,
                                           frontend.control_subframe);
}

inline void append_pcfich_reserved_control_re_list(FrontendPdcchIndication& frontend,
                                                   const LteControlSubframeContext& subframe_ctx) {
    append_pcfich_reserved_control_re_list(frontend.reserved_control_res, frontend.cell_id,
                                           subframe_ctx);
}

inline MmseStatus append_phich_reserved_control_re_list(FrontendPdcchIndication& frontend,
                                                        const PhichReservationConfig& config) {
    return append_phich_reserved_control_re_list(frontend.reserved_control_res, frontend.cell_id,
                                                 config);
}

inline MmseStatus append_phich_reserved_control_re_list(FrontendPdcchIndication& frontend,
                                                        PhichResource resource,
                                                        PhichDuration duration, std::uint8_t mi) {
    return append_phich_reserved_control_re_list(frontend,
                                                 {.resource = resource,
                                                  .duration = duration,
                                                  .mi = mi,
                                                  .subframe_ctx = frontend.control_subframe});
}

inline void append_fdd_phich_reserved_control_re_list(FrontendPdcchIndication& frontend,
                                                      PhichResource resource) {
    (void)append_phich_reserved_control_re_list(
        frontend, {.resource = resource,
                   .duration = PhichDuration::kNormal,
                   .subframe_ctx = {.duplex_mode = PhichDuplexMode::kFdd}});
}

inline MmseStatus
validate_lte_control_subframe_context(const LteControlSubframeContext& control_subframe,
                                      std::uint32_t sfn_subframe) {
    if (control_subframe.subframe != static_cast<std::uint8_t>(sfn_subframe % 10U)) {
        return MmseStatus::kInvalidArgument;
    }
    if (control_subframe.duplex_mode == PhichDuplexMode::kFdd &&
        control_subframe.ul_dl_config != 0U) {
        return MmseStatus::kInvalidArgument;
    }
    return MmseStatus::kOk;
}

inline MmseStatus validate_pdcch_mmse_input(const PdcchMmseInput& in) {
    if (in.grid.n_rx_ant == 0U || in.grid.n_rx_ant > kMmseV1MaxNumRxAntennas ||
        in.grid.n_symbols != kLteNumSymbolsNormalCp ||
        in.grid.n_subcarriers != kLteNumSubcarriers20MHz) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::uint32_t rx = 0; rx < in.grid.n_rx_ant; ++rx) {
        if (in.grid.re[rx] == nullptr || in.grid.im[rx] == nullptr) {
            return MmseStatus::kInvalidArgument;
        }
    }
    if (in.control_symbol_count == 0U || in.control_symbol_count > kLteMaxControlSymbolsNormalCp) {
        return MmseStatus::kInvalidArgument;
    }
    if (in.n_prb == 0U || in.n_prb > kLteNumPrb20MHz) {
        return MmseStatus::kInvalidArgument;
    }
    if (in.n_tx_ports == 0U || in.n_tx_ports > kMmseV1MaxNumCrsTxPorts) {
        return MmseStatus::kInvalidArgument;
    }
    if (in.tx_mode != 1U && in.tx_mode != 2U) {
        return MmseStatus::kInvalidArgument;
    }
    return validate_lte_control_subframe_context(in.control_subframe, in.sfn_subframe);
}

template <typename Range>
inline void apply_reserved_control_re_list(PdcchMmseInput& in, const Range& reserved_control_res) {
    clear_control_re_exclusion_masks(in);
    for (const auto& re : reserved_control_res) {
        exclude_control_re(in, re.symbol, re.prb, re.tone_in_prb);
    }
}

inline PdcchMmseInput make_pdcch_mmse_input(const PlanarGridViewF32& grid,
                                            const FrontendPdcchIndication& frontend) {
    PdcchMmseInput in{};
    in.grid = grid;
    in.sfn_subframe = frontend.sfn_subframe;
    in.cell_id = frontend.cell_id;
    in.n_tx_ports = frontend.n_tx_ports;
    in.tx_mode = frontend.tx_mode;
    in.control_symbol_count = frontend.control_symbol_count;
    in.n_prb = frontend.n_prb;
    in.prb_bitmap = frontend.prb_bitmap;
    in.control_subframe = frontend.control_subframe;
    in.chain = frontend.chain;
    apply_reserved_control_re_list(in, frontend.reserved_control_res);
    return in;
}

inline BackendPdcchEqualizedIndication
make_backend_pdcch_equalized_indication(const PdcchMmseResult& meta,
                                        const PdcchMmseOutputView& out) {
    BackendPdcchEqualizedIndication backend{};
    backend.sfn_subframe = meta.sfn_subframe;
    backend.cell_id = meta.cell_id;
    backend.n_prb = meta.n_prb;
    backend.n_tx_ports = meta.n_tx_ports;
    backend.n_rx_ant = meta.n_rx_ant;
    backend.n_layers = meta.n_layers;
    backend.tx_mode = meta.tx_mode;
    backend.control_symbol_count = meta.control_symbol_count;
    backend.mod_order = meta.mod_order;
    backend.sigma2 = meta.sigma2;
    backend.chain = meta.chain;
    backend.x_hat_re.assign(out.x_hat_re, out.x_hat_re + meta.n_re);
    backend.x_hat_im.assign(out.x_hat_im, out.x_hat_im + meta.n_re);
    backend.sinr.assign(out.sinr, out.sinr + meta.n_re);
    backend.re_grid_indices.assign(out.re_grid_indices, out.re_grid_indices + meta.n_re);
    return backend;
}

inline BackendPdcchDescrambledLlrIndication
make_backend_pdcch_descrambled_llr_indication(const BackendPdcchEqualizedIndication& backend) {
    BackendPdcchDescrambledLlrIndication llr_backend{};
    llr_backend.sfn_subframe = backend.sfn_subframe;
    llr_backend.cell_id = backend.cell_id;
    llr_backend.n_prb = backend.n_prb;
    llr_backend.n_tx_ports = backend.n_tx_ports;
    llr_backend.n_rx_ant = backend.n_rx_ant;
    llr_backend.n_layers = backend.n_layers;
    llr_backend.tx_mode = backend.tx_mode;
    llr_backend.control_symbol_count = backend.control_symbol_count;
    llr_backend.mod_order = backend.mod_order;
    llr_backend.sigma2 = backend.sigma2;
    llr_backend.chain = backend.chain;
    llr_backend.re_grid_indices = backend.re_grid_indices;

    const std::uint8_t n_layers = backend.n_layers == 0U ? 1U : backend.n_layers;
    const std::uint32_t n_re_per_layer =
        static_cast<std::uint32_t>(backend.x_hat_re.size() / n_layers);
    (void)mmse::lte::build_max_log_llrs(backend.x_hat_re.data(), backend.x_hat_im.data(),
                                        backend.sinr.data(), n_re_per_layer, n_re_per_layer,
                                        n_layers, backend.mod_order, llr_backend.llrs);
    mmse::lte::descramble_llrs_inplace(
        llr_backend.llrs.data(), static_cast<std::uint32_t>(llr_backend.llrs.size()),
        mmse::lte::pdcch_c_init(backend.cell_id, backend.sfn_subframe));
    return llr_backend;
}

inline BackendPdcchTdEqualizedIndication
make_backend_pdcch_td_equalized_indication(const PdcchTdMmseResult& meta,
                                           const PdcchTdMmseOutputView& out) {
    BackendPdcchTdEqualizedIndication backend{};
    backend.sfn_subframe = meta.sfn_subframe;
    backend.cell_id = meta.cell_id;
    backend.n_prb = meta.n_prb;
    backend.n_tx_ports = meta.n_tx_ports;
    backend.n_rx_ant = meta.n_rx_ant;
    backend.n_layers = meta.n_layers;
    backend.tx_mode = meta.tx_mode;
    backend.control_symbol_count = meta.control_symbol_count;
    backend.mod_order = meta.mod_order;
    backend.sigma2 = meta.sigma2;
    backend.chain = meta.chain;
    backend.x_hat_re.assign(out.x_hat_re, out.x_hat_re + meta.n_symbols);
    backend.x_hat_im.assign(out.x_hat_im, out.x_hat_im + meta.n_symbols);
    backend.sinr.assign(out.sinr, out.sinr + meta.n_symbols);
    backend.re_grid_indices0.assign(out.re_grid_indices0, out.re_grid_indices0 + meta.n_symbols);
    backend.re_grid_indices1.assign(out.re_grid_indices1, out.re_grid_indices1 + meta.n_symbols);
    return backend;
}

inline MmseStatus
normalize_pdcch_td_cce_order(const BackendPdcchTdEqualizedIndication& td_backend,
                             BackendPdcchEqualizedIndication& cce_ordered_backend) {
    cce_ordered_backend = {};
    const std::size_t symbol_count = td_backend.x_hat_re.size();
    if (td_backend.n_tx_ports != 2U || td_backend.n_layers != 1U || td_backend.tx_mode != 2U ||
        td_backend.mod_order != 2U || symbol_count == 0U || (symbol_count & 1U) != 0U ||
        td_backend.x_hat_im.size() != symbol_count || td_backend.sinr.size() != symbol_count ||
        td_backend.re_grid_indices0.size() != symbol_count ||
        td_backend.re_grid_indices1.size() != symbol_count) {
        return MmseStatus::kInvalidArgument;
    }

    cce_ordered_backend.sfn_subframe = td_backend.sfn_subframe;
    cce_ordered_backend.cell_id = td_backend.cell_id;
    cce_ordered_backend.n_prb = td_backend.n_prb;
    cce_ordered_backend.n_tx_ports = td_backend.n_tx_ports;
    cce_ordered_backend.n_rx_ant = td_backend.n_rx_ant;
    cce_ordered_backend.n_layers = td_backend.n_layers;
    cce_ordered_backend.tx_mode = td_backend.tx_mode;
    cce_ordered_backend.control_symbol_count = td_backend.control_symbol_count;
    cce_ordered_backend.mod_order = td_backend.mod_order;
    cce_ordered_backend.sigma2 = td_backend.sigma2;
    cce_ordered_backend.chain = td_backend.chain;
    cce_ordered_backend.x_hat_re = td_backend.x_hat_re;
    cce_ordered_backend.x_hat_im = td_backend.x_hat_im;
    cce_ordered_backend.sinr = td_backend.sinr;
    cce_ordered_backend.re_grid_indices.resize(symbol_count);

    for (std::size_t symbol = 0U; symbol < symbol_count; symbol += 2U) {
        if (td_backend.re_grid_indices0[symbol] != td_backend.re_grid_indices0[symbol + 1U] ||
            td_backend.re_grid_indices1[symbol] != td_backend.re_grid_indices1[symbol + 1U]) {
            cce_ordered_backend = {};
            return MmseStatus::kInvalidArgument;
        }
        cce_ordered_backend.re_grid_indices[symbol] = td_backend.re_grid_indices0[symbol];
        cce_ordered_backend.re_grid_indices[symbol + 1U] = td_backend.re_grid_indices1[symbol];
    }
    return MmseStatus::kOk;
}

inline BackendPdcchTdDescrambledLlrIndication
make_backend_pdcch_td_descrambled_llr_indication(const BackendPdcchTdEqualizedIndication& backend) {
    BackendPdcchTdDescrambledLlrIndication llr_backend{};
    llr_backend.sfn_subframe = backend.sfn_subframe;
    llr_backend.cell_id = backend.cell_id;
    llr_backend.n_prb = backend.n_prb;
    llr_backend.n_tx_ports = backend.n_tx_ports;
    llr_backend.n_rx_ant = backend.n_rx_ant;
    llr_backend.n_layers = backend.n_layers;
    llr_backend.tx_mode = backend.tx_mode;
    llr_backend.control_symbol_count = backend.control_symbol_count;
    llr_backend.mod_order = backend.mod_order;
    llr_backend.sigma2 = backend.sigma2;
    llr_backend.chain = backend.chain;
    llr_backend.re_grid_indices0 = backend.re_grid_indices0;
    llr_backend.re_grid_indices1 = backend.re_grid_indices1;

    const std::uint8_t n_layers = backend.n_layers == 0U ? 1U : backend.n_layers;
    const std::uint32_t n_re_per_layer =
        static_cast<std::uint32_t>(backend.x_hat_re.size() / n_layers);
    (void)mmse::lte::build_max_log_llrs(backend.x_hat_re.data(), backend.x_hat_im.data(),
                                        backend.sinr.data(), n_re_per_layer, n_re_per_layer,
                                        n_layers, backend.mod_order, llr_backend.llrs);
    mmse::lte::descramble_llrs_inplace(
        llr_backend.llrs.data(), static_cast<std::uint32_t>(llr_backend.llrs.size()),
        mmse::lte::pdcch_c_init(backend.cell_id, backend.sfn_subframe));
    return llr_backend;
}

inline MmseStatus build_pdcch_control_region(std::uint16_t cell_id,
                                             std::uint8_t control_symbol_count,
                                             const std::uint16_t* re_grid_indices,
                                             std::uint32_t n_re,
                                             PdcchControlRegion& control_region) {
    control_region = {};
    if (re_grid_indices == nullptr || n_re == 0U || control_symbol_count == 0U ||
        control_symbol_count > kLteMaxControlSymbolsNormalCp ||
        n_re > kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz || (n_re % kPdcchRePerReg) != 0U) {
        return MmseStatus::kInvalidArgument;
    }

    std::array<bool, kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz> seen_grid_indices{};
    for (std::uint32_t re = 0U; re < n_re; ++re) {
        const std::uint16_t grid_index = re_grid_indices[re];
        const std::uint32_t symbol = grid_index / kLteNumSubcarriers20MHz;
        if (symbol >= control_symbol_count || seen_grid_indices[grid_index]) {
            return MmseStatus::kInvalidArgument;
        }
        seen_grid_indices[grid_index] = true;
    }

    control_region.cell_id = cell_id;
    control_region.control_symbol_count = control_symbol_count;
    control_region.n_source_re = n_re;
    control_region.regs.reserve(n_re / kPdcchRePerReg);
    for (std::uint32_t re = 0U; re < n_re; re += kPdcchRePerReg) {
        PdcchReg reg{};
        for (std::uint32_t re_in_reg = 0U; re_in_reg < kPdcchRePerReg; ++re_in_reg) {
            reg.source_re_indices[re_in_reg] = re + re_in_reg;
            reg.grid_indices[re_in_reg] = re_grid_indices[re + re_in_reg];
        }
        control_region.regs.push_back(reg);
    }

    const std::uint32_t n_cce =
        static_cast<std::uint32_t>(control_region.regs.size()) / kPdcchRegsPerCce;
    control_region.n_unassigned_reg =
        static_cast<std::uint32_t>(control_region.regs.size()) % kPdcchRegsPerCce;
    control_region.cces.reserve(n_cce);
    for (std::uint32_t cce = 0U; cce < n_cce; ++cce) {
        PdcchCce entry{};
        for (std::uint32_t reg_in_cce = 0U; reg_in_cce < kPdcchRegsPerCce; ++reg_in_cce) {
            entry.reg_indices[reg_in_cce] =
                static_cast<std::uint16_t>(cce * kPdcchRegsPerCce + reg_in_cce);
        }
        control_region.cces.push_back(entry);
    }
    return MmseStatus::kOk;
}

inline void
build_pdcch_common_search_candidates(const PdcchControlRegion& control_region,
                                     std::vector<PdcchCommonSearchCandidate>& candidates) {
    candidates.clear();

    struct SearchLevel {
        std::uint8_t aggregation_level = 0;
        std::uint8_t max_candidate_count = 0;
    };
    constexpr std::array<SearchLevel, 2> kCommonSearchLevels = {
        SearchLevel{.aggregation_level = 4U, .max_candidate_count = 4U},
        SearchLevel{.aggregation_level = 8U, .max_candidate_count = 2U},
    };

    const std::uint32_t n_cce = static_cast<std::uint32_t>(control_region.cces.size());
    std::uint32_t candidate_id = 0U;
    for (const SearchLevel level : kCommonSearchLevels) {
        const std::uint32_t candidate_count =
            std::min(static_cast<std::uint32_t>(level.max_candidate_count),
                     n_cce / static_cast<std::uint32_t>(level.aggregation_level));
        for (std::uint32_t candidate = 0U; candidate < candidate_count; ++candidate) {
            candidates.push_back({
                .candidate_id = candidate_id++,
                .first_cce = static_cast<std::uint16_t>(candidate * level.aggregation_level),
                .aggregation_level = level.aggregation_level,
                .encoded_bit_count = 72U * level.aggregation_level,
            });
        }
    }
}

inline PdcchSearchCandidate
make_pdcch_search_candidate(const PdcchCommonSearchCandidate& candidate) noexcept {
    return {
        .candidate_id = candidate.candidate_id,
        .first_cce = candidate.first_cce,
        .aggregation_level = candidate.aggregation_level,
        .encoded_bit_count = candidate.encoded_bit_count,
    };
}

inline bool pdcch_aggregation_level_enabled(std::uint8_t aggregation_level,
                                            std::uint8_t aggregation_level_mask) noexcept {
    switch (aggregation_level) {
    case 1U:
        return (aggregation_level_mask & kPdcchAggregationLevelMaskL1) != 0U;
    case 2U:
        return (aggregation_level_mask & kPdcchAggregationLevelMaskL2) != 0U;
    case 4U:
        return (aggregation_level_mask & kPdcchAggregationLevelMaskL4) != 0U;
    case 8U:
        return (aggregation_level_mask & kPdcchAggregationLevelMaskL8) != 0U;
    default:
        return false;
    }
}

inline std::uint32_t pdcch_ue_specific_search_y(std::uint16_t rnti,
                                                std::uint32_t sfn_subframe) noexcept {
    constexpr std::uint32_t kMultiplier = 39827U;
    constexpr std::uint32_t kModulus = 65537U;
    std::uint32_t value = rnti;
    for (std::uint32_t subframe = 0U; subframe <= sfn_subframe % 10U; ++subframe) {
        value = (kMultiplier * value) % kModulus;
    }
    return value;
}

inline MmseStatus
build_pdcch_ue_specific_search_candidates(const PdcchControlRegion& control_region,
                                          std::uint32_t sfn_subframe,
                                          const PdcchUeSpecificSearchConfig& config,
                                          std::vector<PdcchUeSpecificSearchCandidate>& candidates) {
    candidates.clear();
    if (config.rntis.empty() ||
        (config.aggregation_level_mask & ~kPdcchAggregationLevelMaskAll) != 0U) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::size_t index = 0U; index < config.rntis.size(); ++index) {
        const std::uint16_t rnti = config.rntis[index];
        if (rnti == 0U || rnti == kSiRnti) {
            return MmseStatus::kInvalidArgument;
        }
        for (std::size_t prior = 0U; prior < index; ++prior) {
            if (config.rntis[prior] == rnti) {
                return MmseStatus::kInvalidArgument;
            }
        }
    }

    struct SearchLevel {
        std::uint8_t aggregation_level = 0U;
        std::uint8_t max_candidate_count = 0U;
    };
    constexpr std::array<SearchLevel, 4> kSearchLevels = {
        SearchLevel{.aggregation_level = 1U, .max_candidate_count = 6U},
        SearchLevel{.aggregation_level = 2U, .max_candidate_count = 6U},
        SearchLevel{.aggregation_level = 4U, .max_candidate_count = 2U},
        SearchLevel{.aggregation_level = 8U, .max_candidate_count = 2U},
    };

    const std::uint32_t n_cce = static_cast<std::uint32_t>(control_region.cces.size());
    std::uint32_t candidate_id = 0U;
    for (const std::uint16_t rnti : config.rntis) {
        const std::uint32_t y = pdcch_ue_specific_search_y(rnti, sfn_subframe);
        for (const SearchLevel level : kSearchLevels) {
            if (!pdcch_aggregation_level_enabled(level.aggregation_level,
                                                 config.aggregation_level_mask)) {
                continue;
            }
            const std::uint32_t cce_groups = n_cce / level.aggregation_level;
            const std::uint32_t candidate_count =
                std::min(static_cast<std::uint32_t>(level.max_candidate_count), cce_groups);
            for (std::uint32_t candidate = 0U; candidate < candidate_count; ++candidate) {
                candidates.push_back({
                    .rnti = rnti,
                    .search_candidate =
                        {
                            .candidate_id = candidate_id++,
                            .first_cce = static_cast<std::uint16_t>(level.aggregation_level *
                                                                    ((y + candidate) % cce_groups)),
                            .aggregation_level = level.aggregation_level,
                            .encoded_bit_count = 72U * level.aggregation_level,
                        },
                });
            }
        }
    }
    return MmseStatus::kOk;
}

inline MmseStatus
build_pdcch_search_candidate_llrs(const BackendPdcchEqualizedIndication& backend,
                                  const std::vector<PdcchSearchCandidate>& candidates,
                                  std::vector<PdcchCandidateLlr>& candidate_llrs) {
    candidate_llrs.clear();
    if (backend.n_layers != 1U || backend.mod_order != 2U || backend.x_hat_re.empty() ||
        backend.x_hat_re.size() != backend.x_hat_im.size() ||
        backend.x_hat_re.size() != backend.sinr.size() ||
        backend.x_hat_re.size() != backend.re_grid_indices.size()) {
        return MmseStatus::kInvalidArgument;
    }

    PdcchControlRegion control_region{};
    if (const MmseStatus status = build_pdcch_control_region(
            backend.cell_id, backend.control_symbol_count, backend.re_grid_indices.data(),
            static_cast<std::uint32_t>(backend.re_grid_indices.size()), control_region);
        status != MmseStatus::kOk) {
        return status;
    }

    std::vector<float> full_descrambled_llrs{};
    if (const MmseStatus status = mmse::lte::build_max_log_descrambled_llrs(
            backend.x_hat_re.data(), backend.x_hat_im.data(), backend.sinr.data(),
            static_cast<std::uint32_t>(backend.x_hat_re.size()),
            static_cast<std::uint32_t>(backend.x_hat_re.size()), 1U, backend.mod_order,
            mmse::lte::pdcch_c_init(backend.cell_id, backend.sfn_subframe), full_descrambled_llrs);
        status != MmseStatus::kOk) {
        return status;
    }

    candidate_llrs.reserve(candidates.size());
    for (const PdcchSearchCandidate& candidate : candidates) {
        if (!pdcch_aggregation_level_enabled(candidate.aggregation_level,
                                             kPdcchAggregationLevelMaskAll) ||
            candidate.encoded_bit_count != 72U * candidate.aggregation_level ||
            candidate.first_cce > control_region.cces.size() ||
            candidate.aggregation_level > control_region.cces.size() - candidate.first_cce) {
            candidate_llrs.clear();
            return MmseStatus::kInvalidArgument;
        }
        const std::uint32_t llr_offset = static_cast<std::uint32_t>(candidate.first_cce) * 72U;
        if (llr_offset > full_descrambled_llrs.size() ||
            candidate.encoded_bit_count > full_descrambled_llrs.size() - llr_offset) {
            candidate_llrs.clear();
            return MmseStatus::kInternalError;
        }

        PdcchCandidateLlr candidate_llr{};
        candidate_llr.sfn_subframe = backend.sfn_subframe;
        candidate_llr.cell_id = backend.cell_id;
        candidate_llr.chain = backend.chain;
        candidate_llr.chain.candidate_id = candidate.candidate_id;
        candidate_llr.chain.first_cce = candidate.first_cce;
        candidate_llr.chain.aggregation_level = candidate.aggregation_level;
        candidate_llr.encoded_bit_count = candidate.encoded_bit_count;
        candidate_llr.llrs.assign(full_descrambled_llrs.begin() + llr_offset,
                                  full_descrambled_llrs.begin() + llr_offset +
                                      candidate.encoded_bit_count);
        candidate_llrs.push_back(std::move(candidate_llr));
    }
    return MmseStatus::kOk;
}

inline MmseStatus
build_pdcch_common_search_candidate_llrs(const BackendPdcchEqualizedIndication& backend,
                                         std::vector<PdcchCandidateLlr>& candidate_llrs) {
    candidate_llrs.clear();
    PdcchControlRegion control_region{};
    if (backend.n_layers != 1U || backend.mod_order != 2U || backend.x_hat_re.empty() ||
        backend.x_hat_re.size() != backend.x_hat_im.size() ||
        backend.x_hat_re.size() != backend.sinr.size() ||
        backend.x_hat_re.size() != backend.re_grid_indices.size() ||
        build_pdcch_control_region(backend.cell_id, backend.control_symbol_count,
                                   backend.re_grid_indices.data(),
                                   static_cast<std::uint32_t>(backend.re_grid_indices.size()),
                                   control_region) != MmseStatus::kOk) {
        return MmseStatus::kInvalidArgument;
    }
    std::vector<PdcchCommonSearchCandidate> common_candidates{};
    build_pdcch_common_search_candidates(control_region, common_candidates);
    std::vector<PdcchSearchCandidate> candidates{};
    candidates.reserve(common_candidates.size());
    for (const PdcchCommonSearchCandidate& candidate : common_candidates) {
        candidates.push_back(make_pdcch_search_candidate(candidate));
    }
    return build_pdcch_search_candidate_llrs(backend, candidates, candidate_llrs);
}

inline MmseStatus recover_pdcch_convolutional_rate_matched_llrs(const PdcchCandidateLlr& candidate,
                                                                std::uint32_t dci_payload_bit_count,
                                                                PdcchRateRecoveredLlr& recovered) {
    recovered = {};
    if (dci_payload_bit_count == 0U || dci_payload_bit_count > kPdcchMaxDciPayloadBits ||
        candidate.encoded_bit_count == 0U || candidate.encoded_bit_count != candidate.llrs.size() ||
        candidate.encoded_bit_count > kPdcchMaxCandidateEncodedBits ||
        !pdcch_aggregation_level_enabled(candidate.chain.aggregation_level,
                                         kPdcchAggregationLevelMaskAll) ||
        candidate.encoded_bit_count != 72U * candidate.chain.aggregation_level) {
        return MmseStatus::kInvalidArgument;
    }

    constexpr std::uint32_t kColumns = 32U;
    constexpr std::array<std::uint8_t, kColumns> kPermutation = {
        1U, 17U, 9U, 25U, 5U, 21U, 13U, 29U, 3U, 19U, 11U, 27U, 7U, 23U, 15U, 31U,
        0U, 16U, 8U, 24U, 4U, 20U, 12U, 28U, 2U, 18U, 10U, 26U, 6U, 22U, 14U, 30U,
    };
    constexpr std::array<std::uint8_t, kColumns> kInversePermutation = {
        16U, 0U, 24U, 8U, 20U, 4U, 28U, 12U, 18U, 2U, 26U, 10U, 22U, 6U, 30U, 14U,
        17U, 1U, 25U, 9U, 21U, 5U, 29U, 13U, 19U, 3U, 27U, 11U, 23U, 7U, 31U, 15U,
    };

    const std::uint32_t codeword_bit_count = dci_payload_bit_count + kPdcchCrcBitCount;
    const std::uint32_t row_count = (codeword_bit_count + kColumns - 1U) / kColumns;
    const std::uint32_t interleaver_size = row_count * kColumns;
    const std::uint32_t dummy_bit_count = interleaver_size - codeword_bit_count;
    const std::uint32_t collection_size = kPdcchConvolutionalCodeRate * interleaver_size;
    std::vector<float> collected_llrs(collection_size, 0.0F);
    std::vector<std::uint8_t> collected_present(collection_size, 0U);

    std::uint32_t input_index = 0U;
    std::uint32_t collection_index = 0U;
    while (input_index < candidate.encoded_bit_count) {
        const std::uint32_t intra_stream_index = collection_index % interleaver_size;
        const std::uint32_t permutation_column = intra_stream_index / row_count;
        const std::uint32_t row = intra_stream_index % row_count;
        const std::uint32_t original_bit_index = row * kColumns + kPermutation[permutation_column];
        if (original_bit_index >= dummy_bit_count) {
            if (collected_present[collection_index] == 0U) {
                collected_llrs[collection_index] = candidate.llrs[input_index];
                collected_present[collection_index] = 1U;
            } else {
                collected_llrs[collection_index] += candidate.llrs[input_index];
            }
            ++input_index;
        }
        collection_index = (collection_index + 1U) % collection_size;
    }

    recovered.sfn_subframe = candidate.sfn_subframe;
    recovered.cell_id = candidate.cell_id;
    recovered.chain = candidate.chain;
    recovered.encoded_bit_count = candidate.encoded_bit_count;
    recovered.codeword_bit_count = codeword_bit_count;
    recovered.convolutional_llrs.assign(kPdcchConvolutionalCodeRate * codeword_bit_count, 0.0F);
    for (std::uint32_t bit = 0U; bit < codeword_bit_count; ++bit) {
        const std::uint32_t padded_bit = bit + dummy_bit_count;
        const std::uint32_t row = padded_bit / kColumns;
        const std::uint32_t column = padded_bit % kColumns;
        const std::uint32_t interleaved_index = kInversePermutation[column] * row_count + row;
        for (std::uint32_t stream = 0U; stream < kPdcchConvolutionalCodeRate; ++stream) {
            const std::uint32_t collection_slot = stream * interleaver_size + interleaved_index;
            recovered.convolutional_llrs[bit * kPdcchConvolutionalCodeRate + stream] =
                collected_present[collection_slot] != 0U ? collected_llrs[collection_slot] : 0.0F;
        }
    }
    return MmseStatus::kOk;
}

inline MmseStatus
invoke_pdcch_tail_biting_convolutional_decoder(const PdcchRateRecoveredLlr& recovered,
                                               const PdcchTailBitingConvolutionalDecoder& decoder,
                                               std::vector<std::uint8_t>& decoded_bits) {
    decoded_bits.clear();
    if (decoder.decode == nullptr || recovered.codeword_bit_count == 0U ||
        recovered.convolutional_llrs.size() !=
            static_cast<std::size_t>(kPdcchConvolutionalCodeRate) * recovered.codeword_bit_count) {
        return MmseStatus::kInvalidArgument;
    }

    decoded_bits.resize(recovered.codeword_bit_count);
    const PdcchTailBitingConvolutionalDecodeRequest request{
        .convolutional_llrs = recovered.convolutional_llrs.data(),
        .convolutional_llr_count = static_cast<std::uint32_t>(recovered.convolutional_llrs.size()),
        .decoded_bits = decoded_bits.data(),
        .decoded_bit_count = static_cast<std::uint32_t>(decoded_bits.size()),
        .soft_bit_polarity = recovered.soft_bit_polarity,
        .llr_order = recovered.llr_order,
        .tail_biting = true,
    };
    const MmseStatus status = decoder.decode(decoder.context, request);
    if (status != MmseStatus::kOk) {
        decoded_bits.clear();
    }
    return status;
}

inline std::uint16_t calculate_pdcch_crc16(const std::uint8_t* bits,
                                           std::uint32_t bit_count) noexcept {
    constexpr std::uint16_t kCrc16Polynomial = 0x1021U;
    std::uint16_t crc = 0U;
    if (bits == nullptr && bit_count != 0U) {
        return 0U;
    }
    for (std::uint32_t bit_index = 0U; bit_index < bit_count; ++bit_index) {
        const std::uint16_t feedback =
            static_cast<std::uint16_t>(((crc >> 15U) ^ (bits[bit_index] & 1U)) & 1U);
        crc = static_cast<std::uint16_t>(crc << 1U);
        if (feedback != 0U) {
            crc = static_cast<std::uint16_t>(crc ^ kCrc16Polynomial);
        }
    }
    return crc;
}

inline MmseStatus check_pdcch_crc_rnti(const std::uint8_t* decoded_bits,
                                       std::uint32_t decoded_bit_count, std::uint16_t expected_rnti,
                                       PdcchCrcRntiCheck& check) {
    check = {};
    if (decoded_bits == nullptr || decoded_bit_count <= kPdcchCrcBitCount) {
        return MmseStatus::kInvalidArgument;
    }
    const std::uint32_t payload_bit_count = decoded_bit_count - kPdcchCrcBitCount;
    std::uint16_t transmitted_crc = 0U;
    for (std::uint32_t bit = 0U; bit < kPdcchCrcBitCount; ++bit) {
        if (decoded_bits[payload_bit_count + bit] > 1U) {
            return MmseStatus::kInvalidArgument;
        }
        transmitted_crc = static_cast<std::uint16_t>(
            (transmitted_crc << 1U) |
            static_cast<std::uint16_t>(decoded_bits[payload_bit_count + bit]));
    }
    for (std::uint32_t bit = 0U; bit < payload_bit_count; ++bit) {
        if (decoded_bits[bit] > 1U) {
            return MmseStatus::kInvalidArgument;
        }
    }

    check.transmitted_crc = transmitted_crc;
    check.calculated_crc = calculate_pdcch_crc16(decoded_bits, payload_bit_count);
    check.unmasked_rnti = static_cast<std::uint16_t>(transmitted_crc ^ check.calculated_crc);
    check.matches_expected_rnti = check.unmasked_rnti == expected_rnti;
    return MmseStatus::kOk;
}

inline constexpr std::uint8_t pdcch_dci_riv_bit_count(std::uint16_t n_prb) noexcept {
    std::uint32_t max_value = static_cast<std::uint32_t>(n_prb) * (n_prb + 1U) / 2U;
    std::uint8_t bits = 0U;
    --max_value;
    while (max_value != 0U) {
        ++bits;
        max_value >>= 1U;
    }
    return bits;
}

inline constexpr std::uint32_t
pdcch_dci_format1a_payload_bit_count(const PdcchDciFormat1AConfig& config) noexcept {
    if (config.n_prb != kLteNumPrb20MHz) {
        return 0U;
    }
    const std::uint32_t harq_pid_bits = config.duplex_mode == PhichDuplexMode::kFdd ? 3U : 4U;
    return (config.cif_enabled ? 3U : 0U) + 1U + 1U + pdcch_dci_riv_bit_count(config.n_prb) + 5U +
           harq_pid_bits + 1U + 2U + 2U + (config.duplex_mode == PhichDuplexMode::kTdd ? 2U : 0U);
}

inline MmseStatus decode_pdcch_type2_riv(std::uint16_t resource_indication_value,
                                         std::uint16_t n_prb, std::uint16_t& start_prb,
                                         std::uint16_t& allocated_prb_count) {
    start_prb = 0U;
    allocated_prb_count = 0U;
    if (n_prb == 0U || n_prb > kLteNumPrb20MHz ||
        resource_indication_value >= static_cast<std::uint32_t>(n_prb) * (n_prb + 1U) / 2U) {
        return MmseStatus::kInvalidArgument;
    }

    std::uint32_t length = resource_indication_value / n_prb + 1U;
    std::uint32_t start = resource_indication_value % n_prb;
    if (length > n_prb - start) {
        length = n_prb - resource_indication_value / n_prb + 1U;
        start = n_prb - resource_indication_value % n_prb - 1U;
    }
    if (length == 0U || start + length > n_prb) {
        return MmseStatus::kInvalidArgument;
    }
    start_prb = static_cast<std::uint16_t>(start);
    allocated_prb_count = static_cast<std::uint16_t>(length);
    return MmseStatus::kOk;
}

namespace detail {

struct PdcchBitReader {
    const std::uint8_t* bits = nullptr;
    std::uint32_t count = 0U;
    std::uint32_t position = 0U;

    bool read(std::uint8_t width, std::uint32_t& value) {
        value = 0U;
        if (bits == nullptr || width > 24U || position + width > count) {
            return false;
        }
        for (std::uint8_t bit = 0U; bit < width; ++bit) {
            if (bits[position] > 1U) {
                return false;
            }
            value = (value << 1U) | bits[position++];
        }
        return true;
    }

    bool remaining_bits_are_zero() const {
        if (bits == nullptr) {
            return false;
        }
        for (std::uint32_t bit = position; bit < count; ++bit) {
            if (bits[bit] != 0U) {
                return false;
            }
        }
        return true;
    }
};

} // namespace detail

inline MmseStatus parse_pdcch_dci_format1a(const std::uint8_t* payload_bits,
                                           std::uint32_t payload_bit_count,
                                           std::uint32_t sfn_subframe, std::uint16_t cell_id,
                                           std::uint16_t rnti, const PdcchChainMetadata& chain,
                                           const PdcchDciFormat1AConfig& config,
                                           PdcchDciFormat1A& dci) {
    dci = {};
    const std::uint32_t expected_bit_count = pdcch_dci_format1a_payload_bit_count(config);
    if (payload_bits == nullptr || expected_bit_count == 0U ||
        payload_bit_count != expected_bit_count) {
        return MmseStatus::kInvalidArgument;
    }

    detail::PdcchBitReader reader{.bits = payload_bits, .count = payload_bit_count};
    std::uint32_t value = 0U;
    dci.sfn_subframe = sfn_subframe;
    dci.cell_id = cell_id;
    dci.rnti = rnti;
    dci.chain = chain;
    dci.raw_payload_bits.assign(payload_bits, payload_bits + payload_bit_count);

    if (config.cif_enabled) {
        if (!reader.read(3U, value)) {
            return MmseStatus::kInvalidArgument;
        }
        dci.cif_present = true;
        dci.carrier_indicator = static_cast<std::uint8_t>(value);
    }
    if (!reader.read(1U, value) || value != 1U) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (!reader.read(1U, value)) {
        return MmseStatus::kInvalidArgument;
    }
    dci.distributed_vrb_assignment = value != 0U;

    const std::uint8_t riv_bits = pdcch_dci_riv_bit_count(config.n_prb);
    if (!reader.read(riv_bits, value)) {
        return MmseStatus::kInvalidArgument;
    }
    const std::uint16_t riv = static_cast<std::uint16_t>(value);

    if (!dci.distributed_vrb_assignment &&
        riv == static_cast<std::uint16_t>((1U << riv_bits) - 1U)) {
        std::uint32_t preamble_index = 0U;
        std::uint32_t prach_mask_index = 0U;
        if (!reader.read(6U, preamble_index) || !reader.read(4U, prach_mask_index) ||
            !reader.remaining_bits_are_zero()) {
            return MmseStatus::kInvalidArgument;
        }
        dci.is_pdcch_order = true;
        dci.preamble_index = static_cast<std::uint8_t>(preamble_index);
        dci.prach_mask_index = static_cast<std::uint8_t>(prach_mask_index);
        return MmseStatus::kOk;
    }

    dci.resource_indication_value = riv;
    if (!reader.read(5U, value)) {
        return MmseStatus::kInvalidArgument;
    }
    dci.mcs_tbs_index = static_cast<std::uint8_t>(value);
    const std::uint8_t harq_pid_bits = config.duplex_mode == PhichDuplexMode::kFdd ? 3U : 4U;
    if (!reader.read(harq_pid_bits, value)) {
        return MmseStatus::kInvalidArgument;
    }
    dci.harq_process = static_cast<std::uint8_t>(value);

    if (dci.distributed_vrb_assignment && config.n_prb >= 50U) {
        if (!reader.read(1U, value)) {
            return MmseStatus::kInvalidArgument;
        }
        dci.n_gap_is_two = value != 0U;
    } else if (!reader.read(1U, value)) {
        return MmseStatus::kInvalidArgument;
    }
    if (!reader.read(2U, value)) {
        return MmseStatus::kInvalidArgument;
    }
    dci.redundancy_version = static_cast<std::uint8_t>(value);
    if (!reader.read(1U, value) || value != 0U || !reader.read(1U, value)) {
        return MmseStatus::kInvalidArgument;
    }
    dci.n_prb_1a_is_three = value != 0U;
    dci.n_prb_1a = dci.n_prb_1a_is_three ? 3U : 2U;
    if (config.duplex_mode == PhichDuplexMode::kTdd) {
        if (!reader.read(2U, value)) {
            return MmseStatus::kInvalidArgument;
        }
        dci.downlink_assignment_index = static_cast<std::uint8_t>(value);
    }
    if (!reader.remaining_bits_are_zero()) {
        return MmseStatus::kInvalidArgument;
    }
    return decode_pdcch_type2_riv(dci.resource_indication_value, config.n_prb, dci.start_prb,
                                  dci.n_prb);
}

inline MmseStatus validate_and_parse_pdcch_dci_format1a(
    const std::uint8_t* decoded_bits, std::uint32_t decoded_bit_count, std::uint16_t expected_rnti,
    std::uint32_t sfn_subframe, std::uint16_t cell_id, const PdcchChainMetadata& chain,
    const PdcchDciFormat1AConfig& config, PdcchDciFormat1ADecodeResult& result) {
    result = {};
    const std::uint32_t payload_bit_count = pdcch_dci_format1a_payload_bit_count(config);
    if (payload_bit_count == 0U || decoded_bit_count != payload_bit_count + kPdcchCrcBitCount) {
        return MmseStatus::kInvalidArgument;
    }
    if (const MmseStatus status =
            check_pdcch_crc_rnti(decoded_bits, decoded_bit_count, expected_rnti, result.crc);
        status != MmseStatus::kOk) {
        return status;
    }
    if (!result.crc.matches_expected_rnti) {
        return MmseStatus::kOk;
    }
    const MmseStatus status =
        parse_pdcch_dci_format1a(decoded_bits, payload_bit_count, sfn_subframe, cell_id,
                                 expected_rnti, chain, config, result.dci);
    if (status != MmseStatus::kOk) {
        return status;
    }
    result.matched = true;
    return MmseStatus::kOk;
}

inline MmseStatus decode_pdcch_dci_format1a_with_adapter(
    const PdcchRateRecoveredLlr& recovered, const PdcchTailBitingConvolutionalDecoder& decoder,
    std::uint16_t expected_rnti, const PdcchDciFormat1AConfig& config,
    PdcchDciFormat1ADecodeResult& result) {
    result = {};
    const std::uint32_t payload_bit_count = pdcch_dci_format1a_payload_bit_count(config);
    if (payload_bit_count == 0U ||
        recovered.codeword_bit_count != payload_bit_count + kPdcchCrcBitCount) {
        return MmseStatus::kInvalidArgument;
    }

    std::vector<std::uint8_t> decoded_bits{};
    if (const MmseStatus status =
            invoke_pdcch_tail_biting_convolutional_decoder(recovered, decoder, decoded_bits);
        status != MmseStatus::kOk) {
        return status;
    }
    return validate_and_parse_pdcch_dci_format1a(
        decoded_bits.data(), static_cast<std::uint32_t>(decoded_bits.size()), expected_rnti,
        recovered.sfn_subframe, recovered.cell_id, recovered.chain, config, result);
}

inline MmseStatus decode_pdcch_dci_format1a(const PdcchRateRecoveredLlr& recovered,
                                            const PdcchTailBitingConvolutionalDecoder& decoder,
                                            std::uint16_t expected_rnti,
                                            const PdcchDciFormat1AConfig& config,
                                            PdcchDciFormat1ADecodeResult& result) {
    if (decoder.decode != nullptr) {
        return decode_pdcch_dci_format1a_with_adapter(recovered, decoder, expected_rnti, config,
                                                      result);
    }

    result = {};
    const std::uint32_t payload_bit_count = pdcch_dci_format1a_payload_bit_count(config);
    if (payload_bit_count == 0U ||
        recovered.codeword_bit_count != payload_bit_count + kPdcchCrcBitCount) {
        return MmseStatus::kInvalidArgument;
    }

    std::vector<std::uint8_t> decoded_bits{};
    if (const MmseStatus status = decode_pdcch_tail_biting_convolutional(recovered, decoded_bits);
        status != MmseStatus::kOk) {
        return status;
    }
    return validate_and_parse_pdcch_dci_format1a(
        decoded_bits.data(), static_cast<std::uint32_t>(decoded_bits.size()), expected_rnti,
        recovered.sfn_subframe, recovered.cell_id, recovered.chain, config, result);
}

inline MmseStatus
decode_pdcch_common_search_dci_format1a(const BackendPdcchEqualizedIndication& backend,
                                        const PdcchCommonSearchDecodeConfig& config,
                                        PdcchCommonSearchDecodeResult& result) {
    result = {};
    const std::uint32_t payload_bit_count =
        pdcch_dci_format1a_payload_bit_count(config.dci_format1a);
    if (payload_bit_count == 0U) {
        return MmseStatus::kInvalidArgument;
    }

    std::vector<PdcchCandidateLlr> candidates{};
    if (const MmseStatus status = build_pdcch_common_search_candidate_llrs(backend, candidates);
        status != MmseStatus::kOk) {
        return status;
    }

    result.candidate_count = static_cast<std::uint32_t>(candidates.size());
    result.hits.reserve(candidates.size());
    for (const PdcchCandidateLlr& candidate : candidates) {
        PdcchRateRecoveredLlr recovered{};
        if (const MmseStatus status = recover_pdcch_convolutional_rate_matched_llrs(
                candidate, payload_bit_count, recovered);
            status != MmseStatus::kOk) {
            result = {};
            return status;
        }

        PdcchDciFormat1ADecodeResult candidate_result{};
        if (const MmseStatus status =
                decode_pdcch_dci_format1a(recovered, config.decoder, config.expected_rnti,
                                          config.dci_format1a, candidate_result);
            status != MmseStatus::kOk) {
            result = {};
            return status;
        }
        if (candidate_result.matched) {
            result.hits.push_back(std::move(candidate_result));
        }
    }
    return MmseStatus::kOk;
}

inline MmseStatus
decode_pdcch_ue_specific_search_dci_format1a(const BackendPdcchEqualizedIndication& backend,
                                             const PdcchUeSpecificSearchConfig& config,
                                             PdcchUeSpecificSearchResult& result) {
    result = {};
    const std::uint32_t payload_bit_count =
        pdcch_dci_format1a_payload_bit_count(config.dci_format1a);
    if (payload_bit_count == 0U) {
        return MmseStatus::kInvalidArgument;
    }

    PdcchControlRegion control_region{};
    if (backend.n_layers != 1U || backend.mod_order != 2U || backend.x_hat_re.empty() ||
        backend.x_hat_re.size() != backend.x_hat_im.size() ||
        backend.x_hat_re.size() != backend.sinr.size() ||
        backend.x_hat_re.size() != backend.re_grid_indices.size() ||
        build_pdcch_control_region(backend.cell_id, backend.control_symbol_count,
                                   backend.re_grid_indices.data(),
                                   static_cast<std::uint32_t>(backend.re_grid_indices.size()),
                                   control_region) != MmseStatus::kOk) {
        return MmseStatus::kInvalidArgument;
    }

    std::vector<PdcchUeSpecificSearchCandidate> ue_candidates{};
    if (const MmseStatus status = build_pdcch_ue_specific_search_candidates(
            control_region, backend.sfn_subframe, config, ue_candidates);
        status != MmseStatus::kOk) {
        return status;
    }
    std::vector<PdcchSearchCandidate> search_candidates{};
    search_candidates.reserve(ue_candidates.size());
    for (const PdcchUeSpecificSearchCandidate& candidate : ue_candidates) {
        search_candidates.push_back(candidate.search_candidate);
    }

    std::vector<PdcchCandidateLlr> candidates{};
    if (const MmseStatus status =
            build_pdcch_search_candidate_llrs(backend, search_candidates, candidates);
        status != MmseStatus::kOk) {
        return status;
    }

    result.candidate_count = static_cast<std::uint32_t>(candidates.size());
    result.hits.reserve(candidates.size());
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        const PdcchCandidateLlr& candidate = candidates[index];
        const std::uint16_t expected_rnti = ue_candidates[index].rnti;
        PdcchRateRecoveredLlr recovered{};
        if (const MmseStatus status = recover_pdcch_convolutional_rate_matched_llrs(
                candidate, payload_bit_count, recovered);
            status != MmseStatus::kOk) {
            result = {};
            return status;
        }

        std::vector<std::uint8_t> decoded_bits{};
        const MmseStatus decode_status =
            config.decoder.decode != nullptr
                ? invoke_pdcch_tail_biting_convolutional_decoder(recovered, config.decoder,
                                                                 decoded_bits)
                : decode_pdcch_tail_biting_convolutional(recovered, decoded_bits);
        if (decode_status != MmseStatus::kOk) {
            result = {};
            return decode_status;
        }
        ++result.decoded_candidate_count;

        PdcchDciFormat1ADecodeResult candidate_result{};
        const MmseStatus validation_status = validate_and_parse_pdcch_dci_format1a(
            decoded_bits.data(), static_cast<std::uint32_t>(decoded_bits.size()), expected_rnti,
            recovered.sfn_subframe, recovered.cell_id, recovered.chain, config.dci_format1a,
            candidate_result);
        if (validation_status == MmseStatus::kInvalidArgument ||
            validation_status == MmseStatus::kUnsupportedConfig) {
            ++result.semantic_reject_count;
            continue;
        }
        if (validation_status != MmseStatus::kOk) {
            result = {};
            return validation_status;
        }
        if (!candidate_result.matched) {
            ++result.crc_rnti_miss_count;
            continue;
        }
        result.hits.push_back({
            .rnti = expected_rnti,
            .first_cce = candidate_result.dci.chain.first_cce,
            .aggregation_level = candidate_result.dci.chain.aggregation_level,
            .decoded = std::move(candidate_result),
        });
    }
    return MmseStatus::kOk;
}

inline MmseStatus decode_pdcch_si_rnti_dci_format1a(const BackendPdcchEqualizedIndication& backend,
                                                    const PdcchSiRntiSearchConfig& config,
                                                    PdcchSiRntiSearchResult& result) {
    result = {};
    const PdcchDciFormat1AConfig dci_config{};
    if (backend.n_prb != kLteNumPrb20MHz ||
        pdcch_dci_format1a_payload_bit_count(dci_config) == 0U) {
        return MmseStatus::kInvalidArgument;
    }

    std::vector<PdcchCandidateLlr> candidates{};
    if (const MmseStatus status = build_pdcch_common_search_candidate_llrs(backend, candidates);
        status != MmseStatus::kOk) {
        return status;
    }

    result.candidate_count = static_cast<std::uint32_t>(candidates.size());
    result.hits.reserve(candidates.size());
    const std::uint32_t payload_bit_count = pdcch_dci_format1a_payload_bit_count(dci_config);
    for (const PdcchCandidateLlr& candidate : candidates) {
        PdcchRateRecoveredLlr recovered{};
        if (const MmseStatus status = recover_pdcch_convolutional_rate_matched_llrs(
                candidate, payload_bit_count, recovered);
            status != MmseStatus::kOk) {
            result = {};
            return status;
        }

        PdcchDciFormat1ADecodeResult candidate_result{};
        const MmseStatus status = decode_pdcch_dci_format1a(recovered, config.decoder, kSiRnti,
                                                            dci_config, candidate_result);
        if (status == MmseStatus::kInvalidArgument || status == MmseStatus::kUnsupportedConfig) {
            continue;
        }
        if (status != MmseStatus::kOk) {
            result = {};
            return status;
        }
        if (!candidate_result.matched || candidate_result.dci.rnti != kSiRnti ||
            candidate_result.dci.is_pdcch_order || candidate_result.dci.n_prb_1a < 2U ||
            candidate_result.dci.n_prb_1a > 3U ||
            candidate_result.dci.start_prb + candidate_result.dci.n_prb > kLteNumPrb20MHz) {
            continue;
        }
        result.hits.push_back({
            .first_cce = candidate_result.dci.chain.first_cce,
            .aggregation_level = candidate_result.dci.chain.aggregation_level,
            .decoded = std::move(candidate_result),
        });
    }
    return MmseStatus::kOk;
}

inline ReCoord decode_re_grid_index(std::uint16_t grid_index) {
    const std::uint32_t symbol = grid_index / kLteNumSubcarriers20MHz;
    const std::uint32_t subcarrier = grid_index % kLteNumSubcarriers20MHz;
    return {
        .symbol = symbol,
        .subcarrier = subcarrier,
        .prb = subcarrier / kLteNumSubcarriersPerPrb,
        .tone_in_prb = subcarrier % kLteNumSubcarriersPerPrb,
    };
}

} // namespace mmse::pdcch
