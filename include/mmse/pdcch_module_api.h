#pragma once

#include <cstdint>

#include "mmse/constants.h"
#include "mmse/lte_descrambling.h"
#include "mmse/lte_soft_demod.h"
#include "mmse/pdcch_chain_dto.h"
#include "mmse/types.h"

namespace mmse::pdcch {

enum class PhichResource : std::uint8_t {
    kOneSixth = 0,
    kHalf,
    kOne,
    kTwo,
};

enum class PhichDuration : std::uint8_t {
    kNormal = 0,
    kExtended,
};

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
    if (in.grid.n_rx_ant != kLteNumRxAntV1 || in.grid.n_symbols != kLteNumSymbolsNormalCp ||
        in.grid.n_subcarriers != kLteNumSubcarriers20MHz) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::uint32_t rx = 0; rx < kLteNumRxAntV1; ++rx) {
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
    if (in.n_tx_ports == 0U || in.n_tx_ports > kLteNumTxPortsV1) {
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
