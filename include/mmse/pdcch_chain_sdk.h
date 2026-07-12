#pragma once

#include "mmse/mmse_equalizer.h"
#include "mmse/pdcch_chain_dto.h"
#include "mmse/pdcch_module_api.h"

namespace mmse::pdcch {

namespace detail {

inline MmseStatus build_pdcch_cpu_backend(MmseEqualizerCpuContext& context,
                                          const PdcchMmseInput& input,
                                          BackendPdcchEqualizedIndication& backend) {
    backend = {};
    constexpr std::uint32_t kMaxPdcchRe = kLteMaxControlSymbolsNormalCp * kLteNumSubcarriers20MHz;
    std::vector<float> x_hat_re(kMaxPdcchRe);
    std::vector<float> x_hat_im(kMaxPdcchRe);
    std::vector<float> sinr(kMaxPdcchRe);

    if (input.n_tx_ports == 1U) {
        std::vector<std::uint16_t> re_grid_indices(kMaxPdcchRe);
        PdcchMmseOutputView output{
            .x_hat_re = x_hat_re.data(),
            .x_hat_im = x_hat_im.data(),
            .sinr = sinr.data(),
            .re_grid_indices = re_grid_indices.data(),
            .capacity_re_per_layer = kMaxPdcchRe,
            .capacity_re_metadata = kMaxPdcchRe,
        };
        PdcchMmseResult metadata{};
        if (const MmseStatus status = context.run_pdcch(input, output, metadata);
            status != MmseStatus::kOk) {
            return status;
        }
        backend = make_backend_pdcch_equalized_indication(metadata, output);
        return MmseStatus::kOk;
    }
    if (input.n_tx_ports == 2U) {
        std::vector<std::uint16_t> re_grid_indices0(kMaxPdcchRe);
        std::vector<std::uint16_t> re_grid_indices1(kMaxPdcchRe);
        PdcchTdMmseOutputView output{
            .x_hat_re = x_hat_re.data(),
            .x_hat_im = x_hat_im.data(),
            .sinr = sinr.data(),
            .re_grid_indices0 = re_grid_indices0.data(),
            .re_grid_indices1 = re_grid_indices1.data(),
            .capacity_symbols = kMaxPdcchRe,
        };
        PdcchTdMmseResult metadata{};
        if (const MmseStatus status = context.run_pdcch_td(input, output, metadata);
            status != MmseStatus::kOk) {
            return status;
        }
        const BackendPdcchTdEqualizedIndication td_backend =
            make_backend_pdcch_td_equalized_indication(metadata, output);
        return normalize_pdcch_td_cce_order(td_backend, backend);
    }
    return MmseStatus::kUnsupportedConfig;
}

} // namespace detail

inline MmseStatus run_pdcch_cpu_common_search_decode(MmseEqualizerCpuContext& context,
                                                     const PdcchMmseInput& input,
                                                     const PdcchCommonSearchDecodeConfig& config,
                                                     PdcchCommonSearchDecodeResult& result) {
    result = {};
    BackendPdcchEqualizedIndication backend{};
    if (const MmseStatus status = detail::build_pdcch_cpu_backend(context, input, backend);
        status != MmseStatus::kOk) {
        return status;
    }

    return decode_pdcch_common_search_dci_format1a(backend, config, result);
}

inline MmseStatus run_pdcch_cpu_si_rnti_search(MmseEqualizerCpuContext& context,
                                               const PdcchMmseInput& input,
                                               const PdcchSiRntiSearchConfig& config,
                                               PdcchSiRntiSearchResult& result) {
    result = {};
    if (input.n_prb != kLteNumPrb20MHz ||
        input.control_subframe.duplex_mode != PhichDuplexMode::kFdd) {
        return MmseStatus::kUnsupportedConfig;
    }

    BackendPdcchEqualizedIndication backend{};
    if (const MmseStatus status = detail::build_pdcch_cpu_backend(context, input, backend);
        status != MmseStatus::kOk) {
        return status;
    }

    return decode_pdcch_si_rnti_dci_format1a(backend, config, result);
}

inline MmseStatus run_pdcch_cpu_ue_specific_search(MmseEqualizerCpuContext& context,
                                                   const PdcchMmseInput& input,
                                                   const PdcchUeSpecificSearchConfig& config,
                                                   PdcchUeSpecificSearchResult& result) {
    result = {};
    if (input.n_prb != kLteNumPrb20MHz ||
        input.control_subframe.duplex_mode != PhichDuplexMode::kFdd ||
        config.dci_format1a.duplex_mode != PhichDuplexMode::kFdd) {
        return MmseStatus::kUnsupportedConfig;
    }
    BackendPdcchEqualizedIndication backend{};
    if (const MmseStatus status = detail::build_pdcch_cpu_backend(context, input, backend);
        status != MmseStatus::kOk) {
        return status;
    }
    return decode_pdcch_ue_specific_search_dci_format1a(backend, config, result);
}

namespace detail {

inline bool
pdcch_si_rnti_geometry_cache_matches(const PdcchSiRntiGeometrySearchCache& cache,
                                     const PdcchSiRntiGeometrySearchRequest& request) noexcept {
    return cache.locked && cache.cell_id == request.cell_id &&
           cache.n_tx_ports == request.n_tx_ports && cache.tx_mode == request.tx_mode &&
           cache.subframe_kind == request.control_subframe.kind;
}

inline MmseStatus
build_pdcch_si_rnti_geometry_input(const PdcchSiRntiGeometrySearchRequest& request,
                                   const PdcchControlGeometry& geometry, PdcchMmseInput& input) {
    input = {};
    if (!geometry.standard_reg_order || geometry.control_symbol_count == 0U ||
        geometry.control_symbol_count > kLteMaxControlSymbolsNormalCp ||
        request.control_subframe.duplex_mode != PhichDuplexMode::kFdd ||
        request.control_subframe.subframe != request.sfn_subframe % 10U) {
        return MmseStatus::kInvalidArgument;
    }

    FrontendPdcchIndication frontend{};
    frontend.sfn_subframe = request.sfn_subframe;
    frontend.cell_id = request.cell_id;
    frontend.n_tx_ports = request.n_tx_ports;
    frontend.tx_mode = request.tx_mode;
    frontend.control_symbol_count = geometry.control_symbol_count;
    frontend.n_prb = kLteNumPrb20MHz;
    frontend.prb_bitmap.fill(0xFFFFU);
    frontend.prb_bitmap.back() = 0x000FU;
    frontend.control_subframe = request.control_subframe;
    frontend.chain = request.chain;
    append_pcfich_reserved_control_re_list(frontend);
    if (const MmseStatus status = append_phich_reserved_control_re_list(
            frontend, {.resource = geometry.phich_resource,
                       .duration = geometry.phich_duration,
                       .mi = 1U,
                       .subframe_ctx = frontend.control_subframe});
        status != MmseStatus::kOk) {
        return status;
    }
    input = make_pdcch_mmse_input(request.grid, frontend);
    return validate_pdcch_mmse_input(input);
}

inline constexpr std::array<PhichResource, 4> kPdcchGeometryPhichResources = {
    PhichResource::kOneSixth,
    PhichResource::kHalf,
    PhichResource::kOne,
    PhichResource::kTwo,
};

inline constexpr std::array<PhichDuration, 2> kPdcchGeometryPhichDurations = {
    PhichDuration::kNormal,
    PhichDuration::kExtended,
};

inline void build_pdcch_si_rnti_geometry_candidates(std::vector<PdcchControlGeometry>& geometries) {
    geometries.clear();
    geometries.reserve(kLteMaxControlSymbolsNormalCp * kPdcchGeometryPhichResources.size() *
                       kPdcchGeometryPhichDurations.size());
    for (std::uint8_t cfi = 1U; cfi <= kLteMaxControlSymbolsNormalCp; ++cfi) {
        for (const PhichResource resource : kPdcchGeometryPhichResources) {
            for (const PhichDuration duration : kPdcchGeometryPhichDurations) {
                if (duration == PhichDuration::kExtended && cfi < kLteMaxControlSymbolsNormalCp) {
                    continue;
                }
                geometries.push_back({
                    .control_symbol_count = cfi,
                    .phich_resource = resource,
                    .phich_duration = duration,
                    .standard_reg_order = true,
                });
            }
        }
    }
}

inline void update_pdcch_si_rnti_geometry_cache(PdcchSiRntiGeometrySearchCache& cache,
                                                const PdcchSiRntiGeometrySearchRequest& request,
                                                const PdcchControlGeometry& geometry) {
    cache = {
        .locked = true,
        .cell_id = request.cell_id,
        .n_tx_ports = request.n_tx_ports,
        .tx_mode = request.tx_mode,
        .subframe_kind = request.control_subframe.kind,
        .geometry = geometry,
        .consecutive_miss_count = 0U,
    };
}

} // namespace detail

inline MmseStatus run_pdcch_cpu_si_rnti_geometry_search(
    MmseEqualizerCpuContext& context, const PdcchSiRntiGeometrySearchRequest& request,
    const PdcchSiRntiGeometrySearchConfig& config, PdcchSiRntiGeometrySearchCache& cache,
    PdcchSiRntiGeometrySearchResult& result) {
    result = {};
    if (request.control_subframe.duplex_mode != PhichDuplexMode::kFdd ||
        request.control_subframe.subframe != request.sfn_subframe % 10U) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (cache.locked && !detail::pdcch_si_rnti_geometry_cache_matches(cache, request)) {
        cache = {};
    }

    const auto try_geometry = [&](const PdcchControlGeometry& geometry,
                                  PdcchSiRntiSearchResult& decoded) {
        PdcchMmseInput input{};
        if (const MmseStatus status =
                detail::build_pdcch_si_rnti_geometry_input(request, geometry, input);
            status != MmseStatus::kOk) {
            return status;
        }
        BackendPdcchEqualizedIndication backend{};
        if (const MmseStatus status = detail::build_pdcch_cpu_backend(context, input, backend);
            status != MmseStatus::kOk) {
            return status;
        }
        return decode_pdcch_si_rnti_dci_format1a(backend, {.decoder = config.decoder}, decoded);
    };

    if (cache.locked && cache.consecutive_miss_count < 4U) {
        result.geometry_attempt_count = 1U;
        result.geometry = cache.geometry;
        if (const MmseStatus status = try_geometry(cache.geometry, result.decoded);
            status != MmseStatus::kOk) {
            result = {};
            return status;
        }
        result.candidate_count = result.decoded.candidate_count;
        if (!result.decoded.hits.empty()) {
            cache.consecutive_miss_count = 0U;
            result.status = PdcchSiRntiGeometrySearchStatus::kLocked;
            return MmseStatus::kOk;
        }
        ++cache.consecutive_miss_count;
        result.status = PdcchSiRntiGeometrySearchStatus::kMiss;
        return MmseStatus::kOk;
    }

    if (cache.locked) {
        cache = {};
    }
    std::vector<PdcchControlGeometry> geometries{};
    detail::build_pdcch_si_rnti_geometry_candidates(geometries);
    PdcchControlGeometry matched_geometry{};
    PdcchSiRntiSearchResult matched_decoded{};
    std::uint32_t matching_geometry_count = 0U;
    for (const PdcchControlGeometry& geometry : geometries) {
        PdcchSiRntiSearchResult decoded{};
        ++result.geometry_attempt_count;
        if (const MmseStatus status = try_geometry(geometry, decoded); status != MmseStatus::kOk) {
            result = {};
            return status;
        }
        result.candidate_count += decoded.candidate_count;
        if (decoded.hits.empty()) {
            continue;
        }
        ++matching_geometry_count;
        if (matching_geometry_count == 1U) {
            matched_geometry = geometry;
            matched_decoded = std::move(decoded);
        }
    }
    if (matching_geometry_count == 0U) {
        result.status = PdcchSiRntiGeometrySearchStatus::kMiss;
        return MmseStatus::kOk;
    }
    if (matching_geometry_count != 1U) {
        result.status = PdcchSiRntiGeometrySearchStatus::kAmbiguous;
        result.decoded = {};
        return MmseStatus::kOk;
    }
    detail::update_pdcch_si_rnti_geometry_cache(cache, request, matched_geometry);
    result.status = PdcchSiRntiGeometrySearchStatus::kAcquired;
    result.geometry = matched_geometry;
    result.decoded = std::move(matched_decoded);
    result.candidate_count = result.decoded.candidate_count;
    return MmseStatus::kOk;
}

} // namespace mmse::pdcch
