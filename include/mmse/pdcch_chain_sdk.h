#pragma once

#include "mmse/mmse_equalizer.h"
#include "mmse/pdcch_chain_dto.h"
#include "mmse/pdcch_module_api.h"

namespace mmse::pdcch {

inline MmseStatus run_pdcch_cpu_common_search_decode(MmseEqualizerCpuContext& context,
                                                     const PdcchMmseInput& input,
                                                     const PdcchCommonSearchDecodeConfig& config,
                                                     PdcchCommonSearchDecodeResult& result) {
    result = {};
    constexpr std::uint32_t kMaxPdcchRe = kLteMaxControlSymbolsNormalCp * kLteNumSubcarriers20MHz;

    std::vector<float> x_hat_re(kMaxPdcchRe);
    std::vector<float> x_hat_im(kMaxPdcchRe);
    std::vector<float> sinr(kMaxPdcchRe);
    BackendPdcchEqualizedIndication backend{};

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
    } else if (input.n_tx_ports == 2U) {
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
        if (const MmseStatus status = normalize_pdcch_td_cce_order(td_backend, backend);
            status != MmseStatus::kOk) {
            return status;
        }
    } else {
        return MmseStatus::kUnsupportedConfig;
    }

    return decode_pdcch_common_search_dci_format1a(backend, config, result);
}

} // namespace mmse::pdcch
