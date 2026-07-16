#pragma once

#include <cstdint>

namespace mmse {

inline constexpr std::uint32_t kLteNumSubcarriers20MHz = 1200;
inline constexpr std::uint32_t kLteNumSymbolsNormalCp = 14;
inline constexpr std::uint32_t kLteNumPrb20MHz = 100;
inline constexpr std::uint32_t kLteNumSubcarriersPerPrb = 12;
inline constexpr std::uint32_t kMmseV1MaxNumRxAntennas = 2;
inline constexpr std::uint32_t kMmseV1MaxNumCrsTxPorts = 4;
inline constexpr std::uint32_t kLteNumCrsSymbols = 4;
inline constexpr std::uint32_t kLteNumPilotTonesPerCrsSymbol = 200;
inline constexpr std::uint32_t kLteNumCellIds = 504;
inline constexpr std::uint32_t kLteMaxControlSymbolsNormalCp = 3;
inline constexpr std::uint32_t kLteMaxPdcchControlSymbolsNormalCp = 4;
inline constexpr std::uint32_t kLtePbchNumPrb = 6;
inline constexpr std::uint32_t kLtePbchStartPrb = (kLteNumPrb20MHz - kLtePbchNumPrb) / 2U;
inline constexpr std::uint32_t kLtePbchStartSymbolNormalCp = 7;
inline constexpr std::uint32_t kLtePbchNumSymbols = 4;
inline constexpr std::uint32_t kLtePcfichNumRegs = 4;

inline constexpr bool is_supported_lte_downlink_bandwidth(std::uint32_t n_prb) noexcept {
    switch (n_prb) {
    case 6U:
    case 15U:
    case 25U:
    case 50U:
    case 75U:
    case 100U:
        return true;
    default:
        return false;
    }
}

inline constexpr std::uint32_t lte_downlink_subcarrier_count(std::uint32_t n_prb) noexcept {
    return n_prb * kLteNumSubcarriersPerPrb;
}

inline constexpr std::uint32_t
lte_max_pdcch_control_symbols_normal_cp(std::uint32_t n_prb) noexcept {
    return n_prb == 6U ? kLteMaxPdcchControlSymbolsNormalCp : kLteMaxControlSymbolsNormalCp;
}

inline constexpr float kDefaultDetFloor = 1.0e-6F;
inline constexpr float kDefaultSigma2Min = 1.0e-4F;
inline constexpr float kDefaultGMin = 1.0e-4F;
inline constexpr float kDefaultGammaMax = 1.0e4F;
inline constexpr float kDefaultSigma2IirAlpha = 0.8F;

} // namespace mmse
