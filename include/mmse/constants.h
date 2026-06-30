#pragma once

#include <cstdint>

namespace mmse {

inline constexpr std::uint32_t kLteNumSubcarriers20MHz = 1200;
inline constexpr std::uint32_t kLteNumSymbolsNormalCp = 14;
inline constexpr std::uint32_t kLteNumPrb20MHz = 100;
inline constexpr std::uint32_t kLteNumSubcarriersPerPrb = 12;
inline constexpr std::uint32_t kLteNumRxAntV1 = 2;
inline constexpr std::uint32_t kLteNumTxPortsV1 = 2;
inline constexpr std::uint32_t kLteNumCrsSymbols = 4;
inline constexpr std::uint32_t kLteNumPilotTonesPerCrsSymbol = 200;
inline constexpr std::uint32_t kLteNumCellIds = 504;
inline constexpr std::uint32_t kLteMaxControlSymbolsNormalCp = 3;

inline constexpr float kDefaultDetFloor = 1.0e-6F;
inline constexpr float kDefaultSigma2Min = 1.0e-4F;
inline constexpr float kDefaultGMin = 1.0e-4F;
inline constexpr float kDefaultGammaMax = 1.0e4F;
inline constexpr float kDefaultSigma2IirAlpha = 0.8F;

} // namespace mmse
