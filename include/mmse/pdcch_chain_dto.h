#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "mmse/constants.h"
#include "mmse/types.h"

namespace mmse::pdcch {

inline constexpr std::uint16_t kSiRnti = 0xFFFFU;
inline constexpr std::uint8_t kPdcchRePerReg = 4U;
inline constexpr std::uint8_t kPdcchRegsPerCce = 9U;
inline constexpr std::uint8_t kPdcchCrcBitCount = 16U;
inline constexpr std::uint8_t kPdcchConvolutionalCodeRate = 3U;
inline constexpr std::uint32_t kPdcchMaxCandidateEncodedBits = 576U;
inline constexpr std::uint32_t kPdcchMaxDciPayloadBits = 128U;
inline constexpr std::uint8_t kPdcchAggregationLevelMaskL1 = 1U << 0U;
inline constexpr std::uint8_t kPdcchAggregationLevelMaskL2 = 1U << 1U;
inline constexpr std::uint8_t kPdcchAggregationLevelMaskL4 = 1U << 2U;
inline constexpr std::uint8_t kPdcchAggregationLevelMaskL8 = 1U << 3U;
inline constexpr std::uint8_t kPdcchAggregationLevelMaskAll =
    kPdcchAggregationLevelMaskL1 | kPdcchAggregationLevelMaskL2 | kPdcchAggregationLevelMaskL4 |
    kPdcchAggregationLevelMaskL8;

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

struct ReservedControlRe {
    std::uint32_t symbol = 0;
    std::uint32_t prb = 0;
    std::uint32_t tone_in_prb = 0;
};

struct FrontendPdcchIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 1;
    std::uint8_t tx_mode = 1;
    std::uint8_t control_symbol_count = 0;
    std::uint16_t n_prb = 0;
    std::array<std::uint16_t, 7> prb_bitmap{};
    LteControlSubframeContext control_subframe{};
    std::vector<ReservedControlRe> reserved_control_res{};
    PdcchChainMetadata chain{};
};

struct BackendPdcchEqualizedIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = kLteNumPrb20MHz;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PdcchChainMetadata chain{};
    std::vector<float> x_hat_re{};
    std::vector<float> x_hat_im{};
    std::vector<float> sinr{};
    std::vector<std::uint16_t> re_grid_indices{};
};

struct BackendPdcchDescrambledLlrIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = kLteNumPrb20MHz;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PdcchChainMetadata chain{};
    std::vector<float> llrs{};
    std::vector<std::uint16_t> re_grid_indices{};
};

struct BackendPdcchTdEqualizedIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = kLteNumPrb20MHz;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PdcchChainMetadata chain{};
    std::vector<float> x_hat_re{};
    std::vector<float> x_hat_im{};
    std::vector<float> sinr{};
    std::vector<std::uint16_t> re_grid_indices0{};
    std::vector<std::uint16_t> re_grid_indices1{};
};

struct BackendPdcchTdDescrambledLlrIndication {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = kLteNumPrb20MHz;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PdcchChainMetadata chain{};
    std::vector<float> llrs{};
    std::vector<std::uint16_t> re_grid_indices0{};
    std::vector<std::uint16_t> re_grid_indices1{};
};

struct PdcchReg {
    std::array<std::uint32_t, kPdcchRePerReg> source_re_indices{};
    std::array<std::uint16_t, kPdcchRePerReg> grid_indices{};
};

struct PdcchCce {
    std::array<std::uint16_t, kPdcchRegsPerCce> reg_indices{};
};

struct PdcchControlRegion {
    std::uint16_t cell_id = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint32_t n_source_re = 0;
    std::uint32_t n_unassigned_reg = 0;
    std::vector<PdcchReg> regs{};
    std::vector<PdcchCce> cces{};
};

struct PdcchCommonSearchCandidate {
    std::uint32_t candidate_id = 0;
    std::uint16_t first_cce = 0;
    std::uint8_t aggregation_level = 0;
    std::uint32_t encoded_bit_count = 0;
};

struct PdcchSearchCandidate {
    std::uint32_t candidate_id = 0;
    std::uint16_t first_cce = 0;
    std::uint8_t aggregation_level = 0;
    std::uint32_t encoded_bit_count = 0;
};

struct PdcchUeSpecificSearchCandidate {
    std::uint16_t rnti = 0;
    PdcchSearchCandidate search_candidate{};
};

struct PdcchCandidateLlr {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    PdcchChainMetadata chain{};
    std::uint32_t encoded_bit_count = 0;
    std::vector<float> llrs{};
};

enum class PdcchSoftBitPolarity : std::uint8_t {
    kNegativeFavorsZero = 0,
};

enum class PdcchConvolutionalLlrOrder : std::uint8_t {
    kLteRateRecoveredTriplets = 0,
};

struct PdcchRateRecoveredLlr {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    PdcchChainMetadata chain{};
    std::uint32_t encoded_bit_count = 0;
    std::uint32_t codeword_bit_count = 0;
    PdcchSoftBitPolarity soft_bit_polarity = PdcchSoftBitPolarity::kNegativeFavorsZero;
    PdcchConvolutionalLlrOrder llr_order = PdcchConvolutionalLlrOrder::kLteRateRecoveredTriplets;
    std::vector<float> convolutional_llrs{};
};

struct PdcchTailBitingConvolutionalDecodeRequest {
    const float* convolutional_llrs = nullptr;
    std::uint32_t convolutional_llr_count = 0;
    std::uint8_t* decoded_bits = nullptr;
    std::uint32_t decoded_bit_count = 0;
    PdcchSoftBitPolarity soft_bit_polarity = PdcchSoftBitPolarity::kNegativeFavorsZero;
    PdcchConvolutionalLlrOrder llr_order = PdcchConvolutionalLlrOrder::kLteRateRecoveredTriplets;
    bool tail_biting = true;
};

using PdcchTailBitingConvolutionalDecodeFn =
    MmseStatus (*)(void* context, const PdcchTailBitingConvolutionalDecodeRequest& request);

struct PdcchTailBitingConvolutionalDecoder {
    void* context = nullptr;
    PdcchTailBitingConvolutionalDecodeFn decode = nullptr;
};

struct PdcchDciFormat1AConfig {
    std::uint16_t n_prb = kLteNumPrb20MHz;
    PhichDuplexMode duplex_mode = PhichDuplexMode::kFdd;
    bool cif_enabled = false;
};

struct PdcchCrcRntiCheck {
    std::uint16_t transmitted_crc = 0;
    std::uint16_t calculated_crc = 0;
    std::uint16_t unmasked_rnti = 0;
    bool matches_expected_rnti = false;
};

struct PdcchDciFormat1A {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t rnti = 0;
    PdcchChainMetadata chain{};
    bool cif_present = false;
    std::uint8_t carrier_indicator = 0;
    bool is_pdcch_order = false;
    std::uint8_t preamble_index = 0;
    std::uint8_t prach_mask_index = 0;
    bool distributed_vrb_assignment = false;
    bool n_gap_is_two = false;
    bool n_prb_1a_is_three = false;
    std::uint8_t n_prb_1a = 0;
    std::uint16_t resource_indication_value = 0;
    std::uint16_t start_prb = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t mcs_tbs_index = 0;
    std::uint8_t harq_process = 0;
    std::uint8_t redundancy_version = 0;
    std::uint8_t downlink_assignment_index = 0;
    std::vector<std::uint8_t> raw_payload_bits{};
};

struct PdcchDciFormat1ADecodeResult {
    PdcchCrcRntiCheck crc{};
    bool matched = false;
    PdcchDciFormat1A dci{};
};

struct PdcchCommonSearchDecodeConfig {
    PdcchTailBitingConvolutionalDecoder decoder{};
    std::uint16_t expected_rnti = kSiRnti;
    PdcchDciFormat1AConfig dci_format1a{};
};

struct PdcchCommonSearchDecodeResult {
    std::uint32_t candidate_count = 0;
    std::vector<PdcchDciFormat1ADecodeResult> hits{};
};

struct PdcchGpuCommonSearchDecodeRequest {
    PdcchMmseInput input{};
    PdcchCommonSearchDecodeConfig config{};
};

struct PdcchGpuCommonSearchDecodeProfile {
    double h2d_us = 0.0;
    double ce_mmse_gpu_us = 0.0;
    double llr_gpu_us = 0.0;
    double rate_recovery_gpu_us = 0.0;
    double viterbi_gpu_us = 0.0;
    double crc_gpu_us = 0.0;
    double d2h_us = 0.0;
    double host_submit_us = 0.0;
    double host_collect_us = 0.0;
    std::uint64_t h2d_bytes = 0U;
    std::uint64_t d2h_bytes = 0U;
    std::uint32_t crc_hit_count = 0U;
    std::uint32_t crc_miss_count = 0U;
};

struct PdcchGpuCommonSearchDecodeResult {
    std::uint32_t candidate_count = 0U;
    std::vector<PdcchDciFormat1ADecodeResult> hits{};
    PdcchGpuCommonSearchDecodeProfile profile{};
};

struct PdcchUeSpecificSearchConfig {
    std::vector<std::uint16_t> rntis{};
    std::uint8_t aggregation_level_mask = kPdcchAggregationLevelMaskAll;
    PdcchTailBitingConvolutionalDecoder decoder{};
    PdcchDciFormat1AConfig dci_format1a{};
};

struct PdcchUeSpecificDciFormat1AHit {
    std::uint16_t rnti = 0;
    std::uint16_t first_cce = 0;
    std::uint8_t aggregation_level = 0;
    PdcchDciFormat1ADecodeResult decoded{};
};

struct PdcchUeSpecificSearchResult {
    std::uint32_t candidate_count = 0;
    std::uint32_t decoded_candidate_count = 0;
    std::uint32_t crc_rnti_miss_count = 0;
    std::uint32_t semantic_reject_count = 0;
    std::vector<PdcchUeSpecificDciFormat1AHit> hits{};
};

struct PdcchSiRntiSearchConfig {
    // A non-null callback preserves an SDK-provided decoder; otherwise native Viterbi is used.
    PdcchTailBitingConvolutionalDecoder decoder{};
};

struct PdcchSiRntiDciFormat1AHit {
    std::uint16_t first_cce = 0;
    std::uint8_t aggregation_level = 0;
    PdcchDciFormat1ADecodeResult decoded{};
};

struct PdcchSiRntiSearchResult {
    std::uint32_t candidate_count = 0;
    std::vector<PdcchSiRntiDciFormat1AHit> hits{};
};

struct PdcchControlGeometry {
    std::uint8_t control_symbol_count = 0;
    PhichResource phich_resource = PhichResource::kOne;
    PhichDuration phich_duration = PhichDuration::kNormal;
    bool standard_reg_order = true;
};

enum class PdcchSiRntiGeometrySearchStatus : std::uint8_t {
    kAcquired = 0,
    kLocked,
    kMiss,
    kAmbiguous,
};

struct PdcchSiRntiGeometrySearchRequest {
    PlanarGridViewF32 grid{};
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 1;
    std::uint8_t tx_mode = 1;
    std::uint16_t n_prb = kLteNumPrb20MHz;
    LteControlSubframeContext control_subframe{};
    PdcchChainMetadata chain{};
};

struct PdcchSiRntiGeometrySearchConfig {
    PdcchTailBitingConvolutionalDecoder decoder{};
};

struct PdcchSiRntiGeometrySearchCache {
    bool locked = false;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t tx_mode = 0;
    std::uint16_t n_prb = 0;
    LteControlSubframeKind subframe_kind = LteControlSubframeKind::kRegular;
    PdcchControlGeometry geometry{};
    std::uint8_t consecutive_miss_count = 0;
};

struct PdcchSiRntiGeometrySearchResult {
    PdcchSiRntiGeometrySearchStatus status = PdcchSiRntiGeometrySearchStatus::kMiss;
    std::uint32_t geometry_attempt_count = 0;
    std::uint32_t candidate_count = 0;
    PdcchControlGeometry geometry{};
    PdcchSiRntiSearchResult decoded{};
};

} // namespace mmse::pdcch
