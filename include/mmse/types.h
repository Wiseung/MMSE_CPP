#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "mmse/constants.h"

namespace mmse {

namespace pdcch {
enum class PhichDuplexMode : std::uint8_t {
    kFdd = 0,
    kTdd,
};

enum class LteControlSubframeKind : std::uint8_t {
    kRegular = 0,
    kMbsfn,
};

struct LteControlSubframeContext {
    PhichDuplexMode duplex_mode = PhichDuplexMode::kFdd;
    std::uint8_t subframe = 0;
    std::uint8_t ul_dl_config = 0;
    LteControlSubframeKind kind = LteControlSubframeKind::kRegular;
};
} // namespace pdcch

enum class MmseStatus : std::uint8_t {
    kOk = 0,
    kNotInitialized,
    kInvalidArgument,
    kUnsupportedConfig,
    kBufferTooSmall,
    kInternalError,
};

enum class MmseCpuBackend : std::uint8_t {
    kAuto = 0,
    kScalar,
    kAvx2,
};

enum class MmseGpuBackend : std::uint8_t {
    kAuto = 0,
    kCuda,
};

enum class MmseGpuSigma2Ownership : std::uint8_t {
    kHostOwnedIir = 0,
    kDeviceOwnedState,
};

enum class MmseGpuValidationPolicy : std::uint8_t {
    kReleaseSanity = 0,
    kTestDeepTrace,
};

enum class MmseChannelType : std::uint8_t {
    kPdsch = 0,
    kPdcch,
    kPbch,
    kPcfich,
};

// FFT-domain resource grid stored as one contiguous plane per RX antenna:
// plane[rx][symbol * n_subcarriers + subcarrier]. `generation` participates in
// prepared-subframe caching and must change when the pointed-to samples change.
struct PlanarGridViewF32 {
    std::array<const float*, 2> re{};
    std::array<const float*, 2> im{};
    std::uint32_t n_rx_ant = 0;
    std::uint32_t n_symbols = 0;
    std::uint32_t n_subcarriers = 0;
    std::uint64_t generation = 0;
};

// LTE extraction parameters shared by the generic, PBCH, PCFICH and PDCCH paths.
// `prb_bitmap` uses bit `prb % 16` in word `prb / 16`; the count must agree with
// `n_prb`, while channel-specific validation supplies the remaining constraints.
struct ExtractDescriptor {
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    MmseChannelType channel_type = MmseChannelType::kPdsch;
    std::uint8_t start_symbol = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    std::uint16_t n_prb = 0;
    std::array<std::uint16_t, 7> prb_bitmap{};
    std::array<std::uint16_t, kLteMaxPdcchControlSymbolsNormalCp * kLteNumPrb20MHz>
        control_re_exclusion_masks{};
    std::int8_t pmi = -1;
};

struct PdcchChainMetadata {
    std::uint64_t request_id = 0;
    std::uint32_t candidate_id = 0;
    std::uint16_t first_cce = 0;
    std::uint8_t aggregation_level = 0;
};

struct PbchChainMetadata {
    std::uint64_t request_id = 0;
};

struct PcfichChainMetadata {
    std::uint64_t request_id = 0;
};

struct PbchMmseInput {
    PlanarGridViewF32 grid{};
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 1;
    std::uint8_t tx_mode = 1;
    PbchChainMetadata chain{};
};

struct PcfichMmseInput {
    PlanarGridViewF32 grid{};
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 1;
    std::uint8_t tx_mode = 1;
    PcfichChainMetadata chain{};
};

struct PdcchMmseInput {
    PlanarGridViewF32 grid{};
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 1;
    std::uint8_t tx_mode = 1;
    std::uint8_t control_symbol_count = 0;
    std::uint16_t n_prb = 0;
    std::array<std::uint16_t, 7> prb_bitmap{};
    pdcch::LteControlSubframeContext control_subframe{};
    std::array<std::uint16_t, kLteMaxPdcchControlSymbolsNormalCp * kLteNumPrb20MHz>
        control_re_exclusion_masks{};
    PdcchChainMetadata chain{};
};

// Layer-major output view. For layer l and extracted RE r, the sample is at
// l * capacity_re_per_layer + r. `n_re_per_layer` and `n_layers` are written by
// the runtime after it has built the channel-specific RE layout.
struct EqualizerOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint32_t capacity_re_per_layer = 0;
    std::uint32_t n_re_per_layer = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t mod_order = 0;
};

struct PbchMmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices = nullptr;
    std::uint32_t capacity_re_per_layer = 0;
    std::uint32_t capacity_re_metadata = 0;
};

struct PcfichMmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices = nullptr;
    std::uint32_t capacity_re_per_layer = 0;
    std::uint32_t capacity_re_metadata = 0;
};

struct PdcchMmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices = nullptr;
    std::uint32_t capacity_re_per_layer = 0;
    std::uint32_t capacity_re_metadata = 0;
};

struct PdcchTdMmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices0 = nullptr;
    std::uint16_t* re_grid_indices1 = nullptr;
    std::uint32_t capacity_symbols = 0;
};

struct PbchTdMmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices0 = nullptr;
    std::uint16_t* re_grid_indices1 = nullptr;
    std::uint32_t capacity_symbols = 0;
};

struct PcfichTdMmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices0 = nullptr;
    std::uint16_t* re_grid_indices1 = nullptr;
    std::uint32_t capacity_symbols = 0;
};

// Four-port transmit diversity emits four recovered symbols per frequency-
// switched block. Every output slot records the four source REs of its block.
struct PdcchTd4MmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices0 = nullptr;
    std::uint16_t* re_grid_indices1 = nullptr;
    std::uint16_t* re_grid_indices2 = nullptr;
    std::uint16_t* re_grid_indices3 = nullptr;
    std::uint32_t capacity_symbols = 0;
};

struct PbchTd4MmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices0 = nullptr;
    std::uint16_t* re_grid_indices1 = nullptr;
    std::uint16_t* re_grid_indices2 = nullptr;
    std::uint16_t* re_grid_indices3 = nullptr;
    std::uint32_t capacity_symbols = 0;
};

struct PcfichTd4MmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices0 = nullptr;
    std::uint16_t* re_grid_indices1 = nullptr;
    std::uint16_t* re_grid_indices2 = nullptr;
    std::uint16_t* re_grid_indices3 = nullptr;
    std::uint32_t capacity_symbols = 0;
};

// One transmit-diversity output sequence. Adjacent symbols are recovered from
// the paired source REs referenced by re_grid_indices0/re_grid_indices1.
struct PdschTdMmseOutputView {
    float* x_hat_re = nullptr;
    float* x_hat_im = nullptr;
    float* sinr = nullptr;
    std::uint16_t* re_grid_indices0 = nullptr;
    std::uint16_t* re_grid_indices1 = nullptr;
    std::uint32_t capacity_symbols = 0;
};

struct PbchMmseResult {
    std::uint32_t n_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t n_symbols = 0;
    std::uint32_t n_subcarriers = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t start_prb = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PbchChainMetadata chain{};
};

struct PcfichMmseResult {
    std::uint32_t n_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t n_symbols = 0;
    std::uint32_t n_subcarriers = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t reg_count = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PcfichChainMetadata chain{};
};

struct PdcchMmseResult {
    std::uint32_t n_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t n_symbols = 0;
    std::uint32_t n_subcarriers = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    std::array<std::uint16_t, 7> prb_bitmap{};
    PdcchChainMetadata chain{};
};

struct PdcchTdMmseResult {
    std::uint32_t n_symbols = 0;
    std::uint32_t n_source_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t grid_symbol_count = 0;
    std::uint32_t grid_subcarrier_count = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    std::array<std::uint16_t, 7> prb_bitmap{};
    PdcchChainMetadata chain{};
};

struct PbchTdMmseResult {
    std::uint32_t n_symbols = 0;
    std::uint32_t n_source_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t grid_symbol_count = 0;
    std::uint32_t grid_subcarrier_count = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t start_prb = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PbchChainMetadata chain{};
};

struct PcfichTdMmseResult {
    std::uint32_t n_symbols = 0;
    std::uint32_t n_source_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t grid_symbol_count = 0;
    std::uint32_t grid_subcarrier_count = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t reg_count = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PcfichChainMetadata chain{};
};

struct PdcchTd4MmseResult {
    std::uint32_t n_symbols = 0;
    std::uint32_t n_source_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t grid_symbol_count = 0;
    std::uint32_t grid_subcarrier_count = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t control_symbol_count = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    std::array<std::uint16_t, 7> prb_bitmap{};
    PdcchChainMetadata chain{};
};

struct PbchTd4MmseResult {
    std::uint32_t n_symbols = 0;
    std::uint32_t n_source_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t grid_symbol_count = 0;
    std::uint32_t grid_subcarrier_count = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t start_prb = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PbchChainMetadata chain{};
};

struct PcfichTd4MmseResult {
    std::uint32_t n_symbols = 0;
    std::uint32_t n_source_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t grid_symbol_count = 0;
    std::uint32_t grid_subcarrier_count = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t reg_count = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    float sigma2 = 0.0F;
    PcfichChainMetadata chain{};
};

struct PdschTdMmseResult {
    std::uint32_t n_symbols = 0;
    std::uint32_t n_source_re = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint32_t grid_symbol_count = 0;
    std::uint32_t grid_subcarrier_count = 0;
    std::uint16_t cell_id = 0;
    std::uint16_t n_prb = 0;
    std::uint8_t start_symbol = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t tx_mode = 0;
    std::uint8_t mod_order = 0;
    std::int8_t pmi = -1;
    float sigma2 = 0.0F;
    std::array<std::uint16_t, 7> prb_bitmap{};
};

// Numerical safeguards and execution policy for the CPU MMSE runtime.
struct MmseEqualizerCpuConfig {
    std::uint32_t worker_count = 1;
    float sigma2_iir_alpha = 0.8F;
    float sigma2_min = 1.0e-4F;
    float det_floor = 1.0e-6F;
    float g_min = 1.0e-4F;
    float gamma_max = 1.0e4F;
    MmseCpuBackend backend = MmseCpuBackend::kAuto;
};

// GPU runtime policy. CUDA keeps staging buffers per stream; sigma2 ownership
// selects whether the host IIR state or the device-side state is authoritative.
struct MmseEqualizerGpuConfig {
    std::uint32_t device_ordinal = 0;
    std::uint32_t stream_count = 1;
    float sigma2_iir_alpha = 0.8F;
    float sigma2_min = 1.0e-4F;
    float det_floor = 1.0e-6F;
    float g_min = 1.0e-4F;
    float gamma_max = 1.0e4F;
    MmseGpuBackend backend = MmseGpuBackend::kAuto;
    MmseGpuSigma2Ownership sigma2_ownership = MmseGpuSigma2Ownership::kHostOwnedIir;
    // ReleaseSanity checks finite sampled outputs; TestDeepTrace also compares
    // sampled intermediate values against the scalar CPU implementation.
    MmseGpuValidationPolicy validation_policy = MmseGpuValidationPolicy::kReleaseSanity;
};

struct MmseGpuHostProfileSnapshot {
    double quantize_us = 0.0;
    double layout_build_us = 0.0;
    double grid_meta_pack_us = 0.0;
    double grid_h2d_us = 0.0;
    double estimate_launch_us = 0.0;
    double sigma2_d2h_us = 0.0;
    double sigma2_mid_sync_us = 0.0;
    double sigma2_host_update_us = 0.0;
    double sigma2_h2d_us = 0.0;
    double equalize_launch_us = 0.0;
    double outputs_d2h_us = 0.0;
    double scratch_d2h_us = 0.0;
    double completion_d2h_us = 0.0;
    double final_sync_us = 0.0;
    double output_stage_us = 0.0;
    double total_host_us = 0.0;
    double estimate_gpu_us = 0.0;
    double estimate_residual_gpu_us = 0.0;
    double estimate_channel_gpu_us = 0.0;
    double equalize_gpu_us = 0.0;
    double stream_gpu_us = 0.0;
};

} // namespace mmse
