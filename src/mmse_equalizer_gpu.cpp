#include "mmse/mmse_equalizer.h"
#include "mmse/lte_descrambling.h"
#include "mmse/pbch_module_api.h"
#include "mmse/pdcch_module_api.h"
#include "mmse/pcfich_module_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "internal/mmse_cuda_runtime.h"
#include "internal/mmse_internal.h"

namespace mmse {

namespace {

constexpr std::uint32_t kPinnedRingSlotCount = 16U;
constexpr std::size_t kGridPlaneCount = detail::kMaxGridRe;
constexpr std::size_t kEstimateStubFloatCount = detail::kCudaEstimateStubFloatCount;
constexpr std::size_t kOutputPlaneCount = detail::kMaxDataRe * 2U;

MmseEqualizerCpuConfig make_cpu_fallback_config(const MmseEqualizerGpuConfig& config) {
    MmseEqualizerCpuConfig cpu{};
    cpu.worker_count = 1;
    cpu.sigma2_iir_alpha = config.sigma2_iir_alpha;
    cpu.sigma2_min = config.sigma2_min;
    cpu.det_floor = config.det_floor;
    cpu.g_min = config.g_min;
    cpu.gamma_max = config.gamma_max;
    cpu.backend = MmseCpuBackend::kAuto;
    return cpu;
}

bool use_deep_validation(const MmseEqualizerGpuConfig& config) {
    return config.validation_policy == MmseGpuValidationPolicy::kTestDeepTrace;
}

bool use_device_owned_sigma2(const MmseEqualizerGpuConfig& config) {
    return config.sigma2_ownership == MmseGpuSigma2Ownership::kDeviceOwnedState;
}

bool pack_transmit_diversity_pairs(const detail::ReLayout& layout,
                                   detail::CudaGridMeta& grid_meta) {
    if ((layout.n_re & 1U) != 0U || layout.n_re / 2U > detail::kCudaMaxDataRe / 2U) {
        return false;
    }
    grid_meta.td_pair_count = layout.n_re / 2U;
    for (std::uint32_t pair = 0; pair < grid_meta.td_pair_count; ++pair) {
        grid_meta.td_pair_grid_indices0[pair] = layout.grid_indices[pair * 2U];
        grid_meta.td_pair_grid_indices1[pair] = layout.grid_indices[pair * 2U + 1U];
    }
    return true;
}

bool is_pdsch_transmit_diversity_descriptor(const ExtractDescriptor& desc) {
    return desc.channel_type == MmseChannelType::kPdsch && desc.n_tx_ports == 2U &&
           desc.n_layers == 1U && desc.tx_mode == 2U && desc.pmi == -1;
}

bool transmit_diversity_pairs_stay_in_one_symbol(const detail::CudaGridMeta& grid_meta) {
    if (grid_meta.n_subcarriers == 0U) {
        return false;
    }
    for (std::uint32_t pair = 0U; pair < grid_meta.td_pair_count; ++pair) {
        if (grid_meta.td_pair_grid_indices0[pair] / grid_meta.n_subcarriers !=
            grid_meta.td_pair_grid_indices1[pair] / grid_meta.n_subcarriers) {
            return false;
        }
    }
    return true;
}

std::uint32_t populate_pdcch_gold_words(std::uint16_t cell_id, std::uint32_t sfn_subframe,
                                        std::uint32_t n_re, std::uint32_t* words) {
    const std::uint32_t bit_count = n_re * 2U;
    const std::uint32_t word_count = (bit_count + 31U) / 32U;
    std::fill_n(words, detail::kCudaPdcchGoldWordCount, 0U);
    std::uint32_t x1 = 0U;
    std::uint32_t x2 = 0U;
    lte::detail::init_gold(lte::pdcch_c_init(cell_id, sfn_subframe), x1, x2);
    for (std::uint32_t bit = 0U; bit < bit_count; ++bit) {
        words[bit >> 5U] |= ((x1 ^ x2) & 1U) << (bit & 31U);
        lte::detail::gold_step(x1, x2);
    }
    return word_count;
}

} // namespace

struct MmseEqualizerGpuContext::Impl {
    using Clock = std::chrono::steady_clock;
    struct PreparedSubframeState {
        detail::PreparedSubframeKey key{};
        bool valid = false;
        std::uint32_t slot_index = 0;
        float sigma2 = 0.0F;
    };

    struct PendingPdcchDecode {
        std::uint32_t slot_index = 0U;
        std::uint32_t candidate_count = 0U;
        PdcchMmseInput input{};
        pdcch::PdcchCommonSearchDecodeConfig decode_config{};
        Clock::time_point submit_start{};
        Clock::time_point d2h_start{};
        double ce_mmse_gpu_us = 0.0;
        std::uint64_t h2d_bytes = 0U;
    };

    struct StreamState {
        std::uint32_t ordinal = 0;
        std::uintptr_t handle = 0;
        bool ready = false;
    };

    struct DeviceState {
        std::uint32_t requested_device_ordinal = 0;
        std::uint32_t active_device_ordinal = 0;
        bool selection_ready = false;
        bool cuda_runtime_available = false;
        std::vector<StreamState> streams{};
    };

    // One slot owns host/device buffers and events for one in-flight stream.
    // Slots form a ring so input quantization/copy for the next request can be
    // separated from output collection of the previous request.
    struct HostPinnedSlot {
        std::uint32_t slot_ordinal = 0;
        std::uint32_t stream_ordinal = 0;
        std::uintptr_t stream_handle = 0;
        std::uintptr_t gpu_event_h2d_start = 0;
        std::uintptr_t gpu_event_stream_start = 0;
        std::uintptr_t gpu_event_residual_done = 0;
        std::uintptr_t gpu_event_estimate_done = 0;
        std::uintptr_t gpu_event_equalize_done = 0;
        std::uintptr_t gpu_event_pdcch_llr_done = 0;
        std::uintptr_t gpu_event_pdcch_rate_recovery_done = 0;
        std::uintptr_t gpu_event_pdcch_viterbi_done = 0;
        std::uintptr_t gpu_event_pdcch_crc_done = 0;
        std::uintptr_t gpu_event_stream_done = 0;
        bool pinned_ready = false;
        bool input_ready = false;
        bool output_ready = false;
        bool completion_ready = false;
        bool cpu_seeded = false;
        bool h2d_submitted = false;
        bool kernel_submitted = false;
        bool d2h_submitted = false;
        bool device_buffers_ready = false;
        bool host_grid_is_pinned = false;
        bool host_output_is_pinned = false;
        std::uint32_t completion_value = 0;
        std::uint64_t sequence = 0;
        std::size_t grid_plane_bytes = 0;
        std::size_t estimate_bytes = 0;
        std::size_t xhat_plane_bytes = 0;
        std::size_t sinr_plane_bytes = 0;
        detail::CudaGridMeta grid_meta{};
        detail::CudaGridMeta* grid_meta_staging = nullptr;
        bool grid_meta_staging_pinned = false;
        detail::ReLayout layout{};
        std::array<float*, 2> transport_re{};
        std::array<float*, 2> transport_im{};
        float* h_estimate = nullptr;
        std::array<float, detail::kCudaScratchTraceFloatCount> scratch_host{};
        detail::CudaPdcchCandidateDescriptor* pdcch_candidates_host = nullptr;
        bool pdcch_candidates_host_pinned = false;
        detail::CudaPdcchCandidateResult* pdcch_results_host = nullptr;
        bool pdcch_results_host_pinned = false;
        std::uint32_t* pdcch_hit_count = nullptr;
        bool pdcch_hit_count_host_pinned = false;
        bool pdcch_decode_submitted = false;
        std::uint32_t* pdcch_gold_words_host = nullptr;
        bool pdcch_gold_words_host_pinned = false;
        float* sigma2_host = nullptr;
        bool sigma2_host_pinned = false;
        std::array<float, kOutputPlaneCount> ref_xhat_re{};
        std::array<float, kOutputPlaneCount> ref_xhat_im{};
        std::array<float, kOutputPlaneCount> ref_sinr{};
        float* xhat_re = nullptr;
        float* xhat_im = nullptr;
        float* sinr = nullptr;
        PlanarGridViewF32 grid_view{};
        EqualizerOutputView out_view{};
        detail::CudaDeviceBuffers device{};
    };

    // The ring is bounded by stream_count. A slot is reused only after its
    // stream has been synchronized or its pending decode has been collected.
    struct StagingBuffers {
        std::array<HostPinnedSlot, kPinnedRingSlotCount> slots{};
        std::uint32_t slot_count = 0U;
        std::uint32_t next_stage_slot = 0;
        std::uint32_t active_slot = 0;
        std::uint32_t completed_slot = 0;
        std::uint64_t next_sequence = 1;
        bool ring_ready = false;
    };

    static MmseStatus validate_config(const MmseEqualizerGpuConfig& config);
    static bool nearly_equal(float lhs, float rhs, float rel_tol, float abs_tol);
    MmseStatus compare_equalize_trace_sample(const HostPinnedSlot& slot,
                                             std::uint32_t sample_index) const;
    bool deep_validation_enabled() const;
    std::size_t scratch_float_count() const;
    void reset_runtime_state();
    void release_runtime_state();
    MmseStatus configure_device_state();
    MmseStatus configure_staging_buffers();
    MmseStatus initialize_slot_storage(HostPinnedSlot& slot);
    MmseStatus initialize_slot_device_buffers(HostPinnedSlot& slot);
    MmseStatus stage_inputs(const PlanarGridViewF32& grid, const ExtractDescriptor& desc);
    MmseStatus prepare_subframe_if_needed(const PlanarGridViewF32& grid,
                                          const ExtractDescriptor& desc);
    MmseStatus validate_estimate_stub(const ExtractDescriptor& desc,
                                      const HostPinnedSlot& slot) const;
    MmseStatus validate_equalize_stub(const HostPinnedSlot& slot) const;
    MmseStatus execute_backend(const ExtractDescriptor& desc);
    MmseStatus execute_cuda_transport_stub(const ExtractDescriptor& desc, HostPinnedSlot& slot,
                                           bool reuse_estimate, bool copy_equalized_outputs = true,
                                           bool synchronize = true);
    MmseStatus
    submit_pdcch_gpu_common_search_decode(const pdcch::PdcchGpuCommonSearchDecodeRequest& request,
                                          PendingPdcchDecode& pending);
    MmseStatus
    collect_pdcch_gpu_common_search_decode(const PendingPdcchDecode& pending,
                                           pdcch::PdcchGpuCommonSearchDecodeResult& result);
    MmseStatus
    run_pdcch_gpu_common_search_decode(const pdcch::PdcchGpuCommonSearchDecodeRequest& request,
                                       pdcch::PdcchGpuCommonSearchDecodeResult& result);
    MmseStatus stage_outputs(EqualizerOutputView& out) const;
    MmseStatus run_transmit_diversity(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                                      EqualizerOutputView& out);
    static double elapsed_us(const Clock::time_point& start, const Clock::time_point& end);

    MmseEqualizerGpuConfig config{};
    bool initialized = false;
    bool using_cpu_fallback = false;
    DeviceState device{};
    std::array<detail::Sigma2State, kLteNumCellIds> sigma2_by_cell{};
    MmseEqualizerCpuContext cpu_fallback{};
    StagingBuffers buffers{};
    PreparedSubframeState prepared{};
    MmseGpuHostProfileSnapshot last_host_profile{};
};

MmseEqualizerGpuContext::MmseEqualizerGpuContext() : impl_(new Impl()) {}

MmseEqualizerGpuContext::~MmseEqualizerGpuContext() {
    impl_->release_runtime_state();
    delete impl_;
}

MmseStatus MmseEqualizerGpuContext::Impl::validate_config(const MmseEqualizerGpuConfig& config) {
    if (config.stream_count == 0U || config.stream_count > kPinnedRingSlotCount ||
        config.g_min <= 0.0F || config.g_min >= 0.5F || config.gamma_max <= 0.0F ||
        config.det_floor <= 0.0F || config.sigma2_min <= 0.0F || config.sigma2_iir_alpha < 0.0F ||
        config.sigma2_iir_alpha > 1.0F) {
        return MmseStatus::kInvalidArgument;
    }
    return MmseStatus::kOk;
}

bool MmseEqualizerGpuContext::Impl::nearly_equal(float lhs, float rhs, float rel_tol,
                                                 float abs_tol) {
    const float abs_err = std::abs(lhs - rhs);
    const float scale = std::max({std::abs(lhs), std::abs(rhs), 1.0F});
    return abs_err <= std::max(abs_tol, rel_tol * scale);
}

bool MmseEqualizerGpuContext::Impl::deep_validation_enabled() const {
    return use_deep_validation(config);
}

double MmseEqualizerGpuContext::Impl::elapsed_us(const Clock::time_point& start,
                                                 const Clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start)
        .count();
}

std::size_t MmseEqualizerGpuContext::Impl::scratch_float_count() const {
    return deep_validation_enabled() ? detail::kCudaScratchTraceFloatCount
                                     : detail::kCudaScratchHeaderFloatCount;
}

MmseStatus
MmseEqualizerGpuContext::Impl::compare_equalize_trace_sample(const HostPinnedSlot& slot,
                                                             std::uint32_t sample_index) const {
    if (slot.h_estimate == nullptr) {
        return MmseStatus::kInternalError;
    }
    if (sample_index >= slot.grid_meta.trace_sample_count) {
        return MmseStatus::kInvalidArgument;
    }

    const std::uint32_t re = slot.grid_meta.spot_check_re_slots[sample_index];
    if (re >= slot.layout.n_re) {
        return MmseStatus::kInternalError;
    }

    const std::uint32_t grid_idx = slot.layout.grid_indices[re];
    const std::uint32_t symbol = grid_idx / slot.grid_meta.n_subcarriers;
    const std::uint32_t sc = grid_idx % slot.grid_meta.n_subcarriers;
    const float y0_re = slot.transport_re[0][grid_idx];
    const float y0_im = slot.transport_im[0][grid_idx];
    const float y1_re = slot.transport_re[1][grid_idx];
    const float y1_im = slot.transport_im[1][grid_idx];
    const auto h_at = [&](std::uint32_t tx, std::uint32_t rx) {
        const std::size_t base =
            2U * (((tx * kMmseV1MaxNumRxAntennas + rx) * kLteNumSymbolsNormalCp + symbol) *
                      kLteNumSubcarriers20MHz +
                  sc);
        return detail::Complex32{slot.h_estimate[base], slot.h_estimate[base + 1U]};
    };

    const detail::Equalize2x2Trace cpu_trace = detail::trace_equalize_2x2_scalar(
        h_at(0U, 0U), h_at(1U, 0U), h_at(0U, 1U), h_at(1U, 1U), {y0_re, y0_im}, {y1_re, y1_im},
        slot.grid_meta.sigma2, config.det_floor, config.g_min, config.gamma_max);

    const std::size_t offset =
        detail::kCudaScratchHeaderFloatCount +
        static_cast<std::size_t>(sample_index) * detail::kCudaEqualizeTraceFloatCount;
    const float* gpu = slot.scratch_host.data() + offset;
    const auto check = [&](float actual, float expected, float rel_tol = 1.0e-4F,
                           float abs_tol = 1.0e-5F) {
        return nearly_equal(actual, expected, rel_tol, abs_tol);
    };

    if (!check(gpu[0], static_cast<float>(re), 0.0F, 0.5F) ||
        !check(gpu[1], static_cast<float>(symbol), 0.0F, 0.5F) ||
        !check(gpu[2], static_cast<float>(sc), 0.0F, 0.5F) || !check(gpu[3], cpu_trace.a11) ||
        !check(gpu[4], cpu_trace.a22) || !check(gpu[5], cpu_trace.a12.re) ||
        !check(gpu[6], cpu_trace.a12.im) || !check(gpu[7], cpu_trace.det) ||
        !check(gpu[8], cpu_trace.w00.re) || !check(gpu[9], cpu_trace.w00.im) ||
        !check(gpu[10], cpu_trace.w01.re) || !check(gpu[11], cpu_trace.w01.im) ||
        !check(gpu[12], cpu_trace.w10.re) || !check(gpu[13], cpu_trace.w10.im) ||
        !check(gpu[14], cpu_trace.w11.re) || !check(gpu[15], cpu_trace.w11.im) ||
        !check(gpu[16], cpu_trace.z0.re) || !check(gpu[17], cpu_trace.z0.im) ||
        !check(gpu[18], cpu_trace.z1.re) || !check(gpu[19], cpu_trace.z1.im) ||
        !check(gpu[20], cpu_trace.g0) || !check(gpu[21], cpu_trace.g1) ||
        !check(gpu[22], cpu_trace.xhat0.re) || !check(gpu[23], cpu_trace.xhat0.im) ||
        !check(gpu[24], cpu_trace.xhat1.re) || !check(gpu[25], cpu_trace.xhat1.im) ||
        !check(gpu[26], cpu_trace.gamma0, 5.0e-4F, 1.0e-3F) ||
        !check(gpu[27], cpu_trace.gamma1, 5.0e-4F, 1.0e-3F) || !check(gpu[28], y0_re) ||
        !check(gpu[29], y0_im) || !check(gpu[30], y1_re) || !check(gpu[31], y1_im)) {
        return MmseStatus::kInternalError;
    }
    return MmseStatus::kOk;
}

void MmseEqualizerGpuContext::Impl::release_runtime_state() {
    // Device allocations, events, and pinned host memory are released in the
    // reverse ownership domain in which they are used; the wrapper functions
    // select cudaFree/cudaFreeHost versus ordinary host deletion.
    for (auto& slot : buffers.slots) {
        detail::cuda_free_device_buffer(slot.device.grid_re[0]);
        detail::cuda_free_device_buffer(slot.device.grid_re[1]);
        detail::cuda_free_device_buffer(slot.device.grid_im[0]);
        detail::cuda_free_device_buffer(slot.device.grid_im[1]);
        detail::cuda_free_device_buffer(slot.device.grid_meta);
        detail::cuda_free_device_buffer(slot.device.sigma2);
        detail::cuda_free_device_buffer(slot.device.scratch);
        detail::cuda_free_device_buffer(slot.device.h_estimate);
        detail::cuda_free_device_buffer(slot.device.xhat_re);
        detail::cuda_free_device_buffer(slot.device.xhat_im);
        detail::cuda_free_device_buffer(slot.device.sinr);
        detail::cuda_free_device_buffer(slot.device.completion);
        detail::cuda_free_device_buffer(slot.device.pdcch_llrs);
        detail::cuda_free_device_buffer(slot.device.pdcch_gold_words);
        detail::cuda_free_device_buffer(slot.device.pdcch_candidates);
        detail::cuda_free_device_buffer(slot.device.pdcch_recovered_llrs);
        detail::cuda_free_device_buffer(slot.device.pdcch_survivors);
        detail::cuda_free_device_buffer(slot.device.pdcch_terminal_metrics);
        detail::cuda_free_device_buffer(slot.device.pdcch_candidate_results);
        detail::cuda_free_device_buffer(slot.device.pdcch_results);
        detail::cuda_free_device_buffer(slot.device.pdcch_hit_count);
        detail::cuda_destroy_event(slot.gpu_event_stream_start);
        detail::cuda_destroy_event(slot.gpu_event_h2d_start);
        detail::cuda_destroy_event(slot.gpu_event_residual_done);
        detail::cuda_destroy_event(slot.gpu_event_estimate_done);
        detail::cuda_destroy_event(slot.gpu_event_equalize_done);
        detail::cuda_destroy_event(slot.gpu_event_pdcch_llr_done);
        detail::cuda_destroy_event(slot.gpu_event_pdcch_rate_recovery_done);
        detail::cuda_destroy_event(slot.gpu_event_pdcch_viterbi_done);
        detail::cuda_destroy_event(slot.gpu_event_pdcch_crc_done);
        detail::cuda_destroy_event(slot.gpu_event_stream_done);

        detail::cuda_free_host_f32(slot.transport_re[0], slot.host_grid_is_pinned);
        detail::cuda_free_host_f32(slot.transport_re[1], slot.host_grid_is_pinned);
        detail::cuda_free_host_f32(slot.transport_im[0], slot.host_grid_is_pinned);
        detail::cuda_free_host_f32(slot.transport_im[1], slot.host_grid_is_pinned);
        detail::cuda_free_host_f32(slot.h_estimate, slot.host_output_is_pinned);
        detail::cuda_free_host_f32(slot.xhat_re, slot.host_output_is_pinned);
        detail::cuda_free_host_f32(slot.xhat_im, slot.host_output_is_pinned);
        detail::cuda_free_host_f32(slot.sinr, slot.host_output_is_pinned);
        detail::cuda_free_host_bytes(slot.grid_meta_staging, slot.grid_meta_staging_pinned);
        detail::cuda_free_host_bytes(slot.pdcch_candidates_host, slot.pdcch_candidates_host_pinned);
        detail::cuda_free_host_bytes(slot.pdcch_results_host, slot.pdcch_results_host_pinned);
        detail::cuda_free_host_bytes(slot.pdcch_hit_count, slot.pdcch_hit_count_host_pinned);
        detail::cuda_free_host_bytes(slot.pdcch_gold_words_host, slot.pdcch_gold_words_host_pinned);
        detail::cuda_free_host_f32(slot.sigma2_host, slot.sigma2_host_pinned);
        slot.slot_ordinal = 0;
        slot.stream_ordinal = 0;
        slot.stream_handle = 0;
        slot.gpu_event_h2d_start = 0;
        slot.gpu_event_stream_start = 0;
        slot.gpu_event_residual_done = 0;
        slot.gpu_event_estimate_done = 0;
        slot.gpu_event_equalize_done = 0;
        slot.gpu_event_pdcch_llr_done = 0;
        slot.gpu_event_pdcch_rate_recovery_done = 0;
        slot.gpu_event_pdcch_viterbi_done = 0;
        slot.gpu_event_pdcch_crc_done = 0;
        slot.gpu_event_stream_done = 0;
        slot.pinned_ready = false;
        slot.input_ready = false;
        slot.output_ready = false;
        slot.completion_ready = false;
        slot.cpu_seeded = false;
        slot.h2d_submitted = false;
        slot.kernel_submitted = false;
        slot.d2h_submitted = false;
        slot.device_buffers_ready = false;
        slot.host_grid_is_pinned = false;
        slot.host_output_is_pinned = false;
        slot.completion_value = 0;
        slot.sequence = 0;
        slot.grid_plane_bytes = 0;
        slot.estimate_bytes = 0;
        slot.xhat_plane_bytes = 0;
        slot.sinr_plane_bytes = 0;
        slot.grid_meta = {};
        slot.grid_meta_staging = nullptr;
        slot.grid_meta_staging_pinned = false;
        slot.layout = {};
        slot.transport_re = {};
        slot.transport_im = {};
        slot.h_estimate = nullptr;
        slot.pdcch_candidates_host = nullptr;
        slot.pdcch_candidates_host_pinned = false;
        slot.pdcch_results_host = nullptr;
        slot.pdcch_results_host_pinned = false;
        slot.pdcch_hit_count = nullptr;
        slot.pdcch_hit_count_host_pinned = false;
        slot.pdcch_decode_submitted = false;
        slot.pdcch_gold_words_host = nullptr;
        slot.pdcch_gold_words_host_pinned = false;
        slot.sigma2_host = nullptr;
        slot.sigma2_host_pinned = false;
        slot.xhat_re = nullptr;
        slot.xhat_im = nullptr;
        slot.sinr = nullptr;
        slot.grid_view = {};
        slot.out_view = {};
        slot.device = {};
    }

    for (auto& stream : device.streams) {
        detail::cuda_destroy_stream(stream.handle);
        stream = {};
    }
    device.streams.clear();
}

void MmseEqualizerGpuContext::Impl::reset_runtime_state() {
    release_runtime_state();
    initialized = false;
    using_cpu_fallback = false;
    device = {};
    buffers.next_stage_slot = 0;
    buffers.slot_count = 0U;
    buffers.active_slot = 0;
    buffers.completed_slot = 0;
    buffers.next_sequence = 1;
    buffers.ring_ready = false;
    prepared = {};
}

MmseStatus MmseEqualizerGpuContext::Impl::configure_device_state() {
    device.requested_device_ordinal = config.device_ordinal;
    device.active_device_ordinal = config.device_ordinal;
    device.selection_ready = true;
    device.cuda_runtime_available = false;
    device.streams.clear();

    // Auto deliberately leaves the CUDA device unselected and uses the CPU
    // fallback. Explicit CUDA must successfully select a device and create all
    // non-blocking streams before staging buffers are allocated.
    const bool request_cuda_runtime = (config.backend == MmseGpuBackend::kCuda);
    if (!request_cuda_runtime) {
        return MmseStatus::kOk;
    }
    if (!detail::cuda_backend_compiled()) {
        return MmseStatus::kUnsupportedConfig;
    }

    bool runtime_available = false;
    if (const MmseStatus status =
            detail::cuda_select_device(config.device_ordinal, runtime_available);
        status != MmseStatus::kOk) {
        device.cuda_runtime_available = runtime_available;
        return status;
    }

    device.cuda_runtime_available = runtime_available;
    device.streams.reserve(config.stream_count);
    for (std::uint32_t stream_index = 0; stream_index < config.stream_count; ++stream_index) {
        StreamState stream{};
        stream.ordinal = stream_index;
        if (const MmseStatus status = detail::cuda_create_stream(stream.handle);
            status != MmseStatus::kOk) {
            return status;
        }
        stream.ready = true;
        device.streams.push_back(stream);
    }
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::initialize_slot_storage(HostPinnedSlot& slot) {
    // Transport and output arrays are pinned when CUDA is requested so the
    // asynchronous copies below can overlap with host work. The no-CUDA stub
    // transparently falls back to ordinary host allocations.
    bool grid_pinned = false;
    bool output_pinned = false;
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.transport_re[0], kGridPlaneCount, true, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.host_grid_is_pinned = grid_pinned;
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.transport_re[1], kGridPlaneCount, true, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.transport_im[0], kGridPlaneCount, true, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.transport_im[1], kGridPlaneCount, true, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (deep_validation_enabled()) {
        if (const MmseStatus status = detail::cuda_alloc_host_f32(
                slot.h_estimate, kEstimateStubFloatCount, true, output_pinned);
            status != MmseStatus::kOk) {
            return status;
        }
        slot.host_output_is_pinned = output_pinned;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.xhat_re, kOutputPlaneCount, true, output_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.host_output_is_pinned = output_pinned;
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.xhat_im, kOutputPlaneCount, true, output_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.sinr, kOutputPlaneCount, true, output_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    void* grid_meta = nullptr;
    if (const MmseStatus status = detail::cuda_alloc_host_bytes(
            grid_meta, sizeof(detail::CudaGridMeta), true, slot.grid_meta_staging_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.grid_meta_staging = static_cast<detail::CudaGridMeta*>(grid_meta);
    *slot.grid_meta_staging = {};
    void* pdcch_candidates = nullptr;
    if (const MmseStatus status = detail::cuda_alloc_host_bytes(
            pdcch_candidates,
            detail::kCudaPdcchMaxCandidates * sizeof(detail::CudaPdcchCandidateDescriptor), true,
            slot.pdcch_candidates_host_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.pdcch_candidates_host =
        static_cast<detail::CudaPdcchCandidateDescriptor*>(pdcch_candidates);
    void* pdcch_results = nullptr;
    if (const MmseStatus status = detail::cuda_alloc_host_bytes(
            pdcch_results,
            detail::kCudaPdcchMaxCandidates * sizeof(detail::CudaPdcchCandidateResult), true,
            slot.pdcch_results_host_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.pdcch_results_host = static_cast<detail::CudaPdcchCandidateResult*>(pdcch_results);
    void* pdcch_hit_count = nullptr;
    if (const MmseStatus status = detail::cuda_alloc_host_bytes(
            pdcch_hit_count, sizeof(std::uint32_t), true, slot.pdcch_hit_count_host_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.pdcch_hit_count = static_cast<std::uint32_t*>(pdcch_hit_count);
    *slot.pdcch_hit_count = 0U;
    void* pdcch_gold_words = nullptr;
    if (const MmseStatus status = detail::cuda_alloc_host_bytes(
            pdcch_gold_words, detail::kCudaPdcchGoldWordCount * sizeof(std::uint32_t), true,
            slot.pdcch_gold_words_host_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.pdcch_gold_words_host = static_cast<std::uint32_t*>(pdcch_gold_words);
    std::fill_n(slot.pdcch_gold_words_host, detail::kCudaPdcchGoldWordCount, 0U);
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.sigma2_host, 1U, true, slot.sigma2_host_pinned);
        status != MmseStatus::kOk) {
        return status;
    }

    slot.grid_view.re = {slot.transport_re[0], slot.transport_re[1]};
    slot.grid_view.im = {slot.transport_im[0], slot.transport_im[1]};
    slot.grid_view.n_rx_ant = kMmseV1MaxNumRxAntennas;
    slot.grid_view.n_symbols = kLteNumSymbolsNormalCp;
    slot.grid_view.n_subcarriers = kLteNumSubcarriers20MHz;

    slot.out_view.x_hat_re = slot.xhat_re;
    slot.out_view.x_hat_im = slot.xhat_im;
    slot.out_view.sinr = slot.sinr;
    slot.out_view.capacity_re_per_layer = detail::kMaxDataRe;
    slot.out_view.n_re_per_layer = 0;
    slot.out_view.n_layers = 0;
    slot.out_view.mod_order = 0;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::initialize_slot_device_buffers(HostPinnedSlot& slot) {
    if (config.backend != MmseGpuBackend::kCuda) {
        slot.device_buffers_ready = false;
        return MmseStatus::kOk;
    }

    // Allocate the complete fixed-capacity workspace once. Per-call code then
    // only changes metadata prefixes and launches kernels, avoiding allocator
    // synchronization in the decode path.
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.grid_re[0], slot.grid_plane_bytes);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.grid_re[1], slot.grid_plane_bytes);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.grid_im[0], slot.grid_plane_bytes);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.grid_im[1], slot.grid_plane_bytes);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.grid_meta, sizeof(slot.grid_meta));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.sigma2, sizeof(float));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.scratch, scratch_float_count() * sizeof(float));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.h_estimate, kEstimateStubFloatCount * sizeof(float));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.xhat_re, slot.xhat_plane_bytes);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.xhat_im, slot.xhat_plane_bytes);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.sinr, slot.sinr_plane_bytes);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.completion, sizeof(std::uint32_t));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.pdcch_llrs, 2U * detail::kCudaMaxDataRe * sizeof(float));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.pdcch_gold_words, detail::kCudaPdcchGoldWordCount * sizeof(std::uint32_t));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.pdcch_candidates,
            detail::kCudaPdcchMaxCandidates * sizeof(detail::CudaPdcchCandidateDescriptor));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.pdcch_recovered_llrs,
            detail::kCudaPdcchMaxCandidates * detail::kCudaPdcchRecoveredLlrCount * sizeof(float));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.pdcch_survivors, static_cast<std::size_t>(detail::kCudaPdcchMaxCandidates) *
                                             64U * detail::kCudaPdcchCodewordBitCount *
                                             sizeof(std::uint64_t));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.pdcch_terminal_metrics,
            static_cast<std::size_t>(detail::kCudaPdcchMaxCandidates) * 64U * sizeof(double));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.pdcch_candidate_results,
            detail::kCudaPdcchMaxCandidates * sizeof(detail::CudaPdcchCandidateResult));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.pdcch_results,
            detail::kCudaPdcchMaxCandidates * sizeof(detail::CudaPdcchCandidateResult));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.pdcch_hit_count, sizeof(std::uint32_t));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_stream_start);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_h2d_start);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_residual_done);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_estimate_done);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_equalize_done);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_pdcch_llr_done);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_create_event(slot.gpu_event_pdcch_rate_recovery_done);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_pdcch_viterbi_done);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_pdcch_crc_done);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_create_event(slot.gpu_event_stream_done);
        status != MmseStatus::kOk) {
        return status;
    }

    slot.device_buffers_ready = true;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::configure_staging_buffers() {
    buffers.ring_ready = true;
    buffers.next_sequence = 1;
    buffers.slot_count = config.stream_count;

    // One slot maps to one stream. In CPU fallback mode there are no CUDA
    // streams, but retaining the same ring shape keeps output contracts equal.
    for (std::uint32_t slot_index = 0; slot_index < buffers.slot_count; ++slot_index) {
        auto& slot = buffers.slots[slot_index];
        slot.slot_ordinal = slot_index;
        slot.stream_ordinal = device.streams.empty() ? 0U : slot_index % config.stream_count;
        slot.stream_handle =
            device.streams.empty() ? 0U : device.streams[slot.stream_ordinal].handle;
        slot.grid_plane_bytes = kGridPlaneCount * sizeof(float);
        slot.estimate_bytes = kEstimateStubFloatCount * sizeof(float);
        slot.xhat_plane_bytes = kOutputPlaneCount * sizeof(float);
        slot.sinr_plane_bytes = kOutputPlaneCount * sizeof(float);
        if (const MmseStatus status = initialize_slot_storage(slot); status != MmseStatus::kOk) {
            return status;
        }
        if (const MmseStatus status = initialize_slot_device_buffers(slot);
            status != MmseStatus::kOk) {
            return status;
        }
        slot.pinned_ready = true;
    }
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::stage_inputs(const PlanarGridViewF32& grid,
                                                       const ExtractDescriptor& desc) {
    if (!buffers.ring_ready || buffers.slot_count == 0U) {
        return MmseStatus::kInternalError;
    }

    const detail::PreparedSubframeKey key =
        detail::make_prepared_subframe_key(grid, desc, static_cast<std::uint32_t>(config.backend));
    if (prepared.valid && detail::prepared_subframe_key_equal(prepared.key, key)) {
        // Reusing a prepared subframe selects its owning slot and skips another
        // full-grid host copy; only dynamic channel layout metadata is refreshed
        // later when needed.
        buffers.active_slot = prepared.slot_index;
        return MmseStatus::kOk;
    }

    // Cache miss: copy both RX planes into a pinned slot before any asynchronous
    // H2D operation can read them. The generation is part of the cache key, so
    // pointer reuse alone cannot accidentally retain stale channel state.
    auto& slot = buffers.slots[buffers.next_stage_slot];
    const Clock::time_point transport_start = Clock::now();
    for (std::uint32_t rx = 0; rx < grid.n_rx_ant; ++rx) {
        std::copy_n(grid.re[rx], detail::kMaxGridRe, slot.transport_re[rx]);
        std::copy_n(grid.im[rx], detail::kMaxGridRe, slot.transport_im[rx]);
    }
    for (std::uint32_t rx = grid.n_rx_ant; rx < kMmseV1MaxNumRxAntennas; ++rx) {
        std::fill_n(slot.transport_re[rx], detail::kMaxGridRe, 0.0F);
        std::fill_n(slot.transport_im[rx], detail::kMaxGridRe, 0.0F);
    }
    slot.grid_view.n_rx_ant = grid.n_rx_ant;
    slot.grid_view.generation = grid.generation;
    last_host_profile.quantize_us = elapsed_us(transport_start, Clock::now());
    slot.layout = {};
    slot.grid_meta.n_rx_ant = grid.n_rx_ant;
    slot.grid_meta.n_symbols = grid.n_symbols;
    slot.grid_meta.n_subcarriers = grid.n_subcarriers;
    slot.grid_meta.n_valid_re = 0;
    slot.grid_meta.first_valid_grid_idx = 0;
    slot.grid_meta.subframe = 0;
    slot.grid_meta.start_symbol = 0;
    slot.grid_meta.n_layers = 0;
    slot.grid_meta.n_tx_ports = 0;
    slot.grid_meta.n_segments = 0;
    slot.grid_meta.spot_check_sample_count = 0;
    slot.grid_meta.trace_sample_count = 0;
    slot.grid_meta.channel_type = 0U;
    slot.grid_meta.td_pair_count = 0U;
    slot.grid_meta.sigma2_device_owned = 0U;
    slot.grid_meta.sigma2_iir_alpha = config.sigma2_iir_alpha;
    std::fill_n(slot.grid_meta.output_slot_by_grid_re, detail::kCudaMaxGridRe,
                std::numeric_limits<std::uint32_t>::max());
    std::fill(slot.grid_meta.spot_check_re_slots,
              slot.grid_meta.spot_check_re_slots + detail::kCudaValidationSampleCount,
              static_cast<std::uint16_t>(0));
    std::fill_n(slot.scratch_host.data(), scratch_float_count(), 0.0F);
    std::fill_n(slot.xhat_re, kOutputPlaneCount, 0.0F);
    std::fill_n(slot.xhat_im, kOutputPlaneCount, 0.0F);
    std::fill_n(slot.sinr, kOutputPlaneCount, 0.0F);
    slot.input_ready = true;
    slot.output_ready = false;
    slot.completion_ready = false;
    slot.cpu_seeded = false;
    slot.h2d_submitted = false;
    slot.kernel_submitted = false;
    slot.d2h_submitted = false;
    slot.completion_value = 0;
    *slot.pdcch_hit_count = 0U;
    std::fill_n(slot.pdcch_candidates_host, detail::kCudaPdcchMaxCandidates,
                detail::CudaPdcchCandidateDescriptor{});
    if (slot.pdcch_results_host != nullptr) {
        std::fill_n(slot.pdcch_results_host, detail::kCudaPdcchMaxCandidates,
                    detail::CudaPdcchCandidateResult{});
    }
    slot.pdcch_decode_submitted = false;
    slot.sequence = buffers.next_sequence++;
    slot.out_view.n_re_per_layer = 0;
    slot.out_view.n_layers = 0;
    slot.out_view.mod_order = 0;

    buffers.active_slot = buffers.next_stage_slot;
    buffers.next_stage_slot = (buffers.next_stage_slot + 1U) % buffers.slot_count;
    prepared.key = key;
    prepared.valid = false;
    prepared.slot_index = buffers.active_slot;
    return MmseStatus::kOk;
}

MmseStatus
MmseEqualizerGpuContext::Impl::prepare_subframe_if_needed(const PlanarGridViewF32& grid,
                                                          const ExtractDescriptor& desc) {
    return stage_inputs(grid, desc);
}

MmseStatus MmseEqualizerGpuContext::Impl::validate_estimate_stub(const ExtractDescriptor& desc,
                                                                 const HostPinnedSlot& slot) const {
    if (slot.h_estimate == nullptr) {
        return MmseStatus::kInternalError;
    }
    detail::HGridStorage h_full{};
    float sigma2_estimate = 0.0F;
    detail::estimate_channel(slot.grid_view, desc, h_full, sigma2_estimate);

    constexpr std::array<std::uint32_t, 3> kCheckSymbols = {0U, 6U, 13U};
    constexpr std::array<std::uint32_t, 4> kCheckSc = {0U, 127U, 601U, 1199U};
    for (std::uint32_t tx = 0; tx < desc.n_tx_ports; ++tx) {
        for (std::uint32_t rx = 0; rx < desc.n_rx_ant; ++rx) {
            for (std::uint32_t symbol : kCheckSymbols) {
                for (std::uint32_t sc : kCheckSc) {
                    const std::size_t base =
                        2U *
                        (((tx * kMmseV1MaxNumRxAntennas + rx) * kLteNumSymbolsNormalCp + symbol) *
                             kLteNumSubcarriers20MHz +
                         sc);
                    const detail::Complex32 ref =
                        h_full[((static_cast<std::size_t>(tx) * kMmseV1MaxNumRxAntennas + rx) *
                                    kLteNumSymbolsNormalCp +
                                symbol) *
                                   kLteNumSubcarriers20MHz +
                               sc];
                    const float got_re = slot.h_estimate[base];
                    const float got_im = slot.h_estimate[base + 1U];
                    if (std::abs(got_re - ref.re) > 2.0e-2F ||
                        std::abs(got_im - ref.im) > 2.0e-2F) {
                        return MmseStatus::kInternalError;
                    }
                }
            }
        }
    }
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::validate_equalize_stub(const HostPinnedSlot& slot) const {
    if (slot.grid_meta.n_valid_re == 0U || slot.out_view.n_re_per_layer == 0U) {
        return MmseStatus::kInternalError;
    }

    // Release policy keeps lightweight output spot-checks on sampled REs.
    for (std::uint32_t sample = 0; sample < slot.grid_meta.spot_check_sample_count; ++sample) {
        const std::uint32_t re = slot.grid_meta.spot_check_re_slots[sample];
        for (std::uint32_t layer = 0; layer < slot.out_view.n_layers; ++layer) {
            if (re >= slot.out_view.n_re_per_layer) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(layer) * slot.out_view.capacity_re_per_layer + re;
            if (!std::isfinite(slot.xhat_re[idx]) || !std::isfinite(slot.xhat_im[idx]) ||
                !std::isfinite(slot.sinr[idx])) {
                return MmseStatus::kInternalError;
            }
            if (!(slot.sinr[idx] > 0.0F)) {
                return MmseStatus::kInternalError;
            }
            if (!deep_validation_enabled()) {
                continue;
            }
            if (slot.grid_meta.n_layers == 2U) {
                continue;
            }
            const float ref_re = slot.ref_xhat_re[idx];
            const float ref_im = slot.ref_xhat_im[idx];
            const float ref_sinr = slot.ref_sinr[idx];
            if (!nearly_equal(slot.xhat_re[idx], ref_re, 5.0e-2F, 1.0e-4F) ||
                !nearly_equal(slot.xhat_im[idx], ref_im, 5.0e-2F, 1.0e-4F) ||
                !nearly_equal(slot.sinr[idx], ref_sinr, 7.0e-2F, 1.0e-4F)) {
                return MmseStatus::kInternalError;
            }
        }
    }

    // Debug/test policy adds CPU-vs-GPU trace comparison on sampled REs.
    if (deep_validation_enabled()) {
        if (slot.grid_meta.n_layers == 2U) {
            for (std::uint32_t sample = 0; sample < slot.grid_meta.trace_sample_count; ++sample) {
                if (compare_equalize_trace_sample(slot, sample) != MmseStatus::kOk) {
                    return MmseStatus::kInternalError;
                }
            }
        }
    }
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::execute_cuda_transport_stub(const ExtractDescriptor& desc,
                                                                      HostPinnedSlot& slot,
                                                                      bool reuse_estimate,
                                                                      bool copy_equalized_outputs,
                                                                      bool synchronize) {
    if (!detail::cuda_backend_compiled() || !device.cuda_runtime_available ||
        !slot.device_buffers_ready || slot.stream_handle == 0U) {
        return MmseStatus::kUnsupportedConfig;
    }
    // Non-synchronizing submissions must keep sigma2 on the device; otherwise
    // the host would need a mid-stream sync before equalization. The explicit
    // ownership setting controls the same behavior for synchronous calls.
    const bool device_sigma_for_call = use_device_owned_sigma2(config) || !synchronize;

    last_host_profile.layout_build_us = 0.0;
    last_host_profile.grid_meta_pack_us = 0.0;
    last_host_profile.grid_h2d_us = 0.0;
    last_host_profile.estimate_launch_us = 0.0;
    last_host_profile.sigma2_d2h_us = 0.0;
    last_host_profile.sigma2_mid_sync_us = 0.0;
    last_host_profile.sigma2_host_update_us = 0.0;
    last_host_profile.sigma2_h2d_us = 0.0;
    last_host_profile.equalize_launch_us = 0.0;
    last_host_profile.outputs_d2h_us = 0.0;
    last_host_profile.scratch_d2h_us = 0.0;
    last_host_profile.completion_d2h_us = 0.0;
    last_host_profile.final_sync_us = 0.0;
    last_host_profile.estimate_gpu_us = 0.0;
    last_host_profile.estimate_residual_gpu_us = 0.0;
    last_host_profile.estimate_channel_gpu_us = 0.0;
    last_host_profile.equalize_gpu_us = 0.0;
    last_host_profile.stream_gpu_us = 0.0;

    // Host-side layout/meta packing precedes one ordered stream sequence:
    // H2D -> estimate/residual -> sigma2 finalize -> equalize -> optional D2H.
    const Clock::time_point layout_start = Clock::now();
    detail::build_channel_re_layout(desc, slot.layout);
    last_host_profile.layout_build_us = elapsed_us(layout_start, Clock::now());

    const Clock::time_point grid_meta_pack_start = Clock::now();
    slot.grid_meta.n_valid_re = slot.layout.n_re;
    slot.grid_meta.first_valid_grid_idx = slot.layout.n_re == 0U ? 0U : slot.layout.grid_indices[0];
    slot.grid_meta.subframe = detail::subframe_from_descriptor(desc);
    slot.grid_meta.start_symbol = detail::channel_start_symbol(desc);
    slot.grid_meta.n_layers = desc.n_layers;
    slot.grid_meta.n_tx_ports = desc.n_tx_ports;
    slot.grid_meta.channel_type = static_cast<std::uint32_t>(desc.channel_type);
    slot.grid_meta.td_pair_count = 0U;
    if (desc.n_tx_ports == 2U && desc.n_layers == 1U && desc.tx_mode == 2U &&
        !pack_transmit_diversity_pairs(slot.layout, slot.grid_meta)) {
        return MmseStatus::kUnsupportedConfig;
    }
    slot.grid_meta.n_segments = slot.layout.n_segments;
    slot.grid_meta.sigma2_device_owned = device_sigma_for_call ? 1U : 0U;
    slot.grid_meta.sigma2 = config.sigma2_min;
    slot.grid_meta.sigma2_iir_alpha = config.sigma2_iir_alpha;
    slot.grid_meta.det_floor = config.det_floor;
    slot.grid_meta.g_min = config.g_min;
    slot.grid_meta.gamma_max = config.gamma_max;
    std::array<std::uint32_t, detail::kCudaValidationSampleCount> validation_re_slots{};
    slot.grid_meta.spot_check_sample_count = detail::build_validation_re_samples(
        slot.layout, desc.start_symbol, slot.grid_meta.n_symbols, slot.grid_meta.n_subcarriers,
        validation_re_slots.data(), detail::kCudaValidationSampleCount);
    slot.grid_meta.trace_sample_count =
        deep_validation_enabled() ? slot.grid_meta.spot_check_sample_count : 0U;
    for (std::uint32_t i = 0; i < slot.grid_meta.spot_check_sample_count; ++i) {
        slot.grid_meta.spot_check_re_slots[i] = static_cast<std::uint16_t>(validation_re_slots[i]);
    }

    if (deep_validation_enabled() && slot.grid_meta.td_pair_count == 0U) {
        const MmseStatus cpu_status = cpu_fallback.run(slot.grid_view, desc, slot.out_view);
        if (cpu_status != MmseStatus::kOk) {
            return cpu_status;
        }
        slot.cpu_seeded = true;
        std::copy_n(slot.xhat_re, kOutputPlaneCount, slot.ref_xhat_re.data());
        std::copy_n(slot.xhat_im, kOutputPlaneCount, slot.ref_xhat_im.data());
        std::copy_n(slot.sinr, kOutputPlaneCount, slot.ref_sinr.data());
    } else {
        slot.cpu_seeded = false;
        std::fill(slot.ref_xhat_re.begin(), slot.ref_xhat_re.end(), 0.0F);
        std::fill(slot.ref_xhat_im.begin(), slot.ref_xhat_im.end(), 0.0F);
        std::fill(slot.ref_sinr.begin(), slot.ref_sinr.end(), 0.0F);
        slot.out_view.n_re_per_layer = slot.layout.n_re;
        slot.out_view.n_layers = desc.n_layers;
        slot.out_view.mod_order = desc.mod_order;
    }

    std::copy_n(slot.layout.grid_indices.begin(), detail::kCudaMaxDataRe,
                slot.grid_meta.grid_indices);
    std::copy_n(slot.layout.output_slot_by_grid_re.begin(), detail::kCudaMaxGridRe,
                slot.grid_meta.output_slot_by_grid_re);
    std::copy_n(slot.layout.prb_segment_offsets.begin(), detail::kCudaMaxDataRe + 1U,
                slot.grid_meta.prb_segment_offsets);
    std::copy_n(desc.prb_bitmap.begin(), desc.prb_bitmap.size(), slot.grid_meta.prb_bitmap);
    std::copy_n(detail::kCrsSymbols.begin(), detail::kCrsSymbols.size(),
                slot.grid_meta.crs_symbols);
    for (std::uint32_t tx = 0; tx < kMmseV1MaxNumCrsTxPorts; ++tx) {
        for (std::uint32_t cs = 0; cs < kLteNumCrsSymbols; ++cs) {
            const std::uint8_t symbol = detail::kCrsSymbols[cs];
            slot.grid_meta.crs_freq_offsets[tx][cs] = static_cast<std::uint8_t>(
                detail::crs_frequency_offset(desc.cell_id, static_cast<std::uint8_t>(tx), symbol));
            for (std::uint32_t pilot = 0; pilot < kLteNumPilotTonesPerCrsSymbol; ++pilot) {
                const detail::Complex32& crs = detail::crs_value(
                    {.cell_id = desc.cell_id,
                     .subframe = static_cast<std::uint8_t>(slot.grid_meta.subframe),
                     .port = static_cast<std::uint8_t>(tx),
                     .crs_symbol_index = static_cast<std::uint8_t>(cs)},
                    pilot);
                slot.grid_meta.crs_pilot_re[tx][cs][pilot] = crs.re;
                slot.grid_meta.crs_pilot_im[tx][cs][pilot] = crs.im;
            }
        }
    }
    last_host_profile.grid_meta_pack_us = elapsed_us(grid_meta_pack_start, Clock::now());
    if (!reuse_estimate) {
        // First use of a grid slot transfers samples and runs channel estimation.
        // Host-owned sigma2 intentionally synchronizes once to feed the updated
        // IIR value back to the same stream before equalization.
        const Clock::time_point grid_h2d_start = Clock::now();
        if (const MmseStatus status =
                detail::cuda_event_record(slot.gpu_event_h2d_start, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        *slot.grid_meta_staging = slot.grid_meta;
        if (const MmseStatus status = detail::cuda_copy_grid_h2d_async(
                slot.device, slot.transport_re, slot.transport_im, *slot.grid_meta_staging,
                slot.grid_plane_bytes, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        if (const MmseStatus status =
                detail::cuda_event_record(slot.gpu_event_stream_start, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        last_host_profile.grid_h2d_us = elapsed_us(grid_h2d_start, Clock::now());
        slot.h2d_submitted = true;

        if (device_sigma_for_call) {
            const detail::Sigma2State& sigma2_state = sigma2_by_cell[desc.cell_id];
            *slot.sigma2_host = sigma2_state.initialized ? sigma2_state.value : 0.0F;
            if (const MmseStatus status = detail::cuda_copy_sigma2_h2d_async(
                    slot.device, *slot.sigma2_host, slot.stream_handle);
                status != MmseStatus::kOk) {
                return status;
            }
        }

        const Clock::time_point estimate_launch_start = Clock::now();
        if (const MmseStatus status = detail::cuda_launch_estimate_stub(
                slot.device, slot.stream_handle, slot.gpu_event_residual_done);
            status != MmseStatus::kOk) {
            return status;
        }
        if (const MmseStatus status =
                detail::cuda_event_record(slot.gpu_event_estimate_done, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        last_host_profile.estimate_launch_us = elapsed_us(estimate_launch_start, Clock::now());
        slot.kernel_submitted = true;

        if (!device_sigma_for_call) {
            float sigma2_estimate = 0.0F;
            const Clock::time_point sigma2_d2h_start = Clock::now();
            if (const MmseStatus status = detail::cuda_copy_sigma2_d2h_async(
                    slot.device, sigma2_estimate, slot.stream_handle);
                status != MmseStatus::kOk) {
                return status;
            }
            last_host_profile.sigma2_d2h_us = elapsed_us(sigma2_d2h_start, Clock::now());
            const Clock::time_point sigma2_sync_start = Clock::now();
            if (const MmseStatus status = detail::cuda_stream_synchronize(slot.stream_handle);
                status != MmseStatus::kOk) {
                return status;
            }
            last_host_profile.sigma2_mid_sync_us = elapsed_us(sigma2_sync_start, Clock::now());
            const Clock::time_point sigma2_host_update_start = Clock::now();
            slot.grid_meta.sigma2 = detail::update_sigma2_state(
                sigma2_by_cell[desc.cell_id], sigma2_estimate, make_cpu_fallback_config(config));
            prepared.sigma2 = slot.grid_meta.sigma2;
            prepared.valid = true;
            last_host_profile.sigma2_host_update_us =
                elapsed_us(sigma2_host_update_start, Clock::now());
            const Clock::time_point sigma2_h2d_start = Clock::now();
            *slot.sigma2_host = slot.grid_meta.sigma2;
            if (const MmseStatus status = detail::cuda_copy_sigma2_h2d_async(
                    slot.device, *slot.sigma2_host, slot.stream_handle);
                status != MmseStatus::kOk) {
                return status;
            }
            last_host_profile.sigma2_h2d_us = elapsed_us(sigma2_h2d_start, Clock::now());
        }
    } else {
        // A prepared subframe reuses H and sigma2. Only dynamic layout fields
        // and the stream ordering event need to be submitted.
        slot.grid_meta.sigma2 = prepared.sigma2;
        *slot.grid_meta_staging = slot.grid_meta;
        if (const MmseStatus status = detail::cuda_copy_grid_meta_dynamic_h2d_async(
                slot.device, *slot.grid_meta_staging, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        if (const MmseStatus status =
                detail::cuda_event_record(slot.gpu_event_stream_start, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
    }

    // The completion token is a monotonically changing slot sequence truncated
    // to 32 bits; deep validation copies it back to detect stale-slot reuse.
    const std::uint32_t completion_value = static_cast<std::uint32_t>(slot.sequence & 0xFFFFFFFFU);
    const Clock::time_point equalize_launch_start = Clock::now();
    if (const MmseStatus status = detail::cuda_launch_equalize_stub(
            slot.device, slot.grid_meta.n_valid_re, completion_value, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_event_record(slot.gpu_event_equalize_done, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    last_host_profile.equalize_launch_us = elapsed_us(equalize_launch_start, Clock::now());
    if (copy_equalized_outputs) {
        const Clock::time_point outputs_d2h_start = Clock::now();
        if (const MmseStatus status = detail::cuda_copy_outputs_d2h_async(
                slot.device, slot.xhat_re, slot.xhat_im, slot.sinr, slot.xhat_plane_bytes,
                slot.sinr_plane_bytes, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        last_host_profile.outputs_d2h_us = elapsed_us(outputs_d2h_start, Clock::now());
        slot.d2h_submitted = true;
    } else {
        last_host_profile.outputs_d2h_us = 0.0;
        slot.d2h_submitted = false;
    }
    if (deep_validation_enabled()) {
        if (const MmseStatus status = detail::cuda_copy_estimate_d2h_async(
                slot.device, slot.h_estimate, slot.estimate_bytes, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
    }
    const bool need_scratch_d2h =
        deep_validation_enabled() || (use_device_owned_sigma2(config) && synchronize);
    if (need_scratch_d2h) {
        const Clock::time_point scratch_d2h_start = Clock::now();
        if (const MmseStatus status = detail::cuda_copy_scratch_d2h_async(
                slot.device, slot.scratch_host.data(), scratch_float_count() * sizeof(float),
                slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        last_host_profile.scratch_d2h_us = elapsed_us(scratch_d2h_start, Clock::now());
    } else {
        last_host_profile.scratch_d2h_us = 0.0;
    }
    if (deep_validation_enabled()) {
        const Clock::time_point completion_d2h_start = Clock::now();
        if (const MmseStatus status = detail::cuda_copy_completion_d2h_async(
                slot.device, slot.completion_value, slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        last_host_profile.completion_d2h_us = elapsed_us(completion_d2h_start, Clock::now());
    } else {
        last_host_profile.completion_d2h_us = 0.0;
        slot.completion_ready = true;
        slot.completion_value = completion_value;
    }
    if (const MmseStatus status =
            detail::cuda_event_record(slot.gpu_event_stream_done, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (synchronize) {
        const Clock::time_point final_sync_start = Clock::now();
        if (const MmseStatus status = detail::cuda_stream_synchronize(slot.stream_handle);
            status != MmseStatus::kOk) {
            return status;
        }
        last_host_profile.final_sync_us = elapsed_us(final_sync_start, Clock::now());
    } else {
        last_host_profile.final_sync_us = 0.0;
    }
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_stream_start, slot.gpu_event_residual_done,
                                        last_host_profile.estimate_residual_gpu_us);
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_residual_done, slot.gpu_event_estimate_done,
                                        last_host_profile.estimate_channel_gpu_us);
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_stream_start, slot.gpu_event_estimate_done,
                                        last_host_profile.estimate_gpu_us);
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_estimate_done, slot.gpu_event_equalize_done,
                                        last_host_profile.equalize_gpu_us);
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_stream_start, slot.gpu_event_stream_done,
                                        last_host_profile.stream_gpu_us);
    if (use_device_owned_sigma2(config) && synchronize) {
        slot.grid_meta.sigma2 = slot.scratch_host[3];
        detail::Sigma2State& sigma2_state = sigma2_by_cell[desc.cell_id];
        sigma2_state.value = slot.grid_meta.sigma2;
        sigma2_state.initialized = true;
        prepared.sigma2 = slot.grid_meta.sigma2;
        prepared.valid = true;
    }

    if (copy_equalized_outputs && deep_validation_enabled()) {
        if (desc.n_layers == 2U || slot.grid_meta.td_pair_count != 0U) {
            if (const MmseStatus status = validate_estimate_stub(desc, slot);
                status != MmseStatus::kOk) {
                return status;
            }
        }
        if (desc.n_layers == 2U) {
            if (const MmseStatus status = validate_equalize_stub(slot); status != MmseStatus::kOk) {
                return status;
            }
        }
    } else if (copy_equalized_outputs) {
        if (const MmseStatus status = validate_equalize_stub(slot); status != MmseStatus::kOk) {
            return status;
        }
    }
    if (copy_equalized_outputs && deep_validation_enabled()) {
        if (!(slot.scratch_host[3] >= config.sigma2_min)) {
            return MmseStatus::kInternalError;
        }
        if (!nearly_equal(slot.scratch_host[3], slot.grid_meta.sigma2, 5.0e-2F, 1.0e-6F)) {
            return MmseStatus::kInternalError;
        }
        slot.completion_ready = (slot.completion_value == completion_value);
        if (!slot.completion_ready) {
            return MmseStatus::kInternalError;
        }
    }
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::execute_backend(const ExtractDescriptor& desc) {
    auto& slot = buffers.slots[buffers.active_slot];
    if (!slot.pinned_ready || !slot.input_ready) {
        return MmseStatus::kInternalError;
    }

    if (!using_cpu_fallback) {
        return MmseStatus::kUnsupportedConfig;
    }

    // Backend selection is intentionally explicit here: CUDA uses the staged
    // transport path, while Auto uses the CPU context but the same slot output.
    MmseStatus status = MmseStatus::kUnsupportedConfig;
    switch (config.backend) {
    case MmseGpuBackend::kCuda:
        if (!device.selection_ready || device.streams.empty() ||
            !device.streams[slot.stream_ordinal].ready) {
            return MmseStatus::kInternalError;
        }
        status = execute_cuda_transport_stub(
            desc, slot, prepared.valid && prepared.slot_index == buffers.active_slot);
        break;
    case MmseGpuBackend::kAuto:
        status = cpu_fallback.run(slot.grid_view, desc, slot.out_view);
        break;
    default:
        return MmseStatus::kInvalidArgument;
    }

    if (status == MmseStatus::kOk) {
        slot.output_ready = true;
        buffers.completed_slot = buffers.active_slot;
    }
    return status;
}

MmseStatus MmseEqualizerGpuContext::Impl::stage_outputs(EqualizerOutputView& out) const {
    const auto& slot = buffers.slots[buffers.completed_slot];
    if (!slot.output_ready) {
        return MmseStatus::kInternalError;
    }
    if (out.capacity_re_per_layer < slot.out_view.n_re_per_layer) {
        return MmseStatus::kBufferTooSmall;
    }

    // Copy only the valid prefix for each layer. The slot capacity is fixed for
    // CUDA, whereas the caller's output capacity may be smaller or larger.
    const Clock::time_point output_stage_start = Clock::now();
    for (std::uint32_t layer = 0; layer < slot.out_view.n_layers; ++layer) {
        const std::size_t src_base =
            static_cast<std::size_t>(layer) * slot.out_view.capacity_re_per_layer;
        const std::size_t dst_base = static_cast<std::size_t>(layer) * out.capacity_re_per_layer;
        std::copy_n(slot.out_view.x_hat_re + src_base, slot.out_view.n_re_per_layer,
                    out.x_hat_re + dst_base);
        std::copy_n(slot.out_view.x_hat_im + src_base, slot.out_view.n_re_per_layer,
                    out.x_hat_im + dst_base);
        std::copy_n(slot.out_view.sinr + src_base, slot.out_view.n_re_per_layer,
                    out.sinr + dst_base);
    }
    out.n_re_per_layer = slot.out_view.n_re_per_layer;
    out.n_layers = slot.out_view.n_layers;
    out.mod_order = slot.out_view.mod_order;
    const_cast<Impl*>(this)->last_host_profile.output_stage_us =
        elapsed_us(output_stage_start, Clock::now());
    const_cast<Impl*>(this)->last_host_profile.total_host_us =
        const_cast<Impl*>(this)->last_host_profile.quantize_us +
        const_cast<Impl*>(this)->last_host_profile.layout_build_us +
        const_cast<Impl*>(this)->last_host_profile.grid_meta_pack_us +
        const_cast<Impl*>(this)->last_host_profile.grid_h2d_us +
        const_cast<Impl*>(this)->last_host_profile.estimate_launch_us +
        const_cast<Impl*>(this)->last_host_profile.sigma2_d2h_us +
        const_cast<Impl*>(this)->last_host_profile.sigma2_mid_sync_us +
        const_cast<Impl*>(this)->last_host_profile.sigma2_host_update_us +
        const_cast<Impl*>(this)->last_host_profile.sigma2_h2d_us +
        const_cast<Impl*>(this)->last_host_profile.equalize_launch_us +
        const_cast<Impl*>(this)->last_host_profile.outputs_d2h_us +
        const_cast<Impl*>(this)->last_host_profile.scratch_d2h_us +
        const_cast<Impl*>(this)->last_host_profile.completion_d2h_us +
        const_cast<Impl*>(this)->last_host_profile.final_sync_us +
        const_cast<Impl*>(this)->last_host_profile.output_stage_us;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::run_transmit_diversity(const PlanarGridViewF32& grid,
                                                                 const ExtractDescriptor& desc,
                                                                 EqualizerOutputView& out) {
    if (const MmseStatus status = detail::validate_grid(grid); status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::validate_output(out); status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = prepare_subframe_if_needed(grid, desc);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = execute_backend(desc); status != MmseStatus::kOk) {
        return status;
    }
    return stage_outputs(out);
}

MmseStatus MmseEqualizerGpuContext::Impl::submit_pdcch_gpu_common_search_decode(
    const pdcch::PdcchGpuCommonSearchDecodeRequest& request, PendingPdcchDecode& pending) {
    pending = {};
    const PdcchMmseInput& input = request.input;
    const pdcch::PdcchCommonSearchDecodeConfig& decode_config = request.config;
    if (!initialized) {
        return MmseStatus::kNotInitialized;
    }
    const bool supported_tx_mode = (input.n_tx_ports == 1U && input.tx_mode == 1U) ||
                                   (input.n_tx_ports == 2U && input.tx_mode == 2U);
    if (config.backend != MmseGpuBackend::kCuda || !device.cuda_runtime_available ||
        decode_config.decoder.decode != nullptr || input.n_prb != kLteNumPrb20MHz ||
        !supported_tx_mode || input.grid.n_rx_ant == 0U ||
        input.grid.n_rx_ant > kMmseV1MaxNumRxAntennas ||
        input.grid.n_symbols != kLteNumSymbolsNormalCp ||
        input.grid.n_subcarriers != kLteNumSubcarriers20MHz ||
        input.control_subframe.duplex_mode != pdcch::PhichDuplexMode::kFdd ||
        input.control_subframe.kind != pdcch::LteControlSubframeKind::kRegular ||
        decode_config.dci_format1a.n_prb != kLteNumPrb20MHz ||
        decode_config.dci_format1a.duplex_mode != pdcch::PhichDuplexMode::kFdd ||
        decode_config.dci_format1a.cif_enabled) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (const MmseStatus status = pdcch::validate_pdcch_mmse_input(input);
        status != MmseStatus::kOk) {
        return status;
    }
    if (pdcch::pdcch_dci_format1a_payload_bit_count(decode_config.dci_format1a) !=
        detail::kCudaPdcchCodewordBitCount - pdcch::kPdcchCrcBitCount) {
        return MmseStatus::kUnsupportedConfig;
    }

    ExtractDescriptor desc{};
    desc.sfn_subframe = input.sfn_subframe;
    desc.cell_id = input.cell_id;
    desc.n_tx_ports = input.n_tx_ports;
    desc.n_rx_ant = static_cast<std::uint8_t>(input.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = input.tx_mode;
    desc.channel_type = MmseChannelType::kPdcch;
    desc.start_symbol = 0U;
    desc.control_symbol_count = input.control_symbol_count;
    desc.mod_order = 2U;
    desc.n_prb = input.n_prb;
    desc.prb_bitmap = input.prb_bitmap;
    desc.control_re_exclusion_masks = input.control_re_exclusion_masks;
    desc.pmi = -1;

    // Submission stops after the compact result D2H is queued. No stream sync
    // occurs here, allowing the batch API to enqueue independent requests.
    const Clock::time_point submit_start = Clock::now();
    prepared.valid = false;
    if (const MmseStatus status = prepare_subframe_if_needed(input.grid, desc);
        status != MmseStatus::kOk) {
        return status;
    }
    HostPinnedSlot& slot = buffers.slots[buffers.active_slot];
    if (!slot.device_buffers_ready || slot.stream_handle == 0U ||
        slot.grid_meta_staging == nullptr || slot.pdcch_candidates_host == nullptr ||
        slot.pdcch_results_host == nullptr || slot.pdcch_hit_count == nullptr ||
        slot.pdcch_gold_words_host == nullptr) {
        return MmseStatus::kUnsupportedConfig;
    }
    struct SubmissionGuard {
        HostPinnedSlot& slot;
        bool armed = true;
        ~SubmissionGuard() {
            if (armed) {
                (void)detail::cuda_stream_synchronize(slot.stream_handle);
                slot.pdcch_decode_submitted = false;
            }
        }
    } submission_guard{slot};
    if (const MmseStatus status = execute_cuda_transport_stub(desc, slot, false, false, false);
        status != MmseStatus::kOk) {
        return status;
    }
    if (slot.grid_meta.n_valid_re != slot.layout.n_re ||
        (input.n_tx_ports == 2U && ((slot.layout.n_re & 1U) != 0U ||
                                    slot.grid_meta.td_pair_count > detail::kCudaMaxDataRe / 2U ||
                                    slot.grid_meta.td_pair_count * 2U != slot.layout.n_re))) {
        return MmseStatus::kUnsupportedConfig;
    }

    // Reuse the equalized PDCCH layout to build REG/CCE candidates, then copy
    // only candidate descriptors and Gold words to the device.
    pdcch::PdcchControlRegion control_region{};
    if (const MmseStatus status = pdcch::build_pdcch_control_region(
            input.cell_id, input.control_symbol_count, slot.layout.grid_indices.data(),
            slot.layout.n_re, control_region);
        status != MmseStatus::kOk) {
        return status;
    }
    std::vector<pdcch::PdcchCommonSearchCandidate> candidates{};
    pdcch::build_pdcch_common_search_candidates(control_region, candidates);
    if (candidates.empty() || candidates.size() > detail::kCudaPdcchMaxCandidates) {
        return MmseStatus::kUnsupportedConfig;
    }
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        slot.pdcch_candidates_host[index] = {
            .candidate_id = candidates[index].candidate_id,
            .first_cce = candidates[index].first_cce,
            .aggregation_level = candidates[index].aggregation_level,
            .encoded_bit_count = candidates[index].encoded_bit_count,
        };
    }
    const std::uint32_t candidate_count = static_cast<std::uint32_t>(candidates.size());
    if (const MmseStatus status = detail::cuda_copy_pdcch_candidates_h2d_async(
            slot.device, slot.pdcch_candidates_host, candidate_count, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    const std::uint32_t gold_word_count = populate_pdcch_gold_words(
        input.cell_id, input.sfn_subframe, slot.layout.n_re, slot.pdcch_gold_words_host);
    if (const MmseStatus status = detail::cuda_copy_pdcch_gold_words_h2d_async(
            slot.device, slot.pdcch_gold_words_host, gold_word_count, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_launch_pdcch_llr_descramble(
            slot.device, slot.layout.n_re, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_event_record(slot.gpu_event_pdcch_llr_done, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_launch_pdcch_rate_recovery(
            slot.device, candidate_count, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_event_record(slot.gpu_event_pdcch_rate_recovery_done, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_launch_pdcch_viterbi(slot.device, candidate_count, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_event_record(slot.gpu_event_pdcch_viterbi_done, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_launch_pdcch_crc_compact(
            slot.device, candidate_count, decode_config.expected_rnti, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_event_record(slot.gpu_event_pdcch_crc_done, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }

    const Clock::time_point d2h_start = Clock::now();
    if (const MmseStatus status = detail::cuda_copy_pdcch_results_d2h_async(
            slot.device, slot.pdcch_results_host, *slot.pdcch_hit_count, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_event_record(slot.gpu_event_stream_done, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    pending.slot_index = buffers.active_slot;
    pending.candidate_count = candidate_count;
    pending.input = input;
    pending.decode_config = decode_config;
    pending.submit_start = submit_start;
    pending.d2h_start = d2h_start;
    pending.ce_mmse_gpu_us = last_host_profile.estimate_gpu_us + last_host_profile.equalize_gpu_us;
    pending.h2d_bytes =
        2U * static_cast<std::uint64_t>(input.grid.n_rx_ant) * slot.grid_plane_bytes +
        detail::cuda_grid_meta_h2d_bytes(slot.grid_meta) + sizeof(*slot.sigma2_host) +
        static_cast<std::uint64_t>(candidate_count) * sizeof(detail::CudaPdcchCandidateDescriptor) +
        static_cast<std::uint64_t>(gold_word_count) * sizeof(std::uint32_t);
    slot.pdcch_decode_submitted = true;
    slot.output_ready = false;
    submission_guard.armed = false;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::collect_pdcch_gpu_common_search_decode(
    const PendingPdcchDecode& pending, pdcch::PdcchGpuCommonSearchDecodeResult& result) {
    result = {};
    if (pending.slot_index >= buffers.slots.size()) {
        return MmseStatus::kInternalError;
    }
    HostPinnedSlot& slot = buffers.slots[pending.slot_index];
    if (!slot.pdcch_decode_submitted) {
        return MmseStatus::kInternalError;
    }
    // Collection is the synchronization boundary. The compact hit list is
    // validated again through the shared CPU DCI parser before it is returned.
    const Clock::time_point collect_start = Clock::now();
    if (const MmseStatus status = detail::cuda_stream_synchronize(slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.pdcch_decode_submitted = false;
    if (*slot.pdcch_hit_count > pending.candidate_count) {
        return MmseStatus::kInternalError;
    }

    result.candidate_count = pending.candidate_count;
    result.hits.reserve(*slot.pdcch_hit_count);
    for (std::uint32_t hit = 0U; hit < *slot.pdcch_hit_count; ++hit) {
        const detail::CudaPdcchCandidateResult& device_result = slot.pdcch_results_host[hit];
        if (device_result.matched == 0U || device_result.candidate_id >= pending.candidate_count) {
            return MmseStatus::kInternalError;
        }
        std::array<std::uint8_t, detail::kCudaPdcchCodewordBitCount> decoded_bits{};
        for (std::uint32_t bit = 0U; bit < decoded_bits.size(); ++bit) {
            decoded_bits[bit] = static_cast<std::uint8_t>((device_result.decoded_bits >> bit) & 1U);
        }
        PdcchChainMetadata chain = pending.input.chain;
        chain.candidate_id = device_result.candidate_id;
        chain.first_cce = device_result.first_cce;
        chain.aggregation_level = device_result.aggregation_level;
        pdcch::PdcchDciFormat1ADecodeResult decoded{};
        if (const MmseStatus status = pdcch::validate_and_parse_pdcch_dci_format1a(
                decoded_bits.data(), static_cast<std::uint32_t>(decoded_bits.size()),
                pending.decode_config.expected_rnti, pending.input.sfn_subframe,
                pending.input.cell_id, chain, pending.decode_config.dci_format1a, decoded);
            status != MmseStatus::kOk || !decoded.matched ||
            decoded.crc.transmitted_crc != device_result.transmitted_crc ||
            decoded.crc.calculated_crc != device_result.calculated_crc ||
            decoded.crc.unmasked_rnti != device_result.unmasked_rnti) {
            return status == MmseStatus::kOk ? MmseStatus::kInternalError : status;
        }
        result.hits.push_back(std::move(decoded));
    }

    (void)detail::cuda_event_elapsed_us(slot.gpu_event_h2d_start, slot.gpu_event_stream_start,
                                        result.profile.h2d_us);
    double estimate_gpu_us = 0.0;
    double equalize_gpu_us = 0.0;
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_stream_start, slot.gpu_event_estimate_done,
                                        estimate_gpu_us);
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_estimate_done, slot.gpu_event_equalize_done,
                                        equalize_gpu_us);
    result.profile.ce_mmse_gpu_us = estimate_gpu_us + equalize_gpu_us;
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_pdcch_crc_done, slot.gpu_event_stream_done,
                                        result.profile.d2h_us);
    result.profile.host_submit_us = elapsed_us(pending.submit_start, pending.d2h_start);
    result.profile.host_collect_us = elapsed_us(collect_start, Clock::now());
    result.profile.h2d_bytes = pending.h2d_bytes;
    result.profile.d2h_bytes = sizeof(*slot.pdcch_hit_count) +
                               static_cast<std::uint64_t>(detail::kCudaPdcchMaxCandidates) *
                                   sizeof(detail::CudaPdcchCandidateResult);
    result.profile.crc_hit_count = static_cast<std::uint32_t>(result.hits.size());
    result.profile.crc_miss_count = pending.candidate_count - result.profile.crc_hit_count;
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_equalize_done, slot.gpu_event_pdcch_llr_done,
                                        result.profile.llr_gpu_us);
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_pdcch_llr_done,
                                        slot.gpu_event_pdcch_rate_recovery_done,
                                        result.profile.rate_recovery_gpu_us);
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_pdcch_rate_recovery_done,
                                        slot.gpu_event_pdcch_viterbi_done,
                                        result.profile.viterbi_gpu_us);
    (void)detail::cuda_event_elapsed_us(slot.gpu_event_pdcch_viterbi_done,
                                        slot.gpu_event_pdcch_crc_done, result.profile.crc_gpu_us);
    buffers.completed_slot = pending.slot_index;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::run_pdcch_gpu_common_search_decode(
    const pdcch::PdcchGpuCommonSearchDecodeRequest& request,
    pdcch::PdcchGpuCommonSearchDecodeResult& result) {
    PendingPdcchDecode pending{};
    if (const MmseStatus status = submit_pdcch_gpu_common_search_decode(request, pending);
        status != MmseStatus::kOk) {
        return status;
    }
    return collect_pdcch_gpu_common_search_decode(pending, result);
}

MmseStatus MmseEqualizerGpuContext::init(const MmseEqualizerGpuConfig& config) {
    if (const MmseStatus status = Impl::validate_config(config); status != MmseStatus::kOk) {
        return status;
    }

    impl_->config = config;
    impl_->reset_runtime_state();

    switch (config.backend) {
    case MmseGpuBackend::kAuto:
    case MmseGpuBackend::kCuda:
        break;
    default:
        return MmseStatus::kInvalidArgument;
    }

    if (const MmseStatus status = impl_->configure_device_state(); status != MmseStatus::kOk) {
        impl_->reset_runtime_state();
        return status;
    }
    if (const MmseStatus status = impl_->configure_staging_buffers(); status != MmseStatus::kOk) {
        impl_->reset_runtime_state();
        return status;
    }

    const MmseStatus fallback_status = impl_->cpu_fallback.init(make_cpu_fallback_config(config));
    if (fallback_status != MmseStatus::kOk) {
        impl_->reset_runtime_state();
        return fallback_status;
    }

    impl_->initialized = true;
    impl_->using_cpu_fallback = true;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run(const PlanarGridViewF32& grid,
                                        const ExtractDescriptor& desc, EqualizerOutputView& out) {
    if (!impl_->initialized) {
        return MmseStatus::kNotInitialized;
    }
    if (const MmseStatus status = detail::validate_grid(grid); status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::validate_output(out); status != MmseStatus::kOk) {
        return status;
    }
    if (!detail::descriptor_supported(desc)) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (is_pdsch_transmit_diversity_descriptor(desc)) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (const MmseStatus status = impl_->prepare_subframe_if_needed(grid, desc);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = impl_->execute_backend(desc); status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = impl_->stage_outputs(out); status != MmseStatus::kOk) {
        return status;
    }
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run_pdsch_td(const PlanarGridViewF32& grid,
                                                 const ExtractDescriptor& desc,
                                                 PdschTdMmseOutputView& out,
                                                 PdschTdMmseResult& meta) {
    if (!impl_->initialized) {
        return MmseStatus::kNotInitialized;
    }
    if (!is_pdsch_transmit_diversity_descriptor(desc) || !detail::descriptor_supported(desc)) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (out.x_hat_re == nullptr || out.x_hat_im == nullptr || out.sinr == nullptr ||
        out.re_grid_indices0 == nullptr || out.re_grid_indices1 == nullptr) {
        return MmseStatus::kInvalidArgument;
    }
    if (impl_->config.backend == MmseGpuBackend::kAuto) {
        return impl_->cpu_fallback.run_pdsch_td(grid, desc, out, meta);
    }

    EqualizerOutputView base_out{out.x_hat_re, out.x_hat_im, out.sinr, out.capacity_symbols};
    if (const MmseStatus status = impl_->run_transmit_diversity(grid, desc, base_out);
        status != MmseStatus::kOk) {
        return status;
    }

    const auto& slot = impl_->buffers.slots[impl_->buffers.completed_slot];
    if (slot.grid_meta.td_pair_count * 2U != base_out.n_re_per_layer ||
        !transmit_diversity_pairs_stay_in_one_symbol(slot.grid_meta)) {
        return MmseStatus::kInternalError;
    }
    for (std::uint32_t pair = 0U; pair < slot.grid_meta.td_pair_count; ++pair) {
        const std::uint32_t index = pair * 2U;
        const std::uint16_t grid0 = slot.grid_meta.td_pair_grid_indices0[pair];
        const std::uint16_t grid1 = slot.grid_meta.td_pair_grid_indices1[pair];
        out.re_grid_indices0[index] = out.re_grid_indices0[index + 1U] = grid0;
        out.re_grid_indices1[index] = out.re_grid_indices1[index + 1U] = grid1;
    }

    meta = {};
    meta.n_symbols = base_out.n_re_per_layer;
    meta.n_source_re = base_out.n_re_per_layer;
    meta.sfn_subframe = desc.sfn_subframe;
    meta.grid_symbol_count = grid.n_symbols;
    meta.grid_subcarrier_count = grid.n_subcarriers;
    meta.cell_id = desc.cell_id;
    meta.n_prb = desc.n_prb;
    meta.start_symbol = desc.start_symbol;
    meta.n_tx_ports = desc.n_tx_ports;
    meta.n_rx_ant = desc.n_rx_ant;
    meta.n_layers = desc.n_layers;
    meta.tx_mode = desc.tx_mode;
    meta.mod_order = desc.mod_order;
    meta.pmi = desc.pmi;
    meta.sigma2 = slot.grid_meta.sigma2;
    meta.prb_bitmap = desc.prb_bitmap;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run_pbch(const PbchMmseInput& in, PbchMmseOutputView& out,
                                             PbchMmseResult& meta) {
    if (const MmseStatus status = mmse::pbch::validate_pbch_mmse_input(in);
        status != MmseStatus::kOk) {
        return status;
    }

    ExtractDescriptor desc{};
    desc.sfn_subframe = in.sfn_subframe;
    desc.cell_id = in.cell_id;
    desc.n_tx_ports = in.n_tx_ports;
    desc.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = in.tx_mode;
    desc.channel_type = MmseChannelType::kPbch;
    desc.start_symbol = kLtePbchStartSymbolNormalCp;
    desc.control_symbol_count = 0U;
    desc.mod_order = 2U;
    desc.n_prb = kLtePbchNumPrb;
    desc.prb_bitmap.fill(0U);
    for (std::uint32_t prb = kLtePbchStartPrb; prb < kLtePbchStartPrb + kLtePbchNumPrb; ++prb) {
        desc.prb_bitmap[prb / 16U] |= static_cast<std::uint16_t>(1U << (prb % 16U));
    }
    desc.pmi = -1;

    EqualizerOutputView base_out{};
    base_out.x_hat_re = out.x_hat_re;
    base_out.x_hat_im = out.x_hat_im;
    base_out.sinr = out.sinr;
    base_out.capacity_re_per_layer = out.capacity_re_per_layer;

    const MmseStatus status = run(in.grid, desc, base_out);
    if (status != MmseStatus::kOk) {
        return status;
    }
    if (out.re_grid_indices == nullptr || out.capacity_re_metadata < base_out.n_re_per_layer) {
        return MmseStatus::kBufferTooSmall;
    }

    const auto& slot = impl_->buffers.slots[impl_->buffers.completed_slot];
    for (std::uint32_t i = 0; i < base_out.n_re_per_layer; ++i) {
        out.re_grid_indices[i] = slot.layout.grid_indices[i];
    }

    meta = {};
    meta.n_re = base_out.n_re_per_layer;
    meta.sfn_subframe = in.sfn_subframe;
    meta.n_symbols = in.grid.n_symbols;
    meta.n_subcarriers = in.grid.n_subcarriers;
    meta.cell_id = in.cell_id;
    meta.start_prb = kLtePbchStartPrb;
    meta.n_prb = kLtePbchNumPrb;
    meta.start_symbol = kLtePbchStartSymbolNormalCp;
    meta.n_tx_ports = in.n_tx_ports;
    meta.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    meta.n_layers = base_out.n_layers;
    meta.tx_mode = in.tx_mode;
    meta.mod_order = base_out.mod_order;
    meta.sigma2 = slot.grid_meta.sigma2;
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run_pbch_td(const PbchMmseInput& in, PbchTdMmseOutputView& out,
                                                PbchTdMmseResult& meta) {
    if (!impl_->initialized) {
        return MmseStatus::kNotInitialized;
    }
    if (const MmseStatus status = mmse::pbch::validate_pbch_mmse_input(in);
        status != MmseStatus::kOk) {
        return status;
    }
    if (in.n_tx_ports != 2U || in.tx_mode != 2U) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (impl_->config.backend == MmseGpuBackend::kAuto) {
        return impl_->cpu_fallback.run_pbch_td(in, out, meta);
    }

    ExtractDescriptor desc{};
    desc.sfn_subframe = in.sfn_subframe;
    desc.cell_id = in.cell_id;
    desc.n_tx_ports = 2U;
    desc.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = 2U;
    desc.channel_type = MmseChannelType::kPbch;
    desc.start_symbol = kLtePbchStartSymbolNormalCp;
    desc.mod_order = 2U;
    desc.n_prb = kLtePbchNumPrb;
    for (std::uint32_t prb = kLtePbchStartPrb; prb < kLtePbchStartPrb + kLtePbchNumPrb; ++prb) {
        desc.prb_bitmap[prb / 16U] |= static_cast<std::uint16_t>(1U << (prb % 16U));
    }
    EqualizerOutputView base_out{out.x_hat_re, out.x_hat_im, out.sinr, out.capacity_symbols};
    if (out.re_grid_indices0 == nullptr || out.re_grid_indices1 == nullptr) {
        return MmseStatus::kInvalidArgument;
    }
    if (const MmseStatus status = impl_->run_transmit_diversity(in.grid, desc, base_out);
        status != MmseStatus::kOk) {
        return status;
    }
    const auto& slot = impl_->buffers.slots[impl_->buffers.completed_slot];
    if (slot.grid_meta.td_pair_count * 2U != base_out.n_re_per_layer) {
        return MmseStatus::kInternalError;
    }
    for (std::uint32_t pair = 0U; pair < slot.grid_meta.td_pair_count; ++pair) {
        const std::uint32_t index = pair * 2U;
        const std::uint16_t grid0 = slot.grid_meta.td_pair_grid_indices0[pair];
        const std::uint16_t grid1 = slot.grid_meta.td_pair_grid_indices1[pair];
        out.re_grid_indices0[index] = out.re_grid_indices0[index + 1U] = grid0;
        out.re_grid_indices1[index] = out.re_grid_indices1[index + 1U] = grid1;
    }
    meta = {};
    meta.n_symbols = base_out.n_re_per_layer;
    meta.n_source_re = base_out.n_re_per_layer;
    meta.sfn_subframe = in.sfn_subframe;
    meta.grid_symbol_count = in.grid.n_symbols;
    meta.grid_subcarrier_count = in.grid.n_subcarriers;
    meta.cell_id = in.cell_id;
    meta.start_prb = kLtePbchStartPrb;
    meta.n_prb = kLtePbchNumPrb;
    meta.start_symbol = kLtePbchStartSymbolNormalCp;
    meta.n_tx_ports = 2U;
    meta.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    meta.n_layers = 1U;
    meta.tx_mode = 2U;
    meta.mod_order = 2U;
    meta.sigma2 = slot.grid_meta.sigma2;
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run_pcfich(const PcfichMmseInput& in, PcfichMmseOutputView& out,
                                               PcfichMmseResult& meta) {
    if (const MmseStatus status = mmse::pcfich::validate_pcfich_mmse_input(in);
        status != MmseStatus::kOk) {
        return status;
    }

    ExtractDescriptor desc{};
    desc.sfn_subframe = in.sfn_subframe;
    desc.cell_id = in.cell_id;
    desc.n_tx_ports = in.n_tx_ports;
    desc.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = in.tx_mode;
    desc.channel_type = MmseChannelType::kPcfich;
    desc.start_symbol = 0U;
    desc.control_symbol_count = 1U;
    desc.mod_order = 2U;
    desc.n_prb = kLteNumPrb20MHz;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    desc.pmi = -1;

    EqualizerOutputView base_out{};
    base_out.x_hat_re = out.x_hat_re;
    base_out.x_hat_im = out.x_hat_im;
    base_out.sinr = out.sinr;
    base_out.capacity_re_per_layer = out.capacity_re_per_layer;

    const MmseStatus status = run(in.grid, desc, base_out);
    if (status != MmseStatus::kOk) {
        return status;
    }
    if (out.re_grid_indices == nullptr || out.capacity_re_metadata < base_out.n_re_per_layer) {
        return MmseStatus::kBufferTooSmall;
    }

    const auto& slot = impl_->buffers.slots[impl_->buffers.completed_slot];
    for (std::uint32_t i = 0; i < base_out.n_re_per_layer; ++i) {
        out.re_grid_indices[i] = slot.layout.grid_indices[i];
    }

    meta = {};
    meta.n_re = base_out.n_re_per_layer;
    meta.sfn_subframe = in.sfn_subframe;
    meta.n_symbols = in.grid.n_symbols;
    meta.n_subcarriers = in.grid.n_subcarriers;
    meta.cell_id = in.cell_id;
    meta.n_prb = kLteNumPrb20MHz;
    meta.start_symbol = 0U;
    meta.reg_count = static_cast<std::uint8_t>(kLtePcfichNumRegs);
    meta.n_tx_ports = in.n_tx_ports;
    meta.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    meta.n_layers = base_out.n_layers;
    meta.tx_mode = in.tx_mode;
    meta.mod_order = base_out.mod_order;
    meta.sigma2 = slot.grid_meta.sigma2;
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run_pcfich_td(const PcfichMmseInput& in,
                                                  PcfichTdMmseOutputView& out,
                                                  PcfichTdMmseResult& meta) {
    if (!impl_->initialized) {
        return MmseStatus::kNotInitialized;
    }
    if (const MmseStatus status = mmse::pcfich::validate_pcfich_mmse_input(in);
        status != MmseStatus::kOk) {
        return status;
    }
    if (in.n_tx_ports != 2U || in.tx_mode != 2U) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (impl_->config.backend == MmseGpuBackend::kAuto) {
        return impl_->cpu_fallback.run_pcfich_td(in, out, meta);
    }

    ExtractDescriptor desc{};
    desc.sfn_subframe = in.sfn_subframe;
    desc.cell_id = in.cell_id;
    desc.n_tx_ports = 2U;
    desc.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = 2U;
    desc.channel_type = MmseChannelType::kPcfich;
    desc.control_symbol_count = 1U;
    desc.mod_order = 2U;
    desc.n_prb = kLteNumPrb20MHz;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    EqualizerOutputView base_out{out.x_hat_re, out.x_hat_im, out.sinr, out.capacity_symbols};
    if (out.re_grid_indices0 == nullptr || out.re_grid_indices1 == nullptr) {
        return MmseStatus::kInvalidArgument;
    }
    if (const MmseStatus status = impl_->run_transmit_diversity(in.grid, desc, base_out);
        status != MmseStatus::kOk) {
        return status;
    }
    const auto& slot = impl_->buffers.slots[impl_->buffers.completed_slot];
    if (slot.grid_meta.td_pair_count * 2U != base_out.n_re_per_layer) {
        return MmseStatus::kInternalError;
    }
    for (std::uint32_t pair = 0U; pair < slot.grid_meta.td_pair_count; ++pair) {
        const std::uint32_t index = pair * 2U;
        const std::uint16_t grid0 = slot.grid_meta.td_pair_grid_indices0[pair];
        const std::uint16_t grid1 = slot.grid_meta.td_pair_grid_indices1[pair];
        out.re_grid_indices0[index] = out.re_grid_indices0[index + 1U] = grid0;
        out.re_grid_indices1[index] = out.re_grid_indices1[index + 1U] = grid1;
    }
    meta = {};
    meta.n_symbols = base_out.n_re_per_layer;
    meta.n_source_re = base_out.n_re_per_layer;
    meta.sfn_subframe = in.sfn_subframe;
    meta.grid_symbol_count = in.grid.n_symbols;
    meta.grid_subcarrier_count = in.grid.n_subcarriers;
    meta.cell_id = in.cell_id;
    meta.n_prb = kLteNumPrb20MHz;
    meta.start_symbol = 0U;
    meta.reg_count = static_cast<std::uint8_t>(kLtePcfichNumRegs);
    meta.n_tx_ports = 2U;
    meta.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    meta.n_layers = 1U;
    meta.tx_mode = 2U;
    meta.mod_order = 2U;
    meta.sigma2 = slot.grid_meta.sigma2;
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run_pdcch(const PdcchMmseInput& in, PdcchMmseOutputView& out,
                                              PdcchMmseResult& meta) {
    if (const MmseStatus status = mmse::pdcch::validate_pdcch_mmse_input(in);
        status != MmseStatus::kOk) {
        return status;
    }
    if (in.n_prb != kLteNumPrb20MHz || in.control_symbol_count > kLteMaxControlSymbolsNormalCp ||
        in.n_tx_ports != 1U) {
        return MmseStatus::kUnsupportedConfig;
    }

    ExtractDescriptor desc{};
    desc.sfn_subframe = in.sfn_subframe;
    desc.cell_id = in.cell_id;
    desc.n_tx_ports = in.n_tx_ports;
    desc.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = in.tx_mode;
    desc.channel_type = MmseChannelType::kPdcch;
    desc.start_symbol = 0U;
    desc.control_symbol_count = in.control_symbol_count;
    desc.mod_order = 2U;
    desc.n_prb = in.n_prb;
    desc.prb_bitmap = in.prb_bitmap;
    desc.control_re_exclusion_masks = in.control_re_exclusion_masks;
    desc.pmi = -1;

    EqualizerOutputView base_out{};
    base_out.x_hat_re = out.x_hat_re;
    base_out.x_hat_im = out.x_hat_im;
    base_out.sinr = out.sinr;
    base_out.capacity_re_per_layer = out.capacity_re_per_layer;

    const MmseStatus status = run(in.grid, desc, base_out);
    if (status != MmseStatus::kOk) {
        return status;
    }
    if (out.re_grid_indices == nullptr || out.capacity_re_metadata < base_out.n_re_per_layer) {
        return MmseStatus::kBufferTooSmall;
    }

    const auto& slot = impl_->buffers.slots[impl_->buffers.completed_slot];
    for (std::uint32_t i = 0; i < base_out.n_re_per_layer; ++i) {
        out.re_grid_indices[i] = slot.layout.grid_indices[i];
    }

    meta = {};
    meta.n_re = base_out.n_re_per_layer;
    meta.sfn_subframe = in.sfn_subframe;
    meta.n_symbols = in.grid.n_symbols;
    meta.n_subcarriers = in.grid.n_subcarriers;
    meta.cell_id = in.cell_id;
    meta.n_prb = in.n_prb;
    meta.n_tx_ports = in.n_tx_ports;
    meta.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    meta.n_layers = base_out.n_layers;
    meta.tx_mode = in.tx_mode;
    meta.control_symbol_count = in.control_symbol_count;
    meta.mod_order = base_out.mod_order;
    meta.sigma2 = slot.grid_meta.sigma2;
    meta.prb_bitmap = in.prb_bitmap;
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run_pdcch_td(const PdcchMmseInput& in,
                                                 PdcchTdMmseOutputView& out,
                                                 PdcchTdMmseResult& meta) {
    if (!impl_->initialized) {
        return MmseStatus::kNotInitialized;
    }
    if (const MmseStatus status = mmse::pdcch::validate_pdcch_mmse_input(in);
        status != MmseStatus::kOk) {
        return status;
    }
    if (in.n_prb != kLteNumPrb20MHz || in.control_symbol_count > kLteMaxControlSymbolsNormalCp ||
        in.n_tx_ports != 2U || in.tx_mode != 2U) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (impl_->config.backend == MmseGpuBackend::kAuto) {
        if (!impl_->using_cpu_fallback) {
            return MmseStatus::kUnsupportedConfig;
        }
        return impl_->cpu_fallback.run_pdcch_td(in, out, meta);
    }

    ExtractDescriptor desc{};
    desc.sfn_subframe = in.sfn_subframe;
    desc.cell_id = in.cell_id;
    desc.n_tx_ports = in.n_tx_ports;
    desc.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = in.tx_mode;
    desc.channel_type = MmseChannelType::kPdcch;
    desc.start_symbol = 0U;
    desc.control_symbol_count = in.control_symbol_count;
    desc.mod_order = 2U;
    desc.n_prb = in.n_prb;
    desc.prb_bitmap = in.prb_bitmap;
    desc.control_re_exclusion_masks = in.control_re_exclusion_masks;
    desc.pmi = -1;

    EqualizerOutputView base_out{};
    base_out.x_hat_re = out.x_hat_re;
    base_out.x_hat_im = out.x_hat_im;
    base_out.sinr = out.sinr;
    base_out.capacity_re_per_layer = out.capacity_symbols;

    if (out.re_grid_indices0 == nullptr || out.re_grid_indices1 == nullptr) {
        return MmseStatus::kInvalidArgument;
    }
    const MmseStatus status = impl_->run_transmit_diversity(in.grid, desc, base_out);
    if (status != MmseStatus::kOk) {
        return status;
    }
    if (out.capacity_symbols < base_out.n_re_per_layer) {
        return MmseStatus::kBufferTooSmall;
    }

    const auto& slot = impl_->buffers.slots[impl_->buffers.completed_slot];
    if (slot.grid_meta.td_pair_count * 2U != base_out.n_re_per_layer) {
        return MmseStatus::kInternalError;
    }
    for (std::uint32_t pair = 0U; pair < slot.grid_meta.td_pair_count; ++pair) {
        const std::uint32_t index = pair * 2U;
        const std::uint16_t grid0 = slot.grid_meta.td_pair_grid_indices0[pair];
        const std::uint16_t grid1 = slot.grid_meta.td_pair_grid_indices1[pair];
        out.re_grid_indices0[index] = grid0;
        out.re_grid_indices1[index] = grid1;
        out.re_grid_indices0[index + 1U] = grid0;
        out.re_grid_indices1[index + 1U] = grid1;
    }

    meta = {};
    meta.n_symbols = base_out.n_re_per_layer;
    meta.n_source_re = base_out.n_re_per_layer;
    meta.sfn_subframe = in.sfn_subframe;
    meta.grid_symbol_count = in.grid.n_symbols;
    meta.grid_subcarrier_count = in.grid.n_subcarriers;
    meta.cell_id = in.cell_id;
    meta.n_prb = in.n_prb;
    meta.n_tx_ports = in.n_tx_ports;
    meta.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    meta.n_layers = 1U;
    meta.tx_mode = in.tx_mode;
    meta.control_symbol_count = in.control_symbol_count;
    meta.mod_order = 2U;
    meta.sigma2 = slot.grid_meta.sigma2;
    meta.prb_bitmap = in.prb_bitmap;
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::run_pdcch_gpu_common_search_decode(
    const pdcch::PdcchGpuCommonSearchDecodeRequest& request,
    pdcch::PdcchGpuCommonSearchDecodeResult& result) {
    return impl_->run_pdcch_gpu_common_search_decode(request, result);
}

MmseStatus MmseEqualizerGpuContext::run_pdcch_gpu_common_search_decode_batch(
    std::span<const pdcch::PdcchGpuCommonSearchDecodeRequest> requests,
    std::span<pdcch::PdcchGpuCommonSearchDecodeResult> results) {
    if (requests.size() != results.size()) {
        return MmseStatus::kInvalidArgument;
    }
    for (auto& result : results) {
        result = {};
    }
    if (requests.empty()) {
        return MmseStatus::kOk;
    }

    const std::size_t group_capacity = impl_->buffers.slot_count;
    if (group_capacity == 0U) {
        return MmseStatus::kInternalError;
    }
    for (std::size_t first = 0U; first < requests.size(); first += group_capacity) {
        const std::size_t count = std::min(group_capacity, requests.size() - first);
        std::array<Impl::PendingPdcchDecode, kPinnedRingSlotCount> pending{};
        for (std::size_t offset = 0U; offset < count; ++offset) {
            if (const MmseStatus status = impl_->submit_pdcch_gpu_common_search_decode(
                    requests[first + offset], pending[offset]);
                status != MmseStatus::kOk) {
                for (std::size_t submitted = 0U; submitted < offset; ++submitted) {
                    pdcch::PdcchGpuCommonSearchDecodeResult discarded{};
                    (void)impl_->collect_pdcch_gpu_common_search_decode(pending[submitted],
                                                                        discarded);
                }
                return status;
            }
        }
        for (std::size_t offset = 0U; offset < count; ++offset) {
            if (const MmseStatus status = impl_->collect_pdcch_gpu_common_search_decode(
                    pending[offset], results[first + offset]);
                status != MmseStatus::kOk) {
                return status;
            }
        }
    }
    return MmseStatus::kOk;
}

MmseGpuHostProfileSnapshot MmseEqualizerGpuContext::last_host_profile() const {
    return impl_->last_host_profile;
}

} // namespace mmse
