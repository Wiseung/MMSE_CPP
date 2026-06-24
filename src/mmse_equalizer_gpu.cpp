#include "mmse/mmse_equalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "internal/mmse_cuda_runtime.h"
#include "internal/mmse_internal.h"

namespace mmse {

namespace {

constexpr std::uint32_t kPinnedRingSlotCount = 3U;
constexpr std::size_t kGridPlaneCount = detail::kMaxGridRe;
constexpr std::size_t kEstimateStubFloatCount = detail::kCudaEstimateStubFloatCount;
constexpr std::size_t kOutputPlaneCount = detail::kMaxDataRe * 2U;
constexpr float kGridTransportScale = 1.0F / 32767.0F;
constexpr bool kEnableDeepCudaStubValidation = false;

std::int16_t quantize_grid_sample(float value, float scale) {
    const float scaled = value / scale;
    const float clamped = std::clamp(scaled, -32767.0F, 32767.0F);
    return static_cast<std::int16_t>(clamped >= 0.0F ? clamped + 0.5F : clamped - 0.5F);
}

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

} // namespace

struct MmseEqualizerGpuContext::Impl {
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

    struct HostPinnedSlot {
        std::uint32_t slot_ordinal = 0;
        std::uint32_t stream_ordinal = 0;
        std::uintptr_t stream_handle = 0;
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
        std::array<float, 2> grid_scale = {kGridTransportScale, kGridTransportScale};
        detail::CudaGridMeta grid_meta{};
        detail::ReLayout layout{};
        std::array<float*, 2> staging_re{};
        std::array<float*, 2> staging_im{};
        std::array<std::int16_t*, 2> transport_re{};
        std::array<std::int16_t*, 2> transport_im{};
        float* h_estimate = nullptr;
        std::array<float, detail::kCudaScratchFloatCount> scratch_host{};
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

    struct StagingBuffers {
        std::array<HostPinnedSlot, kPinnedRingSlotCount> slots{};
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
    void reset_runtime_state();
    void release_runtime_state();
    MmseStatus configure_device_state();
    MmseStatus configure_staging_buffers();
    MmseStatus initialize_slot_storage(HostPinnedSlot& slot);
    MmseStatus initialize_slot_device_buffers(HostPinnedSlot& slot);
    MmseStatus stage_inputs(const PlanarGridViewF32& grid);
    MmseStatus validate_estimate_stub(const ExtractDescriptor& desc,
                                      const HostPinnedSlot& slot) const;
    MmseStatus validate_equalize_stub(const HostPinnedSlot& slot) const;
    MmseStatus execute_backend(const ExtractDescriptor& desc);
    MmseStatus execute_cuda_transport_stub(const ExtractDescriptor& desc, HostPinnedSlot& slot);
    MmseStatus stage_outputs(EqualizerOutputView& out) const;

    MmseEqualizerGpuConfig config{};
    bool initialized = false;
    bool using_cpu_fallback = false;
    DeviceState device{};
    std::array<detail::Sigma2State, kLteNumCellIds> sigma2_by_cell{};
    MmseEqualizerCpuContext cpu_fallback{};
    StagingBuffers buffers{};
};

MmseEqualizerGpuContext::MmseEqualizerGpuContext() : impl_(new Impl()) {}

MmseEqualizerGpuContext::~MmseEqualizerGpuContext() {
    delete impl_;
}

MmseStatus MmseEqualizerGpuContext::Impl::validate_config(const MmseEqualizerGpuConfig& config) {
    if (config.stream_count == 0U || config.g_min <= 0.0F || config.g_min >= 0.5F ||
        config.gamma_max <= 0.0F || config.det_floor <= 0.0F || config.sigma2_min <= 0.0F ||
        config.sigma2_iir_alpha < 0.0F || config.sigma2_iir_alpha > 1.0F) {
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

MmseStatus
MmseEqualizerGpuContext::Impl::compare_equalize_trace_sample(const HostPinnedSlot& slot,
                                                             std::uint32_t sample_index) const {
    if (sample_index >= slot.grid_meta.validation_sample_count) {
        return MmseStatus::kInvalidArgument;
    }

    const std::uint32_t re = slot.grid_meta.validation_re_slots[sample_index];
    if (re >= slot.layout.n_re) {
        return MmseStatus::kInternalError;
    }

    const std::uint32_t grid_idx = slot.layout.grid_indices[re];
    const std::uint32_t symbol = grid_idx / slot.grid_meta.n_subcarriers;
    const std::uint32_t sc = grid_idx % slot.grid_meta.n_subcarriers;
    const float y0_re = static_cast<float>(slot.transport_re[0][grid_idx]) * slot.grid_scale[0];
    const float y0_im = static_cast<float>(slot.transport_im[0][grid_idx]) * slot.grid_scale[0];
    const float y1_re = static_cast<float>(slot.transport_re[1][grid_idx]) * slot.grid_scale[1];
    const float y1_im = static_cast<float>(slot.transport_im[1][grid_idx]) * slot.grid_scale[1];
    const auto h_at = [&](std::uint32_t tx, std::uint32_t rx) {
        const std::size_t base =
            2U * (((tx * kLteNumRxAntV1 + rx) * kLteNumSymbolsNormalCp + symbol) *
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
    for (auto& slot : buffers.slots) {
        detail::cuda_free_device_buffer(slot.device.grid_re[0]);
        detail::cuda_free_device_buffer(slot.device.grid_re[1]);
        detail::cuda_free_device_buffer(slot.device.grid_im[0]);
        detail::cuda_free_device_buffer(slot.device.grid_im[1]);
        detail::cuda_free_device_buffer(slot.device.grid_scale);
        detail::cuda_free_device_buffer(slot.device.grid_meta);
        detail::cuda_free_device_buffer(slot.device.sigma2);
        detail::cuda_free_device_buffer(slot.device.scratch);
        detail::cuda_free_device_buffer(slot.device.h_estimate);
        detail::cuda_free_device_buffer(slot.device.xhat_re);
        detail::cuda_free_device_buffer(slot.device.xhat_im);
        detail::cuda_free_device_buffer(slot.device.sinr);
        detail::cuda_free_device_buffer(slot.device.completion);

        detail::cuda_free_host_f32(slot.staging_re[0], false);
        detail::cuda_free_host_f32(slot.staging_re[1], false);
        detail::cuda_free_host_f32(slot.staging_im[0], false);
        detail::cuda_free_host_f32(slot.staging_im[1], false);
        detail::cuda_free_host_i16(slot.transport_re[0], slot.host_grid_is_pinned);
        detail::cuda_free_host_i16(slot.transport_re[1], slot.host_grid_is_pinned);
        detail::cuda_free_host_i16(slot.transport_im[0], slot.host_grid_is_pinned);
        detail::cuda_free_host_i16(slot.transport_im[1], slot.host_grid_is_pinned);
        detail::cuda_free_host_f32(slot.h_estimate, slot.host_output_is_pinned);
        detail::cuda_free_host_f32(slot.xhat_re, slot.host_output_is_pinned);
        detail::cuda_free_host_f32(slot.xhat_im, slot.host_output_is_pinned);
        detail::cuda_free_host_f32(slot.sinr, slot.host_output_is_pinned);
        slot.slot_ordinal = 0;
        slot.stream_ordinal = 0;
        slot.stream_handle = 0;
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
        slot.grid_scale = {kGridTransportScale, kGridTransportScale};
        slot.grid_meta = {};
        slot.layout = {};
        slot.staging_re = {};
        slot.staging_im = {};
        slot.transport_re = {};
        slot.transport_im = {};
        slot.h_estimate = nullptr;
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
    buffers.active_slot = 0;
    buffers.completed_slot = 0;
    buffers.next_sequence = 1;
    buffers.ring_ready = false;
}

MmseStatus MmseEqualizerGpuContext::Impl::configure_device_state() {
    device.requested_device_ordinal = config.device_ordinal;
    device.active_device_ordinal = config.device_ordinal;
    device.selection_ready = true;
    device.cuda_runtime_available = false;
    device.streams.clear();

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
    bool grid_pinned = false;
    bool output_pinned = false;
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.staging_re[0], kGridPlaneCount, false, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.staging_re[1], kGridPlaneCount, false, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.staging_im[0], kGridPlaneCount, false, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_f32(slot.staging_im[1], kGridPlaneCount, false, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_i16(slot.transport_re[0], kGridPlaneCount, true, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.host_grid_is_pinned = grid_pinned;
    if (const MmseStatus status =
            detail::cuda_alloc_host_i16(slot.transport_re[1], kGridPlaneCount, true, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_i16(slot.transport_im[0], kGridPlaneCount, true, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_host_i16(slot.transport_im[1], kGridPlaneCount, true, grid_pinned);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_alloc_host_f32(
            slot.h_estimate, kEstimateStubFloatCount, true, output_pinned);
        status != MmseStatus::kOk) {
        return status;
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

    slot.grid_view.re = {slot.staging_re[0], slot.staging_re[1]};
    slot.grid_view.im = {slot.staging_im[0], slot.staging_im[1]};
    slot.grid_view.n_rx_ant = kLteNumRxAntV1;
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
    if (const MmseStatus status = detail::cuda_alloc_device_buffer(
            slot.device.grid_scale, slot.grid_scale.size() * sizeof(float));
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
            slot.device.scratch, detail::kCudaScratchFloatCount * sizeof(float));
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.h_estimate, slot.estimate_bytes);
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

    slot.device_buffers_ready = true;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::configure_staging_buffers() {
    buffers.ring_ready = true;
    buffers.next_sequence = 1;

    for (std::uint32_t slot_index = 0; slot_index < kPinnedRingSlotCount; ++slot_index) {
        auto& slot = buffers.slots[slot_index];
        slot.slot_ordinal = slot_index;
        slot.stream_ordinal = device.streams.empty() ? 0U : slot_index % config.stream_count;
        slot.stream_handle =
            device.streams.empty() ? 0U : device.streams[slot.stream_ordinal].handle;
        slot.grid_plane_bytes = kGridPlaneCount * sizeof(std::int16_t);
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

MmseStatus MmseEqualizerGpuContext::Impl::stage_inputs(const PlanarGridViewF32& grid) {
    if (!buffers.ring_ready) {
        return MmseStatus::kInternalError;
    }

    auto& slot = buffers.slots[buffers.next_stage_slot];
    for (std::uint32_t rx = 0; rx < kLteNumRxAntV1; ++rx) {
        std::copy_n(grid.re[rx], detail::kMaxGridRe, slot.staging_re[rx]);
        std::copy_n(grid.im[rx], detail::kMaxGridRe, slot.staging_im[rx]);
        for (std::size_t i = 0; i < kGridPlaneCount; ++i) {
            slot.transport_re[rx][i] =
                quantize_grid_sample(slot.staging_re[rx][i], slot.grid_scale[rx]);
            slot.transport_im[rx][i] =
                quantize_grid_sample(slot.staging_im[rx][i], slot.grid_scale[rx]);
        }
    }
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
    slot.grid_meta.validation_sample_count = 0;
    std::fill_n(slot.grid_meta.output_slot_by_grid_re, detail::kCudaMaxGridRe,
                std::numeric_limits<std::uint32_t>::max());
    std::fill(slot.grid_meta.validation_re_slots,
              slot.grid_meta.validation_re_slots + detail::kCudaValidationSampleCount,
              static_cast<std::uint16_t>(0));
    std::fill(slot.scratch_host.begin(), slot.scratch_host.end(), 0.0F);
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
    slot.sequence = buffers.next_sequence++;
    slot.out_view.n_re_per_layer = 0;
    slot.out_view.n_layers = 0;
    slot.out_view.mod_order = 0;

    buffers.active_slot = buffers.next_stage_slot;
    buffers.next_stage_slot = (buffers.next_stage_slot + 1U) % kPinnedRingSlotCount;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::validate_estimate_stub(const ExtractDescriptor& desc,
                                                                 const HostPinnedSlot& slot) const {
    detail::HGridStorage h_full{};
    float sigma2_estimate = 0.0F;
    detail::estimate_channel(slot.grid_view, desc, h_full, sigma2_estimate);

    constexpr std::array<std::uint32_t, 3> kCheckSymbols = {0U, 6U, 13U};
    constexpr std::array<std::uint32_t, 3> kCheckSc = {0U, 127U, 1199U};
    for (std::uint32_t tx = 0; tx < kLteNumTxPortsV1; ++tx) {
        for (std::uint32_t rx = 0; rx < kLteNumRxAntV1; ++rx) {
            for (std::uint32_t symbol : kCheckSymbols) {
                for (std::uint32_t sc : kCheckSc) {
                    const std::size_t base =
                        2U * (((tx * kLteNumRxAntV1 + rx) * kLteNumSymbolsNormalCp + symbol) *
                                  kLteNumSubcarriers20MHz +
                              sc);
                    const detail::Complex32 ref =
                        h_full[((static_cast<std::size_t>(tx) * kLteNumRxAntV1 + rx) *
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
    for (std::uint32_t sample = 0; sample < slot.grid_meta.validation_sample_count; ++sample) {
        const std::uint32_t re = slot.grid_meta.validation_re_slots[sample];
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
            if constexpr (!kEnableDeepCudaStubValidation) {
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
    if constexpr (kEnableDeepCudaStubValidation) {
        if (slot.grid_meta.n_layers == 2U) {
            for (std::uint32_t sample = 0; sample < slot.grid_meta.validation_sample_count;
                 ++sample) {
                if (compare_equalize_trace_sample(slot, sample) != MmseStatus::kOk) {
                    return MmseStatus::kInternalError;
                }
            }
        }
    }
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerGpuContext::Impl::execute_cuda_transport_stub(const ExtractDescriptor& desc,
                                                                      HostPinnedSlot& slot) {
    if (!detail::cuda_backend_compiled() || !device.cuda_runtime_available ||
        !slot.device_buffers_ready || slot.stream_handle == 0U) {
        return MmseStatus::kUnsupportedConfig;
    }

    detail::build_data_re_layout(desc, slot.layout);
    slot.grid_meta.n_valid_re = slot.layout.n_re;
    slot.grid_meta.first_valid_grid_idx = slot.layout.n_re == 0U ? 0U : slot.layout.grid_indices[0];
    slot.grid_meta.subframe = detail::subframe_from_descriptor(desc);
    slot.grid_meta.start_symbol = desc.start_symbol;
    slot.grid_meta.n_layers = desc.n_layers;
    slot.grid_meta.n_tx_ports = desc.n_tx_ports;
    slot.grid_meta.n_segments = slot.layout.n_segments;
    slot.grid_meta.sigma2 = config.sigma2_min;
    slot.grid_meta.det_floor = config.det_floor;
    slot.grid_meta.g_min = config.g_min;
    slot.grid_meta.gamma_max = config.gamma_max;
    std::array<std::uint32_t, detail::kCudaValidationSampleCount> validation_re_slots{};
    slot.grid_meta.validation_sample_count = detail::build_validation_re_samples(
        slot.layout, desc.start_symbol, slot.grid_meta.n_symbols, slot.grid_meta.n_subcarriers,
        validation_re_slots.data(), detail::kCudaValidationSampleCount);
    for (std::uint32_t i = 0; i < slot.grid_meta.validation_sample_count; ++i) {
        slot.grid_meta.validation_re_slots[i] = static_cast<std::uint16_t>(validation_re_slots[i]);
    }
    if constexpr (kEnableDeepCudaStubValidation) {
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
    for (std::uint32_t tx = 0; tx < kLteNumTxPortsV1; ++tx) {
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

    if (const MmseStatus status = detail::cuda_copy_grid_h2d_async(
            slot.device, slot.transport_re, slot.transport_im, slot.grid_scale, slot.grid_meta,
            slot.grid_plane_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.h2d_submitted = true;

    const std::uint32_t completion_value = static_cast<std::uint32_t>(slot.sequence & 0xFFFFFFFFU);
    if (const MmseStatus status =
            detail::cuda_launch_estimate_stub(slot.device, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.kernel_submitted = true;

    if (const MmseStatus status = detail::cuda_copy_estimate_d2h_async(
            slot.device, slot.h_estimate, slot.estimate_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    float sigma2_estimate = 0.0F;
    if (const MmseStatus status =
            detail::cuda_copy_sigma2_d2h_async(slot.device, sigma2_estimate, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_stream_synchronize(slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.grid_meta.sigma2 = detail::update_sigma2_state(
        sigma2_by_cell[desc.cell_id], sigma2_estimate, make_cpu_fallback_config(config));
    if (const MmseStatus status =
            detail::cuda_copy_grid_meta_h2d_async(slot.device, slot.grid_meta, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_copy_outputs_h2d_async(
            slot.device, slot.xhat_re, slot.xhat_im, slot.sinr, slot.xhat_plane_bytes,
            slot.sinr_plane_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_launch_equalize_stub(
            slot.device, slot.grid_meta.n_valid_re, completion_value, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_copy_outputs_d2h_async(
            slot.device, slot.xhat_re, slot.xhat_im, slot.sinr, slot.xhat_plane_bytes,
            slot.sinr_plane_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.d2h_submitted = true;
    if (const MmseStatus status = detail::cuda_copy_scratch_d2h_async(
            slot.device, slot.scratch_host.data(), slot.scratch_host.size() * sizeof(float),
            slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_copy_completion_d2h_async(
            slot.device, slot.completion_value, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_stream_synchronize(slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if constexpr (kEnableDeepCudaStubValidation) {
        if (desc.n_layers == 2U) {
            if (const MmseStatus status = validate_estimate_stub(desc, slot);
                status != MmseStatus::kOk) {
                return status;
            }
            if (const MmseStatus status = validate_equalize_stub(slot); status != MmseStatus::kOk) {
                return status;
            }
        }
    } else {
        if (const MmseStatus status = validate_equalize_stub(slot); status != MmseStatus::kOk) {
            return status;
        }
    }
    if (!(slot.scratch_host[3] >= config.sigma2_min)) {
        return MmseStatus::kInternalError;
    }

    slot.completion_ready = (slot.completion_value == completion_value);
    if (!slot.completion_ready) {
        return MmseStatus::kInternalError;
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

    MmseStatus status = MmseStatus::kUnsupportedConfig;
    switch (config.backend) {
    case MmseGpuBackend::kCuda:
        if (!device.selection_ready || device.streams.empty() ||
            !device.streams[slot.stream_ordinal].ready) {
            return MmseStatus::kInternalError;
        }
        status = execute_cuda_transport_stub(desc, slot);
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
    return MmseStatus::kOk;
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
    if (const MmseStatus status = impl_->stage_inputs(grid); status != MmseStatus::kOk) {
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

} // namespace mmse
