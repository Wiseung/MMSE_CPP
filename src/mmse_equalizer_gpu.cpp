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
        std::array<float, 4> scratch_host{};
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
    if (const MmseStatus status =
            detail::cuda_alloc_device_buffer(slot.device.scratch, 4U * sizeof(float));
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
    std::fill_n(slot.grid_meta.output_slot_by_grid_re, detail::kCudaMaxGridRe,
                std::numeric_limits<std::uint32_t>::max());
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
    constexpr std::array<std::uint32_t, 3> kCheckRe = {0U, 17U, 255U};
    for (std::uint32_t layer = 0; layer < slot.out_view.n_layers; ++layer) {
        for (std::uint32_t re : kCheckRe) {
            if (re >= slot.out_view.n_re_per_layer) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(layer) * slot.out_view.capacity_re_per_layer + re;
            if (!std::isfinite(slot.xhat_re[idx]) || !std::isfinite(slot.xhat_im[idx]) ||
                !std::isfinite(slot.sinr[idx])) {
                return MmseStatus::kInternalError;
            }
            if (layer == 0U && re == 0U &&
                std::abs(slot.scratch_host[0] - static_cast<float>(re)) > 1.0F) {
                return MmseStatus::kInternalError;
            }
            if (!(slot.sinr[idx] > 0.0F)) {
                return MmseStatus::kInternalError;
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

    const MmseStatus cpu_status = cpu_fallback.run(slot.grid_view, desc, slot.out_view);
    if (cpu_status != MmseStatus::kOk) {
        return cpu_status;
    }
    slot.cpu_seeded = true;
    std::copy_n(slot.xhat_re, kOutputPlaneCount, slot.ref_xhat_re.data());
    std::copy_n(slot.xhat_im, kOutputPlaneCount, slot.ref_xhat_im.data());
    std::copy_n(slot.sinr, kOutputPlaneCount, slot.ref_sinr.data());
    detail::build_data_re_layout(desc, slot.layout);
    slot.grid_meta.n_valid_re = slot.layout.n_re;
    slot.grid_meta.first_valid_grid_idx = slot.layout.n_re == 0U ? 0U : slot.layout.grid_indices[0];
    slot.grid_meta.subframe = detail::subframe_from_descriptor(desc);
    slot.grid_meta.start_symbol = desc.start_symbol;
    slot.grid_meta.n_layers = desc.n_layers;
    slot.grid_meta.n_tx_ports = desc.n_tx_ports;
    slot.grid_meta.n_segments = slot.layout.n_segments;
    slot.grid_meta.sigma2 = config.sigma2_min;
    slot.grid_meta.g_min = config.g_min;
    slot.grid_meta.gamma_max = config.gamma_max;
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

    if (const MmseStatus status = detail::cuda_copy_outputs_h2d_async(
            slot.device, slot.xhat_re, slot.xhat_im, slot.sinr, slot.xhat_plane_bytes,
            slot.sinr_plane_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }

    const std::uint32_t completion_value = static_cast<std::uint32_t>(slot.sequence & 0xFFFFFFFFU);
    if (const MmseStatus status =
            detail::cuda_launch_estimate_stub(slot.device, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.kernel_submitted = true;

    if (const MmseStatus status = detail::cuda_copy_outputs_d2h_async(
            slot.device, slot.xhat_re, slot.xhat_im, slot.sinr, slot.xhat_plane_bytes,
            slot.sinr_plane_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    slot.d2h_submitted = true;

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
    slot.grid_meta.sigma2 = detail::update_sigma2_state(
        sigma2_by_cell[desc.cell_id], sigma2_estimate, make_cpu_fallback_config(config));
    if (const MmseStatus status = detail::cuda_copy_grid_h2d_async(
            slot.device, slot.transport_re, slot.transport_im, slot.grid_scale, slot.grid_meta,
            slot.grid_plane_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_copy_outputs_h2d_async(
            slot.device, slot.xhat_re, slot.xhat_im, slot.sinr, slot.xhat_plane_bytes,
            slot.sinr_plane_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status =
            detail::cuda_launch_equalize_stub(slot.device, completion_value, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::cuda_copy_outputs_d2h_async(
            slot.device, slot.xhat_re, slot.xhat_im, slot.sinr, slot.xhat_plane_bytes,
            slot.sinr_plane_bytes, slot.stream_handle);
        status != MmseStatus::kOk) {
        return status;
    }
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
    if (const MmseStatus status = validate_estimate_stub(desc, slot); status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = validate_equalize_stub(slot); status != MmseStatus::kOk) {
        return status;
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
    return impl_->stage_outputs(out);
}

} // namespace mmse
