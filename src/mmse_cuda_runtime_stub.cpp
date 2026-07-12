#include "internal/mmse_cuda_runtime.h"

#include <algorithm>
#include <cstring>

namespace mmse::detail {

bool cuda_backend_compiled() {
    return false;
}

MmseStatus cuda_select_device(std::uint32_t, bool& runtime_available) {
    runtime_available = false;
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_create_stream(std::uintptr_t&) {
    return MmseStatus::kUnsupportedConfig;
}

void cuda_destroy_stream(std::uintptr_t) {}

MmseStatus cuda_query_device_info(std::uint32_t, CudaDeviceInfo&) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_query_current_memory_info(std::size_t&, std::size_t&) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_query_estimate_kernel_resources(CudaKernelResourceInfo&) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_query_equalize_kernel_resources(std::int32_t, CudaKernelResourceInfo&) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_create_event(std::uintptr_t&) {
    return MmseStatus::kUnsupportedConfig;
}

void cuda_destroy_event(std::uintptr_t) {}

MmseStatus cuda_event_record(std::uintptr_t, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_event_elapsed_us(std::uintptr_t, std::uintptr_t, double&) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_alloc_host_f32(float*& ptr, std::size_t count, bool, bool& pinned_allocation) {
    ptr = new (std::nothrow) float[count];
    if (ptr == nullptr) {
        pinned_allocation = false;
        return MmseStatus::kInternalError;
    }
    std::fill_n(ptr, count, 0.0F);
    pinned_allocation = false;
    return MmseStatus::kOk;
}

void cuda_free_host_f32(float* ptr, bool) {
    delete[] ptr;
}

MmseStatus cuda_alloc_host_bytes(void*& ptr, std::size_t bytes, bool, bool& pinned_allocation) {
    ptr = ::operator new(bytes, std::nothrow);
    if (ptr == nullptr) {
        pinned_allocation = false;
        return MmseStatus::kInternalError;
    }
    std::memset(ptr, 0, bytes);
    pinned_allocation = false;
    return MmseStatus::kOk;
}

void cuda_free_host_bytes(void* ptr, bool) {
    ::operator delete(ptr);
}

MmseStatus cuda_alloc_device_buffer(void*&, std::size_t) {
    return MmseStatus::kUnsupportedConfig;
}

void cuda_free_device_buffer(void*) {}

MmseStatus cuda_copy_grid_h2d_async(const CudaDeviceBuffers&, const std::array<float*, 2>&,
                                    const std::array<float*, 2>&, const CudaGridMeta&, std::size_t,
                                    std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_grid_meta_h2d_async(const CudaDeviceBuffers&, const CudaGridMeta&,
                                         std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_grid_meta_layout_slice_h2d_async(const CudaDeviceBuffers&, const CudaGridMeta&,
                                                      std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_grid_meta_dynamic_h2d_async(const CudaDeviceBuffers&, const CudaGridMeta&,
                                                 std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_sigma2_h2d_async(const CudaDeviceBuffers&, float, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_outputs_h2d_async(const CudaDeviceBuffers&, const float*, const float*,
                                       const float*, std::size_t, std::size_t, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_launch_estimate_stub(const CudaDeviceBuffers&, std::uintptr_t, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_launch_equalize_stub(const CudaDeviceBuffers&, std::uint32_t, std::uint32_t,
                                     std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_outputs_d2h_async(const CudaDeviceBuffers&, float*, float*, float*,
                                       std::size_t, std::size_t, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_estimate_d2h_async(const CudaDeviceBuffers&, float*, std::size_t,
                                        std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_sigma2_d2h_async(const CudaDeviceBuffers&, float&, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_scratch_d2h_async(const CudaDeviceBuffers&, float*, std::size_t,
                                       std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_completion_d2h_async(const CudaDeviceBuffers&, std::uint32_t&,
                                          std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_pdcch_candidates_h2d_async(const CudaDeviceBuffers&,
                                                const CudaPdcchCandidateDescriptor*, std::uint32_t,
                                                std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_pdcch_gold_words_h2d_async(const CudaDeviceBuffers&, const std::uint32_t*,
                                                std::uint32_t, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_launch_pdcch_llr_descramble(const CudaDeviceBuffers&, std::uint32_t,
                                            std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_launch_pdcch_rate_recovery(const CudaDeviceBuffers&, std::uint32_t,
                                           std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_launch_pdcch_viterbi(const CudaDeviceBuffers&, std::uint32_t, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_launch_pdcch_crc_compact(const CudaDeviceBuffers&, std::uint32_t, std::uint16_t,
                                         std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_copy_pdcch_results_d2h_async(const CudaDeviceBuffers&, CudaPdcchCandidateResult*,
                                             std::uint32_t&, std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

MmseStatus cuda_stream_synchronize(std::uintptr_t) {
    return MmseStatus::kUnsupportedConfig;
}

} // namespace mmse::detail
