#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "mmse/constants.h"
#include "mmse/types.h"

namespace mmse::detail {

inline constexpr std::uint32_t kCudaMaxGridRe = kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz;
inline constexpr std::uint32_t kCudaMaxDataRe = kCudaMaxGridRe;
inline constexpr std::uint32_t kCudaEstimateStubComplexCount =
    kLteNumSymbolsNormalCp * kLteNumTxPortsV1 * kLteNumRxAntV1 * kLteNumSubcarriers20MHz;
inline constexpr std::uint32_t kCudaEstimateStubFloatCount = 2U * kCudaEstimateStubComplexCount;
inline constexpr std::uint32_t kCudaValidationSampleCount = 12U;
inline constexpr std::uint32_t kCudaEqualizeTraceFloatCount = 32U;
inline constexpr std::uint32_t kCudaScratchHeaderFloatCount = 4U;
inline constexpr std::uint32_t kCudaScratchTraceFloatCount =
    kCudaScratchHeaderFloatCount + kCudaValidationSampleCount * kCudaEqualizeTraceFloatCount;

struct CudaDeviceBuffers {
    std::array<void*, 2> grid_re{};
    std::array<void*, 2> grid_im{};
    void* grid_meta = nullptr;
    void* sigma2 = nullptr;
    void* scratch = nullptr;
    void* h_estimate = nullptr;
    void* xhat_re = nullptr;
    void* xhat_im = nullptr;
    void* sinr = nullptr;
    void* completion = nullptr;
};

struct CudaGridMeta {
    std::uint32_t n_rx_ant = 0;
    std::uint32_t n_symbols = 0;
    std::uint32_t n_subcarriers = 0;
    std::uint32_t n_valid_re = 0;
    std::uint32_t first_valid_grid_idx = 0;
    std::uint32_t subframe = 0;
    std::uint32_t start_symbol = 0;
    std::uint32_t n_layers = 0;
    std::uint32_t n_tx_ports = 0;
    std::uint32_t channel_type = 0;
    std::uint32_t n_segments = 0;
    std::uint32_t spot_check_sample_count = 0;
    std::uint32_t trace_sample_count = 0;
    std::uint32_t sigma2_device_owned = 0;
    float sigma2 = 0.0F;
    float sigma2_iir_alpha = 0.0F;
    float det_floor = 0.0F;
    float g_min = 0.0F;
    float gamma_max = 0.0F;
    std::uint16_t grid_indices[kCudaMaxDataRe]{};
    std::uint16_t spot_check_re_slots[kCudaValidationSampleCount]{};
    std::uint32_t output_slot_by_grid_re[kCudaMaxGridRe]{};
    std::uint32_t prb_segment_offsets[kCudaMaxDataRe + 1]{};
    std::uint16_t prb_bitmap[7]{};
    std::uint8_t crs_symbols[kLteNumCrsSymbols]{};
    std::uint8_t crs_freq_offsets[kLteNumTxPortsV1][kLteNumCrsSymbols]{};
    float crs_pilot_re[kLteNumTxPortsV1][kLteNumCrsSymbols][kLteNumPilotTonesPerCrsSymbol]{};
    float crs_pilot_im[kLteNumTxPortsV1][kLteNumCrsSymbols][kLteNumPilotTonesPerCrsSymbol]{};
};

struct CudaDeviceInfo {
    std::array<char, 128> name{};
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t multi_processor_count = 0;
    std::uint32_t warp_size = 0;
    std::uint32_t max_threads_per_block = 0;
    std::uint32_t max_threads_per_multiprocessor = 0;
    std::uint32_t regs_per_block = 0;
    std::uint32_t regs_per_multiprocessor = 0;
    std::size_t shared_mem_per_block = 0;
    std::size_t shared_mem_per_multiprocessor = 0;
    std::size_t total_global_mem = 0;
    std::size_t l2_cache_size = 0;
};

struct CudaKernelResourceInfo {
    std::array<char, 64> name{};
    std::int32_t block_size = 0;
    std::int32_t max_threads_per_block = 0;
    std::int32_t num_regs = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t local_bytes = 0;
    std::size_t const_bytes = 0;
    std::int32_t binary_version = 0;
    std::int32_t ptx_version = 0;
    std::int32_t max_blocks_per_multiprocessor = 0;
    float theoretical_occupancy = 0.0F;
};

bool cuda_backend_compiled();

MmseStatus cuda_select_device(std::uint32_t device_ordinal, bool& runtime_available);
MmseStatus cuda_create_stream(std::uintptr_t& stream_handle);
void cuda_destroy_stream(std::uintptr_t stream_handle);
MmseStatus cuda_query_device_info(std::uint32_t device_ordinal, CudaDeviceInfo& info);
MmseStatus cuda_query_current_memory_info(std::size_t& free_bytes, std::size_t& total_bytes);
MmseStatus cuda_query_estimate_kernel_resources(CudaKernelResourceInfo& info);
MmseStatus cuda_query_equalize_kernel_resources(std::int32_t block_size,
                                                CudaKernelResourceInfo& info);
MmseStatus cuda_create_event(std::uintptr_t& event_handle);
void cuda_destroy_event(std::uintptr_t event_handle);
MmseStatus cuda_event_record(std::uintptr_t event_handle, std::uintptr_t stream_handle);
MmseStatus cuda_event_elapsed_us(std::uintptr_t start_event_handle, std::uintptr_t end_event_handle,
                                 double& elapsed_us);

MmseStatus cuda_alloc_host_f32(float*& ptr, std::size_t count, bool request_pinned,
                               bool& pinned_allocation);
void cuda_free_host_f32(float* ptr, bool pinned_allocation);

MmseStatus cuda_alloc_device_buffer(void*& ptr, std::size_t bytes);
void cuda_free_device_buffer(void* ptr);

MmseStatus cuda_copy_grid_h2d_async(const CudaDeviceBuffers& buffers,
                                    const std::array<float*, 2>& re,
                                    const std::array<float*, 2>& im, const CudaGridMeta& grid_meta,
                                    std::size_t grid_plane_bytes, std::uintptr_t stream_handle);
MmseStatus cuda_copy_grid_meta_h2d_async(const CudaDeviceBuffers& buffers,
                                         const CudaGridMeta& grid_meta,
                                         std::uintptr_t stream_handle);
MmseStatus cuda_copy_grid_meta_layout_slice_h2d_async(const CudaDeviceBuffers& buffers,
                                                      const CudaGridMeta& grid_meta,
                                                      std::uintptr_t stream_handle);
MmseStatus cuda_copy_grid_meta_dynamic_h2d_async(const CudaDeviceBuffers& buffers,
                                                 const CudaGridMeta& grid_meta,
                                                 std::uintptr_t stream_handle);

MmseStatus cuda_copy_sigma2_h2d_async(const CudaDeviceBuffers& buffers, float sigma2,
                                      std::uintptr_t stream_handle);

MmseStatus cuda_copy_outputs_h2d_async(const CudaDeviceBuffers& buffers, const float* xhat_re,
                                       const float* xhat_im, const float* sinr,
                                       std::size_t xhat_plane_bytes, std::size_t sinr_plane_bytes,
                                       std::uintptr_t stream_handle);

MmseStatus cuda_launch_estimate_stub(const CudaDeviceBuffers& buffers, std::uintptr_t stream_handle,
                                     std::uintptr_t residual_done_event_handle = 0);
MmseStatus cuda_launch_equalize_stub(const CudaDeviceBuffers& buffers, std::uint32_t n_valid_re,
                                     std::uint32_t completion_value, std::uintptr_t stream_handle);

MmseStatus cuda_copy_outputs_d2h_async(const CudaDeviceBuffers& buffers, float* xhat_re,
                                       float* xhat_im, float* sinr, std::size_t xhat_plane_bytes,
                                       std::size_t sinr_plane_bytes, std::uintptr_t stream_handle);

MmseStatus cuda_copy_estimate_d2h_async(const CudaDeviceBuffers& buffers, float* h_estimate,
                                        std::size_t estimate_bytes, std::uintptr_t stream_handle);

MmseStatus cuda_copy_sigma2_d2h_async(const CudaDeviceBuffers& buffers, float& sigma2,
                                      std::uintptr_t stream_handle);

MmseStatus cuda_copy_scratch_d2h_async(const CudaDeviceBuffers& buffers, float* scratch,
                                       std::size_t scratch_bytes, std::uintptr_t stream_handle);

MmseStatus cuda_copy_completion_d2h_async(const CudaDeviceBuffers& buffers,
                                          std::uint32_t& completion_value,
                                          std::uintptr_t stream_handle);

MmseStatus cuda_stream_synchronize(std::uintptr_t stream_handle);

} // namespace mmse::detail
