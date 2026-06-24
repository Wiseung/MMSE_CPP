#include "internal/mmse_cuda_runtime.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace mmse::detail {

namespace {

MmseStatus map_cuda_error(cudaError_t status) {
    return status == cudaSuccess ? MmseStatus::kOk : MmseStatus::kUnsupportedConfig;
}

__global__ void estimate_stub_kernel(const std::int16_t* grid_re0,
                                     const std::int16_t* grid_re1,
                                     const std::int16_t* grid_im0,
                                     const std::int16_t* grid_im1,
                                     const float* grid_scale,
                                     const CudaGridMeta* grid_meta,
                                     float* sigma2_out,
                                     float* h_estimate) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (h_estimate == nullptr || grid_meta == nullptr || grid_scale == nullptr) {
            return;
        }
        float sigma2_accum = 0.0F;
        std::uint32_t residual_count = 0U;
        float freq_re[kLteNumCrsSymbols][kLteNumSubcarriers20MHz]{};
        float freq_im[kLteNumCrsSymbols][kLteNumSubcarriers20MHz]{};
        for (std::uint32_t tx = 0; tx < grid_meta->n_tx_ports && tx < kLteNumTxPortsV1; ++tx) {
            for (std::uint32_t rx = 0; rx < grid_meta->n_rx_ant && rx < kLteNumRxAntV1; ++rx) {
                for (std::uint32_t cs = 0; cs < kLteNumCrsSymbols; ++cs) {
                    const std::uint32_t grid_idx =
                        grid_meta->crs_symbols[cs] * grid_meta->n_subcarriers;
                    const std::int16_t* re_plane =
                        rx == 0U ? grid_re0 : grid_re1;
                    const std::int16_t* im_plane =
                        rx == 0U ? grid_im0 : grid_im1;
                    float ls_re[kLteNumPilotTonesPerCrsSymbol]{};
                    float ls_im[kLteNumPilotTonesPerCrsSymbol]{};
                    for (std::uint32_t pilot = 0; pilot < kLteNumPilotTonesPerCrsSymbol; ++pilot) {
                        const std::uint32_t sc =
                            6U * pilot + grid_meta->crs_freq_offsets[tx][cs];
                        const std::uint32_t re_idx = grid_idx + sc;
                        float y_re = 0.0F;
                        float y_im = 0.0F;
                        if (re_plane != nullptr) {
                            y_re = static_cast<float>(re_plane[re_idx]) * grid_scale[rx];
                        }
                        if (im_plane != nullptr) {
                            y_im = static_cast<float>(im_plane[re_idx]) * grid_scale[rx];
                        }
                        const float crs_re = grid_meta->crs_pilot_re[tx][cs][pilot];
                        const float crs_im = grid_meta->crs_pilot_im[tx][cs][pilot];
                        ls_re[pilot] = y_re * crs_re + y_im * crs_im;
                        ls_im[pilot] = y_im * crs_re - y_re * crs_im;
                    }
                    for (std::uint32_t pilot = 1U; pilot + 1U < kLteNumPilotTonesPerCrsSymbol;
                         ++pilot) {
                        const float smooth_re = 0.5F * (ls_re[pilot - 1U] + ls_re[pilot + 1U]);
                        const float smooth_im = 0.5F * (ls_im[pilot - 1U] + ls_im[pilot + 1U]);
                        const float diff_re = ls_re[pilot] - smooth_re;
                        const float diff_im = ls_im[pilot] - smooth_im;
                        sigma2_accum += diff_re * diff_re + diff_im * diff_im;
                        ++residual_count;
                    }
                    for (std::uint32_t sc = 0; sc < grid_meta->n_subcarriers; ++sc) {
                        const std::uint32_t lower_pilot =
                            (sc < grid_meta->crs_freq_offsets[tx][cs])
                                ? 0U
                                : (sc - grid_meta->crs_freq_offsets[tx][cs]) / 6U;
                        const std::uint32_t upper_pilot =
                            min(lower_pilot + 1U, kLteNumPilotTonesPerCrsSymbol - 1U);
                        const std::uint32_t left_sc =
                            6U * lower_pilot + grid_meta->crs_freq_offsets[tx][cs];
                        const std::uint32_t right_sc =
                            6U * upper_pilot + grid_meta->crs_freq_offsets[tx][cs];
                        float out_re = ls_re[lower_pilot];
                        float out_im = ls_im[lower_pilot];
                        if (left_sc != right_sc && sc > left_sc && sc < right_sc) {
                            const float t = static_cast<float>(sc - left_sc) /
                                            static_cast<float>(right_sc - left_sc);
                            out_re = ls_re[lower_pilot] +
                                     (ls_re[upper_pilot] - ls_re[lower_pilot]) * t;
                            out_im = ls_im[lower_pilot] +
                                     (ls_im[upper_pilot] - ls_im[lower_pilot]) * t;
                        } else if (sc >= right_sc) {
                            out_re = ls_re[upper_pilot];
                            out_im = ls_im[upper_pilot];
                        }
                        freq_re[cs][sc] = out_re;
                        freq_im[cs][sc] = out_im;
                    }
                }
                for (std::uint32_t symbol = 0; symbol < kLteNumSymbolsNormalCp; ++symbol) {
                    std::uint32_t upper = 0U;
                    while (upper < kLteNumCrsSymbols && grid_meta->crs_symbols[upper] < symbol) {
                        ++upper;
                    }
                    std::uint32_t lower = (upper == 0U) ? 0U : upper - 1U;
                    if (upper >= kLteNumCrsSymbols) {
                        upper = kLteNumCrsSymbols - 1U;
                        lower = upper;
                    }
                    const std::uint32_t left_symbol = grid_meta->crs_symbols[lower];
                    const std::uint32_t right_symbol = grid_meta->crs_symbols[upper];
                    for (std::uint32_t sc = 0; sc < grid_meta->n_subcarriers; ++sc) {
                        float out_re = freq_re[lower][sc];
                        float out_im = freq_im[lower][sc];
                        if (left_symbol != right_symbol) {
                            const float t = static_cast<float>(symbol - left_symbol) /
                                            static_cast<float>(right_symbol - left_symbol);
                            out_re =
                                freq_re[lower][sc] + (freq_re[upper][sc] - freq_re[lower][sc]) * t;
                            out_im =
                                freq_im[lower][sc] + (freq_im[upper][sc] - freq_im[lower][sc]) * t;
                        }
                        const std::uint32_t base =
                            2U * (((tx * kLteNumRxAntV1 + rx) * kLteNumSymbolsNormalCp + symbol) *
                                      kLteNumSubcarriers20MHz +
                                  sc);
                        h_estimate[base] = out_re;
                        h_estimate[base + 1U] = out_im;
                    }
                }
            }
        }
        if (sigma2_out != nullptr) {
            float sigma2 = residual_count == 0U ? 0.0F : sigma2_accum / static_cast<float>(residual_count);
            sigma2 = sigma2 < grid_meta->sigma2 ? grid_meta->sigma2 : sigma2;
            *sigma2_out = sigma2;
        }
    }
}

__global__ void equalize_stub_kernel(const std::int16_t* grid_re0,
                                     const std::int16_t* grid_re1,
                                     const std::int16_t* grid_im0,
                                     const std::int16_t* grid_im1,
                                     const float* grid_scale,
                                     const float* h_estimate,
                                     const float* sigma2_in,
                                     const CudaGridMeta* grid_meta,
                                     float* xhat_re,
                                     float* xhat_im,
                                     float* sinr,
                                     float* scratch,
                                     std::uint32_t* completion,
                                     std::uint32_t completion_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (grid_meta != nullptr && h_estimate != nullptr && xhat_re != nullptr &&
            xhat_im != nullptr && sinr != nullptr && grid_scale != nullptr) {
            const float sigma2 = sigma2_in != nullptr ? *sigma2_in : grid_meta->sigma2;
            const float g_min = grid_meta->g_min;
            const float gamma_max = grid_meta->gamma_max;
            for (std::uint32_t re = 0; re < grid_meta->n_valid_re; ++re) {
                const std::uint32_t grid_idx = grid_meta->grid_indices[re];
                const std::uint32_t out_slot = grid_meta->output_slot_by_grid_re[grid_idx];
                const std::uint32_t symbol = grid_idx / grid_meta->n_subcarriers;
                const std::uint32_t sc = grid_idx % grid_meta->n_subcarriers;

                const float y0_re = static_cast<float>(grid_re0[grid_idx]) * grid_scale[0];
                const float y0_im = static_cast<float>(grid_im0[grid_idx]) * grid_scale[0];
                const float y1_re = static_cast<float>(grid_re1[grid_idx]) * grid_scale[1];
                const float y1_im = static_cast<float>(grid_im1[grid_idx]) * grid_scale[1];

                const std::uint32_t h00_base =
                    2U * (((0U * kLteNumRxAntV1 + 0U) * kLteNumSymbolsNormalCp + symbol) *
                              kLteNumSubcarriers20MHz +
                          sc);
                const std::uint32_t h01_base =
                    2U * (((1U * kLteNumRxAntV1 + 0U) * kLteNumSymbolsNormalCp + symbol) *
                              kLteNumSubcarriers20MHz +
                          sc);
                const std::uint32_t h10_base =
                    2U * (((0U * kLteNumRxAntV1 + 1U) * kLteNumSymbolsNormalCp + symbol) *
                              kLteNumSubcarriers20MHz +
                          sc);
                const std::uint32_t h11_base =
                    2U * (((1U * kLteNumRxAntV1 + 1U) * kLteNumSymbolsNormalCp + symbol) *
                              kLteNumSubcarriers20MHz +
                          sc);

                const float h00_re = h_estimate[h00_base];
                const float h00_im = h_estimate[h00_base + 1U];
                const float h10_re = h_estimate[h10_base];
                const float h10_im = h_estimate[h10_base + 1U];

                if (grid_meta->n_layers == 1U) {
                    const float denom = h00_re * h00_re + h00_im * h00_im + h10_re * h10_re +
                                        h10_im * h10_im + sigma2;
                    const float w0_re = h00_re / denom;
                    const float w0_im = -h00_im / denom;
                    const float w1_re = h10_re / denom;
                    const float w1_im = -h10_im / denom;
                    const float z0_re =
                        (w0_re * y0_re - w0_im * y0_im) + (w1_re * y1_re - w1_im * y1_im);
                    const float z0_im =
                        (w0_re * y0_im + w0_im * y0_re) + (w1_re * y1_im + w1_im * y1_re);
                    float g0 =
                        (h00_re * h00_re + h00_im * h00_im + h10_re * h10_re + h10_im * h10_im) /
                        denom;
                    g0 = g0 < g_min ? g_min : (g0 > 1.0F - g_min ? 1.0F - g_min : g0);
                    xhat_re[out_slot] = z0_re / g0;
                    xhat_im[out_slot] = z0_im / g0;
                    float gamma0 = g0 / (1.0F - g0);
                    gamma0 = gamma0 > gamma_max ? gamma_max : gamma0;
                    sinr[out_slot] = gamma0;
                } else {
                    const float h01_re = h_estimate[h01_base];
                    const float h01_im = h_estimate[h01_base + 1U];
                    const float h11_re = h_estimate[h11_base];
                    const float h11_im = h_estimate[h11_base + 1U];

                    const float a11 = h00_re * h00_re + h00_im * h00_im + h10_re * h10_re +
                                      h10_im * h10_im + sigma2;
                    const float a22 = h01_re * h01_re + h01_im * h01_im + h11_re * h11_re +
                                      h11_im * h11_im + sigma2;
                    const float a12_re = h00_re * h01_re + h00_im * h01_im + h10_re * h11_re +
                                         h10_im * h11_im;
                    const float a12_im = -h00_re * h01_im + h00_im * h01_re - h10_re * h11_im +
                                         h10_im * h11_re;
                    float det = a11 * a22 - (a12_re * a12_re + a12_im * a12_im);
                    det = det < 1.0e-6F ? 1.0e-6F : det;
                    const float inv_det = 1.0F / det;

                    const float inv11 = a22 * inv_det;
                    const float inv22 = a11 * inv_det;
                    const float inv12_re = -a12_re * inv_det;
                    const float inv12_im = -a12_im * inv_det;
                    const float inv21_re = inv12_re;
                    const float inv21_im = -inv12_im;

                    const float hh00_re = h00_re;
                    const float hh00_im = -h00_im;
                    const float hh01_re = h10_re;
                    const float hh01_im = -h10_im;
                    const float hh10_re = h01_re;
                    const float hh10_im = -h01_im;
                    const float hh11_re = h11_re;
                    const float hh11_im = -h11_im;

                    const float w00_re =
                        inv11 * hh00_re + (inv12_re * hh10_re - inv12_im * hh10_im);
                    const float w00_im =
                        inv11 * hh00_im + (inv12_re * hh10_im + inv12_im * hh10_re);
                    const float w01_re =
                        inv11 * hh01_re + (inv12_re * hh11_re - inv12_im * hh11_im);
                    const float w01_im =
                        inv11 * hh01_im + (inv12_re * hh11_im + inv12_im * hh11_re);
                    const float w10_re =
                        (inv21_re * hh00_re - inv21_im * hh00_im) + inv22 * hh10_re;
                    const float w10_im =
                        (inv21_re * hh00_im + inv21_im * hh00_re) + inv22 * hh10_im;
                    const float w11_re =
                        (inv21_re * hh01_re - inv21_im * hh01_im) + inv22 * hh11_re;
                    const float w11_im =
                        (inv21_re * hh01_im + inv21_im * hh01_re) + inv22 * hh11_im;

                    const float z0_re =
                        (w00_re * y0_re - w00_im * y0_im) + (w01_re * y1_re - w01_im * y1_im);
                    const float z0_im =
                        (w00_re * y0_im + w00_im * y0_re) + (w01_re * y1_im + w01_im * y1_re);
                    const float z1_re =
                        (w10_re * y0_re - w10_im * y0_im) + (w11_re * y1_re - w11_im * y1_im);
                    const float z1_im =
                        (w10_re * y0_im + w10_im * y0_re) + (w11_re * y1_im + w11_im * y1_re);

                    const float g0_re =
                        (w00_re * h00_re - w00_im * h00_im) + (w01_re * h10_re - w01_im * h10_im);
                    const float g1_re =
                        (w10_re * h01_re - w10_im * h01_im) + (w11_re * h11_re - w11_im * h11_im);
                    float g0 =
                        g0_re < g_min ? g_min : (g0_re > 1.0F - g_min ? 1.0F - g_min : g0_re);
                    float g1 =
                        g1_re < g_min ? g_min : (g1_re > 1.0F - g_min ? 1.0F - g_min : g1_re);

                    xhat_re[out_slot] = z0_re / g0;
                    xhat_im[out_slot] = z0_im / g0;
                    float gamma0 = g0 / (1.0F - g0);
                    gamma0 = gamma0 > gamma_max ? gamma_max : gamma0;
                    sinr[out_slot] = gamma0;

                    const std::uint32_t layer1_base = kCudaMaxDataRe + out_slot;
                    xhat_re[layer1_base] = z1_re / g1;
                    xhat_im[layer1_base] = z1_im / g1;
                    float gamma1 = g1 / (1.0F - g1);
                    gamma1 = gamma1 > gamma_max ? gamma_max : gamma1;
                    sinr[layer1_base] = gamma1;
                }

                if (re == 0U && scratch != nullptr) {
                    scratch[0] = static_cast<float>(out_slot);
                    scratch[1] = static_cast<float>(symbol);
                    scratch[2] = static_cast<float>(sc);
                    scratch[3] = sigma2;
                }
            }
        } else if (scratch != nullptr && h_estimate != nullptr) {
            scratch[0] = h_estimate[0];
            scratch[1] = h_estimate[1];
            scratch[2] = 0.0F;
            scratch[3] = 0.0F;
        }
        if (completion != nullptr) {
            *completion = completion_value;
        }
    }
}

}  // namespace

bool cuda_backend_compiled() {
    return true;
}

MmseStatus cuda_select_device(std::uint32_t device_ordinal, bool& runtime_available) {
    const int ordinal = static_cast<int>(device_ordinal);
    const cudaError_t set_status = cudaSetDevice(ordinal);
    runtime_available = (set_status == cudaSuccess);
    return map_cuda_error(set_status);
}

MmseStatus cuda_create_stream(std::uintptr_t& stream_handle) {
    cudaStream_t stream = nullptr;
    const cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        stream_handle = 0;
        return map_cuda_error(status);
    }
    stream_handle = reinterpret_cast<std::uintptr_t>(stream);
    return MmseStatus::kOk;
}

void cuda_destroy_stream(std::uintptr_t stream_handle) {
    if (stream_handle != 0) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_handle));
    }
}

MmseStatus cuda_alloc_host_f32(float*& ptr,
                               std::size_t count,
                               bool request_pinned,
                               bool& pinned_allocation) {
    ptr = nullptr;
    pinned_allocation = false;
    if (!request_pinned) {
        ptr = new (std::nothrow) float[count];
        if (ptr == nullptr) {
            return MmseStatus::kInternalError;
        }
        for (std::size_t i = 0; i < count; ++i) {
            ptr[i] = 0.0F;
        }
        return MmseStatus::kOk;
    }

    const cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                                             count * sizeof(float),
                                             cudaHostAllocDefault);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }
    pinned_allocation = true;
    for (std::size_t i = 0; i < count; ++i) {
        ptr[i] = 0.0F;
    }
    return MmseStatus::kOk;
}

void cuda_free_host_f32(float* ptr, bool pinned_allocation) {
    if (ptr == nullptr) {
        return;
    }
    if (pinned_allocation) {
        cudaFreeHost(ptr);
        return;
    }
    delete[] ptr;
}

MmseStatus cuda_alloc_host_i16(std::int16_t*& ptr,
                               std::size_t count,
                               bool request_pinned,
                               bool& pinned_allocation) {
    ptr = nullptr;
    pinned_allocation = false;
    if (!request_pinned) {
        ptr = new (std::nothrow) std::int16_t[count];
        if (ptr == nullptr) {
            return MmseStatus::kInternalError;
        }
        for (std::size_t i = 0; i < count; ++i) {
            ptr[i] = 0;
        }
        return MmseStatus::kOk;
    }

    const cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                                             count * sizeof(std::int16_t),
                                             cudaHostAllocDefault);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }
    pinned_allocation = true;
    for (std::size_t i = 0; i < count; ++i) {
        ptr[i] = 0;
    }
    return MmseStatus::kOk;
}

void cuda_free_host_i16(std::int16_t* ptr, bool pinned_allocation) {
    if (ptr == nullptr) {
        return;
    }
    if (pinned_allocation) {
        cudaFreeHost(ptr);
        return;
    }
    delete[] ptr;
}

MmseStatus cuda_alloc_device_buffer(void*& ptr, std::size_t bytes) {
    ptr = nullptr;
    return map_cuda_error(cudaMalloc(&ptr, bytes));
}

void cuda_free_device_buffer(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

MmseStatus cuda_copy_grid_h2d_async(const CudaDeviceBuffers& buffers,
                                    const std::array<std::int16_t*, 2>& re,
                                    const std::array<std::int16_t*, 2>& im,
                                    const std::array<float, 2>& grid_scale,
                                    const CudaGridMeta& grid_meta,
                                    std::size_t grid_plane_bytes,
                                    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    for (std::size_t rx = 0; rx < re.size(); ++rx) {
        if (const cudaError_t status =
                cudaMemcpyAsync(buffers.grid_re[rx], re[rx], grid_plane_bytes, cudaMemcpyHostToDevice, stream);
            status != cudaSuccess) {
            return map_cuda_error(status);
        }
        if (const cudaError_t status =
                cudaMemcpyAsync(buffers.grid_im[rx], im[rx], grid_plane_bytes, cudaMemcpyHostToDevice, stream);
            status != cudaSuccess) {
            return map_cuda_error(status);
        }
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(buffers.grid_scale,
                            grid_scale.data(),
                            grid_scale.size() * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(buffers.grid_meta,
                            &grid_meta,
                            sizeof(grid_meta),
                            cudaMemcpyHostToDevice,
                            stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    return MmseStatus::kOk;
}

MmseStatus cuda_copy_outputs_h2d_async(const CudaDeviceBuffers& buffers,
                                       const float* xhat_re,
                                       const float* xhat_im,
                                       const float* sinr,
                                       std::size_t xhat_plane_bytes,
                                       std::size_t sinr_plane_bytes,
                                       std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (const cudaError_t status =
            cudaMemcpyAsync(buffers.xhat_re, xhat_re, xhat_plane_bytes, cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(buffers.xhat_im, xhat_im, xhat_plane_bytes, cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(buffers.sinr, sinr, sinr_plane_bytes, cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    return MmseStatus::kOk;
}

MmseStatus cuda_launch_estimate_stub(const CudaDeviceBuffers& buffers,
                                     std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    estimate_stub_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const std::int16_t*>(buffers.grid_re[0]),
        reinterpret_cast<const std::int16_t*>(buffers.grid_re[1]),
        reinterpret_cast<const std::int16_t*>(buffers.grid_im[0]),
        reinterpret_cast<const std::int16_t*>(buffers.grid_im[1]),
        reinterpret_cast<const float*>(buffers.grid_scale),
        reinterpret_cast<const CudaGridMeta*>(buffers.grid_meta),
        reinterpret_cast<float*>(buffers.sigma2),
        reinterpret_cast<float*>(buffers.h_estimate));
    return map_cuda_error(cudaGetLastError());
}

MmseStatus cuda_launch_equalize_stub(const CudaDeviceBuffers& buffers,
                                     std::uint32_t completion_value,
                                     std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    equalize_stub_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const std::int16_t*>(buffers.grid_re[0]),
        reinterpret_cast<const std::int16_t*>(buffers.grid_re[1]),
        reinterpret_cast<const std::int16_t*>(buffers.grid_im[0]),
        reinterpret_cast<const std::int16_t*>(buffers.grid_im[1]),
        reinterpret_cast<const float*>(buffers.grid_scale),
        reinterpret_cast<const float*>(buffers.h_estimate),
        reinterpret_cast<const float*>(buffers.sigma2),
        reinterpret_cast<const CudaGridMeta*>(buffers.grid_meta),
        reinterpret_cast<float*>(buffers.xhat_re),
        reinterpret_cast<float*>(buffers.xhat_im),
        reinterpret_cast<float*>(buffers.sinr),
        reinterpret_cast<float*>(buffers.scratch),
        reinterpret_cast<std::uint32_t*>(buffers.completion),
        completion_value);
    return map_cuda_error(cudaGetLastError());
}

MmseStatus cuda_copy_outputs_d2h_async(const CudaDeviceBuffers& buffers,
                                       float* xhat_re,
                                       float* xhat_im,
                                       float* sinr,
                                       std::size_t xhat_plane_bytes,
                                       std::size_t sinr_plane_bytes,
                                       std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (const cudaError_t status =
            cudaMemcpyAsync(xhat_re, buffers.xhat_re, xhat_plane_bytes, cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(xhat_im, buffers.xhat_im, xhat_plane_bytes, cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(sinr, buffers.sinr, sinr_plane_bytes, cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    return MmseStatus::kOk;
}

MmseStatus cuda_copy_estimate_d2h_async(const CudaDeviceBuffers& buffers,
                                        float* h_estimate,
                                        std::size_t estimate_bytes,
                                        std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    return map_cuda_error(cudaMemcpyAsync(
        h_estimate, buffers.h_estimate, estimate_bytes, cudaMemcpyDeviceToHost, stream));
}

MmseStatus cuda_copy_sigma2_d2h_async(const CudaDeviceBuffers& buffers,
                                      float& sigma2,
                                      std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    return map_cuda_error(
        cudaMemcpyAsync(&sigma2, buffers.sigma2, sizeof(sigma2), cudaMemcpyDeviceToHost, stream));
}

MmseStatus cuda_copy_scratch_d2h_async(const CudaDeviceBuffers& buffers,
                                       float* scratch,
                                       std::size_t scratch_bytes,
                                       std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    return map_cuda_error(cudaMemcpyAsync(
        scratch, buffers.scratch, scratch_bytes, cudaMemcpyDeviceToHost, stream));
}

MmseStatus cuda_copy_completion_d2h_async(const CudaDeviceBuffers& buffers,
                                          std::uint32_t& completion_value,
                                          std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    return map_cuda_error(cudaMemcpyAsync(&completion_value,
                                          buffers.completion,
                                          sizeof(completion_value),
                                          cudaMemcpyDeviceToHost,
                                          stream));
}

MmseStatus cuda_stream_synchronize(std::uintptr_t stream_handle) {
    return map_cuda_error(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_handle)));
}

}  // namespace mmse::detail
