#include "internal/mmse_cuda_runtime.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace mmse::detail {

namespace {

constexpr std::uint32_t kEstimateThreadsPerBlock = 256U;
constexpr std::uint32_t kEstimateResidualBlocks = 64U;

struct CudaComplex32 {
    float re = 0.0F;
    float im = 0.0F;
};

__device__ inline CudaComplex32 cadd(CudaComplex32 a, CudaComplex32 b) {
    return {a.re + b.re, a.im + b.im};
}

__device__ inline CudaComplex32 cmul(CudaComplex32 a, CudaComplex32 b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ inline CudaComplex32 cconj(CudaComplex32 a) {
    return {a.re, -a.im};
}

__device__ inline CudaComplex32 cscale(CudaComplex32 a, float s) {
    return {a.re * s, a.im * s};
}

__device__ inline float cnorm2(CudaComplex32 a) {
    return a.re * a.re + a.im * a.im;
}

__device__ inline float clampf(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__host__ __device__ inline std::uint32_t min_u32(std::uint32_t lhs, std::uint32_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__host__ __device__ inline std::uint32_t ceil_div_u32(std::uint32_t num, std::uint32_t den) {
    return den == 0U ? 0U : (num + den - 1U) / den;
}

__device__ inline CudaComplex32 load_grid_sample(const float* re_plane,
                                                 const float* im_plane,
                                                 std::uint32_t grid_idx) {
    float y_re = 0.0F;
    float y_im = 0.0F;
    if (re_plane != nullptr) {
        y_re = re_plane[grid_idx];
    }
    if (im_plane != nullptr) {
        y_im = im_plane[grid_idx];
    }
    return {y_re, y_im};
}

__device__ inline CudaComplex32 ls_at(const float* re_plane,
                                      const float* im_plane,
                                      const CudaGridMeta* grid_meta,
                                      std::uint32_t tx,
                                      std::uint32_t rx,
                                      std::uint32_t cs,
                                      std::uint32_t pilot) {
    const std::uint32_t symbol = grid_meta->crs_symbols[cs];
    const std::uint32_t sc = 6U * pilot + grid_meta->crs_freq_offsets[tx][cs];
    const std::uint32_t grid_idx = symbol * grid_meta->n_subcarriers + sc;
    const CudaComplex32 y = load_grid_sample(re_plane, im_plane, grid_idx);
    const float crs_re = grid_meta->crs_pilot_re[tx][cs][pilot];
    const float crs_im = grid_meta->crs_pilot_im[tx][cs][pilot];
    return {y.re * crs_re + y.im * crs_im, y.im * crs_re - y.re * crs_im};
}

__device__ inline CudaComplex32 interpolate_freq(const float* re_plane,
                                                 const float* im_plane,
                                                 const CudaGridMeta* grid_meta,
                                                 std::uint32_t tx,
                                                 std::uint32_t rx,
                                                 std::uint32_t cs,
                                                 std::uint32_t sc) {
    const std::uint32_t first_offset = grid_meta->crs_freq_offsets[tx][cs];
    const std::uint32_t lower_pilot =
        (sc < first_offset) ? 0U : (sc - first_offset) / 6U;
    const std::uint32_t clamped_lower =
        min_u32(lower_pilot, kLteNumPilotTonesPerCrsSymbol - 1U);
    const std::uint32_t upper_pilot =
        min_u32(clamped_lower + 1U, kLteNumPilotTonesPerCrsSymbol - 1U);
    const std::uint32_t left_sc = 6U * clamped_lower + first_offset;
    const std::uint32_t right_sc = 6U * upper_pilot + first_offset;
    const CudaComplex32 left = ls_at(re_plane, im_plane, grid_meta, tx, rx, cs, clamped_lower);
    const CudaComplex32 right = ls_at(re_plane, im_plane, grid_meta, tx, rx, cs, upper_pilot);
    if (left_sc != right_sc && sc > left_sc && sc < right_sc) {
        const float t =
            static_cast<float>(sc - left_sc) / static_cast<float>(right_sc - left_sc);
        return {left.re + (right.re - left.re) * t, left.im + (right.im - left.im) * t};
    }
    return (sc >= right_sc) ? right : left;
}

__device__ inline CudaComplex32 estimate_h_at(const float* re_plane,
                                              const float* im_plane,
                                              const CudaGridMeta* grid_meta,
                                              std::uint32_t tx,
                                              std::uint32_t rx,
                                              std::uint32_t symbol,
                                              std::uint32_t sc) {
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
    const CudaComplex32 left = interpolate_freq(re_plane, im_plane, grid_meta, tx, rx, lower, sc);
    const CudaComplex32 right = interpolate_freq(re_plane, im_plane, grid_meta, tx, rx, upper, sc);
    if (left_symbol != right_symbol) {
        const float t = static_cast<float>(symbol - left_symbol) /
                        static_cast<float>(right_symbol - left_symbol);
        return {left.re + (right.re - left.re) * t, left.im + (right.im - left.im) * t};
    }
    return left;
}

MmseStatus map_cuda_error(cudaError_t status) {
    return status == cudaSuccess ? MmseStatus::kOk : MmseStatus::kUnsupportedConfig;
}

template <typename TKernel>
MmseStatus fill_kernel_resource_info(TKernel kernel,
                                     const char* kernel_name,
                                     std::int32_t block_size,
                                     CudaKernelResourceInfo& info) {
    cudaFuncAttributes attrs{};
    if (const cudaError_t status = cudaFuncGetAttributes(&attrs, kernel); status != cudaSuccess) {
        return map_cuda_error(status);
    }

    info = {};
    for (std::size_t i = 0; kernel_name[i] != '\0' && i + 1U < info.name.size(); ++i) {
        info.name[i] = kernel_name[i];
    }
    info.block_size = block_size;
    info.max_threads_per_block = attrs.maxThreadsPerBlock;
    info.num_regs = attrs.numRegs;
    info.static_shared_bytes = static_cast<std::size_t>(attrs.sharedSizeBytes);
    info.local_bytes = static_cast<std::size_t>(attrs.localSizeBytes);
    info.const_bytes = static_cast<std::size_t>(attrs.constSizeBytes);
    info.binary_version = attrs.binaryVersion;
    info.ptx_version = attrs.ptxVersion;

    int active_blocks = 0;
    if (const cudaError_t status =
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, block_size, 0);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    info.max_blocks_per_multiprocessor = active_blocks;

    int device = 0;
    if (const cudaError_t status = cudaGetDevice(&device); status != cudaSuccess) {
        return map_cuda_error(status);
    }
    cudaDeviceProp prop{};
    if (const cudaError_t status = cudaGetDeviceProperties(&prop, device); status != cudaSuccess) {
        return map_cuda_error(status);
    }

    const float active_threads = static_cast<float>(active_blocks * block_size);
    info.theoretical_occupancy =
        prop.maxThreadsPerMultiProcessor == 0
            ? 0.0F
            : active_threads / static_cast<float>(prop.maxThreadsPerMultiProcessor);
    return MmseStatus::kOk;
}

__global__ void estimate_residual_kernel(const float* grid_re0,
                                         const float* grid_re1,
                                         const float* grid_im0,
                                         const float* grid_im1,
                                         const CudaGridMeta* grid_meta,
                                         float* sigma2_accum_out,
                                         std::uint32_t* residual_count_out) {
    if (grid_meta == nullptr || sigma2_accum_out == nullptr || residual_count_out == nullptr) {
        return;
    }

    const std::uint32_t total_items =
        min_u32(grid_meta->n_tx_ports, kLteNumTxPortsV1) *
        min_u32(grid_meta->n_rx_ant, kLteNumRxAntV1) * kLteNumCrsSymbols *
        (kLteNumPilotTonesPerCrsSymbol - 2U);

    float local_sigma2 = 0.0F;
    std::uint32_t local_count = 0U;
    const std::uint32_t stride = blockDim.x * gridDim.x;
    for (std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < total_items;
         item += stride) {
        std::uint32_t index = item;
        const std::uint32_t pilot = (index % (kLteNumPilotTonesPerCrsSymbol - 2U)) + 1U;
        index /= (kLteNumPilotTonesPerCrsSymbol - 2U);
        const std::uint32_t cs = index % kLteNumCrsSymbols;
        index /= kLteNumCrsSymbols;
        const std::uint32_t rx = index % min_u32(grid_meta->n_rx_ant, kLteNumRxAntV1);
        index /= min_u32(grid_meta->n_rx_ant, kLteNumRxAntV1);
        const std::uint32_t tx = index;

        const float* re_plane = (rx == 0U) ? grid_re0 : grid_re1;
        const float* im_plane = (rx == 0U) ? grid_im0 : grid_im1;
        const CudaComplex32 prev = ls_at(re_plane, im_plane, grid_meta, tx, rx, cs, pilot - 1U);
        const CudaComplex32 curr = ls_at(re_plane, im_plane, grid_meta, tx, rx, cs, pilot);
        const CudaComplex32 next = ls_at(re_plane, im_plane, grid_meta, tx, rx, cs, pilot + 1U);
        const float smooth_re = 0.5F * (prev.re + next.re);
        const float smooth_im = 0.5F * (prev.im + next.im);
        const float diff_re = curr.re - smooth_re;
        const float diff_im = curr.im - smooth_im;
        local_sigma2 += diff_re * diff_re + diff_im * diff_im;
        local_count += 1U;
    }

    atomicAdd(sigma2_accum_out, local_sigma2);
    atomicAdd(residual_count_out, local_count);
}

__global__ void estimate_channel_kernel(const float* grid_re0,
                                        const float* grid_re1,
                                        const float* grid_im0,
                                        const float* grid_im1,
                                        const CudaGridMeta* grid_meta,
                                        float* h_estimate) {
    if (h_estimate == nullptr || grid_meta == nullptr) {
        return;
    }

    const std::uint32_t tx_count = min_u32(grid_meta->n_tx_ports, kLteNumTxPortsV1);
    const std::uint32_t rx_count = min_u32(grid_meta->n_rx_ant, kLteNumRxAntV1);
    const std::uint32_t total_h = tx_count * rx_count * kLteNumSymbolsNormalCp *
                                  grid_meta->n_subcarriers;
    const std::uint32_t stride = blockDim.x * gridDim.x;
    for (std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total_h;
         linear += stride) {
        std::uint32_t index = linear;
        const std::uint32_t sc = index % grid_meta->n_subcarriers;
        index /= grid_meta->n_subcarriers;
        const std::uint32_t symbol = index % kLteNumSymbolsNormalCp;
        index /= kLteNumSymbolsNormalCp;
        const std::uint32_t rx = index % rx_count;
        index /= rx_count;
        const std::uint32_t tx = index;

        const float* re_plane = (rx == 0U) ? grid_re0 : grid_re1;
        const float* im_plane = (rx == 0U) ? grid_im0 : grid_im1;
        const CudaComplex32 h = estimate_h_at(re_plane, im_plane, grid_meta, tx, rx, symbol, sc);
        const std::uint32_t base =
            2U * (((tx * kLteNumRxAntV1 + rx) * kLteNumSymbolsNormalCp + symbol) *
                      kLteNumSubcarriers20MHz +
                  sc);
        h_estimate[base] = h.re;
        h_estimate[base + 1U] = h.im;
    }
}

__global__ void finalize_sigma2_kernel(const CudaGridMeta* grid_meta,
                                       const float* sigma2_accum_in,
                                       const std::uint32_t* residual_count_in,
                                       float* sigma2_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || grid_meta == nullptr || sigma2_state == nullptr ||
        sigma2_accum_in == nullptr || residual_count_in == nullptr) {
        return;
    }

    const std::uint32_t residual_count = *residual_count_in;
    float sigma2 =
        residual_count == 0U ? 0.0F : *sigma2_accum_in / static_cast<float>(residual_count);
    sigma2 = sigma2 < grid_meta->sigma2 ? grid_meta->sigma2 : sigma2;
    if (grid_meta->sigma2_device_owned == 0U) {
        *sigma2_state = sigma2;
    } else {
        const float previous = *sigma2_state;
        if (!(previous >= grid_meta->sigma2)) {
            *sigma2_state = sigma2;
        } else {
            const float alpha = clampf(grid_meta->sigma2_iir_alpha, 0.0F, 1.0F);
            const float updated = alpha * previous + (1.0F - alpha) * sigma2;
            *sigma2_state = updated < grid_meta->sigma2 ? grid_meta->sigma2 : updated;
        }
    }
}

__global__ void equalize_stub_kernel(const float* grid_re0,
                                     const float* grid_re1,
                                     const float* grid_im0,
                                     const float* grid_im1,
                                     const float* h_estimate,
                                     const float* sigma2_in,
                                     const CudaGridMeta* grid_meta,
                                     float* xhat_re,
                                     float* xhat_im,
                                     float* sinr,
                                     float* scratch,
                                      std::uint32_t* completion,
                                      std::uint32_t completion_value) {
    if (grid_meta == nullptr || h_estimate == nullptr || xhat_re == nullptr || xhat_im == nullptr ||
        sinr == nullptr) {
        if (threadIdx.x == 0 && blockIdx.x == 0 && scratch != nullptr && h_estimate != nullptr) {
            scratch[0] = h_estimate[0];
            scratch[1] = h_estimate[1];
            scratch[2] = 0.0F;
            scratch[3] = 0.0F;
        }
        if (threadIdx.x == 0 && blockIdx.x == 0 && completion != nullptr) {
            *completion = completion_value;
        }
        return;
    }

    const std::uint32_t re = blockIdx.x * blockDim.x + threadIdx.x;
    if (re < grid_meta->n_valid_re) {
        const float sigma2 = sigma2_in != nullptr ? *sigma2_in : grid_meta->sigma2;
        const float det_floor = grid_meta->det_floor;
        const float g_min = grid_meta->g_min;
        const float gamma_max = grid_meta->gamma_max;
        const std::uint32_t grid_idx = grid_meta->grid_indices[re];
        const std::uint32_t out_slot = grid_meta->output_slot_by_grid_re[grid_idx];
        const std::uint32_t symbol = grid_idx / grid_meta->n_subcarriers;
        const std::uint32_t sc = grid_idx % grid_meta->n_subcarriers;

        const float y0_re = grid_re0[grid_idx];
        const float y0_im = grid_im0[grid_idx];
        const float y1_re = grid_re1[grid_idx];
        const float y1_im = grid_im1[grid_idx];

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
                (h00_re * h00_re + h00_im * h00_im + h10_re * h10_re + h10_im * h10_im) / denom;
            g0 = g0 < g_min ? g_min : (g0 > 1.0F - g_min ? 1.0F - g_min : g0);
            xhat_re[out_slot] = z0_re / g0;
            xhat_im[out_slot] = z0_im / g0;
            float gamma0 = g0 / (1.0F - g0);
            gamma0 = gamma0 > gamma_max ? gamma_max : gamma0;
            sinr[out_slot] = gamma0;
        } else {
            const CudaComplex32 h00{h00_re, h00_im};
            const CudaComplex32 h01{h_estimate[h01_base], h_estimate[h01_base + 1U]};
            const CudaComplex32 h10{h10_re, h10_im};
            const CudaComplex32 h11{h_estimate[h11_base], h_estimate[h11_base + 1U]};
            const CudaComplex32 y0{y0_re, y0_im};
            const CudaComplex32 y1{y1_re, y1_im};

            const float a11 = cnorm2(h00) + cnorm2(h10) + sigma2;
            const float a22 = cnorm2(h01) + cnorm2(h11) + sigma2;
            const CudaComplex32 a12 = cadd(cmul(cconj(h00), h01), cmul(cconj(h10), h11));
            float det = a11 * a22 - cnorm2(a12);
            det = det < det_floor ? det_floor : det;
            const float inv_det = 1.0F / det;

            const float inv11 = a22 * inv_det;
            const float inv22 = a11 * inv_det;
            const CudaComplex32 inv12 = cscale(a12, -inv_det);
            const CudaComplex32 inv21 = cconj(inv12);

            const CudaComplex32 hh00 = cconj(h00);
            const CudaComplex32 hh01 = cconj(h10);
            const CudaComplex32 hh10 = cconj(h01);
            const CudaComplex32 hh11 = cconj(h11);

            const CudaComplex32 w00 = cadd(cscale(hh00, inv11), cmul(inv12, hh10));
            const CudaComplex32 w01 = cadd(cscale(hh01, inv11), cmul(inv12, hh11));
            const CudaComplex32 w10 = cadd(cmul(inv21, hh00), cscale(hh10, inv22));
            const CudaComplex32 w11 = cadd(cmul(inv21, hh01), cscale(hh11, inv22));

            const CudaComplex32 z0 = cadd(cmul(w00, y0), cmul(w01, y1));
            const CudaComplex32 z1 = cadd(cmul(w10, y0), cmul(w11, y1));

            const float g0_re = cadd(cmul(w00, h00), cmul(w01, h10)).re;
            const float g1_re = cadd(cmul(w10, h01), cmul(w11, h11)).re;
            float g0 = g0_re < g_min ? g_min : (g0_re > 1.0F - g_min ? 1.0F - g_min : g0_re);
            float g1 = g1_re < g_min ? g_min : (g1_re > 1.0F - g_min ? 1.0F - g_min : g1_re);

            xhat_re[out_slot] = z0.re / g0;
            xhat_im[out_slot] = z0.im / g0;
            float gamma0 = g0 / (1.0F - g0);
            gamma0 = gamma0 > gamma_max ? gamma_max : gamma0;
            sinr[out_slot] = gamma0;

            const std::uint32_t layer1_base = kCudaMaxDataRe + out_slot;
            xhat_re[layer1_base] = z1.re / g1;
            xhat_im[layer1_base] = z1.im / g1;
            float gamma1 = g1 / (1.0F - g1);
            gamma1 = gamma1 > gamma_max ? gamma_max : gamma1;
            sinr[layer1_base] = gamma1;

            if (scratch != nullptr && grid_meta->trace_sample_count != 0U) {
                for (std::uint32_t sample = 0; sample < grid_meta->trace_sample_count;
                     ++sample) {
                    if (grid_meta->spot_check_re_slots[sample] != re) {
                        continue;
                    }
                    const std::uint32_t base =
                        kCudaScratchHeaderFloatCount + sample * kCudaEqualizeTraceFloatCount;
                    scratch[base + 0U] = static_cast<float>(re);
                    scratch[base + 1U] = static_cast<float>(symbol);
                    scratch[base + 2U] = static_cast<float>(sc);
                    scratch[base + 3U] = a11;
                    scratch[base + 4U] = a22;
                    scratch[base + 5U] = a12.re;
                    scratch[base + 6U] = a12.im;
                    scratch[base + 7U] = det;
                    scratch[base + 8U] = w00.re;
                    scratch[base + 9U] = w00.im;
                    scratch[base + 10U] = w01.re;
                    scratch[base + 11U] = w01.im;
                    scratch[base + 12U] = w10.re;
                    scratch[base + 13U] = w10.im;
                    scratch[base + 14U] = w11.re;
                    scratch[base + 15U] = w11.im;
                    scratch[base + 16U] = z0.re;
                    scratch[base + 17U] = z0.im;
                    scratch[base + 18U] = z1.re;
                    scratch[base + 19U] = z1.im;
                    scratch[base + 20U] = g0;
                    scratch[base + 21U] = g1;
                    scratch[base + 22U] = xhat_re[out_slot];
                    scratch[base + 23U] = xhat_im[out_slot];
                    scratch[base + 24U] = xhat_re[layer1_base];
                    scratch[base + 25U] = xhat_im[layer1_base];
                    scratch[base + 26U] = sinr[out_slot];
                    scratch[base + 27U] = sinr[layer1_base];
                    scratch[base + 28U] = y0_re;
                    scratch[base + 29U] = y0_im;
                    scratch[base + 30U] = y1_re;
                    scratch[base + 31U] = y1_im;
                }
            }
        }

        if (re == 0U && scratch != nullptr) {
            scratch[0] = static_cast<float>(out_slot);
            scratch[1] = static_cast<float>(symbol);
            scratch[2] = static_cast<float>(sc);
            scratch[3] = sigma2;
        }
    }

    const bool is_completion_thread =
        (blockIdx.x == 0 && threadIdx.x == 0) ||
        (grid_meta->n_valid_re == 0U && blockIdx.x == 0 && threadIdx.x == 0);
    if (is_completion_thread && completion != nullptr) {
        *completion = completion_value;
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

MmseStatus cuda_query_device_info(std::uint32_t device_ordinal, CudaDeviceInfo& info) {
    cudaDeviceProp prop{};
    if (const cudaError_t status =
            cudaGetDeviceProperties(&prop, static_cast<int>(device_ordinal));
        status != cudaSuccess) {
        return map_cuda_error(status);
    }

    info = {};
    for (std::size_t i = 0; prop.name[i] != '\0' && i + 1U < info.name.size(); ++i) {
        info.name[i] = prop.name[i];
    }
    info.major = static_cast<std::uint32_t>(prop.major);
    info.minor = static_cast<std::uint32_t>(prop.minor);
    info.multi_processor_count = static_cast<std::uint32_t>(prop.multiProcessorCount);
    info.warp_size = static_cast<std::uint32_t>(prop.warpSize);
    info.max_threads_per_block = static_cast<std::uint32_t>(prop.maxThreadsPerBlock);
    info.max_threads_per_multiprocessor =
        static_cast<std::uint32_t>(prop.maxThreadsPerMultiProcessor);
    info.regs_per_block = static_cast<std::uint32_t>(prop.regsPerBlock);
    info.regs_per_multiprocessor = static_cast<std::uint32_t>(prop.regsPerMultiprocessor);
    info.shared_mem_per_block = static_cast<std::size_t>(prop.sharedMemPerBlock);
    info.shared_mem_per_multiprocessor =
        static_cast<std::size_t>(prop.sharedMemPerMultiprocessor);
    info.total_global_mem = static_cast<std::size_t>(prop.totalGlobalMem);
    info.l2_cache_size = static_cast<std::size_t>(prop.l2CacheSize);
    return MmseStatus::kOk;
}

MmseStatus cuda_query_current_memory_info(std::size_t& free_bytes, std::size_t& total_bytes) {
    return map_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
}

MmseStatus cuda_query_estimate_kernel_resources(CudaKernelResourceInfo& info) {
    return fill_kernel_resource_info(estimate_channel_kernel, "estimate_channel_kernel",
                                     kEstimateThreadsPerBlock, info);
}

MmseStatus cuda_query_equalize_kernel_resources(std::int32_t block_size,
                                                CudaKernelResourceInfo& info) {
    return fill_kernel_resource_info(equalize_stub_kernel, "equalize_stub_kernel", block_size,
                                     info);
}

MmseStatus cuda_create_event(std::uintptr_t& event_handle) {
    cudaEvent_t event{};
    if (const cudaError_t status = cudaEventCreate(&event); status != cudaSuccess) {
        event_handle = 0U;
        return map_cuda_error(status);
    }
    event_handle = reinterpret_cast<std::uintptr_t>(event);
    return MmseStatus::kOk;
}

void cuda_destroy_event(std::uintptr_t event_handle) {
    if (event_handle != 0U) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event_handle));
    }
}

MmseStatus cuda_event_record(std::uintptr_t event_handle, std::uintptr_t stream_handle) {
    return map_cuda_error(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event_handle),
                                          reinterpret_cast<cudaStream_t>(stream_handle)));
}

MmseStatus cuda_event_elapsed_us(std::uintptr_t start_event_handle, std::uintptr_t end_event_handle,
                                 double& elapsed_us) {
    float elapsed_ms = 0.0F;
    if (const cudaError_t status =
            cudaEventElapsedTime(&elapsed_ms, reinterpret_cast<cudaEvent_t>(start_event_handle),
                                 reinterpret_cast<cudaEvent_t>(end_event_handle));
        status != cudaSuccess) {
        elapsed_us = 0.0;
        return map_cuda_error(status);
    }
    elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    return MmseStatus::kOk;
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
                                    const std::array<float*, 2>& re,
                                    const std::array<float*, 2>& im,
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

MmseStatus cuda_copy_grid_meta_h2d_async(const CudaDeviceBuffers& buffers,
                                         const CudaGridMeta& grid_meta,
                                         std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    return map_cuda_error(cudaMemcpyAsync(buffers.grid_meta, &grid_meta, sizeof(grid_meta),
                                          cudaMemcpyHostToDevice, stream));
}

MmseStatus cuda_copy_sigma2_h2d_async(const CudaDeviceBuffers& buffers,
                                      float sigma2,
                                      std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    return map_cuda_error(
        cudaMemcpyAsync(buffers.sigma2, &sigma2, sizeof(sigma2), cudaMemcpyHostToDevice, stream));
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
    if (buffers.h_estimate == nullptr || buffers.scratch == nullptr) {
        return MmseStatus::kInternalError;
    }
    float zero_sigma2 = 0.0F;
    std::uint32_t zero_count = 0U;
    float* sigma2_accum = reinterpret_cast<float*>(buffers.scratch);
    std::uint32_t* residual_count =
        reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::byte*>(buffers.scratch) + sizeof(float));
    if (const cudaError_t status =
            cudaMemcpyAsync(sigma2_accum, &zero_sigma2, sizeof(zero_sigma2), cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(residual_count, &zero_count, sizeof(zero_count), cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }

    estimate_residual_kernel<<<kEstimateResidualBlocks, kEstimateThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float*>(buffers.grid_re[0]),
        reinterpret_cast<const float*>(buffers.grid_re[1]),
        reinterpret_cast<const float*>(buffers.grid_im[0]),
        reinterpret_cast<const float*>(buffers.grid_im[1]),
        reinterpret_cast<const CudaGridMeta*>(buffers.grid_meta),
        sigma2_accum,
        residual_count);
    if (const cudaError_t residual_status = cudaGetLastError(); residual_status != cudaSuccess) {
        return map_cuda_error(residual_status);
    }

    const std::uint32_t total_h = kLteNumTxPortsV1 * kLteNumRxAntV1 * kLteNumSymbolsNormalCp *
                                  kLteNumSubcarriers20MHz;
    const std::uint32_t estimate_blocks =
        max(1U, ceil_div_u32(total_h, kEstimateThreadsPerBlock));
    estimate_channel_kernel<<<estimate_blocks, kEstimateThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float*>(buffers.grid_re[0]),
        reinterpret_cast<const float*>(buffers.grid_re[1]),
        reinterpret_cast<const float*>(buffers.grid_im[0]),
        reinterpret_cast<const float*>(buffers.grid_im[1]),
        reinterpret_cast<const CudaGridMeta*>(buffers.grid_meta),
        reinterpret_cast<float*>(buffers.h_estimate));
    if (const cudaError_t estimate_status = cudaGetLastError(); estimate_status != cudaSuccess) {
        return map_cuda_error(estimate_status);
    }

    finalize_sigma2_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const CudaGridMeta*>(buffers.grid_meta),
        sigma2_accum,
        residual_count,
        reinterpret_cast<float*>(buffers.sigma2));
    return map_cuda_error(cudaGetLastError());
}

MmseStatus cuda_launch_equalize_stub(const CudaDeviceBuffers& buffers,
                                     std::uint32_t n_valid_re,
                                     std::uint32_t completion_value,
                                     std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    constexpr std::uint32_t kThreadsPerBlock = 256U;
    const std::uint32_t blocks = (n_valid_re == 0U) ? 1U : (n_valid_re + kThreadsPerBlock - 1U) /
                                                       kThreadsPerBlock;
    equalize_stub_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float*>(buffers.grid_re[0]),
        reinterpret_cast<const float*>(buffers.grid_re[1]),
        reinterpret_cast<const float*>(buffers.grid_im[0]),
        reinterpret_cast<const float*>(buffers.grid_im[1]),
        reinterpret_cast<const float*>(buffers.h_estimate),
        reinterpret_cast<const float*>(buffers.sigma2),
        reinterpret_cast<const CudaGridMeta*>(buffers.grid_meta),
        reinterpret_cast<float*>(buffers.xhat_re),
        reinterpret_cast<float*>(buffers.xhat_im),
        reinterpret_cast<float*>(buffers.sinr),
        reinterpret_cast<float*>(buffers.scratch),
        reinterpret_cast<std::uint32_t*>(buffers.completion),
        completion_value);
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        return MmseStatus::kInternalError;
    }
    return MmseStatus::kOk;
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
