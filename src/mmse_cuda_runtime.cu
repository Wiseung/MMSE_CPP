#include "internal/mmse_cuda_runtime.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>

namespace mmse::detail {

namespace {

constexpr std::uint32_t kEstimateThreadsPerBlock = 256U;
constexpr std::uint32_t kEstimateResidualBlocks = 64U;
constexpr double kCudaNegativeInfinity = -1.7976931348623157e308;

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
    if (!isfinite(value)) {
        return lo;
    }
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
    y_re = isfinite(y_re) ? y_re : 0.0F;
    y_im = isfinite(y_im) ? y_im : 0.0F;
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
    const std::uint32_t last_sc =
        6U * (kLteNumPilotTonesPerCrsSymbol - 1U) + first_offset;
    const bool below_first = sc < first_offset;
    const bool above_last = sc > last_sc;
    const std::uint32_t lower_pilot =
        min_u32((sc < first_offset ? 0U : (sc - first_offset) / 6U),
                kLteNumPilotTonesPerCrsSymbol - 1U);
    const std::uint32_t left_pilot =
        below_first ? 0U : (above_last ? kLteNumPilotTonesPerCrsSymbol - 2U : lower_pilot);
    const std::uint32_t right_pilot =
        below_first ? 1U : (above_last ? kLteNumPilotTonesPerCrsSymbol - 1U
                                       : min_u32(lower_pilot + 1U,
                                                 kLteNumPilotTonesPerCrsSymbol - 1U));
    const std::uint32_t left_sc = 6U * left_pilot + first_offset;
    const std::uint32_t right_sc = 6U * right_pilot + first_offset;
    const CudaComplex32 left = ls_at(re_plane, im_plane, grid_meta, tx, rx, cs, left_pilot);
    const CudaComplex32 right = ls_at(re_plane, im_plane, grid_meta, tx, rx, cs, right_pilot);
    if (left_sc == right_sc) {
        return left;
    }
    const float t = static_cast<float>(static_cast<int>(sc) - static_cast<int>(left_sc)) /
                    static_cast<float>(right_sc - left_sc);
    return {left.re + (right.re - left.re) * t, left.im + (right.im - left.im) * t};
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
        lower = kLteNumCrsSymbols - 2U;
        upper = kLteNumCrsSymbols - 1U;
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

    const std::uint32_t tx_count = min_u32(grid_meta->n_tx_ports, kMmseV1MaxNumCrsTxPorts);
    const std::uint32_t total_items =
        tx_count * min_u32(grid_meta->n_rx_ant, kMmseV1MaxNumRxAntennas) * kLteNumCrsSymbols *
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
        const std::uint32_t rx = index % min_u32(grid_meta->n_rx_ant, kMmseV1MaxNumRxAntennas);
        index /= min_u32(grid_meta->n_rx_ant, kMmseV1MaxNumRxAntennas);
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
        local_sigma2 += (2.0F / 3.0F) * (diff_re * diff_re + diff_im * diff_im);
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

    const std::uint32_t tx_count = min_u32(grid_meta->n_tx_ports, kMmseV1MaxNumCrsTxPorts);
    const std::uint32_t rx_count = min_u32(grid_meta->n_rx_ant, kMmseV1MaxNumRxAntennas);
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
            2U * (((tx * kMmseV1MaxNumRxAntennas + rx) * kLteNumSymbolsNormalCp + symbol) *
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
    float sigma2 = residual_count == 0U ? grid_meta->sigma2
                                         : *sigma2_accum_in / static_cast<float>(residual_count);
    if (!isfinite(sigma2) || sigma2 < 0.0F) {
        sigma2 = grid_meta->sigma2;
    }
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

__global__ void zero_estimate_accumulators_kernel(float* sigma2_accum_out,
                                                  std::uint32_t* residual_count_out) {
    if (threadIdx.x == 0U && blockIdx.x == 0U) {
        if (sigma2_accum_out != nullptr) {
            *sigma2_accum_out = 0.0F;
        }
        if (residual_count_out != nullptr) {
            *residual_count_out = 0U;
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
        const bool td_mode = grid_meta->td_pair_count != 0U;
        const std::uint32_t grid_idx =
            td_mode && re < grid_meta->td_pair_count ? grid_meta->td_pair_grid_indices0[re]
                                                     : grid_meta->grid_indices[re];
        const std::uint32_t out_slot =
            grid_meta->channel_type == static_cast<std::uint32_t>(MmseChannelType::kPdcch) &&
                    grid_meta->n_tx_ports == 1U && grid_meta->td_pair_count == 0U
                ? re
                : grid_meta->output_slot_by_grid_re[grid_idx];
        const std::uint32_t symbol = grid_idx / grid_meta->n_subcarriers;
        const std::uint32_t sc = grid_idx % grid_meta->n_subcarriers;
        const bool has_rx1 = grid_meta->n_rx_ant > 1U;

        const float y0_re = grid_re0[grid_idx];
        const float y0_im = grid_im0[grid_idx];
        const float y1_re = has_rx1 ? grid_re1[grid_idx] : 0.0F;
        const float y1_im = has_rx1 ? grid_im1[grid_idx] : 0.0F;

        const std::uint32_t h00_base =
            2U * (((0U * kMmseV1MaxNumRxAntennas + 0U) * kLteNumSymbolsNormalCp + symbol) *
                      kLteNumSubcarriers20MHz +
                  sc);
        const std::uint32_t h01_base =
            2U * (((1U * kMmseV1MaxNumRxAntennas + 0U) * kLteNumSymbolsNormalCp + symbol) *
                      kLteNumSubcarriers20MHz +
                  sc);
        const std::uint32_t h10_base =
            2U * (((0U * kMmseV1MaxNumRxAntennas + 1U) * kLteNumSymbolsNormalCp + symbol) *
                      kLteNumSubcarriers20MHz +
                  sc);
        const std::uint32_t h11_base =
            2U * (((1U * kMmseV1MaxNumRxAntennas + 1U) * kLteNumSymbolsNormalCp + symbol) *
                      kLteNumSubcarriers20MHz +
                  sc);

        const float h00_re = h_estimate[h00_base];
        const float h00_im = h_estimate[h00_base + 1U];
        const float h10_re = has_rx1 ? h_estimate[h10_base] : 0.0F;
        const float h10_im = has_rx1 ? h_estimate[h10_base + 1U] : 0.0F;

        if (td_mode && re < grid_meta->td_pair_count) {
            const std::uint32_t next_grid_idx = grid_meta->td_pair_grid_indices1[re];
            const std::uint32_t next_symbol = next_grid_idx / grid_meta->n_subcarriers;
            const std::uint32_t next_sc = next_grid_idx % grid_meta->n_subcarriers;
            const float y0n_re = grid_re0[next_grid_idx];
            const float y0n_im = grid_im0[next_grid_idx];
            const float y1n_re = has_rx1 ? grid_re1[next_grid_idx] : 0.0F;
            const float y1n_im = has_rx1 ? grid_im1[next_grid_idx] : 0.0F;

            const std::uint32_t h00n_base =
                2U * (((0U * kMmseV1MaxNumRxAntennas + 0U) * kLteNumSymbolsNormalCp + next_symbol) *
                          kLteNumSubcarriers20MHz +
                      next_sc);
            const std::uint32_t h01n_base =
                2U * (((1U * kMmseV1MaxNumRxAntennas + 0U) * kLteNumSymbolsNormalCp + next_symbol) *
                          kLteNumSubcarriers20MHz +
                      next_sc);
            const std::uint32_t h10n_base =
                2U * (((0U * kMmseV1MaxNumRxAntennas + 1U) * kLteNumSymbolsNormalCp + next_symbol) *
                          kLteNumSubcarriers20MHz +
                      next_sc);
            const std::uint32_t h11n_base =
                2U * (((1U * kMmseV1MaxNumRxAntennas + 1U) * kLteNumSymbolsNormalCp + next_symbol) *
                          kLteNumSubcarriers20MHz +
                      next_sc);

            const CudaComplex32 h00{h00_re, h00_im};
            const CudaComplex32 h01{h_estimate[h01_base], h_estimate[h01_base + 1U]};
            const CudaComplex32 h10{h10_re, h10_im};
            const CudaComplex32 h11{
                has_rx1 ? h_estimate[h11_base] : 0.0F,
                has_rx1 ? h_estimate[h11_base + 1U] : 0.0F};
            const CudaComplex32 h00n{h_estimate[h00n_base], h_estimate[h00n_base + 1U]};
            const CudaComplex32 h01n{h_estimate[h01n_base], h_estimate[h01n_base + 1U]};
            const CudaComplex32 h10n{
                has_rx1 ? h_estimate[h10n_base] : 0.0F,
                has_rx1 ? h_estimate[h10n_base + 1U] : 0.0F};
            const CudaComplex32 h11n{
                has_rx1 ? h_estimate[h11n_base] : 0.0F,
                has_rx1 ? h_estimate[h11n_base + 1U] : 0.0F};
            const CudaComplex32 y0{y0_re, y0_im};
            const CudaComplex32 y1{y1_re, y1_im};
            const CudaComplex32 y0n{y0n_re, y0n_im};
            const CudaComplex32 y1n{y1n_re, y1n_im};

            const CudaComplex32 a00[] = {h00, h10, cscale(cconj(h01n), -1.0F),
                                         cscale(cconj(h11n), -1.0F)};
            const CudaComplex32 a01[] = {h01, h11, cconj(h00n), cconj(h10n)};
            const CudaComplex32 observations[] = {y0, y1, cconj(y0n), cconj(y1n)};
            float gram00 = sigma2 > 0.0F ? sigma2 : 0.0F;
            float gram11 = gram00;
            CudaComplex32 gram01{};
            CudaComplex32 z0{};
            CudaComplex32 z1{};
            for (std::uint32_t row = 0U; row < 4U; ++row) {
                gram00 += cnorm2(a00[row]);
                gram11 += cnorm2(a01[row]);
                gram01 = cadd(gram01, cmul(cconj(a00[row]), a01[row]));
                z0 = cadd(z0, cmul(cconj(a00[row]), observations[row]));
                z1 = cadd(z1, cmul(cconj(a01[row]), observations[row]));
            }
            float det = gram00 * gram11 - cnorm2(gram01);
            det = det < det_floor ? det_floor : det;
            const float inv_det = 1.0F / det;
            const float inv00 = gram11 * inv_det;
            const float inv11 = gram00 * inv_det;
            const CudaComplex32 inv01 = cscale(gram01, -inv_det);
            const CudaComplex32 inv10 = cconj(inv01);
            const CudaComplex32 x0 = cadd(cscale(z0, inv00), cmul(inv01, z1));
            const CudaComplex32 x1 = cadd(cmul(inv10, z0), cscale(z1, inv11));
            const float regularization = sigma2 > 0.0F ? sigma2 : 0.0F;
            float g0 = 1.0F - regularization * inv00;
            float g1 = 1.0F - regularization * inv11;
            g0 = clampf(g0, g_min, 1.0F - g_min);
            g1 = clampf(g1, g_min, 1.0F - g_min);

            const std::uint32_t out0 = re * 2U;
            const std::uint32_t out1 = out0 + 1U;
            xhat_re[out0] = x0.re / g0;
            xhat_im[out0] = x0.im / g0;
            xhat_re[out1] = x1.re / g1;
            xhat_im[out1] = x1.im / g1;
            float gamma0 = g0 / (1.0F - g0);
            float gamma1 = g1 / (1.0F - g1);
            sinr[out0] = gamma0 > gamma_max ? gamma_max : gamma0;
            sinr[out1] = gamma1 > gamma_max ? gamma_max : gamma1;
        } else if (td_mode) {
            // One CUDA thread processes a complete explicit TD pair.
        } else if (grid_meta->n_layers == 1U) {
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
            const CudaComplex32 h11{
                has_rx1 ? h_estimate[h11_base] : 0.0F,
                has_rx1 ? h_estimate[h11_base + 1U] : 0.0F};
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

__device__ inline std::uint32_t pdcch_parity(std::uint32_t value) {
    std::uint32_t result = 0U;
    while (value != 0U) {
        result ^= value & 1U;
        value >>= 1U;
    }
    return result;
}

__device__ inline std::uint8_t pdcch_branch_output_class(std::uint32_t shift_register) {
    constexpr std::uint32_t kGenerator0 = 0133U;
    constexpr std::uint32_t kGenerator1 = 0171U;
    constexpr std::uint32_t kGenerator2 = 0165U;
    return static_cast<std::uint8_t>(pdcch_parity(shift_register & kGenerator0) |
                                     (pdcch_parity(shift_register & kGenerator1) << 1U) |
                                     (pdcch_parity(shift_register & kGenerator2) << 2U));
}

__device__ inline double pdcch_branch_metric(const float* llrs, std::uint8_t output_class) {
    const double sign0 = (output_class & 0x1U) != 0U ? 1.0 : -1.0;
    const double sign1 = (output_class & 0x2U) != 0U ? 1.0 : -1.0;
    const double sign2 = (output_class & 0x4U) != 0U ? 1.0 : -1.0;
    return sign0 * static_cast<double>(llrs[0]) + sign1 * static_cast<double>(llrs[1]) +
           sign2 * static_cast<double>(llrs[2]);
}

__global__ void pdcch_llr_descramble_kernel(const float* xhat_re,
                                             const float* xhat_im,
                                             const float* sinr,
                                             std::uint32_t n_re,
                                             const std::uint32_t* gold_words,
                                             float* llrs) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t llr_count = n_re * 2U;
    if (index >= llr_count || xhat_re == nullptr || xhat_im == nullptr || sinr == nullptr ||
        gold_words == nullptr || llrs == nullptr) {
        return;
    }

    const std::uint32_t re = index >> 1U;
    const float gamma = sinr[re] > 0.0F ? sinr[re] : 0.0F;
    constexpr float kScale = -2.0F / 0.70710678118654752440F;
    float value = kScale * gamma * ((index & 1U) == 0U ? xhat_re[re] : xhat_im[re]);
    if (((gold_words[index >> 5U] >> (index & 31U)) & 1U) != 0U) {
        value = __uint_as_float(__float_as_uint(value) ^ 0x80000000U);
    }
    llrs[index] = value;
}

__global__ void pdcch_rate_recovery_kernel(const float* llrs,
                                            const CudaPdcchCandidateDescriptor* candidates,
                                            std::uint32_t candidate_count,
                                            float* recovered_llrs) {
    const std::uint32_t candidate_index = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t output_index = blockIdx.y * 32U + lane;
    if (candidate_index >= candidate_count || output_index >= kCudaPdcchRecoveredLlrCount ||
        lane >= 32U || llrs == nullptr ||
        candidates == nullptr || recovered_llrs == nullptr) {
        return;
    }
    constexpr std::uint32_t kColumns = 32U;
    constexpr std::uint8_t kPermutation[kColumns] = {
        1U, 17U, 9U, 25U, 5U, 21U, 13U, 29U, 3U, 19U, 11U, 27U, 7U, 23U, 15U, 31U,
        0U, 16U, 8U, 24U, 4U, 20U, 12U, 28U, 2U, 18U, 10U, 26U, 6U, 22U, 14U, 30U,
    };
    constexpr std::uint32_t kRows = 2U;
    constexpr std::uint32_t kInterleaverSize = kRows * kColumns;
    constexpr std::uint32_t kDummyBits = kInterleaverSize - kCudaPdcchCodewordBitCount;
    constexpr std::uint32_t kCollectionSize = 3U * kInterleaverSize;
    const CudaPdcchCandidateDescriptor* const candidate = candidates + candidate_index;
    const std::uint8_t output_slot = candidate->rate_recovery_collection_slots[output_index];
    float accumulated = 0.0F;

    const std::uint32_t llr_offset = static_cast<std::uint32_t>(candidate->first_cce) * 72U;
    std::uint32_t input_index = 0U;
    std::uint32_t collection_index = 0U;
    while (input_index < candidate->encoded_bit_count) {
        const std::uint32_t intra_stream_index = collection_index % kInterleaverSize;
        const std::uint32_t permutation_column = intra_stream_index / kRows;
        const std::uint32_t row = intra_stream_index % kRows;
        const std::uint32_t original_bit_index = row * kColumns + kPermutation[permutation_column];
        if (original_bit_index >= kDummyBits) {
            const float value = llrs[llr_offset + input_index];
            if (collection_index == output_slot) {
                accumulated += value;
            }
            ++input_index;
        }
        collection_index = (collection_index + 1U) % kCollectionSize;
    }

    float* const output = recovered_llrs + candidate_index * kCudaPdcchRecoveredLlrCount;
    output[output_index] = accumulated;
}

__global__ void pdcch_viterbi_kernel(const float* recovered_llrs,
                                      std::uint32_t candidate_count,
                                      std::uint64_t* survivors,
                                      double* terminal_metrics) {
    const std::uint32_t candidate_index = blockIdx.x;
    const std::uint32_t warp_index = threadIdx.x >> 5U;
    const std::uint32_t initial_state = blockIdx.y * 2U + warp_index;
    const std::uint32_t lane = threadIdx.x & 31U;
    if (candidate_index >= candidate_count || initial_state >= 64U || lane >= 32U ||
        recovered_llrs == nullptr || survivors == nullptr || terminal_metrics == nullptr) {
        return;
    }

    __shared__ double metrics[2][64];
    __shared__ double next_metrics[2][64];
    const std::uint32_t target_state = lane * 2U;
    metrics[warp_index][target_state] =
        target_state == initial_state ? 0.0 : kCudaNegativeInfinity;
    metrics[warp_index][target_state + 1U] =
        target_state + 1U == initial_state ? 0.0 : kCudaNegativeInfinity;
    __syncwarp();

    std::uint64_t* const candidate_survivors =
        survivors + (candidate_index * 64U + initial_state) * kCudaPdcchCodewordBitCount;
    const float* const candidate_llrs =
        recovered_llrs + candidate_index * kCudaPdcchRecoveredLlrCount;
    for (std::uint32_t bit = 0U; bit < kCudaPdcchCodewordBitCount; ++bit) {
        const std::uint32_t low_predecessor = lane;
        const std::uint32_t high_predecessor = low_predecessor | 32U;
        const std::uint8_t low_zero_class = pdcch_branch_output_class(low_predecessor << 1U);
        const std::uint8_t complement_class = low_zero_class ^ 0x7U;
        const float* const bit_llrs = candidate_llrs + bit * 3U;
        const double direct_branch_metric = pdcch_branch_metric(bit_llrs, low_zero_class);
        const double complement_branch_metric = pdcch_branch_metric(bit_llrs, complement_class);
        const double low_metric = metrics[warp_index][low_predecessor];
        const double high_metric = metrics[warp_index][high_predecessor];
        const double low_zero_metric = low_metric + direct_branch_metric;
        const double high_zero_metric = high_metric + complement_branch_metric;
        const bool choose_high_zero = high_zero_metric > low_zero_metric;
        next_metrics[warp_index][target_state] =
            choose_high_zero ? high_zero_metric : low_zero_metric;
        const double low_one_metric = low_metric + complement_branch_metric;
        const double high_one_metric = high_metric + direct_branch_metric;
        const bool choose_high_one = high_one_metric > low_one_metric;
        next_metrics[warp_index][target_state + 1U] =
            choose_high_one ? high_one_metric : low_one_metric;

        const unsigned int zero_mask = __ballot_sync(0xFFFFFFFFU, choose_high_zero);
        const unsigned int one_mask = __ballot_sync(0xFFFFFFFFU, choose_high_one);
        if (lane == 0U) {
            std::uint64_t word = 0U;
            for (std::uint32_t state_pair = 0U; state_pair < 32U; ++state_pair) {
                if ((zero_mask & (1U << state_pair)) != 0U) {
                    word |= std::uint64_t{1U} << (state_pair * 2U);
                }
                if ((one_mask & (1U << state_pair)) != 0U) {
                    word |= std::uint64_t{1U} << (state_pair * 2U + 1U);
                }
            }
            candidate_survivors[bit] = word;
        }
        __syncwarp();
        metrics[warp_index][target_state] = next_metrics[warp_index][target_state];
        metrics[warp_index][target_state + 1U] = next_metrics[warp_index][target_state + 1U];
        __syncwarp();
    }
    if (lane == 0U) {
        terminal_metrics[candidate_index * 64U + initial_state] = metrics[warp_index][initial_state];
    }
}

__global__ void pdcch_crc_kernel(const CudaPdcchCandidateDescriptor* candidates,
                                  std::uint32_t candidate_count,
                                  const std::uint64_t* survivors,
                                  const double* terminal_metrics,
                                  std::uint16_t expected_rnti,
                                  CudaPdcchCandidateResult* candidate_results) {
    const std::uint32_t candidate_index = blockIdx.x;
    if (candidate_index >= candidate_count || threadIdx.x != 0U || candidates == nullptr ||
        survivors == nullptr || terminal_metrics == nullptr || candidate_results == nullptr) {
        return;
    }
    double best_metric = kCudaNegativeInfinity;
    std::uint64_t decoded_bits = 0U;
    for (std::uint32_t initial_state = 0U; initial_state < 64U; ++initial_state) {
        const double metric = terminal_metrics[candidate_index * 64U + initial_state];
        if (metric <= best_metric) {
            continue;
        }
        std::uint32_t state = initial_state;
        std::uint64_t candidate_bits = 0U;
        const std::uint64_t* const candidate_survivors =
            survivors + (candidate_index * 64U + initial_state) * kCudaPdcchCodewordBitCount;
        for (std::uint32_t bit = kCudaPdcchCodewordBitCount; bit > 0U; --bit) {
            candidate_bits |= static_cast<std::uint64_t>(state & 1U) << (bit - 1U);
            const bool high_predecessor =
                (candidate_survivors[bit - 1U] & (std::uint64_t{1U} << state)) != 0U;
            state = (state >> 1U) | (high_predecessor ? 32U : 0U);
        }
        if (state == initial_state) {
            best_metric = metric;
            decoded_bits = candidate_bits;
        }
    }

    std::uint16_t transmitted_crc = 0U;
    std::uint16_t calculated_crc = 0U;
    for (std::uint32_t bit = 0U; bit < 28U; ++bit) {
        const std::uint16_t value = static_cast<std::uint16_t>((decoded_bits >> bit) & 1U);
        const std::uint16_t feedback =
            static_cast<std::uint16_t>(((calculated_crc >> 15U) ^ value) & 1U);
        calculated_crc = static_cast<std::uint16_t>(calculated_crc << 1U);
        if (feedback != 0U) {
            calculated_crc = static_cast<std::uint16_t>(calculated_crc ^ 0x1021U);
        }
    }
    for (std::uint32_t bit = 28U; bit < kCudaPdcchCodewordBitCount; ++bit) {
        transmitted_crc = static_cast<std::uint16_t>(
            (transmitted_crc << 1U) | static_cast<std::uint16_t>((decoded_bits >> bit) & 1U));
    }
    const std::uint16_t unmasked_rnti =
        static_cast<std::uint16_t>(transmitted_crc ^ calculated_crc);
    const CudaPdcchCandidateDescriptor candidate = candidates[candidate_index];
    candidate_results[candidate_index] = {
        .candidate_id = candidate.candidate_id,
        .first_cce = candidate.first_cce,
        .aggregation_level = candidate.aggregation_level,
        .matched = static_cast<std::uint8_t>(unmasked_rnti == expected_rnti),
        .transmitted_crc = transmitted_crc,
        .calculated_crc = calculated_crc,
        .unmasked_rnti = unmasked_rnti,
        .decoded_bits = decoded_bits,
    };
}

__global__ void pdcch_compact_hits_kernel(const CudaPdcchCandidateResult* candidate_results,
                                          std::uint32_t candidate_count,
                                          CudaPdcchCandidateResult* compact_results,
                                          std::uint32_t* hit_count) {
    if (blockIdx.x != 0U || threadIdx.x != 0U || candidate_results == nullptr ||
        compact_results == nullptr || hit_count == nullptr) {
        return;
    }
    std::uint32_t count = 0U;
    for (std::uint32_t candidate = 0U; candidate < candidate_count; ++candidate) {
        const CudaPdcchCandidateResult result = candidate_results[candidate];
        if (result.matched != 0U) {
            compact_results[count++] = result;
        }
    }
    *hit_count = count;
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

MmseStatus cuda_alloc_host_bytes(void*& ptr, std::size_t bytes, bool request_pinned,
                                 bool& pinned_allocation) {
    ptr = nullptr;
    pinned_allocation = false;
    if (!request_pinned) {
        ptr = ::operator new(bytes, std::nothrow);
        if (ptr == nullptr) {
            return MmseStatus::kInternalError;
        }
        return MmseStatus::kOk;
    }
    const cudaError_t status = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }
    pinned_allocation = true;
    return MmseStatus::kOk;
}

void cuda_free_host_bytes(void* ptr, bool pinned_allocation) {
    if (ptr == nullptr) {
        return;
    }
    if (pinned_allocation) {
        cudaFreeHost(ptr);
        return;
    }
    ::operator delete(ptr);
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
    const std::size_t rx_count = min_u32(grid_meta.n_rx_ant, static_cast<std::uint32_t>(re.size()));
    for (std::size_t rx = 0; rx < rx_count; ++rx) {
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
    const bool compact_pdcch_meta =
        grid_meta.channel_type == static_cast<std::uint32_t>(MmseChannelType::kPdcch) &&
        grid_meta.n_tx_ports == 1U && grid_meta.td_pair_count == 0U;
    if (!compact_pdcch_meta) {
        return map_cuda_error(cudaMemcpyAsync(buffers.grid_meta, &grid_meta, sizeof(grid_meta),
                                              cudaMemcpyHostToDevice, stream));
    }

    const std::size_t valid_re = min_u32(grid_meta.n_valid_re, kCudaMaxDataRe);
    if (const cudaError_t status = cudaMemcpyAsync(
            buffers.grid_meta, &grid_meta,
            offsetof(CudaGridMeta, grid_indices) + valid_re * sizeof(grid_meta.grid_indices[0]),
            cudaMemcpyHostToDevice, stream);
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

MmseStatus cuda_copy_grid_meta_layout_slice_h2d_async(const CudaDeviceBuffers& buffers,
                                                      const CudaGridMeta& grid_meta,
                                                      std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const std::byte* base = reinterpret_cast<const std::byte*>(&grid_meta);
    const std::size_t offset = offsetof(CudaGridMeta, n_valid_re);
    const std::size_t bytes = offsetof(CudaGridMeta, prb_bitmap) + sizeof(grid_meta.prb_bitmap) - offset;
    return map_cuda_error(cudaMemcpyAsync(reinterpret_cast<std::byte*>(buffers.grid_meta) + offset,
                                          base + offset,
                                          bytes,
                                          cudaMemcpyHostToDevice,
                                          stream));
}

MmseStatus cuda_copy_grid_meta_dynamic_h2d_async(const CudaDeviceBuffers& buffers,
                                                 const CudaGridMeta& grid_meta,
                                                 std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const bool compact_pdcch_meta =
        grid_meta.channel_type == static_cast<std::uint32_t>(MmseChannelType::kPdcch) &&
        grid_meta.n_tx_ports == 1U && grid_meta.td_pair_count == 0U;
    if (compact_pdcch_meta) {
        const std::size_t valid_re = min_u32(grid_meta.n_valid_re, kCudaMaxDataRe);
        return map_cuda_error(cudaMemcpyAsync(
            buffers.grid_meta, &grid_meta,
            offsetof(CudaGridMeta, grid_indices) + valid_re * sizeof(grid_meta.grid_indices[0]),
            cudaMemcpyHostToDevice, stream));
    }
    const std::byte* base = reinterpret_cast<const std::byte*>(&grid_meta);
    const std::size_t header_offset = offsetof(CudaGridMeta, n_valid_re);
    const std::size_t header_bytes =
        offsetof(CudaGridMeta, grid_indices) - header_offset;
    if (const cudaError_t status =
            cudaMemcpyAsync(reinterpret_cast<std::byte*>(buffers.grid_meta) + header_offset,
                            base + header_offset,
                            header_bytes,
                            cudaMemcpyHostToDevice,
                            stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(reinterpret_cast<std::byte*>(buffers.grid_meta) + offsetof(CudaGridMeta, grid_indices),
                            grid_meta.grid_indices,
                            sizeof(grid_meta.grid_indices),
                            cudaMemcpyHostToDevice,
                            stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status = cudaMemcpyAsync(
            reinterpret_cast<std::byte*>(buffers.grid_meta) +
                offsetof(CudaGridMeta, td_pair_grid_indices0),
            grid_meta.td_pair_grid_indices0, sizeof(grid_meta.td_pair_grid_indices0),
            cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status = cudaMemcpyAsync(
            reinterpret_cast<std::byte*>(buffers.grid_meta) +
                offsetof(CudaGridMeta, td_pair_grid_indices1),
            grid_meta.td_pair_grid_indices1, sizeof(grid_meta.td_pair_grid_indices1),
            cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    if (const cudaError_t status =
            cudaMemcpyAsync(reinterpret_cast<std::byte*>(buffers.grid_meta) +
                                offsetof(CudaGridMeta, output_slot_by_grid_re),
                            grid_meta.output_slot_by_grid_re,
                            sizeof(grid_meta.output_slot_by_grid_re),
                            cudaMemcpyHostToDevice,
                            stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    return MmseStatus::kOk;
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
                                     std::uintptr_t stream_handle,
                                     std::uintptr_t residual_done_event_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (buffers.h_estimate == nullptr || buffers.scratch == nullptr) {
        return MmseStatus::kInternalError;
    }
    float* sigma2_accum = reinterpret_cast<float*>(buffers.scratch);
    std::uint32_t* residual_count =
        reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::byte*>(buffers.scratch) + sizeof(float));
    zero_estimate_accumulators_kernel<<<1, 1, 0, stream>>>(sigma2_accum, residual_count);
    if (const cudaError_t zero_status = cudaGetLastError(); zero_status != cudaSuccess) {
        return map_cuda_error(zero_status);
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
    if (residual_done_event_handle != 0U) {
        if (const cudaError_t status =
                cudaEventRecord(reinterpret_cast<cudaEvent_t>(residual_done_event_handle), stream);
            status != cudaSuccess) {
            return map_cuda_error(status);
        }
    }

    const std::uint32_t total_h = kMmseV1MaxNumCrsTxPorts * kMmseV1MaxNumRxAntennas * kLteNumSymbolsNormalCp *
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

MmseStatus cuda_copy_pdcch_candidates_h2d_async(
    const CudaDeviceBuffers& buffers, const CudaPdcchCandidateDescriptor* candidates,
    std::uint32_t candidate_count, std::uintptr_t stream_handle) {
    if (buffers.pdcch_candidates == nullptr || candidates == nullptr ||
        candidate_count == 0U || candidate_count > kCudaPdcchMaxCandidates) {
        return MmseStatus::kInvalidArgument;
    }
    return map_cuda_error(cudaMemcpyAsync(
        buffers.pdcch_candidates, candidates,
        static_cast<std::size_t>(candidate_count) * sizeof(CudaPdcchCandidateDescriptor),
        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_handle)));
}

MmseStatus cuda_copy_pdcch_gold_words_h2d_async(const CudaDeviceBuffers& buffers,
                                                 const std::uint32_t* gold_words,
                                                 std::uint32_t word_count,
                                                 std::uintptr_t stream_handle) {
    if (buffers.pdcch_gold_words == nullptr || gold_words == nullptr || word_count == 0U ||
        word_count > kCudaPdcchGoldWordCount) {
        return MmseStatus::kInvalidArgument;
    }
    return map_cuda_error(cudaMemcpyAsync(
        buffers.pdcch_gold_words, gold_words, static_cast<std::size_t>(word_count) * sizeof(*gold_words),
        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_handle)));
}

MmseStatus cuda_launch_pdcch_llr_descramble(const CudaDeviceBuffers& buffers, std::uint32_t n_re,
                                            std::uintptr_t stream_handle) {
    if (buffers.xhat_re == nullptr || buffers.xhat_im == nullptr || buffers.sinr == nullptr ||
        buffers.pdcch_gold_words == nullptr || buffers.pdcch_llrs == nullptr || n_re == 0U ||
        n_re > kCudaMaxDataRe) {
        return MmseStatus::kInvalidArgument;
    }
    constexpr std::uint32_t kThreads = 256U;
    const std::uint32_t blocks = ceil_div_u32(n_re * 2U, kThreads);
    pdcch_llr_descramble_kernel<<<blocks, kThreads, 0,
                                  reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        reinterpret_cast<const float*>(buffers.xhat_re), reinterpret_cast<const float*>(buffers.xhat_im),
        reinterpret_cast<const float*>(buffers.sinr), n_re,
        reinterpret_cast<const std::uint32_t*>(buffers.pdcch_gold_words),
        reinterpret_cast<float*>(buffers.pdcch_llrs));
    return map_cuda_error(cudaGetLastError());
}

MmseStatus cuda_launch_pdcch_rate_recovery(const CudaDeviceBuffers& buffers,
                                           std::uint32_t candidate_count,
                                           std::uintptr_t stream_handle) {
    if (buffers.pdcch_llrs == nullptr || buffers.pdcch_candidates == nullptr ||
        buffers.pdcch_recovered_llrs == nullptr || candidate_count == 0U ||
        candidate_count > kCudaPdcchMaxCandidates) {
        return MmseStatus::kInvalidArgument;
    }
    constexpr std::uint32_t kRateRecoveryWarps =
        (kCudaPdcchRecoveredLlrCount + 31U) / 32U;
    pdcch_rate_recovery_kernel<<<dim3(candidate_count, kRateRecoveryWarps), 32U, 0,
                                  reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        reinterpret_cast<const float*>(buffers.pdcch_llrs),
        reinterpret_cast<const CudaPdcchCandidateDescriptor*>(buffers.pdcch_candidates),
        candidate_count, reinterpret_cast<float*>(buffers.pdcch_recovered_llrs));
    return map_cuda_error(cudaGetLastError());
}

MmseStatus cuda_launch_pdcch_viterbi(const CudaDeviceBuffers& buffers,
                                    std::uint32_t candidate_count,
                                    std::uintptr_t stream_handle) {
    if (buffers.pdcch_recovered_llrs == nullptr || buffers.pdcch_survivors == nullptr ||
        buffers.pdcch_terminal_metrics == nullptr || candidate_count == 0U ||
        candidate_count > kCudaPdcchMaxCandidates) {
        return MmseStatus::kInvalidArgument;
    }
    pdcch_viterbi_kernel<<<dim3(candidate_count, 32U), 64U, 0,
                              reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        reinterpret_cast<const float*>(buffers.pdcch_recovered_llrs), candidate_count,
        reinterpret_cast<std::uint64_t*>(buffers.pdcch_survivors),
        reinterpret_cast<double*>(buffers.pdcch_terminal_metrics));
    return map_cuda_error(cudaGetLastError());
}

MmseStatus cuda_launch_pdcch_crc_compact(const CudaDeviceBuffers& buffers,
                                        std::uint32_t candidate_count,
                                        std::uint16_t expected_rnti,
                                        std::uintptr_t stream_handle) {
    if (buffers.pdcch_candidates == nullptr || buffers.pdcch_survivors == nullptr ||
        buffers.pdcch_terminal_metrics == nullptr || buffers.pdcch_candidate_results == nullptr ||
        buffers.pdcch_results == nullptr || buffers.pdcch_hit_count == nullptr ||
        candidate_count == 0U || candidate_count > kCudaPdcchMaxCandidates) {
        return MmseStatus::kInvalidArgument;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    pdcch_crc_kernel<<<candidate_count, 1U, 0, stream>>>(
        reinterpret_cast<const CudaPdcchCandidateDescriptor*>(buffers.pdcch_candidates),
        candidate_count, reinterpret_cast<const std::uint64_t*>(buffers.pdcch_survivors),
        reinterpret_cast<const double*>(buffers.pdcch_terminal_metrics), expected_rnti,
        reinterpret_cast<CudaPdcchCandidateResult*>(buffers.pdcch_candidate_results));
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
        return map_cuda_error(status);
    }
    pdcch_compact_hits_kernel<<<1U, 1U, 0, stream>>>(
        reinterpret_cast<const CudaPdcchCandidateResult*>(buffers.pdcch_candidate_results),
        candidate_count, reinterpret_cast<CudaPdcchCandidateResult*>(buffers.pdcch_results),
        reinterpret_cast<std::uint32_t*>(buffers.pdcch_hit_count));
    return map_cuda_error(cudaGetLastError());
}

MmseStatus cuda_copy_pdcch_results_d2h_async(const CudaDeviceBuffers& buffers,
                                             CudaPdcchCandidateResult* results,
                                             std::uint32_t& hit_count,
                                             std::uintptr_t stream_handle) {
    if (buffers.pdcch_results == nullptr || buffers.pdcch_hit_count == nullptr ||
        results == nullptr) {
        return MmseStatus::kInvalidArgument;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (const cudaError_t status = cudaMemcpyAsync(
            &hit_count, buffers.pdcch_hit_count, sizeof(hit_count), cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess) {
        return map_cuda_error(status);
    }
    return map_cuda_error(cudaMemcpyAsync(
        results, buffers.pdcch_results,
        static_cast<std::size_t>(kCudaPdcchMaxCandidates) * sizeof(CudaPdcchCandidateResult),
        cudaMemcpyDeviceToHost, stream));
}

MmseStatus cuda_stream_synchronize(std::uintptr_t stream_handle) {
    return map_cuda_error(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_handle)));
}

}  // namespace mmse::detail
