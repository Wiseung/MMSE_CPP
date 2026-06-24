#include "internal/mmse_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <limits>
#include <mutex>

namespace mmse::detail {

namespace {

constexpr std::size_t h_index(std::uint32_t tx, std::uint32_t rx, std::uint32_t symbol,
                              std::uint32_t sc) {
    return (((static_cast<std::size_t>(tx) * kLteNumRxAntV1 + rx) * kLteNumSymbolsNormalCp +
             symbol) *
            kLteNumSubcarriers20MHz) +
           sc;
}

constexpr std::size_t crs_table_index(std::uint16_t cell_id, std::uint8_t subframe,
                                      std::uint8_t port, std::uint8_t crs_symbol_index,
                                      std::uint32_t pilot_index) {
    return ((((static_cast<std::size_t>(cell_id) * 10U + subframe) * kLteNumTxPortsV1 + port) *
                 kLteNumCrsSymbols +
             crs_symbol_index) *
            kLteNumPilotTonesPerCrsSymbol) +
           pilot_index;
}

std::array<Complex32, kLteNumCellIds * 10U * kLteNumTxPortsV1 * kLteNumCrsSymbols *
                          kLteNumPilotTonesPerCrsSymbol>
    g_crs_table{};
std::once_flag g_crs_table_once;

inline float clampf(float value, float lo, float hi) {
    return std::min(std::max(value, lo), hi);
}

inline bool bitmap_has_prb(const ExtractDescriptor& desc, std::uint32_t prb) {
    const std::uint32_t word = prb / 16U;
    const std::uint32_t bit = prb % 16U;
    return (desc.prb_bitmap[word] & (static_cast<std::uint16_t>(1U) << bit)) != 0U;
}

inline Complex32 grid_at(const PlanarGridViewF32& grid, std::uint32_t rx, std::uint32_t symbol,
                         std::uint32_t sc) {
    const std::size_t idx = static_cast<std::size_t>(symbol) * kLteNumSubcarriers20MHz + sc;
    return {grid.re[rx][idx], grid.im[rx][idx]};
}

inline void store_out(EqualizerOutputView& out, std::uint32_t layer, std::uint32_t re_index,
                      const EqualizedSymbol& value) {
    const std::size_t base = static_cast<std::size_t>(layer) * out.capacity_re_per_layer + re_index;
    out.x_hat_re[base] = value.xhat.re;
    out.x_hat_im[base] = value.xhat.im;
    out.sinr[base] = value.gamma;
}

Complex32 lerp_symbol(std::uint32_t target_symbol, std::uint32_t left_symbol, Complex32 left,
                      std::uint32_t right_symbol, Complex32 right) {
    if (left_symbol == right_symbol) {
        return left;
    }
    const float t = static_cast<float>(target_symbol - left_symbol) /
                    static_cast<float>(right_symbol - left_symbol);
    return linear_interp(left, right, t);
}

} // namespace

Complex32 cadd(Complex32 a, Complex32 b) {
    return {a.re + b.re, a.im + b.im};
}

Complex32 csub(Complex32 a, Complex32 b) {
    return {a.re - b.re, a.im - b.im};
}

Complex32 cmul(Complex32 a, Complex32 b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

Complex32 cconj(Complex32 a) {
    return {a.re, -a.im};
}

Complex32 cscale(Complex32 a, float s) {
    return {a.re * s, a.im * s};
}

float cnorm2(Complex32 a) {
    return a.re * a.re + a.im * a.im;
}

Complex32 linear_interp(Complex32 left, Complex32 right, float t) {
    return {left.re + (right.re - left.re) * t, left.im + (right.im - left.im) * t};
}

bool descriptor_supported(const ExtractDescriptor& desc) {
    if (desc.cell_id >= kLteNumCellIds) {
        return false;
    }
    if (desc.n_rx_ant != kLteNumRxAntV1 || desc.n_tx_ports != kLteNumTxPortsV1) {
        return false;
    }
    if (desc.n_layers != 1U && desc.n_layers != 2U) {
        return false;
    }
    if (desc.pmi != -1) {
        return false;
    }
    if (desc.start_symbol >= kLteNumSymbolsNormalCp) {
        return false;
    }
    if (desc.mod_order != 2U && desc.mod_order != 4U && desc.mod_order != 6U &&
        desc.mod_order != 8U) {
        return false;
    }
    if (desc.n_prb == 0U || desc.n_prb > kLteNumPrb20MHz) {
        return false;
    }
    std::uint32_t bitmap_prbs = 0U;
    for (std::uint32_t prb = 0; prb < kLteNumPrb20MHz; ++prb) {
        bitmap_prbs += bitmap_has_prb(desc, prb) ? 1U : 0U;
    }
    if (bitmap_prbs != desc.n_prb) {
        return false;
    }
    if (desc.tx_mode == 4U) {
        return false;
    }
    return true;
}

MmseStatus validate_grid(const PlanarGridViewF32& grid) {
    if (grid.n_rx_ant != kLteNumRxAntV1 || grid.n_symbols != kLteNumSymbolsNormalCp ||
        grid.n_subcarriers != kLteNumSubcarriers20MHz) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::uint32_t rx = 0; rx < kLteNumRxAntV1; ++rx) {
        if (grid.re[rx] == nullptr || grid.im[rx] == nullptr) {
            return MmseStatus::kInvalidArgument;
        }
    }
    return MmseStatus::kOk;
}

MmseStatus validate_output(const EqualizerOutputView& out) {
    if (out.x_hat_re == nullptr || out.x_hat_im == nullptr || out.sinr == nullptr) {
        return MmseStatus::kInvalidArgument;
    }
    return MmseStatus::kOk;
}

std::uint32_t subframe_from_descriptor(const ExtractDescriptor& desc) {
    return desc.sfn_subframe % 10U;
}

std::uint32_t crs_frequency_offset(std::uint16_t cell_id, std::uint8_t port, std::uint8_t symbol) {
    const std::uint32_t v_shift = cell_id % 6U;
    std::uint32_t v = 0U;
    if (symbol == 0U || symbol == 7U) {
        v = (port == 0U) ? 0U : 3U;
    } else {
        v = (port == 0U) ? 3U : 0U;
    }
    return (v + v_shift) % 6U;
}

std::uint32_t crs_subcarrier(std::uint16_t cell_id, std::uint8_t port, std::uint8_t symbol,
                             std::uint32_t pilot_index) {
    return 6U * pilot_index + crs_frequency_offset(cell_id, port, symbol);
}

bool is_crs_re(std::uint16_t cell_id, std::uint8_t symbol, std::uint32_t sc) {
    if (symbol != 0U && symbol != 4U && symbol != 7U && symbol != 11U) {
        return false;
    }
    return sc % 6U == crs_frequency_offset(cell_id, 0U, symbol) % 6U ||
           sc % 6U == crs_frequency_offset(cell_id, 1U, symbol) % 6U;
}

const Complex32& crs_value(const CrsTableKey& key, std::uint32_t pilot_index) {
    return g_crs_table[crs_table_index(key.cell_id, key.subframe, key.port, key.crs_symbol_index,
                                       pilot_index)];
}

void ensure_crs_tables() {
    std::call_once(g_crs_table_once, []() {
        for (std::uint16_t cell_id = 0; cell_id < kLteNumCellIds; ++cell_id) {
            for (std::uint8_t subframe = 0; subframe < 10U; ++subframe) {
                for (std::uint8_t port = 0; port < kLteNumTxPortsV1; ++port) {
                    for (std::uint8_t symbol_idx = 0; symbol_idx < kLteNumCrsSymbols;
                         ++symbol_idx) {
                        const std::uint8_t symbol = kCrsSymbols[symbol_idx];
                        const std::uint32_t c_init = (1U << 10U) *
                                                         (7U * (subframe + 1U) + symbol + 1U) *
                                                         (2U * cell_id + 1U) +
                                                     2U * cell_id + 1U;
                        std::uint32_t x1 = 0x1U;
                        std::uint32_t x2 = c_init;
                        for (int n = 0; n < 1600; ++n) {
                            const std::uint32_t x1_new = ((x1 >> 3U) ^ x1) & 1U;
                            const std::uint32_t x2_new =
                                ((x2 >> 3U) ^ (x2 >> 2U) ^ (x2 >> 1U) ^ x2) & 1U;
                            x1 = (x1 >> 1U) | (x1_new << 30U);
                            x2 = (x2 >> 1U) | (x2_new << 30U);
                        }
                        for (std::uint32_t pilot = 0; pilot < kLteNumPilotTonesPerCrsSymbol;
                             ++pilot) {
                            const std::uint32_t b0 = (x1 ^ x2) & 1U;
                            const std::uint32_t x1_new0 = ((x1 >> 3U) ^ x1) & 1U;
                            const std::uint32_t x2_new0 =
                                ((x2 >> 3U) ^ (x2 >> 2U) ^ (x2 >> 1U) ^ x2) & 1U;
                            x1 = (x1 >> 1U) | (x1_new0 << 30U);
                            x2 = (x2 >> 1U) | (x2_new0 << 30U);
                            const std::uint32_t b1 = (x1 ^ x2) & 1U;
                            const std::uint32_t x1_new1 = ((x1 >> 3U) ^ x1) & 1U;
                            const std::uint32_t x2_new1 =
                                ((x2 >> 3U) ^ (x2 >> 2U) ^ (x2 >> 1U) ^ x2) & 1U;
                            x1 = (x1 >> 1U) | (x1_new1 << 30U);
                            x2 = (x2 >> 1U) | (x2_new1 << 30U);

                            const float scale = 0.7071067811865475F;
                            g_crs_table[crs_table_index(cell_id, subframe, port, symbol_idx,
                                                        pilot)] = {
                                scale * (b0 == 0U ? 1.0F : -1.0F),
                                scale * (b1 == 0U ? 1.0F : -1.0F),
                            };
                        }
                    }
                }
            }
        }
    });
}

std::uint32_t build_data_re_layout(const ExtractDescriptor& desc, ReLayout& layout) {
    layout.n_re = 0U;
    layout.n_segments = 0U;
    layout.output_slot_by_grid_re.fill(std::numeric_limits<std::uint32_t>::max());

    for (std::uint32_t symbol = desc.start_symbol; symbol < kLteNumSymbolsNormalCp; ++symbol) {
        std::uint32_t segment_begin = layout.n_re;
        for (std::uint32_t prb = 0; prb < kLteNumPrb20MHz; ++prb) {
            if (!bitmap_has_prb(desc, prb)) {
                continue;
            }
            for (std::uint32_t tone = 0; tone < 12U; ++tone) {
                const std::uint32_t sc = prb * 12U + tone;
                if (is_crs_re(desc.cell_id, static_cast<std::uint8_t>(symbol), sc)) {
                    continue;
                }
                const std::uint32_t grid_idx = symbol * kLteNumSubcarriers20MHz + sc;
                layout.grid_indices[layout.n_re] = static_cast<std::uint16_t>(grid_idx);
                layout.output_slot_by_grid_re[grid_idx] = layout.n_re;
                ++layout.n_re;
            }
        }
        if (layout.n_re > segment_begin) {
            layout.prb_segment_offsets[layout.n_segments++] = segment_begin;
        }
    }
    layout.prb_segment_offsets[layout.n_segments] = layout.n_re;
    return layout.n_re;
}

void estimate_channel(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                      HGridStorage& h_full, float& sigma2_estimate) {
    ensure_crs_tables();
    sigma2_estimate = 0.0F;
    std::uint32_t residual_count = 0U;
    const std::uint32_t subframe = subframe_from_descriptor(desc);

    for (std::uint32_t tx = 0; tx < kLteNumTxPortsV1; ++tx) {
        for (std::uint32_t rx = 0; rx < kLteNumRxAntV1; ++rx) {
            std::array<std::array<Complex32, kLteNumPilotTonesPerCrsSymbol>, kLteNumCrsSymbols>
                ls{};
            std::array<std::array<Complex32, kLteNumSubcarriers20MHz>, kLteNumCrsSymbols> freq{};

            for (std::uint32_t cs = 0; cs < kLteNumCrsSymbols; ++cs) {
                const std::uint8_t symbol = kCrsSymbols[cs];
                CrsTableKey key{
                    .cell_id = desc.cell_id,
                    .subframe = static_cast<std::uint8_t>(subframe),
                    .port = static_cast<std::uint8_t>(tx),
                    .crs_symbol_index = static_cast<std::uint8_t>(cs),
                };

                for (std::uint32_t pilot = 0; pilot < kLteNumPilotTonesPerCrsSymbol; ++pilot) {
                    const std::uint32_t sc =
                        crs_subcarrier(desc.cell_id, static_cast<std::uint8_t>(tx), symbol, pilot);
                    ls[cs][pilot] =
                        cmul(grid_at(grid, rx, symbol, sc), cconj(crs_value(key, pilot)));
                }

                for (std::uint32_t pilot = 1U; pilot + 1U < kLteNumPilotTonesPerCrsSymbol;
                     ++pilot) {
                    const Complex32 smooth =
                        cscale(cadd(ls[cs][pilot - 1U], ls[cs][pilot + 1U]), 0.5F);
                    sigma2_estimate += cnorm2(csub(ls[cs][pilot], smooth));
                    ++residual_count;
                }

                for (std::uint32_t sc = 0; sc < kLteNumSubcarriers20MHz; ++sc) {
                    const std::uint32_t offset =
                        crs_frequency_offset(desc.cell_id, static_cast<std::uint8_t>(tx), symbol);
                    const std::uint32_t lower_pilot = (sc < offset) ? 0U : (sc - offset) / 6U;
                    const std::uint32_t upper_pilot =
                        std::min(lower_pilot + 1U, kLteNumPilotTonesPerCrsSymbol - 1U);
                    const std::uint32_t left_sc = crs_subcarrier(
                        desc.cell_id, static_cast<std::uint8_t>(tx), symbol, lower_pilot);
                    const std::uint32_t right_sc = crs_subcarrier(
                        desc.cell_id, static_cast<std::uint8_t>(tx), symbol, upper_pilot);
                    if (left_sc == right_sc) {
                        freq[cs][sc] = ls[cs][lower_pilot];
                    } else if (sc <= left_sc) {
                        freq[cs][sc] = ls[cs][lower_pilot];
                    } else if (sc >= right_sc) {
                        freq[cs][sc] = ls[cs][upper_pilot];
                    } else {
                        const float t = static_cast<float>(sc - left_sc) /
                                        static_cast<float>(right_sc - left_sc);
                        freq[cs][sc] = linear_interp(ls[cs][lower_pilot], ls[cs][upper_pilot], t);
                    }
                }
            }

            for (std::uint32_t symbol = 0; symbol < kLteNumSymbolsNormalCp; ++symbol) {
                std::uint32_t upper = 0U;
                while (upper < kLteNumCrsSymbols && kCrsSymbols[upper] < symbol) {
                    ++upper;
                }
                std::uint32_t lower = (upper == 0U) ? 0U : upper - 1U;
                if (upper >= kLteNumCrsSymbols) {
                    upper = kLteNumCrsSymbols - 1U;
                    lower = upper;
                }
                for (std::uint32_t sc = 0; sc < kLteNumSubcarriers20MHz; ++sc) {
                    h_full[h_index(tx, rx, symbol, sc)] =
                        lerp_symbol(symbol, kCrsSymbols[lower], freq[lower][sc], kCrsSymbols[upper],
                                    freq[upper][sc]);
                }
            }
        }
    }

    if (residual_count == 0U) {
        sigma2_estimate = kDefaultSigma2Min;
    } else {
        sigma2_estimate /= static_cast<float>(residual_count);
    }
}

float update_sigma2_state(Sigma2State& state, float sigma2_estimate,
                          const MmseEqualizerCpuConfig& config) {
    sigma2_estimate = std::max(sigma2_estimate, config.sigma2_min);
    if (!state.initialized) {
        state.value = sigma2_estimate;
        state.initialized = true;
        return state.value;
    }
    state.value =
        config.sigma2_iir_alpha * state.value + (1.0F - config.sigma2_iir_alpha) * sigma2_estimate;
    state.value = std::max(state.value, config.sigma2_min);
    return state.value;
}

void pack_equalizer_inputs(const PlanarGridViewF32& grid, const HGridStorage& h_full,
                           const ReLayout& layout, PackedEqualizerInputs& packed) {
    for (std::uint32_t re = 0; re < layout.n_re; ++re) {
        const std::uint32_t grid_idx = layout.grid_indices[re];
        const std::uint32_t symbol = grid_idx / kLteNumSubcarriers20MHz;
        const std::uint32_t sc = grid_idx % kLteNumSubcarriers20MHz;

        const Complex32 h00 = h_full[h_index(0U, 0U, symbol, sc)];
        const Complex32 h01 = h_full[h_index(1U, 0U, symbol, sc)];
        const Complex32 h10 = h_full[h_index(0U, 1U, symbol, sc)];
        const Complex32 h11 = h_full[h_index(1U, 1U, symbol, sc)];
        const Complex32 y0 = grid_at(grid, 0U, symbol, sc);
        const Complex32 y1 = grid_at(grid, 1U, symbol, sc);

        packed.h00_re[re] = h00.re;
        packed.h00_im[re] = h00.im;
        packed.h01_re[re] = h01.re;
        packed.h01_im[re] = h01.im;
        packed.h10_re[re] = h10.re;
        packed.h10_im[re] = h10.im;
        packed.h11_re[re] = h11.re;
        packed.h11_im[re] = h11.im;
        packed.y0_re[re] = y0.re;
        packed.y0_im[re] = y0.im;
        packed.y1_re[re] = y1.re;
        packed.y1_im[re] = y1.im;
    }
}

EqualizedSymbol equalize_1x2_scalar(Complex32 h0, Complex32 h1, Complex32 y0, Complex32 y1,
                                    float sigma2, float g_min, float gamma_max) {
    const float denom = cnorm2(h0) + cnorm2(h1) + sigma2;
    const Complex32 weight0 = cscale(cconj(h0), 1.0F / denom);
    const Complex32 weight1 = cscale(cconj(h1), 1.0F / denom);
    const Complex32 z = cadd(cmul(weight0, y0), cmul(weight1, y1));
    const float g = clampf((cnorm2(h0) + cnorm2(h1)) / denom, g_min, 1.0F - g_min);
    return {
        .xhat = cscale(z, 1.0F / g),
        .gamma = clampf(g / (1.0F - g), g_min, gamma_max),
    };
}

EqualizedSymbol equalize_2x2_scalar(Complex32 h00, Complex32 h01, Complex32 h10, Complex32 h11,
                                    Complex32 y0, Complex32 y1, float sigma2, float det_floor,
                                    float g_min, float gamma_max, std::uint8_t layer_index) {
    const float a11 = cnorm2(h00) + cnorm2(h10) + sigma2;
    const float a22 = cnorm2(h01) + cnorm2(h11) + sigma2;
    const Complex32 a12 = cadd(cmul(cconj(h00), h01), cmul(cconj(h10), h11));
    const float det = std::max(a11 * a22 - cnorm2(a12), det_floor);
    const float inv_det = 1.0F / det;

    const float inv11 = a22 * inv_det;
    const float inv22 = a11 * inv_det;
    const Complex32 inv12 = cscale(a12, -inv_det);
    const Complex32 inv21 = cconj(inv12);

    const Complex32 hh00 = cconj(h00);
    const Complex32 hh01 = cconj(h10);
    const Complex32 hh10 = cconj(h01);
    const Complex32 hh11 = cconj(h11);

    const Complex32 w00 = cadd(cscale(hh00, inv11), cmul(inv12, hh10));
    const Complex32 w01 = cadd(cscale(hh01, inv11), cmul(inv12, hh11));
    const Complex32 w10 = cadd(cmul(inv21, hh00), cscale(hh10, inv22));
    const Complex32 w11 = cadd(cmul(inv21, hh01), cscale(hh11, inv22));

    const Complex32 z0 = cadd(cmul(w00, y0), cmul(w01, y1));
    const Complex32 z1 = cadd(cmul(w10, y0), cmul(w11, y1));

    const float g0 = clampf(cadd(cmul(w00, h00), cmul(w01, h10)).re, g_min, 1.0F - g_min);
    const float g1 = clampf(cadd(cmul(w10, h01), cmul(w11, h11)).re, g_min, 1.0F - g_min);

    const Complex32 xhat = (layer_index == 0U) ? cscale(z0, 1.0F / g0) : cscale(z1, 1.0F / g1);
    const float gamma = (layer_index == 0U) ? g0 / (1.0F - g0) : g1 / (1.0F - g1);
    return {.xhat = xhat, .gamma = clampf(gamma, g_min, gamma_max)};
}

bool cpu_supports_avx2() {
#if defined(_M_X64) || defined(__x86_64__)
#if defined(_MSC_VER)
    int cpu_info[4] = {};
    __cpuid(cpu_info, 0);
    if (cpu_info[0] < 7) {
        return false;
    }
    __cpuidex(cpu_info, 1, 0);
    const bool osxsave = (cpu_info[2] & (1 << 27)) != 0;
    const bool avx = (cpu_info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) {
        return false;
    }
    __cpuidex(cpu_info, 7, 0);
    return (cpu_info[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2");
#endif
#else
    return false;
#endif
}

} // namespace mmse::detail

namespace mmse {

using detail::Complex32;
using detail::EqualizedSymbol;
using detail::HGridStorage;
using detail::PackedEqualizerInputs;
using detail::ReLayout;
using detail::Sigma2State;
using detail::StaticThreadPool;

struct MmseEqualizerCpuContext::Impl {
    struct RunTaskContext {
        Impl* impl = nullptr;
        ExtractDescriptor desc{};
        EqualizerOutputView* out = nullptr;
        float sigma2 = 0.0F;
    };

    static void worker_task(void* raw_ctx, std::uint32_t begin, std::uint32_t end);

    MmseEqualizerCpuConfig config{};
    bool initialized = false;
    bool use_avx2 = false;
    HGridStorage h_full{};
    detail::PackedEqualizerInputs packed{};
    ReLayout layout{};
    std::array<Sigma2State, kLteNumCellIds> sigma2_by_cell{};
    StaticThreadPool pool{};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges{};
};

MmseEqualizerCpuContext::MmseEqualizerCpuContext() : impl_(new Impl()) {}

MmseEqualizerCpuContext::~MmseEqualizerCpuContext() {
    delete impl_;
}

MmseStatus MmseEqualizerCpuContext::init(const MmseEqualizerCpuConfig& config) {
    if (config.worker_count == 0U || config.worker_count > detail::kMaxThreadWorkers ||
        config.g_min <= 0.0F || config.g_min >= 0.5F || config.gamma_max <= 0.0F ||
        config.det_floor <= 0.0F || config.sigma2_min <= 0.0F || config.sigma2_iir_alpha < 0.0F ||
        config.sigma2_iir_alpha > 1.0F) {
        return MmseStatus::kInvalidArgument;
    }

    const bool avx2_supported = detail::cpu_supports_avx2();
    bool use_avx2 = false;
    switch (config.backend) {
    case MmseCpuBackend::kAuto:
        use_avx2 = avx2_supported;
        break;
    case MmseCpuBackend::kScalar:
        use_avx2 = false;
        break;
    case MmseCpuBackend::kAvx2:
        if (!avx2_supported) {
            return MmseStatus::kUnsupportedConfig;
        }
        use_avx2 = true;
        break;
    default:
        return MmseStatus::kInvalidArgument;
    }

    impl_->config = config;
    impl_->use_avx2 = use_avx2;
    impl_->pool.init(config.worker_count);
    impl_->ranges.resize(config.worker_count);
    impl_->initialized = true;
    detail::ensure_crs_tables();
    return MmseStatus::kOk;
}

namespace {

std::uint32_t chunk_count(std::uint32_t workers, std::uint32_t total) {
    if (workers == 0U) {
        return 1U;
    }
    return std::min(workers, std::max(1U, total));
}

} // namespace

void MmseEqualizerCpuContext::Impl::worker_task(void* raw_ctx, std::uint32_t begin,
                                                std::uint32_t end) {
    auto* ctx = static_cast<RunTaskContext*>(raw_ctx);
    auto& impl = *ctx->impl;
    auto& out = *ctx->out;

    if (ctx->desc.n_layers == 2U && impl.use_avx2) {
        detail::equalize_2x2_avx2(impl.packed, begin, end, ctx->sigma2, impl.config.det_floor,
                                  impl.config.g_min, impl.config.gamma_max, out.x_hat_re,
                                  out.x_hat_im, out.sinr, out.x_hat_re + out.capacity_re_per_layer,
                                  out.x_hat_im + out.capacity_re_per_layer,
                                  out.sinr + out.capacity_re_per_layer);
        return;
    }

    for (std::uint32_t i = begin; i < end; ++i) {
        Complex32 h00{impl.packed.h00_re[i], impl.packed.h00_im[i]};
        Complex32 h01{impl.packed.h01_re[i], impl.packed.h01_im[i]};
        Complex32 h10{impl.packed.h10_re[i], impl.packed.h10_im[i]};
        Complex32 h11{impl.packed.h11_re[i], impl.packed.h11_im[i]};
        Complex32 y0{impl.packed.y0_re[i], impl.packed.y0_im[i]};
        Complex32 y1{impl.packed.y1_re[i], impl.packed.y1_im[i]};

        if (ctx->desc.n_layers == 1U) {
            const EqualizedSymbol eq = detail::equalize_1x2_scalar(
                h00, h10, y0, y1, ctx->sigma2, impl.config.g_min, impl.config.gamma_max);
            detail::store_out(out, 0U, i, eq);
            continue;
        }

        const EqualizedSymbol eq0 = detail::equalize_2x2_scalar(
            h00, h01, h10, h11, y0, y1, ctx->sigma2, impl.config.det_floor, impl.config.g_min,
            impl.config.gamma_max, 0U);
        const EqualizedSymbol eq1 = detail::equalize_2x2_scalar(
            h00, h01, h10, h11, y0, y1, ctx->sigma2, impl.config.det_floor, impl.config.g_min,
            impl.config.gamma_max, 1U);
        detail::store_out(out, 0U, i, eq0);
        detail::store_out(out, 1U, i, eq1);
    }
}

MmseStatus MmseEqualizerCpuContext::run(const PlanarGridViewF32& grid,
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

    const std::uint32_t n_re = detail::build_data_re_layout(desc, impl_->layout);
    if (out.capacity_re_per_layer < n_re) {
        return MmseStatus::kBufferTooSmall;
    }

    float sigma2_estimate = 0.0F;
    detail::estimate_channel(grid, desc, impl_->h_full, sigma2_estimate);
    const float sigma2 = detail::update_sigma2_state(impl_->sigma2_by_cell[desc.cell_id],
                                                     sigma2_estimate, impl_->config);
    detail::pack_equalizer_inputs(grid, impl_->h_full, impl_->layout, impl_->packed);

    out.n_re_per_layer = n_re;
    out.n_layers = desc.n_layers;
    out.mod_order = desc.mod_order;

    Impl::RunTaskContext worker_ctx{
        .impl = impl_,
        .desc = desc,
        .out = &out,
        .sigma2 = sigma2,
    };

    const std::uint32_t workers = chunk_count(impl_->pool.worker_count(), n_re == 0U ? 1U : n_re);
    const std::uint32_t chunk = (n_re + workers - 1U) / workers;
    for (std::uint32_t w = 0; w < workers; ++w) {
        const std::uint32_t begin = w * chunk;
        const std::uint32_t end = std::min(begin + chunk, n_re);
        impl_->ranges[w] = {begin, end};
    }
    for (std::uint32_t w = workers; w < impl_->ranges.size(); ++w) {
        impl_->ranges[w] = {0U, 0U};
    }

    impl_->pool.parallel_for(
        std::span<const std::pair<std::uint32_t, std::uint32_t>>(impl_->ranges.data(), workers),
        Impl::worker_task, &worker_ctx);
    return MmseStatus::kOk;
}

const char* to_string(MmseStatus status) {
    switch (status) {
    case MmseStatus::kOk:
        return "ok";
    case MmseStatus::kNotInitialized:
        return "not_initialized";
    case MmseStatus::kInvalidArgument:
        return "invalid_argument";
    case MmseStatus::kUnsupportedConfig:
        return "unsupported_config";
    case MmseStatus::kBufferTooSmall:
        return "buffer_too_small";
    case MmseStatus::kInternalError:
        return "internal_error";
    default:
        return "unknown";
    }
}

} // namespace mmse
