#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "mmse/constants.h"
#include "mmse/mmse_equalizer.h"

namespace mmse::detail {

inline constexpr std::array<std::uint8_t, kLteNumCrsSymbols> kCrsSymbols = {0, 4, 7, 11};
inline constexpr std::uint32_t kMaxGridRe = kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz;
inline constexpr std::uint32_t kMaxDataRe = kMaxGridRe;
inline constexpr std::uint32_t kMaxThreadWorkers = 64;

struct Complex32 {
    float re = 0.0F;
    float im = 0.0F;
};

struct EqualizedSymbol {
    Complex32 xhat{};
    float gamma = 0.0F;
};

struct Equalize2x2Trace {
    float a11 = 0.0F;
    float a22 = 0.0F;
    Complex32 a12{};
    float det = 0.0F;
    float inv_det = 0.0F;
    float inv11 = 0.0F;
    float inv22 = 0.0F;
    Complex32 inv12{};
    Complex32 inv21{};
    Complex32 hh00{};
    Complex32 hh01{};
    Complex32 hh10{};
    Complex32 hh11{};
    Complex32 w00{};
    Complex32 w01{};
    Complex32 w10{};
    Complex32 w11{};
    Complex32 z0{};
    Complex32 z1{};
    float g0 = 0.0F;
    float g1 = 0.0F;
    Complex32 xhat0{};
    Complex32 xhat1{};
    float gamma0 = 0.0F;
    float gamma1 = 0.0F;
};

struct TransmitDiversityEqualizePair {
    EqualizedSymbol symbol0{};
    EqualizedSymbol symbol1{};
};

using HGridStorage =
    std::array<Complex32, 2 * 2 * kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz>;

struct PackedEqualizerInputs {
    std::array<float, kMaxDataRe> h00_re{};
    std::array<float, kMaxDataRe> h00_im{};
    std::array<float, kMaxDataRe> h01_re{};
    std::array<float, kMaxDataRe> h01_im{};
    std::array<float, kMaxDataRe> h10_re{};
    std::array<float, kMaxDataRe> h10_im{};
    std::array<float, kMaxDataRe> h11_re{};
    std::array<float, kMaxDataRe> h11_im{};
    std::array<float, kMaxDataRe> y0_re{};
    std::array<float, kMaxDataRe> y0_im{};
    std::array<float, kMaxDataRe> y1_re{};
    std::array<float, kMaxDataRe> y1_im{};
};

struct ReLayout {
    std::array<std::uint16_t, kMaxDataRe> grid_indices{};
    std::array<std::uint32_t, kMaxGridRe> output_slot_by_grid_re{};
    std::array<std::uint32_t, kMaxDataRe + 1> prb_segment_offsets{};
    std::uint32_t n_re = 0;
    std::uint32_t n_segments = 0;
};

struct PreparedSubframeKey {
    std::array<const float*, 2> re{};
    std::array<const float*, 2> im{};
    std::uint32_t sfn_subframe = 0;
    std::uint16_t cell_id = 0;
    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_rx_ant = 0;
    std::uint32_t n_symbols = 0;
    std::uint32_t n_subcarriers = 0;
    std::uint64_t generation = 0;
    std::uint32_t backend_mode = 0;
};

struct Sigma2State {
    float value = kDefaultSigma2Min;
    bool initialized = false;
};

class StaticThreadPool {
  public:
    using TaskFn = void (*)(void*, std::uint32_t, std::uint32_t);

    StaticThreadPool();
    ~StaticThreadPool();

    StaticThreadPool(const StaticThreadPool&) = delete;
    StaticThreadPool& operator=(const StaticThreadPool&) = delete;

    void init(std::uint32_t worker_count);
    void stop();
    std::uint32_t worker_count() const;

    void parallel_for(const std::span<const std::pair<std::uint32_t, std::uint32_t>>& ranges,
                      TaskFn fn, void* ctx);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct CrsTableKey {
    std::uint16_t cell_id = 0;
    std::uint8_t subframe = 0;
    std::uint8_t port = 0;
    std::uint8_t crs_symbol_index = 0;
};

const Complex32& crs_value(const CrsTableKey& key, std::uint32_t pilot_index);
void ensure_crs_tables();

bool descriptor_supported(const ExtractDescriptor& desc);
MmseStatus validate_grid(const PlanarGridViewF32& grid);
MmseStatus validate_output(const EqualizerOutputView& out);
PreparedSubframeKey make_prepared_subframe_key(const PlanarGridViewF32& grid,
                                               const ExtractDescriptor& desc,
                                               std::uint32_t backend_mode = 0U);
bool prepared_subframe_key_equal(const PreparedSubframeKey& lhs, const PreparedSubframeKey& rhs);
std::uint32_t channel_start_symbol(const ExtractDescriptor& desc);
std::uint32_t subframe_from_descriptor(const ExtractDescriptor& desc);
std::uint32_t crs_frequency_offset(std::uint16_t cell_id, std::uint8_t port, std::uint8_t symbol);
std::uint32_t crs_subcarrier(std::uint16_t cell_id, std::uint8_t port, std::uint8_t symbol,
                             std::uint32_t pilot_index);
bool is_crs_re(std::uint16_t cell_id, std::uint8_t symbol, std::uint32_t sc);
bool is_crs_re(const ExtractDescriptor& desc, std::uint8_t symbol, std::uint32_t sc);
std::uint32_t build_data_re_layout(const ExtractDescriptor& desc, ReLayout& layout);
std::uint32_t build_pdcch_re_layout(const ExtractDescriptor& desc, ReLayout& layout);
std::uint32_t build_pbch_re_layout(const ExtractDescriptor& desc, ReLayout& layout);
std::uint32_t build_pcfich_re_layout(const ExtractDescriptor& desc, ReLayout& layout);
std::uint32_t build_channel_re_layout(const ExtractDescriptor& desc, ReLayout& layout);
std::uint32_t build_validation_re_samples(const ReLayout& layout, std::uint32_t start_symbol,
                                          std::uint32_t n_symbols, std::uint32_t n_subcarriers,
                                          std::uint32_t* out_re_slots, std::uint32_t max_slots);

Complex32 cadd(Complex32 a, Complex32 b);
Complex32 csub(Complex32 a, Complex32 b);
Complex32 cmul(Complex32 a, Complex32 b);
Complex32 cconj(Complex32 a);
Complex32 cscale(Complex32 a, float s);
float cnorm2(Complex32 a);

Complex32 linear_interp(Complex32 left, Complex32 right, float t);

void estimate_channel(
    const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
    std::array<Complex32, 2 * 2 * kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz>& h_full,
    float& sigma2_estimate);

float update_sigma2_state(Sigma2State& state, float sigma2_estimate,
                          const MmseEqualizerCpuConfig& config);
float peek_sigma2_state(const Sigma2State& state, float sigma2_min);
void debug_reset_estimate_channel_call_count();
std::uint64_t debug_get_estimate_channel_call_count();

void pack_equalizer_inputs(
    const PlanarGridViewF32& grid,
    const std::array<Complex32, 2 * 2 * kLteNumSymbolsNormalCp * kLteNumSubcarriers20MHz>& h_full,
    const ReLayout& layout, PackedEqualizerInputs& packed);

EqualizedSymbol equalize_2x2_scalar(Complex32 h00, Complex32 h01, Complex32 h10, Complex32 h11,
                                    Complex32 y0, Complex32 y1, float sigma2, float det_floor,
                                    float g_min, float gamma_max, std::uint8_t layer_index);
Equalize2x2Trace trace_equalize_2x2_scalar(Complex32 h00, Complex32 h01, Complex32 h10,
                                           Complex32 h11, Complex32 y0, Complex32 y1, float sigma2,
                                           float det_floor, float g_min, float gamma_max);

EqualizedSymbol equalize_1x2_scalar(Complex32 h0, Complex32 h1, Complex32 y0, Complex32 y1,
                                    float sigma2, float g_min, float gamma_max);
TransmitDiversityEqualizePair
demap_transmit_diversity_mmse_scalar(Complex32 h00, Complex32 h01, Complex32 h10, Complex32 h11,
                                     Complex32 h00_next, Complex32 h01_next, Complex32 h10_next,
                                     Complex32 h11_next, Complex32 y0, Complex32 y1,
                                     Complex32 y0_next, Complex32 y1_next, float sigma2,
                                     float det_floor, float g_min, float gamma_max);

TransmitDiversityEqualizePair
demap_pdcch_transmit_diversity_scalar(Complex32 h00, Complex32 h01, Complex32 h10, Complex32 h11,
                                      Complex32 h00_next, Complex32 h01_next, Complex32 h10_next,
                                      Complex32 h11_next, Complex32 y0, Complex32 y1,
                                      Complex32 y0_next, Complex32 y1_next, float sigma2,
                                      float det_floor, float g_min, float gamma_max);

bool cpu_supports_avx2();

void equalize_2x2_avx2(const PackedEqualizerInputs& packed, std::uint32_t begin, std::uint32_t end,
                       float sigma2, float det_floor, float g_min, float gamma_max,
                       float* out_re_layer0, float* out_im_layer0, float* out_sinr_layer0,
                       float* out_re_layer1, float* out_im_layer1, float* out_sinr_layer1);

} // namespace mmse::detail
