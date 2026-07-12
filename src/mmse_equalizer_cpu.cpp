#include "internal/mmse_internal.h"
#include "mmse/pbch_module_api.h"
#include "mmse/pdcch_module_api.h"
#include "mmse/pcfich_module_api.h"

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
    return (((static_cast<std::size_t>(tx) * kMmseV1MaxNumRxAntennas + rx) *
                 kLteNumSymbolsNormalCp +
             symbol) *
            kLteNumSubcarriers20MHz) +
           sc;
}

constexpr std::size_t crs_table_index(std::uint16_t cell_id, std::uint8_t subframe,
                                      std::uint8_t port, std::uint8_t crs_symbol_index,
                                      std::uint32_t pilot_index) {
    return ((((static_cast<std::size_t>(cell_id) * 10U + subframe) * kMmseV1MaxNumCrsTxPorts +
              port) *
                 kLteNumCrsSymbols +
             crs_symbol_index) *
            kLteNumPilotTonesPerCrsSymbol) +
           pilot_index;
}

std::array<Complex32, kLteNumCellIds * 10U * kMmseV1MaxNumCrsTxPorts * kLteNumCrsSymbols *
                          kLteNumPilotTonesPerCrsSymbol>
    g_crs_table{};
std::once_flag g_crs_table_once;
std::uint64_t g_estimate_channel_call_count = 0;

inline float clampf(float value, float lo, float hi) {
    if (!std::isfinite(value)) {
        return lo;
    }
    return std::min(std::max(value, lo), hi);
}

inline float finite_or_zero(float value) {
    return std::isfinite(value) ? value : 0.0F;
}

inline bool bitmap_has_prb(const ExtractDescriptor& desc, std::uint32_t prb) {
    const std::uint32_t word = prb / 16U;
    const std::uint32_t bit = prb % 16U;
    return (desc.prb_bitmap[word] & (static_cast<std::uint16_t>(1U) << bit)) != 0U;
}

inline bool control_re_excluded(const ExtractDescriptor& desc, std::uint32_t symbol,
                                std::uint32_t prb, std::uint32_t tone) {
    if (symbol >= desc.control_symbol_count || prb >= desc.n_prb ||
        tone >= kLteNumSubcarriersPerPrb) {
        return false;
    }
    const std::size_t mask_idx =
        static_cast<std::size_t>(symbol) * desc.n_prb + static_cast<std::size_t>(prb);
    return (desc.control_re_exclusion_masks[mask_idx] & (static_cast<std::uint16_t>(1U) << tone)) !=
           0U;
}

inline void reset_layout(ReLayout& layout) {
    layout.n_re = 0U;
    layout.n_segments = 0U;
    layout.output_slot_by_grid_re.fill(std::numeric_limits<std::uint32_t>::max());
}

inline void append_layout_re(ReLayout& layout, std::uint32_t grid_idx) {
    if (layout.output_slot_by_grid_re[grid_idx] != std::numeric_limits<std::uint32_t>::max()) {
        return;
    }
    layout.grid_indices[layout.n_re] = static_cast<std::uint16_t>(grid_idx);
    layout.output_slot_by_grid_re[grid_idx] = layout.n_re;
    ++layout.n_re;
}

inline std::uint32_t pbch_prb_begin() {
    return kLtePbchStartPrb;
}

inline std::uint32_t pbch_prb_end() {
    return pbch_prb_begin() + kLtePbchNumPrb;
}

inline Complex32 grid_at(const PlanarGridViewF32& grid, std::uint32_t rx, std::uint32_t symbol,
                         std::uint32_t sc) {
    const std::size_t idx = static_cast<std::size_t>(symbol) * grid.n_subcarriers + sc;
    return {finite_or_zero(grid.re[rx][idx]), finite_or_zero(grid.im[rx][idx])};
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
    if (desc.n_rx_ant == 0U || desc.n_rx_ant > kMmseV1MaxNumRxAntennas) {
        return false;
    }
    if (desc.n_tx_ports == 0U || desc.n_tx_ports > kMmseV1MaxNumCrsTxPorts) {
        return false;
    }
    if (desc.n_layers != 1U && desc.n_layers != 2U) {
        return false;
    }
    if (desc.n_layers > desc.n_tx_ports) {
        return false;
    }
    if (desc.n_layers > desc.n_rx_ant) {
        return false;
    }
    if (desc.pmi != -1) {
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
    switch (desc.channel_type) {
    case MmseChannelType::kPdsch:
        if (desc.start_symbol >= kLteNumSymbolsNormalCp) {
            return false;
        }
        if (desc.control_symbol_count > kLteMaxControlSymbolsNormalCp) {
            return false;
        }
        if ((desc.n_tx_ports == 1U && (desc.n_layers != 1U || desc.tx_mode != 1U)) ||
            (desc.n_tx_ports == 2U && desc.tx_mode != 2U)) {
            return false;
        }
        break;
    case MmseChannelType::kPdcch:
        if (desc.control_symbol_count == 0U || !is_supported_lte_downlink_bandwidth(desc.n_prb) ||
            desc.control_symbol_count > lte_max_pdcch_control_symbols_normal_cp(desc.n_prb)) {
            return false;
        }
        if (desc.start_symbol != 0U) {
            return false;
        }
        if (desc.mod_order != 2U) {
            return false;
        }
        if (desc.n_layers != 1U) {
            return false;
        }
        if (desc.n_tx_ports != 1U || desc.tx_mode != 1U) {
            return false;
        }
        for (std::uint32_t prb = 0U; prb < kLteNumPrb20MHz; ++prb) {
            if (bitmap_has_prb(desc, prb) != (prb < desc.n_prb)) {
                return false;
            }
        }
        break;
    case MmseChannelType::kPbch:
        if (desc.start_symbol != kLtePbchStartSymbolNormalCp) {
            return false;
        }
        if (desc.control_symbol_count != 0U) {
            return false;
        }
        if (desc.mod_order != 2U) {
            return false;
        }
        if (desc.n_layers != 1U) {
            return false;
        }
        if (desc.n_tx_ports != 1U || desc.tx_mode != 1U) {
            return false;
        }
        if (desc.n_prb != kLtePbchNumPrb) {
            return false;
        }
        for (std::uint32_t prb = 0; prb < kLteNumPrb20MHz; ++prb) {
            const bool expected = prb >= pbch_prb_begin() && prb < pbch_prb_end();
            if (bitmap_has_prb(desc, prb) != expected) {
                return false;
            }
        }
        break;
    case MmseChannelType::kPcfich:
        if (desc.start_symbol != 0U) {
            return false;
        }
        if (desc.control_symbol_count == 0U ||
            desc.control_symbol_count > kLteMaxControlSymbolsNormalCp) {
            return false;
        }
        if (desc.mod_order != 2U) {
            return false;
        }
        if (desc.n_layers != 1U) {
            return false;
        }
        if (desc.n_tx_ports != 1U || desc.tx_mode != 1U) {
            return false;
        }
        if (desc.n_prb != kLteNumPrb20MHz) {
            return false;
        }
        for (std::uint32_t prb = 0; prb < kLteNumPrb20MHz; ++prb) {
            if (!bitmap_has_prb(desc, prb)) {
                return false;
            }
        }
        break;
    default:
        return false;
    }
    return true;
}

MmseStatus validate_grid(const PlanarGridViewF32& grid, const ExtractDescriptor* desc) {
    std::uint32_t expected_subcarriers = kLteNumSubcarriers20MHz;
    if (desc != nullptr && desc->channel_type == MmseChannelType::kPdcch &&
        is_supported_lte_downlink_bandwidth(desc->n_prb)) {
        expected_subcarriers = lte_downlink_subcarrier_count(desc->n_prb);
    }
    if (grid.n_rx_ant == 0U || grid.n_rx_ant > kMmseV1MaxNumRxAntennas ||
        grid.n_symbols != kLteNumSymbolsNormalCp || grid.n_subcarriers != expected_subcarriers) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::uint32_t rx = 0; rx < grid.n_rx_ant; ++rx) {
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

PreparedSubframeKey make_prepared_subframe_key(const PlanarGridViewF32& grid,
                                               const ExtractDescriptor& desc,
                                               std::uint32_t backend_mode) {
    PreparedSubframeKey key{};
    key.re = grid.re;
    key.im = grid.im;
    key.sfn_subframe = desc.sfn_subframe;
    key.cell_id = desc.cell_id;
    key.n_tx_ports = desc.n_tx_ports;
    key.n_rx_ant = desc.n_rx_ant;
    key.n_symbols = grid.n_symbols;
    key.n_subcarriers = grid.n_subcarriers;
    key.generation = grid.generation;
    key.backend_mode = backend_mode;
    return key;
}

bool prepared_subframe_key_equal(const PreparedSubframeKey& lhs, const PreparedSubframeKey& rhs) {
    return lhs.re == rhs.re && lhs.im == rhs.im && lhs.sfn_subframe == rhs.sfn_subframe &&
           lhs.cell_id == rhs.cell_id && lhs.n_tx_ports == rhs.n_tx_ports &&
           lhs.n_rx_ant == rhs.n_rx_ant && lhs.n_symbols == rhs.n_symbols &&
           lhs.n_subcarriers == rhs.n_subcarriers && lhs.generation == rhs.generation &&
           lhs.backend_mode == rhs.backend_mode;
}

std::uint32_t channel_start_symbol(const ExtractDescriptor& desc) {
    switch (desc.channel_type) {
    case MmseChannelType::kPdsch:
        return desc.start_symbol;
    case MmseChannelType::kPdcch:
    case MmseChannelType::kPcfich:
        return 0U;
    case MmseChannelType::kPbch:
        return kLtePbchStartSymbolNormalCp;
    default:
        return desc.start_symbol;
    }
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

bool is_crs_re(const ExtractDescriptor& desc, std::uint8_t symbol, std::uint32_t sc) {
    if (symbol != 0U && symbol != 4U && symbol != 7U && symbol != 11U) {
        return false;
    }
    for (std::uint32_t port = 0; port < desc.n_tx_ports; ++port) {
        if (sc % 6U ==
            crs_frequency_offset(desc.cell_id, static_cast<std::uint8_t>(port), symbol) % 6U) {
            return true;
        }
    }
    return false;
}

const Complex32& crs_value(const CrsTableKey& key, std::uint32_t pilot_index) {
    return g_crs_table[crs_table_index(key.cell_id, key.subframe, key.port, key.crs_symbol_index,
                                       pilot_index)];
}

void ensure_crs_tables() {
    std::call_once(g_crs_table_once, []() {
        for (std::uint16_t cell_id = 0; cell_id < kLteNumCellIds; ++cell_id) {
            for (std::uint8_t subframe = 0; subframe < 10U; ++subframe) {
                for (std::uint8_t port = 0; port < kMmseV1MaxNumCrsTxPorts; ++port) {
                    for (std::uint8_t symbol_idx = 0; symbol_idx < kLteNumCrsSymbols;
                         ++symbol_idx) {
                        const std::uint8_t symbol = kCrsSymbols[symbol_idx];
                        const std::uint32_t slot = 2U * subframe + (symbol >= 7U ? 1U : 0U);
                        const std::uint32_t slot_symbol = symbol % 7U;
                        const std::uint32_t c_init = (1U << 10U) *
                                                         (7U * (slot + 1U) + slot_symbol + 1U) *
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
    reset_layout(layout);

    for (std::uint32_t symbol = desc.start_symbol; symbol < kLteNumSymbolsNormalCp; ++symbol) {
        std::uint32_t segment_begin = layout.n_re;
        for (std::uint32_t prb = 0; prb < kLteNumPrb20MHz; ++prb) {
            if (!bitmap_has_prb(desc, prb)) {
                continue;
            }
            for (std::uint32_t tone = 0; tone < 12U; ++tone) {
                const std::uint32_t sc = prb * 12U + tone;
                if (is_crs_re(desc, static_cast<std::uint8_t>(symbol), sc)) {
                    continue;
                }
                const std::uint32_t grid_idx = symbol * kLteNumSubcarriers20MHz + sc;
                append_layout_re(layout, grid_idx);
            }
        }
        if (layout.n_re > segment_begin) {
            layout.prb_segment_offsets[layout.n_segments++] = segment_begin;
        }
    }
    layout.prb_segment_offsets[layout.n_segments] = layout.n_re;
    return layout.n_re;
}

std::uint32_t build_pdcch_re_layout(const ExtractDescriptor& desc, ReLayout& layout) {
    reset_layout(layout);

    using PdcchRegIndices = std::array<std::uint16_t, mmse::pdcch::kPdcchRePerReg>;
    constexpr std::uint32_t kPdcchPermutationColumns = 32U;
    constexpr std::array<std::uint8_t, kPdcchPermutationColumns> kPdcchPermutation = {
        1U, 17U, 9U, 25U, 5U, 21U, 13U, 29U, 3U, 19U, 11U, 27U, 7U, 23U, 15U, 31U,
        0U, 16U, 8U, 24U, 4U, 20U, 12U, 28U, 2U, 18U, 10U, 26U, 6U, 22U, 14U, 30U,
    };
    std::array<PdcchRegIndices, kMaxDataRe / mmse::pdcch::kPdcchRePerReg> physical_regs{};
    std::array<std::uint32_t, kMaxDataRe / mmse::pdcch::kPdcchRePerReg> interleaved_reg_indices{};
    std::uint32_t n_physical_regs = 0U;
    const std::uint32_t n_subcarriers = lte_downlink_subcarrier_count(desc.n_prb);

    const auto append_reg = [&](std::uint32_t symbol, std::uint32_t prb,
                                const PdcchRegIndices& tones) {
        PdcchRegIndices grid_indices{};
        for (std::uint32_t re_in_reg = 0U; re_in_reg < tones.size(); ++re_in_reg) {
            const std::uint32_t tone = tones[re_in_reg];
            if (control_re_excluded(desc, symbol, prb, tone)) {
                return;
            }
            grid_indices[re_in_reg] = static_cast<std::uint16_t>(
                symbol * n_subcarriers + prb * kLteNumSubcarriersPerPrb + tone);
        }
        physical_regs[n_physical_regs++] = grid_indices;
    };

    constexpr std::array<PdcchRegIndices, 3> kSymbol0FirstRegTones = {
        PdcchRegIndices{1U, 2U, 4U, 5U},
        PdcchRegIndices{0U, 2U, 3U, 5U},
        PdcchRegIndices{0U, 1U, 3U, 4U},
    };
    const PdcchRegIndices symbol0_reg0 = kSymbol0FirstRegTones[desc.cell_id % 3U];
    PdcchRegIndices symbol0_reg1{};
    for (std::uint32_t re_in_reg = 0U; re_in_reg < symbol0_reg1.size(); ++re_in_reg) {
        symbol0_reg1[re_in_reg] = static_cast<std::uint16_t>(symbol0_reg0[re_in_reg] + 6U);
    }
    constexpr std::array<PdcchRegIndices, 3> kLaterSymbolRegs = {
        PdcchRegIndices{0U, 1U, 2U, 3U},
        PdcchRegIndices{4U, 5U, 6U, 7U},
        PdcchRegIndices{8U, 9U, 10U, 11U},
    };

    for (std::uint32_t prb = 0U; prb < desc.n_prb; ++prb) {
        append_reg(0U, prb, symbol0_reg0);
        for (std::uint32_t reg = 0U; reg < 2U; ++reg) {
            for (std::uint32_t symbol = 1U; symbol < desc.control_symbol_count; ++symbol) {
                append_reg(symbol, prb, kLaterSymbolRegs[reg]);
            }
        }
        append_reg(0U, prb, symbol0_reg1);
        for (std::uint32_t symbol = 1U; symbol < desc.control_symbol_count; ++symbol) {
            append_reg(symbol, prb, kLaterSymbolRegs[2]);
        }
    }

    if (n_physical_regs == 0U) {
        layout.prb_segment_offsets[0] = 0U;
        return 0U;
    }

    const std::uint32_t n_rows =
        (n_physical_regs + kPdcchPermutationColumns - 1U) / kPdcchPermutationColumns;
    const std::uint32_t n_dummy = kPdcchPermutationColumns * n_rows - n_physical_regs;
    std::uint32_t cyclic_shift_source_reg = 0U;
    for (std::uint32_t column = 0U; column < kPdcchPermutationColumns; ++column) {
        for (std::uint32_t row = 0U; row < n_rows; ++row) {
            const std::uint32_t matrix_index =
                row * kPdcchPermutationColumns + kPdcchPermutation[column];
            if (matrix_index < n_dummy) {
                continue;
            }
            const std::uint32_t mapped_reg = matrix_index - n_dummy;
            interleaved_reg_indices[mapped_reg] =
                (n_physical_regs + cyclic_shift_source_reg - (desc.cell_id % n_physical_regs)) %
                n_physical_regs;
            ++cyclic_shift_source_reg;
        }
    }

    const std::uint32_t n_mapped_regs =
        (n_physical_regs / mmse::pdcch::kPdcchRegsPerCce) * mmse::pdcch::kPdcchRegsPerCce;
    for (std::uint32_t mapped_reg = 0U; mapped_reg < n_mapped_regs; ++mapped_reg) {
        const PdcchRegIndices& reg = physical_regs[interleaved_reg_indices[mapped_reg]];
        for (const std::uint16_t grid_idx : reg) {
            append_layout_re(layout, grid_idx);
        }
    }

    layout.n_segments = layout.n_re == 0U ? 0U : 1U;
    layout.prb_segment_offsets[0] = 0U;
    layout.prb_segment_offsets[layout.n_segments] = layout.n_re;
    return layout.n_re;
}

std::uint32_t build_pbch_re_layout(const ExtractDescriptor& desc, ReLayout& layout) {
    reset_layout(layout);

    for (std::uint32_t symbol = kLtePbchStartSymbolNormalCp;
         symbol < kLtePbchStartSymbolNormalCp + kLtePbchNumSymbols; ++symbol) {
        const std::uint32_t segment_begin = layout.n_re;
        for (std::uint32_t prb = pbch_prb_begin(); prb < pbch_prb_end(); ++prb) {
            for (std::uint32_t tone = 0; tone < kLteNumSubcarriersPerPrb; ++tone) {
                const std::uint32_t sc = prb * kLteNumSubcarriersPerPrb + tone;
                const bool pbch_symbol_uses_crs_holes = symbol == kLtePbchStartSymbolNormalCp ||
                                                        symbol == kLtePbchStartSymbolNormalCp + 1U;
                const std::uint32_t v_shift = desc.cell_id % 6U;
                const bool pbch_reserved_for_crs =
                    pbch_symbol_uses_crs_holes &&
                    (sc % 6U == v_shift || sc % 6U == ((v_shift + 3U) % 6U));
                if (pbch_reserved_for_crs) {
                    continue;
                }
                const std::uint32_t grid_idx = symbol * kLteNumSubcarriers20MHz + sc;
                append_layout_re(layout, grid_idx);
            }
        }
        if (layout.n_re > segment_begin) {
            layout.prb_segment_offsets[layout.n_segments++] = segment_begin;
        }
    }

    layout.prb_segment_offsets[layout.n_segments] = layout.n_re;
    return layout.n_re;
}

std::uint32_t build_pcfich_re_layout(const ExtractDescriptor& desc, ReLayout& layout) {
    reset_layout(layout);

    const auto regs = mmse::pdcch::detail::pcfich_reg_coords(desc.cell_id);
    const std::uint32_t vo = desc.cell_id % 3U;
    const std::uint32_t segment_begin = layout.n_re;
    for (const auto& reg : regs) {
        const std::uint32_t tone_base = reg.reg_in_symbol_prb * 6U;
        for (std::uint32_t local_tone = 0; local_tone < 6U; ++local_tone) {
            if (local_tone == vo || local_tone == vo + 3U) {
                continue;
            }
            const std::uint32_t tone = tone_base + local_tone;
            const std::uint32_t grid_idx = reg.prb * kLteNumSubcarriersPerPrb + tone;
            append_layout_re(layout, grid_idx);
        }
    }
    if (layout.n_re > segment_begin) {
        layout.prb_segment_offsets[layout.n_segments++] = segment_begin;
    }
    layout.prb_segment_offsets[layout.n_segments] = layout.n_re;
    return layout.n_re;
}

std::uint32_t build_channel_re_layout(const ExtractDescriptor& desc, ReLayout& layout) {
    switch (desc.channel_type) {
    case MmseChannelType::kPdsch:
        return build_data_re_layout(desc, layout);
    case MmseChannelType::kPdcch:
        return build_pdcch_re_layout(desc, layout);
    case MmseChannelType::kPbch:
        return build_pbch_re_layout(desc, layout);
    case MmseChannelType::kPcfich:
        return build_pcfich_re_layout(desc, layout);
    default:
        reset_layout(layout);
        layout.prb_segment_offsets[0] = 0U;
        return 0U;
    }
}

std::uint32_t build_validation_re_samples(const ReLayout& layout, std::uint32_t start_symbol,
                                          std::uint32_t n_symbols, std::uint32_t n_subcarriers,
                                          std::uint32_t* out_re_slots, std::uint32_t max_slots) {
    if (out_re_slots == nullptr || max_slots == 0U || layout.n_re == 0U) {
        return 0U;
    }

    std::uint32_t count = 0U;
    const auto try_add = [&](std::uint32_t re_slot) {
        if (re_slot >= layout.n_re) {
            return;
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            if (out_re_slots[i] == re_slot) {
                return;
            }
        }
        if (count < max_slots) {
            out_re_slots[count++] = re_slot;
        }
    };

    try_add(0U);
    try_add(layout.n_re / 2U);
    try_add(layout.n_re - 1U);

    if (n_symbols == 0U || n_subcarriers == 0U) {
        return count;
    }

    constexpr std::array<std::uint32_t, 3> kSymbolOffsets = {0U, 6U, 12U};
    constexpr std::array<std::uint32_t, 4> kSubcarrierOffsets = {0U, 127U, 599U, 1199U};
    for (std::uint32_t symbol_offset : kSymbolOffsets) {
        const std::uint32_t symbol = start_symbol + symbol_offset;
        if (symbol >= n_symbols) {
            continue;
        }
        for (std::uint32_t sc : kSubcarrierOffsets) {
            if (sc >= n_subcarriers) {
                continue;
            }
            const std::uint32_t grid_idx = symbol * n_subcarriers + sc;
            const std::uint32_t re_slot = layout.output_slot_by_grid_re[grid_idx];
            if (re_slot != std::numeric_limits<std::uint32_t>::max()) {
                try_add(re_slot);
            }
        }
    }
    return count;
}

void estimate_channel(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                      HGridStorage& h_full, float& sigma2_estimate) {
    ++g_estimate_channel_call_count;
    ensure_crs_tables();
    h_full.fill(Complex32{});
    sigma2_estimate = 0.0F;
    std::uint32_t residual_count = 0U;
    const std::uint32_t subframe = subframe_from_descriptor(desc);
    const std::uint32_t n_subcarriers = grid.n_subcarriers;
    const std::uint32_t n_pilot_tones = n_subcarriers / 6U;

    for (std::uint32_t tx = 0; tx < desc.n_tx_ports; ++tx) {
        for (std::uint32_t rx = 0; rx < desc.n_rx_ant; ++rx) {
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

                for (std::uint32_t pilot = 0; pilot < n_pilot_tones; ++pilot) {
                    const std::uint32_t sc =
                        crs_subcarrier(desc.cell_id, static_cast<std::uint8_t>(tx), symbol, pilot);
                    ls[cs][pilot] =
                        cmul(grid_at(grid, rx, symbol, sc), cconj(crs_value(key, pilot)));
                }

                for (std::uint32_t pilot = 1U; pilot + 1U < n_pilot_tones; ++pilot) {
                    const Complex32 smooth =
                        cscale(cadd(ls[cs][pilot - 1U], ls[cs][pilot + 1U]), 0.5F);
                    sigma2_estimate += (2.0F / 3.0F) * cnorm2(csub(ls[cs][pilot], smooth));
                    ++residual_count;
                }

                for (std::uint32_t sc = 0; sc < n_subcarriers; ++sc) {
                    const std::uint32_t offset =
                        crs_frequency_offset(desc.cell_id, static_cast<std::uint8_t>(tx), symbol);
                    const std::uint32_t lower_pilot =
                        sc < offset ? 0U : std::min((sc - offset) / 6U, n_pilot_tones - 1U);
                    const bool below_first = sc < offset;
                    const bool above_last =
                        sc > crs_subcarrier(desc.cell_id, static_cast<std::uint8_t>(tx), symbol,
                                            n_pilot_tones - 1U);
                    const std::uint32_t left_pilot =
                        below_first ? 0U : (above_last ? n_pilot_tones - 2U : lower_pilot);
                    const std::uint32_t right_pilot =
                        below_first ? 1U
                                    : (above_last ? n_pilot_tones - 1U
                                                  : std::min(lower_pilot + 1U, n_pilot_tones - 1U));
                    const std::uint32_t left_sc = crs_subcarrier(
                        desc.cell_id, static_cast<std::uint8_t>(tx), symbol, left_pilot);
                    const std::uint32_t right_sc = crs_subcarrier(
                        desc.cell_id, static_cast<std::uint8_t>(tx), symbol, right_pilot);
                    if (left_sc == right_sc) {
                        freq[cs][sc] = ls[cs][left_pilot];
                    } else {
                        const float t = static_cast<float>(static_cast<std::int32_t>(sc) -
                                                           static_cast<std::int32_t>(left_sc)) /
                                        static_cast<float>(right_sc - left_sc);
                        freq[cs][sc] = linear_interp(ls[cs][left_pilot], ls[cs][right_pilot], t);
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
                    lower = kLteNumCrsSymbols - 2U;
                    upper = kLteNumCrsSymbols - 1U;
                }
                for (std::uint32_t sc = 0; sc < n_subcarriers; ++sc) {
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
        if (!std::isfinite(sigma2_estimate) || sigma2_estimate < 0.0F) {
            sigma2_estimate = kDefaultSigma2Min;
        }
    }
}

float update_sigma2_state(Sigma2State& state, float sigma2_estimate,
                          const MmseEqualizerCpuConfig& config) {
    if (!std::isfinite(sigma2_estimate) || sigma2_estimate < 0.0F) {
        sigma2_estimate = config.sigma2_min;
    }
    sigma2_estimate = std::max(sigma2_estimate, config.sigma2_min);
    if (!state.initialized) {
        state.value = sigma2_estimate;
        state.initialized = true;
        return state.value;
    }
    state.value =
        config.sigma2_iir_alpha * state.value + (1.0F - config.sigma2_iir_alpha) * sigma2_estimate;
    state.value =
        std::isfinite(state.value) ? std::max(state.value, config.sigma2_min) : config.sigma2_min;
    return state.value;
}

float peek_sigma2_state(const Sigma2State& state, float sigma2_min) {
    return state.initialized && std::isfinite(state.value) ? std::max(state.value, sigma2_min)
                                                           : sigma2_min;
}

void debug_reset_estimate_channel_call_count() {
    g_estimate_channel_call_count = 0;
}

std::uint64_t debug_get_estimate_channel_call_count() {
    return g_estimate_channel_call_count;
}

void pack_equalizer_inputs(const PlanarGridViewF32& grid, const HGridStorage& h_full,
                           const ReLayout& layout, PackedEqualizerInputs& packed) {
    for (std::uint32_t re = 0; re < layout.n_re; ++re) {
        const std::uint32_t grid_idx = layout.grid_indices[re];
        const std::uint32_t symbol = grid_idx / grid.n_subcarriers;
        const std::uint32_t sc = grid_idx % grid.n_subcarriers;

        const Complex32 h00 = h_full[h_index(0U, 0U, symbol, sc)];
        const Complex32 h01 = h_full[h_index(1U, 0U, symbol, sc)];
        const Complex32 h10 =
            grid.n_rx_ant > 1U ? h_full[h_index(0U, 1U, symbol, sc)] : Complex32{};
        const Complex32 h11 =
            grid.n_rx_ant > 1U ? h_full[h_index(1U, 1U, symbol, sc)] : Complex32{};
        const Complex32 y0 = grid_at(grid, 0U, symbol, sc);
        const Complex32 y1 = grid.n_rx_ant > 1U ? grid_at(grid, 1U, symbol, sc) : Complex32{};

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

Equalize2x2Trace trace_equalize_2x2_scalar(Complex32 h00, Complex32 h01, Complex32 h10,
                                           Complex32 h11, Complex32 y0, Complex32 y1, float sigma2,
                                           float det_floor, float g_min, float gamma_max) {
    Equalize2x2Trace trace{};
    trace.a11 = cnorm2(h00) + cnorm2(h10) + sigma2;
    trace.a22 = cnorm2(h01) + cnorm2(h11) + sigma2;
    trace.a12 = cadd(cmul(cconj(h00), h01), cmul(cconj(h10), h11));
    trace.det = std::max(trace.a11 * trace.a22 - cnorm2(trace.a12), det_floor);
    trace.inv_det = 1.0F / trace.det;

    trace.inv11 = trace.a22 * trace.inv_det;
    trace.inv22 = trace.a11 * trace.inv_det;
    trace.inv12 = cscale(trace.a12, -trace.inv_det);
    trace.inv21 = cconj(trace.inv12);

    trace.hh00 = cconj(h00);
    trace.hh01 = cconj(h10);
    trace.hh10 = cconj(h01);
    trace.hh11 = cconj(h11);

    trace.w00 = cadd(cscale(trace.hh00, trace.inv11), cmul(trace.inv12, trace.hh10));
    trace.w01 = cadd(cscale(trace.hh01, trace.inv11), cmul(trace.inv12, trace.hh11));
    trace.w10 = cadd(cmul(trace.inv21, trace.hh00), cscale(trace.hh10, trace.inv22));
    trace.w11 = cadd(cmul(trace.inv21, trace.hh01), cscale(trace.hh11, trace.inv22));

    trace.z0 = cadd(cmul(trace.w00, y0), cmul(trace.w01, y1));
    trace.z1 = cadd(cmul(trace.w10, y0), cmul(trace.w11, y1));

    trace.g0 = clampf(cadd(cmul(trace.w00, h00), cmul(trace.w01, h10)).re, g_min, 1.0F - g_min);
    trace.g1 = clampf(cadd(cmul(trace.w10, h01), cmul(trace.w11, h11)).re, g_min, 1.0F - g_min);

    trace.xhat0 = cscale(trace.z0, 1.0F / trace.g0);
    trace.xhat1 = cscale(trace.z1, 1.0F / trace.g1);
    trace.gamma0 = clampf(trace.g0 / (1.0F - trace.g0), g_min, gamma_max);
    trace.gamma1 = clampf(trace.g1 / (1.0F - trace.g1), g_min, gamma_max);
    return trace;
}

EqualizedSymbol equalize_2x2_scalar(Complex32 h00, Complex32 h01, Complex32 h10, Complex32 h11,
                                    Complex32 y0, Complex32 y1, float sigma2, float det_floor,
                                    float g_min, float gamma_max, std::uint8_t layer_index) {
    const Equalize2x2Trace trace =
        trace_equalize_2x2_scalar(h00, h01, h10, h11, y0, y1, sigma2, det_floor, g_min, gamma_max);
    const Complex32 xhat = (layer_index == 0U) ? trace.xhat0 : trace.xhat1;
    const float gamma =
        (layer_index == 0U) ? trace.g0 / (1.0F - trace.g0) : trace.g1 / (1.0F - trace.g1);
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
    struct PreparedSubframeState {
        detail::PreparedSubframeKey key{};
        bool valid = false;
        float sigma2 = 0.0F;
        std::uint32_t prepared_subframe = 0;
        std::uint16_t prepared_cell_id = 0;
        std::uint8_t prepared_n_tx_ports = 0;
    };

    struct RunTaskContext {
        Impl* impl = nullptr;
        ExtractDescriptor desc{};
        EqualizerOutputView* out = nullptr;
        float sigma2 = 0.0F;
    };

    static void worker_task(void* raw_ctx, std::uint32_t begin, std::uint32_t end);
    MmseStatus prepare_subframe_if_needed(const PlanarGridViewF32& grid,
                                          const ExtractDescriptor& desc, float& sigma2);
    MmseStatus run_channel_from_prepared_estimate(const PlanarGridViewF32& grid,
                                                  const ExtractDescriptor& desc,
                                                  EqualizerOutputView& out, float sigma2);

    MmseEqualizerCpuConfig config{};
    bool initialized = false;
    bool use_avx2 = false;
    HGridStorage h_full{};
    detail::PackedEqualizerInputs packed{};
    ReLayout layout{};
    PreparedSubframeState prepared{};
    std::array<Sigma2State, kLteNumCellIds> sigma2_by_cell{};
    StaticThreadPool pool{};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges{};
};

namespace {

struct TransmitDiversityRePair {
    std::uint16_t grid_index0 = 0;
    std::uint16_t grid_index1 = 0;
};

inline float td_clampf(float value, float lo, float hi) {
    if (!std::isfinite(value)) {
        return lo;
    }
    return std::min(std::max(value, lo), hi);
}

inline std::size_t td_h_index(std::uint32_t tx, std::uint32_t rx, std::uint32_t symbol,
                              std::uint32_t sc) {
    return (((static_cast<std::size_t>(tx) * kMmseV1MaxNumRxAntennas + rx) *
                 kLteNumSymbolsNormalCp +
             symbol) *
            kLteNumSubcarriers20MHz) +
           sc;
}

inline Complex32 td_grid_at(const PlanarGridViewF32& grid, std::uint32_t rx, std::uint32_t symbol,
                            std::uint32_t sc) {
    const std::size_t idx = static_cast<std::size_t>(symbol) * grid.n_subcarriers + sc;
    return {grid.re[rx][idx], grid.im[rx][idx]};
}

bool build_transmit_diversity_re_pairs(const ReLayout& layout,
                                       std::vector<TransmitDiversityRePair>& pairs) {
    pairs.clear();
    if ((layout.n_re & 1U) != 0U) {
        return false;
    }
    pairs.reserve(layout.n_re / 2U);
    for (std::uint32_t i = 0; i < layout.n_re; i += 2U) {
        const std::uint16_t grid0 = layout.grid_indices[i];
        const std::uint16_t grid1 = layout.grid_indices[i + 1U];
        pairs.push_back({grid0, grid1});
    }
    return true;
}

detail::TransmitDiversityEqualizePair
demap_transmit_diversity_from_grid(const HGridStorage& h_full, const PlanarGridViewF32& grid,
                                   std::uint16_t grid_index0, std::uint16_t grid_index1,
                                   float sigma2, float det_floor, float g_min, float gamma_max);

} // namespace

namespace detail {

TransmitDiversityEqualizePair
demap_transmit_diversity_mmse_scalar(Complex32 h00, Complex32 h01, Complex32 h10, Complex32 h11,
                                     Complex32 h00_next, Complex32 h01_next, Complex32 h10_next,
                                     Complex32 h11_next, Complex32 y0, Complex32 y1,
                                     Complex32 y0_next, Complex32 y1_next, float sigma2,
                                     float det_floor, float g_min, float gamma_max) {
    const std::array<Complex32, 4> a0 = {h00, h10, cscale(cconj(h01_next), -1.0F),
                                         cscale(cconj(h11_next), -1.0F)};
    const std::array<Complex32, 4> a1 = {h01, h11, cconj(h00_next), cconj(h10_next)};
    const std::array<Complex32, 4> observations = {y0, y1, cconj(y0_next), cconj(y1_next)};

    float a00 = std::max(sigma2, 0.0F);
    float a11 = std::max(sigma2, 0.0F);
    Complex32 a01{};
    Complex32 z0{};
    Complex32 z1{};
    for (std::size_t row = 0; row < observations.size(); ++row) {
        a00 += cnorm2(a0[row]);
        a11 += cnorm2(a1[row]);
        a01 = cadd(a01, cmul(cconj(a0[row]), a1[row]));
        z0 = cadd(z0, cmul(cconj(a0[row]), observations[row]));
        z1 = cadd(z1, cmul(cconj(a1[row]), observations[row]));
    }
    const float det = std::max(a00 * a11 - cnorm2(a01), det_floor);
    const float inv_det = 1.0F / det;
    const float inv00 = a11 * inv_det;
    const float inv11 = a00 * inv_det;
    const Complex32 inv01 = cscale(a01, -inv_det);
    const Complex32 inv10 = cconj(inv01);
    const Complex32 x0 = cadd(cscale(z0, inv00), cmul(inv01, z1));
    const Complex32 x1 = cadd(cmul(inv10, z0), cscale(z1, inv11));

    const float regularization = std::max(sigma2, 0.0F);
    float gain0 = 1.0F - regularization * inv00;
    float gain1 = 1.0F - regularization * inv11;
    gain0 = td_clampf(gain0, g_min, 1.0F - g_min);
    gain1 = td_clampf(gain1, g_min, 1.0F - g_min);
    return {{cscale(x0, 1.0F / gain0), td_clampf(gain0 / (1.0F - gain0), g_min, gamma_max)},
            {cscale(x1, 1.0F / gain1), td_clampf(gain1 / (1.0F - gain1), g_min, gamma_max)}};
}

TransmitDiversityEqualizePair
demap_pdcch_transmit_diversity_scalar(Complex32 h00, Complex32 h01, Complex32 h10, Complex32 h11,
                                      Complex32 h00_next, Complex32 h01_next, Complex32 h10_next,
                                      Complex32 h11_next, Complex32 y0, Complex32 y1,
                                      Complex32 y0_next, Complex32 y1_next, float sigma2,
                                      float det_floor, float g_min, float gamma_max) {
    return demap_transmit_diversity_mmse_scalar(h00, h01, h10, h11, h00_next, h01_next, h10_next,
                                                h11_next, y0, y1, y0_next, y1_next, sigma2,
                                                det_floor, g_min, gamma_max);
}

} // namespace detail

namespace {

detail::TransmitDiversityEqualizePair
demap_transmit_diversity_from_grid(const HGridStorage& h_full, const PlanarGridViewF32& grid,
                                   std::uint16_t grid_index0, std::uint16_t grid_index1,
                                   float sigma2, float det_floor, float g_min, float gamma_max) {
    const std::uint32_t symbol0 = grid_index0 / grid.n_subcarriers;
    const std::uint32_t sc0 = grid_index0 % grid.n_subcarriers;
    const std::uint32_t symbol1 = grid_index1 / grid.n_subcarriers;
    const std::uint32_t sc1 = grid_index1 % grid.n_subcarriers;
    return detail::demap_transmit_diversity_mmse_scalar(
        h_full[td_h_index(0U, 0U, symbol0, sc0)], h_full[td_h_index(1U, 0U, symbol0, sc0)],
        grid.n_rx_ant > 1U ? h_full[td_h_index(0U, 1U, symbol0, sc0)] : Complex32{},
        grid.n_rx_ant > 1U ? h_full[td_h_index(1U, 1U, symbol0, sc0)] : Complex32{},
        h_full[td_h_index(0U, 0U, symbol1, sc1)], h_full[td_h_index(1U, 0U, symbol1, sc1)],
        grid.n_rx_ant > 1U ? h_full[td_h_index(0U, 1U, symbol1, sc1)] : Complex32{},
        grid.n_rx_ant > 1U ? h_full[td_h_index(1U, 1U, symbol1, sc1)] : Complex32{},
        td_grid_at(grid, 0U, symbol0, sc0),
        grid.n_rx_ant > 1U ? td_grid_at(grid, 1U, symbol0, sc0) : Complex32{},
        td_grid_at(grid, 0U, symbol1, sc1),
        grid.n_rx_ant > 1U ? td_grid_at(grid, 1U, symbol1, sc1) : Complex32{}, sigma2, det_floor,
        g_min, gamma_max);
}

MmseStatus
equalize_transmit_diversity_pairs(const HGridStorage& h_full, const PlanarGridViewF32& grid,
                                  const std::vector<TransmitDiversityRePair>& pairs, float sigma2,
                                  float det_floor, float g_min, float gamma_max, float* x_hat_re,
                                  float* x_hat_im, float* sinr, std::uint16_t* indices0,
                                  std::uint16_t* indices1, std::uint32_t capacity_symbols) {
    const std::uint32_t n_symbols = static_cast<std::uint32_t>(pairs.size() * 2U);
    if (x_hat_re == nullptr || x_hat_im == nullptr || sinr == nullptr || indices0 == nullptr ||
        indices1 == nullptr) {
        return MmseStatus::kInvalidArgument;
    }
    if (capacity_symbols < n_symbols) {
        return MmseStatus::kBufferTooSmall;
    }
    for (std::uint32_t i = 0; i < pairs.size(); ++i) {
        const TransmitDiversityRePair& pair = pairs[i];
        const detail::TransmitDiversityEqualizePair eq = demap_transmit_diversity_from_grid(
            h_full, grid, pair.grid_index0, pair.grid_index1, sigma2, det_floor, g_min, gamma_max);
        const std::uint32_t base = i * 2U;
        x_hat_re[base] = eq.symbol0.xhat.re;
        x_hat_im[base] = eq.symbol0.xhat.im;
        sinr[base] = eq.symbol0.gamma;
        indices0[base] = pair.grid_index0;
        indices1[base] = pair.grid_index1;
        x_hat_re[base + 1U] = eq.symbol1.xhat.re;
        x_hat_im[base + 1U] = eq.symbol1.xhat.im;
        sinr[base + 1U] = eq.symbol1.gamma;
        indices0[base + 1U] = pair.grid_index0;
        indices1[base + 1U] = pair.grid_index1;
    }
    return MmseStatus::kOk;
}

} // namespace

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

MmseStatus MmseEqualizerCpuContext::Impl::prepare_subframe_if_needed(const PlanarGridViewF32& grid,
                                                                     const ExtractDescriptor& desc,
                                                                     float& sigma2) {
    const detail::PreparedSubframeKey key = detail::make_prepared_subframe_key(grid, desc);
    if (prepared.valid && detail::prepared_subframe_key_equal(prepared.key, key)) {
        sigma2 = prepared.sigma2;
        return MmseStatus::kOk;
    }

    float sigma2_estimate = 0.0F;
    detail::estimate_channel(grid, desc, h_full, sigma2_estimate);
    sigma2 = detail::update_sigma2_state(sigma2_by_cell[desc.cell_id], sigma2_estimate, config);
    prepared.key = key;
    prepared.valid = true;
    prepared.sigma2 = sigma2;
    prepared.prepared_subframe = desc.sfn_subframe;
    prepared.prepared_cell_id = desc.cell_id;
    prepared.prepared_n_tx_ports = desc.n_tx_ports;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerCpuContext::Impl::run_channel_from_prepared_estimate(
    const PlanarGridViewF32& grid, const ExtractDescriptor& desc, EqualizerOutputView& out,
    float sigma2) {
    const std::uint32_t n_re = detail::build_channel_re_layout(desc, layout);
    if (out.capacity_re_per_layer < n_re) {
        return MmseStatus::kBufferTooSmall;
    }

    detail::pack_equalizer_inputs(grid, h_full, layout, packed);

    out.n_re_per_layer = n_re;
    out.n_layers = desc.n_layers;
    out.mod_order = desc.mod_order;

    RunTaskContext worker_ctx{
        .impl = this,
        .desc = desc,
        .out = &out,
        .sigma2 = sigma2,
    };

    const std::uint32_t workers = chunk_count(pool.worker_count(), n_re == 0U ? 1U : n_re);
    const std::uint32_t chunk = (n_re + workers - 1U) / workers;
    for (std::uint32_t w = 0; w < workers; ++w) {
        const std::uint32_t begin = w * chunk;
        const std::uint32_t end = std::min(begin + chunk, n_re);
        ranges[w] = {begin, end};
    }
    for (std::uint32_t w = workers; w < ranges.size(); ++w) {
        ranges[w] = {0U, 0U};
    }

    pool.parallel_for(
        std::span<const std::pair<std::uint32_t, std::uint32_t>>(ranges.data(), workers),
        Impl::worker_task, &worker_ctx);
    return MmseStatus::kOk;
}

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
    if (const MmseStatus status = detail::validate_grid(grid, &desc); status != MmseStatus::kOk) {
        return status;
    }
    if (const MmseStatus status = detail::validate_output(out); status != MmseStatus::kOk) {
        return status;
    }
    if (!detail::descriptor_supported(desc)) {
        return MmseStatus::kUnsupportedConfig;
    }

    float sigma2 = 0.0F;
    if (const MmseStatus status = impl_->prepare_subframe_if_needed(grid, desc, sigma2);
        status != MmseStatus::kOk) {
        return status;
    }
    return impl_->run_channel_from_prepared_estimate(grid, desc, out, sigma2);
}

MmseStatus MmseEqualizerCpuContext::run_pbch(const PbchMmseInput& in, PbchMmseOutputView& out,
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

    for (std::uint32_t i = 0; i < base_out.n_re_per_layer; ++i) {
        out.re_grid_indices[i] = impl_->layout.grid_indices[i];
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
    meta.sigma2 =
        detail::peek_sigma2_state(impl_->sigma2_by_cell[in.cell_id], impl_->config.sigma2_min);
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerCpuContext::run_pbch_td(const PbchMmseInput& in, PbchTdMmseOutputView& out,
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

    ExtractDescriptor desc{};
    desc.sfn_subframe = in.sfn_subframe;
    desc.cell_id = in.cell_id;
    desc.n_tx_ports = in.n_tx_ports;
    desc.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = in.tx_mode;
    desc.channel_type = MmseChannelType::kPbch;
    desc.start_symbol = kLtePbchStartSymbolNormalCp;
    desc.mod_order = 2U;
    desc.n_prb = kLtePbchNumPrb;
    for (std::uint32_t prb = kLtePbchStartPrb; prb < kLtePbchStartPrb + kLtePbchNumPrb; ++prb) {
        desc.prb_bitmap[prb / 16U] |= static_cast<std::uint16_t>(1U << (prb % 16U));
    }
    if (const MmseStatus status = detail::validate_grid(in.grid); status != MmseStatus::kOk) {
        return status;
    }

    float sigma2 = 0.0F;
    if (const MmseStatus status = impl_->prepare_subframe_if_needed(in.grid, desc, sigma2);
        status != MmseStatus::kOk) {
        return status;
    }
    const std::uint32_t n_source_re = detail::build_pbch_re_layout(desc, impl_->layout);
    std::vector<TransmitDiversityRePair> pairs{};
    if (!build_transmit_diversity_re_pairs(impl_->layout, pairs)) {
        return MmseStatus::kUnsupportedConfig;
    }
    const std::uint32_t n_symbols = static_cast<std::uint32_t>(pairs.size() * 2U);
    if (const MmseStatus status = equalize_transmit_diversity_pairs(
            impl_->h_full, in.grid, pairs, sigma2, impl_->config.det_floor, impl_->config.g_min,
            impl_->config.gamma_max, out.x_hat_re, out.x_hat_im, out.sinr, out.re_grid_indices0,
            out.re_grid_indices1, out.capacity_symbols);
        status != MmseStatus::kOk) {
        return status;
    }

    meta = {};
    meta.n_symbols = n_symbols;
    meta.n_source_re = n_source_re;
    meta.sfn_subframe = in.sfn_subframe;
    meta.grid_symbol_count = in.grid.n_symbols;
    meta.grid_subcarrier_count = in.grid.n_subcarriers;
    meta.cell_id = in.cell_id;
    meta.start_prb = kLtePbchStartPrb;
    meta.n_prb = kLtePbchNumPrb;
    meta.start_symbol = kLtePbchStartSymbolNormalCp;
    meta.n_tx_ports = in.n_tx_ports;
    meta.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    meta.n_layers = 1U;
    meta.tx_mode = in.tx_mode;
    meta.mod_order = 2U;
    meta.sigma2 =
        detail::peek_sigma2_state(impl_->sigma2_by_cell[in.cell_id], impl_->config.sigma2_min);
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerCpuContext::run_pcfich(const PcfichMmseInput& in, PcfichMmseOutputView& out,
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

    for (std::uint32_t i = 0; i < base_out.n_re_per_layer; ++i) {
        out.re_grid_indices[i] = impl_->layout.grid_indices[i];
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
    meta.sigma2 =
        detail::peek_sigma2_state(impl_->sigma2_by_cell[in.cell_id], impl_->config.sigma2_min);
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerCpuContext::run_pcfich_td(const PcfichMmseInput& in,
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

    ExtractDescriptor desc{};
    desc.sfn_subframe = in.sfn_subframe;
    desc.cell_id = in.cell_id;
    desc.n_tx_ports = in.n_tx_ports;
    desc.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    desc.n_layers = 1U;
    desc.tx_mode = in.tx_mode;
    desc.channel_type = MmseChannelType::kPcfich;
    desc.control_symbol_count = 1U;
    desc.mod_order = 2U;
    desc.n_prb = kLteNumPrb20MHz;
    desc.prb_bitmap.fill(0xFFFFU);
    desc.prb_bitmap.back() = 0x000FU;
    if (const MmseStatus status = detail::validate_grid(in.grid); status != MmseStatus::kOk) {
        return status;
    }

    float sigma2 = 0.0F;
    if (const MmseStatus status = impl_->prepare_subframe_if_needed(in.grid, desc, sigma2);
        status != MmseStatus::kOk) {
        return status;
    }
    const std::uint32_t n_source_re = detail::build_pcfich_re_layout(desc, impl_->layout);
    std::vector<TransmitDiversityRePair> pairs{};
    if (!build_transmit_diversity_re_pairs(impl_->layout, pairs)) {
        return MmseStatus::kUnsupportedConfig;
    }
    const std::uint32_t n_symbols = static_cast<std::uint32_t>(pairs.size() * 2U);
    if (const MmseStatus status = equalize_transmit_diversity_pairs(
            impl_->h_full, in.grid, pairs, sigma2, impl_->config.det_floor, impl_->config.g_min,
            impl_->config.gamma_max, out.x_hat_re, out.x_hat_im, out.sinr, out.re_grid_indices0,
            out.re_grid_indices1, out.capacity_symbols);
        status != MmseStatus::kOk) {
        return status;
    }

    meta = {};
    meta.n_symbols = n_symbols;
    meta.n_source_re = n_source_re;
    meta.sfn_subframe = in.sfn_subframe;
    meta.grid_symbol_count = in.grid.n_symbols;
    meta.grid_subcarrier_count = in.grid.n_subcarriers;
    meta.cell_id = in.cell_id;
    meta.n_prb = kLteNumPrb20MHz;
    meta.start_symbol = 0U;
    meta.reg_count = static_cast<std::uint8_t>(kLtePcfichNumRegs);
    meta.n_tx_ports = in.n_tx_ports;
    meta.n_rx_ant = static_cast<std::uint8_t>(in.grid.n_rx_ant);
    meta.n_layers = 1U;
    meta.tx_mode = in.tx_mode;
    meta.mod_order = 2U;
    meta.sigma2 =
        detail::peek_sigma2_state(impl_->sigma2_by_cell[in.cell_id], impl_->config.sigma2_min);
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerCpuContext::run_pdcch(const PdcchMmseInput& in, PdcchMmseOutputView& out,
                                              PdcchMmseResult& meta) {
    if (const MmseStatus status = mmse::pdcch::validate_pdcch_mmse_input(in);
        status != MmseStatus::kOk) {
        return status;
    }
    if (in.n_tx_ports != 1U) {
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

    for (std::uint32_t i = 0; i < base_out.n_re_per_layer; ++i) {
        out.re_grid_indices[i] = impl_->layout.grid_indices[i];
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
    meta.sigma2 =
        detail::peek_sigma2_state(impl_->sigma2_by_cell[in.cell_id], impl_->config.sigma2_min);
    meta.prb_bitmap = in.prb_bitmap;
    meta.chain = in.chain;
    return MmseStatus::kOk;
}

MmseStatus MmseEqualizerCpuContext::run_pdcch_td(const PdcchMmseInput& in,
                                                 PdcchTdMmseOutputView& out,
                                                 PdcchTdMmseResult& meta) {
    if (!impl_->initialized) {
        return MmseStatus::kNotInitialized;
    }
    if (const MmseStatus status = mmse::pdcch::validate_pdcch_mmse_input(in);
        status != MmseStatus::kOk) {
        return status;
    }
    if (in.n_tx_ports != 2U) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (out.x_hat_re == nullptr || out.x_hat_im == nullptr || out.sinr == nullptr ||
        out.re_grid_indices0 == nullptr || out.re_grid_indices1 == nullptr) {
        return MmseStatus::kInvalidArgument;
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

    if (const MmseStatus status = detail::validate_grid(in.grid, &desc);
        status != MmseStatus::kOk) {
        return status;
    }

    float sigma2 = 0.0F;
    if (const MmseStatus status = impl_->prepare_subframe_if_needed(in.grid, desc, sigma2);
        status != MmseStatus::kOk) {
        return status;
    }

    const std::uint32_t n_source_re = detail::build_pdcch_re_layout(desc, impl_->layout);
    std::vector<TransmitDiversityRePair> pairs{};
    if (!build_transmit_diversity_re_pairs(impl_->layout, pairs)) {
        return MmseStatus::kUnsupportedConfig;
    }
    const std::uint32_t n_symbols = static_cast<std::uint32_t>(pairs.size() * 2U);
    if (const MmseStatus status = equalize_transmit_diversity_pairs(
            impl_->h_full, in.grid, pairs, sigma2, impl_->config.det_floor, impl_->config.g_min,
            impl_->config.gamma_max, out.x_hat_re, out.x_hat_im, out.sinr, out.re_grid_indices0,
            out.re_grid_indices1, out.capacity_symbols);
        status != MmseStatus::kOk) {
        return status;
    }

    meta = {};
    meta.n_symbols = n_symbols;
    meta.n_source_re = n_source_re;
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
    meta.sigma2 =
        detail::peek_sigma2_state(impl_->sigma2_by_cell[in.cell_id], impl_->config.sigma2_min);
    meta.prb_bitmap = in.prb_bitmap;
    meta.chain = in.chain;
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
