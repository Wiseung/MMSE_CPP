#include <cstdint>
#include <iostream>
#include <vector>

#include "mmse/lte_chain_sdk.h"

namespace {

void append_bits(std::vector<std::uint8_t>& bits, std::uint32_t value, std::uint8_t width) {
    for (std::uint8_t bit = 0U; bit < width; ++bit) {
        bits.push_back(static_cast<std::uint8_t>((value >> (width - bit - 1U)) & 1U));
    }
}

std::uint16_t type2_riv(std::uint16_t n_prb, std::uint16_t start_prb,
                        std::uint16_t allocated_prb_count) {
    if (allocated_prb_count - 1U <= n_prb / 2U) {
        return static_cast<std::uint16_t>(n_prb * (allocated_prb_count - 1U) + start_prb);
    }
    return static_cast<std::uint16_t>(n_prb * (n_prb - allocated_prb_count + 1U) + n_prb - 1U -
                                      start_prb);
}

} // namespace

int main() {
    constexpr std::uint16_t kRnti = 0x1234U;
    const mmse::pdcch::PdcchDciFormat1AConfig dci_config{};
    std::vector<std::uint8_t> decoded_bits{};
    append_bits(decoded_bits, 1U, 1U);
    append_bits(decoded_bits, 0U, 1U);
    append_bits(decoded_bits, type2_riv(dci_config.n_prb, 10U, 20U),
                mmse::pdcch::pdcch_dci_riv_bit_count(dci_config.n_prb));
    append_bits(decoded_bits, 7U, 5U);
    append_bits(decoded_bits, 5U, 3U);
    append_bits(decoded_bits, 1U, 1U);
    append_bits(decoded_bits, 2U, 2U);
    append_bits(decoded_bits, 1U, 2U);
    while (decoded_bits.size() < mmse::pdcch::pdcch_dci_format1a_payload_bit_count(dci_config)) {
        decoded_bits.push_back(0U);
    }
    const std::uint16_t masked_crc = static_cast<std::uint16_t>(
        mmse::pdcch::calculate_pdcch_crc16(decoded_bits.data(),
                                           static_cast<std::uint32_t>(decoded_bits.size())) ^
        kRnti);
    append_bits(decoded_bits, masked_crc, mmse::pdcch::kPdcchCrcBitCount);

    mmse::pdcch::PdcchDciFormat1ADecodeResult decoded{};
    mmse::PdcchChainMetadata chain{};
    chain.request_id = 1001U;
    chain.candidate_id = 2U;
    chain.first_cce = 8U;
    chain.aggregation_level = 4U;
    if (mmse::pdcch::validate_and_parse_pdcch_dci_format1a(
            decoded_bits.data(), static_cast<std::uint32_t>(decoded_bits.size()), kRnti, 1234U, 19U,
            chain, dci_config, decoded) != mmse::MmseStatus::kOk) {
        return 1;
    }

    mmse::handoff::PdschGrantV1 grant{};
    mmse::handoff::PdcchPdschHandoffConfigV1 handoff_config{};
    handoff_config.start_symbol = 3U;
    handoff_config.n_tx_ports = 1U;
    handoff_config.n_layers = 1U;
    handoff_config.transmission_mode = 1U;
    if (mmse::handoff::make_pdsch_grant_v1(decoded, handoff_config, grant) !=
        mmse::MmseStatus::kOk) {
        return 2;
    }

    mmse::ExtractDescriptor descriptor{};
    if (mmse::handoff::make_pdsch_extract_descriptor_v1(grant, 2U, descriptor) !=
        mmse::MmseStatus::kOk) {
        return 3;
    }

    std::cout << "pdcch_pdsch_handoff.v1"
              << " rnti=" << grant.rnti << " first_cce=" << grant.pdcch_chain.first_cce
              << " aggregation=" << static_cast<unsigned>(grant.pdcch_chain.aggregation_level)
              << " start_prb=" << grant.start_prb << " n_prb=" << grant.n_prb
              << " qm=" << static_cast<unsigned>(grant.modulation_order)
              << " tbs_bits=" << grant.transport_block_size_bits
              << " ndi=" << static_cast<unsigned>(grant.new_data_indicator)
              << " descriptor_prbs=" << descriptor.n_prb << '\n';
    return 0;
}