#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "mmse/constants.h"
#include "mmse/lte_pdsch_transport.h"
#include "mmse/pdcch_chain_dto.h"
#include "mmse/types.h"

namespace mmse::handoff {

inline constexpr std::uint16_t kPdschGrantSchemaVersionV1 = 1U;

enum class DciFormatV1 : std::uint8_t {
    kFormat1A = 0,
};

struct PdcchPdschHandoffConfigV1 {
    std::uint8_t start_symbol = 0;
    std::uint8_t n_tx_ports = 1;
    std::uint8_t n_layers = 1;
    std::uint8_t transmission_mode = 1;
    std::int8_t pmi = -1;
    std::uint8_t codeword = 0;
    std::uint8_t codeword_count = 1;
};

struct PdschGrantV1 {
    std::uint16_t schema_version = 0;
    std::uint32_t sfn_subframe = 0;
    std::uint16_t sfn = 0;
    std::uint8_t subframe = 0;
    std::uint16_t physical_cell_id = 0;
    std::uint16_t rnti = 0;
    pdcch::PdcchDciFormat1ARntiType rnti_type = pdcch::PdcchDciFormat1ARntiType::kCRnti;
    DciFormatV1 dci_format = DciFormatV1::kFormat1A;
    pdcch::PdcchCrcRntiCheck crc{};
    PdcchChainMetadata pdcch_chain{};
    std::vector<std::uint8_t> raw_payload_bits{};

    std::uint16_t downlink_bandwidth_prb = 0;
    std::uint16_t start_prb = 0;
    std::uint16_t n_prb = 0;
    std::array<std::uint16_t, 7> physical_prb_bitmap{};
    std::uint8_t start_symbol = 0;

    std::uint8_t mcs = 0;
    bool mcs_valid = false;
    std::uint8_t modulation_order = 0;
    std::uint8_t transport_block_size_index = 0;
    std::uint32_t transport_block_size_bits = 0;
    std::uint8_t harq_process = 0;
    std::uint8_t new_data_indicator = 0;
    bool new_data_indicator_valid = false;
    std::uint8_t redundancy_version = 0;
    std::uint8_t transmit_power_control = 0;
    bool transmit_power_control_valid = false;
    std::uint8_t downlink_assignment_index = 0;
    bool downlink_assignment_index_valid = false;

    std::uint8_t n_tx_ports = 0;
    std::uint8_t n_layers = 0;
    std::uint8_t transmission_mode = 0;
    std::int8_t pmi = -1;
    std::uint8_t codeword = 0;
    std::uint8_t codeword_count = 0;
};

inline MmseStatus make_pdsch_grant_v1(const pdcch::PdcchDciFormat1ADecodeResult& decoded,
                                      const PdcchPdschHandoffConfigV1& config,
                                      PdschGrantV1& grant) {
    grant = {};
    const pdcch::PdcchDciFormat1A& dci = decoded.dci;
    if (!decoded.matched || !decoded.crc.matches_expected_rnti ||
        decoded.crc.unmasked_rnti != dci.rnti || dci.rnti == 0U ||
        decoded.crc.unmasked_rnti !=
            static_cast<std::uint16_t>(decoded.crc.transmitted_crc ^ decoded.crc.calculated_crc) ||
        dci.cell_id >= kLteNumCellIds ||
        (dci.chain.aggregation_level != 1U && dci.chain.aggregation_level != 2U &&
         dci.chain.aggregation_level != 4U && dci.chain.aggregation_level != 8U) ||
        dci.raw_payload_bits.empty() ||
        dci.raw_payload_bits.size() > pdcch::kPdcchMaxDciPayloadBits ||
        dci.downlink_bandwidth_prb == 0U || dci.n_prb == 0U ||
        dci.start_prb + dci.n_prb > dci.downlink_bandwidth_prb) {
        return MmseStatus::kInvalidArgument;
    }
    for (const std::uint8_t bit : dci.raw_payload_bits) {
        if (bit > 1U) {
            return MmseStatus::kInvalidArgument;
        }
    }
    if ((dci.rnti_type == pdcch::PdcchDciFormat1ARntiType::kSiRnti) !=
        (dci.rnti == pdcch::kSiRnti)) {
        return MmseStatus::kInvalidArgument;
    }
    if (!dci.transport_block_size_index_valid) {
        return MmseStatus::kUnsupportedConfig;
    }
    if (dci.is_pdcch_order || dci.distributed_vrb_assignment || dci.cif_present ||
        dci.downlink_bandwidth_prb != kLteNumPrb20MHz || config.start_symbol == 0U ||
        config.start_symbol > kLteMaxControlSymbolsNormalCp || config.n_layers != 1U ||
        config.codeword != 0U || config.codeword_count != 1U || config.pmi != -1 ||
        !((config.n_tx_ports == 1U && config.transmission_mode == 1U) ||
          (config.n_tx_ports == 2U && config.transmission_mode == 2U))) {
        return MmseStatus::kUnsupportedConfig;
    }

    lte::PdschMcsParameters mcs_parameters{};
    std::uint16_t tbs_prb_count = dci.n_prb;
    std::uint8_t modulation_order = 2U;
    bool mcs_valid = false;
    if (dci.rnti_type == pdcch::PdcchDciFormat1ARntiType::kSiRnti) {
        if (!dci.n_prb_1a_valid || (dci.n_prb_1a != 2U && dci.n_prb_1a != 3U) ||
            dci.new_data_indicator_valid || dci.transmit_power_control_valid) {
            return MmseStatus::kInvalidArgument;
        }
        tbs_prb_count = dci.n_prb_1a;
    } else {
        if (!dci.modulation_and_coding_scheme_valid ||
            !lte::pdsch_mcs_parameters(dci.mcs_tbs_index, mcs_parameters)) {
            return MmseStatus::kUnsupportedConfig;
        }
        if (!dci.new_data_indicator_valid || !dci.transmit_power_control_valid ||
            dci.n_prb_1a_valid ||
            mcs_parameters.transport_block_size_index != dci.transport_block_size_index) {
            return MmseStatus::kInvalidArgument;
        }
        modulation_order = mcs_parameters.modulation_order;
        mcs_valid = true;
    }

    std::uint32_t tbs_bits = 0U;
    if (!lte::pdsch_transport_block_size(dci.transport_block_size_index, tbs_prb_count, tbs_bits)) {
        return MmseStatus::kUnsupportedConfig;
    }

    grant.schema_version = kPdschGrantSchemaVersionV1;
    grant.sfn_subframe = dci.sfn_subframe;
    grant.sfn = static_cast<std::uint16_t>((dci.sfn_subframe / 10U) % 1024U);
    grant.subframe = static_cast<std::uint8_t>(dci.sfn_subframe % 10U);
    grant.physical_cell_id = dci.cell_id;
    grant.rnti = dci.rnti;
    grant.rnti_type = dci.rnti_type;
    grant.crc = decoded.crc;
    grant.pdcch_chain = dci.chain;
    grant.raw_payload_bits = dci.raw_payload_bits;
    grant.downlink_bandwidth_prb = dci.downlink_bandwidth_prb;
    grant.start_prb = dci.start_prb;
    grant.n_prb = dci.n_prb;
    for (std::uint32_t prb = dci.start_prb; prb < dci.start_prb + dci.n_prb; ++prb) {
        grant.physical_prb_bitmap[prb / 16U] |= static_cast<std::uint16_t>(1U << (prb % 16U));
    }
    grant.start_symbol = config.start_symbol;
    grant.mcs = dci.mcs_tbs_index;
    grant.mcs_valid = mcs_valid;
    grant.modulation_order = modulation_order;
    grant.transport_block_size_index = dci.transport_block_size_index;
    grant.transport_block_size_bits = tbs_bits;
    grant.harq_process = dci.harq_process;
    grant.new_data_indicator = dci.new_data_indicator;
    grant.new_data_indicator_valid = dci.new_data_indicator_valid;
    grant.redundancy_version = dci.redundancy_version;
    grant.transmit_power_control = dci.transmit_power_control;
    grant.transmit_power_control_valid = dci.transmit_power_control_valid;
    grant.downlink_assignment_index = dci.downlink_assignment_index;
    grant.downlink_assignment_index_valid = dci.downlink_assignment_index_valid;
    grant.n_tx_ports = config.n_tx_ports;
    grant.n_layers = config.n_layers;
    grant.transmission_mode = config.transmission_mode;
    grant.pmi = config.pmi;
    grant.codeword = config.codeword;
    grant.codeword_count = config.codeword_count;
    return MmseStatus::kOk;
}

inline MmseStatus make_pdsch_extract_descriptor_v1(const PdschGrantV1& grant, std::uint8_t n_rx_ant,
                                                   ExtractDescriptor& descriptor) {
    descriptor = {};
    if (grant.schema_version != kPdschGrantSchemaVersionV1 ||
        grant.physical_cell_id >= kLteNumCellIds || n_rx_ant == 0U ||
        n_rx_ant > kMmseV1MaxNumRxAntennas || grant.n_prb == 0U ||
        grant.downlink_bandwidth_prb != kLteNumPrb20MHz ||
        grant.start_prb + grant.n_prb > kLteNumPrb20MHz || grant.start_symbol == 0U ||
        grant.start_symbol > kLteMaxControlSymbolsNormalCp ||
        (grant.modulation_order != 2U && grant.modulation_order != 4U &&
         grant.modulation_order != 6U) ||
        grant.n_layers != 1U || grant.n_layers > n_rx_ant || grant.codeword != 0U ||
        grant.codeword_count != 1U || grant.pmi != -1 ||
        !((grant.n_tx_ports == 1U && grant.transmission_mode == 1U) ||
          (grant.n_tx_ports == 2U && grant.transmission_mode == 2U))) {
        return MmseStatus::kInvalidArgument;
    }
    for (std::uint32_t prb = 0U; prb < kLteNumPrb20MHz; ++prb) {
        const bool present = (grant.physical_prb_bitmap[prb / 16U] & (1U << (prb % 16U))) != 0U;
        const bool expected = prb >= grant.start_prb && prb < grant.start_prb + grant.n_prb;
        if (present != expected) {
            return MmseStatus::kInvalidArgument;
        }
    }
    descriptor.sfn_subframe = grant.sfn_subframe;
    descriptor.cell_id = grant.physical_cell_id;
    descriptor.n_tx_ports = grant.n_tx_ports;
    descriptor.n_rx_ant = n_rx_ant;
    descriptor.n_layers = grant.n_layers;
    descriptor.tx_mode = grant.transmission_mode;
    descriptor.channel_type = MmseChannelType::kPdsch;
    descriptor.start_symbol = grant.start_symbol;
    descriptor.control_symbol_count = grant.start_symbol;
    descriptor.mod_order = grant.modulation_order;
    descriptor.n_prb = grant.n_prb;
    descriptor.prb_bitmap = grant.physical_prb_bitmap;
    descriptor.pmi = grant.pmi;
    return MmseStatus::kOk;
}

} // namespace mmse::handoff