#pragma once

#include <span>

#include "mmse/constants.h"
#include "mmse/types.h"

namespace mmse {

namespace pdcch {
struct PdcchGpuCommonSearchDecodeRequest;
struct PdcchGpuCommonSearchDecodeResult;
} // namespace pdcch

class MmseEqualizerCpuContext {
  public:
    MmseEqualizerCpuContext();
    ~MmseEqualizerCpuContext();

    MmseEqualizerCpuContext(const MmseEqualizerCpuContext&) = delete;
    MmseEqualizerCpuContext& operator=(const MmseEqualizerCpuContext&) = delete;

    MmseStatus init(const MmseEqualizerCpuConfig& config);
    // `run` estimates CRS channel state once per prepared subframe and then
    // writes equalized symbols in the layer-major layout of EqualizerOutputView.
    MmseStatus run(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                   EqualizerOutputView& out);
    MmseStatus run_pdsch_td(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                            PdschTdMmseOutputView& out, PdschTdMmseResult& meta);
    MmseStatus run_pbch(const PbchMmseInput& in, PbchMmseOutputView& out, PbchMmseResult& meta);
    MmseStatus run_pbch_td(const PbchMmseInput& in, PbchTdMmseOutputView& out,
                           PbchTdMmseResult& meta);
    MmseStatus run_pcfich(const PcfichMmseInput& in, PcfichMmseOutputView& out,
                          PcfichMmseResult& meta);
    MmseStatus run_pcfich_td(const PcfichMmseInput& in, PcfichTdMmseOutputView& out,
                             PcfichTdMmseResult& meta);
    MmseStatus run_pdcch(const PdcchMmseInput& in, PdcchMmseOutputView& out, PdcchMmseResult& meta);
    MmseStatus run_pdcch_td(const PdcchMmseInput& in, PdcchTdMmseOutputView& out,
                            PdcchTdMmseResult& meta);

  private:
    struct Impl;
    Impl* impl_;
};

class MmseEqualizerGpuContext {
  public:
    MmseEqualizerGpuContext();
    ~MmseEqualizerGpuContext();

    MmseEqualizerGpuContext(const MmseEqualizerGpuContext&) = delete;
    MmseEqualizerGpuContext& operator=(const MmseEqualizerGpuContext&) = delete;

    MmseStatus init(const MmseEqualizerGpuConfig& config);
    // Generic entry point. Auto may use the CPU fallback; CUDA uses the staged
    // stream path and preserves the same output contract as the CPU context.
    MmseStatus run(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                   EqualizerOutputView& out);
    MmseStatus run_pdsch_td(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                            PdschTdMmseOutputView& out, PdschTdMmseResult& meta);
    MmseStatus run_pbch(const PbchMmseInput& in, PbchMmseOutputView& out, PbchMmseResult& meta);
    MmseStatus run_pbch_td(const PbchMmseInput& in, PbchTdMmseOutputView& out,
                           PbchTdMmseResult& meta);
    MmseStatus run_pcfich(const PcfichMmseInput& in, PcfichMmseOutputView& out,
                          PcfichMmseResult& meta);
    MmseStatus run_pcfich_td(const PcfichMmseInput& in, PcfichTdMmseOutputView& out,
                             PcfichTdMmseResult& meta);
    MmseStatus run_pdcch(const PdcchMmseInput& in, PdcchMmseOutputView& out, PdcchMmseResult& meta);
    MmseStatus run_pdcch_td(const PdcchMmseInput& in, PdcchTdMmseOutputView& out,
                            PdcchTdMmseResult& meta);
    // GPU common-search decode is split internally into submit/collect so the
    // caller can batch work across streams through the batch overload.
    MmseStatus
    run_pdcch_gpu_common_search_decode(const pdcch::PdcchGpuCommonSearchDecodeRequest& request,
                                       pdcch::PdcchGpuCommonSearchDecodeResult& result);
    MmseStatus run_pdcch_gpu_common_search_decode_batch(
        std::span<const pdcch::PdcchGpuCommonSearchDecodeRequest> requests,
        std::span<pdcch::PdcchGpuCommonSearchDecodeResult> results);
    MmseGpuHostProfileSnapshot last_host_profile() const;

  private:
    struct Impl;
    Impl* impl_;
};

const char* to_string(MmseStatus status);

} // namespace mmse
