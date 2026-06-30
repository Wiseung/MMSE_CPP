#pragma once

#include "mmse/constants.h"
#include "mmse/types.h"

namespace mmse {

class MmseEqualizerCpuContext {
  public:
    MmseEqualizerCpuContext();
    ~MmseEqualizerCpuContext();

    MmseEqualizerCpuContext(const MmseEqualizerCpuContext&) = delete;
    MmseEqualizerCpuContext& operator=(const MmseEqualizerCpuContext&) = delete;

    MmseStatus init(const MmseEqualizerCpuConfig& config);
    MmseStatus run(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                   EqualizerOutputView& out);
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
    MmseStatus run(const PlanarGridViewF32& grid, const ExtractDescriptor& desc,
                   EqualizerOutputView& out);
    MmseStatus run_pdcch(const PdcchMmseInput& in, PdcchMmseOutputView& out, PdcchMmseResult& meta);
    MmseStatus run_pdcch_td(const PdcchMmseInput& in, PdcchTdMmseOutputView& out,
                            PdcchTdMmseResult& meta);
    MmseGpuHostProfileSnapshot last_host_profile() const;

  private:
    struct Impl;
    Impl* impl_;
};

const char* to_string(MmseStatus status);

} // namespace mmse
