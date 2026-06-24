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
    MmseStatus run(const PlanarGridViewF32& grid,
                   const ExtractDescriptor& desc,
                   EqualizerOutputView& out);

private:
    struct Impl;
    Impl* impl_;
};

const char* to_string(MmseStatus status);

}  // namespace mmse
