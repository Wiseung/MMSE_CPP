#include <cstdint>
#include <iostream>

#include "pdcch_benchmark_fixture.h"

using namespace mmse;
namespace fixture = mmse::benchmark::pdcch_fixture;

int main() {
    fixture::MixedWorkload workload{};
    if (!fixture::make_mixed_workload(kLteNumPrb20MHz, 1U, workload)) {
        std::cerr << "pdcch_ncu_workload fixture construction failed\n";
        return 1;
    }

    MmseEqualizerGpuContext context;
    MmseEqualizerGpuConfig config{};
    config.backend = MmseGpuBackend::kCuda;
    config.stream_count = 1U;
    if (context.init(config) != MmseStatus::kOk) {
        std::cerr << "pdcch_ncu_workload CUDA initialization failed\n";
        return 1;
    }

    const fixture::Case& benchmark_case = workload.cases.back();
    pdcch::PdcchGpuCommonSearchDecodeResult result{};
    if (pdcch::run_pdcch_gpu_common_search_decode(context, benchmark_case.request, result) !=
            MmseStatus::kOk ||
        !fixture::validate_result(result, benchmark_case)) {
        std::cerr << "pdcch_ncu_workload decode validation failed\n";
        return 1;
    }

    std::uint64_t checksum = 0U;
    for (const auto& hit : result.hits) {
        checksum = checksum * 1315423911U + hit.dci.chain.first_cce;
        checksum = checksum * 1315423911U + hit.dci.chain.aggregation_level;
        checksum = checksum * 1315423911U + hit.dci.start_prb;
        checksum = checksum * 1315423911U + hit.dci.n_prb;
        checksum = checksum * 1315423911U + hit.dci.mcs_tbs_index;
        checksum = checksum * 1315423911U + hit.dci.redundancy_version;
    }

    std::cout << "pdcch_ncu_workload.schema=mmse.ncu.v1\n";
    std::cout << "pdcch_ncu_workload.measurement_source=nsight_compute_replay\n";
    std::cout << "pdcch_ncu_workload.workload=mixed_l4_l8\n";
    std::cout << "pdcch_ncu_workload.n_tx_ports=1\n";
    std::cout << "pdcch_ncu_workload.candidate_count=" << result.candidate_count << '\n';
    std::cout << "pdcch_ncu_workload.hit_count=" << result.hits.size() << '\n';
    std::cout << "pdcch_ncu_workload.checksum=" << checksum << '\n';
    return 0;
}