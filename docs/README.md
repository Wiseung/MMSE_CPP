# 文档索引

## LTE 总览

- [LTE 下行信道译码总览](G:\MMSE_CPP\docs\lte_pdcch_pdsch_channel_decode_overview.md)
  说明协议层面的 `PBCH / PDCCH / PDSCH` 下行流程，以及当前 `MMSE_CPP` 的实现边界。
- [LTE PDCCH 完整流程说明](G:\MMSE_CPP\docs\lte_pdcch_complete_flow.md)
  逐步说明 `PDCCH` 从控制区确定到最终 `DCI` 输出的完整链路，并标出当前仓库已覆盖与未覆盖的步骤。
- [LTE DCI 输出语义与 CE/MMSE 接口说明](G:\MMSE_CPP\docs\lte_dci_and_ce_mmse_reference.md)
  说明下游 `DCI` 翻译输出的字段语义，以及当前 CE/MMSE 阶段的输入输出、数据长度和 `10 ms` 调用口径。

## LTE Equalized Channel SDK

- [LTE Equalized Channel SDK 文档首页](G:\MMSE_CPP\docs\lte_equalized_channel_sdk_interface.md)
- [LTE DCI 输出语义与 CE/MMSE 接口说明](G:\MMSE_CPP\docs\lte_dci_and_ce_mmse_reference.md)
- [PBCH 快速开始](G:\MMSE_CPP\docs\pbch_chain_sdk_quick_start.md)
- [PCFICH 快速开始与 API 参考](G:\MMSE_CPP\docs\pcfich_chain_sdk_quick_start_api_reference.md)
- [PDSCH 下游 LLR / 解扰接口面快速开始与 API 参考](G:\MMSE_CPP\docs\pdsch_llr_downstream_quick_start_api_reference.md)
  包含通用空间复用与 `2Tx` transmit-diversity 均衡输出接入下游 `LLR` helper 的方式。

## PDCCH SDK 子页面

- [PDCCH Chain SDK 文档首页](G:\MMSE_CPP\docs\pdcch_chain_sdk_interface.md)
- [PDCCH Chain SDK 快速开始](G:\MMSE_CPP\docs\pdcch_chain_sdk_quick_start.md)
- [PDCCH Chain SDK API 参考](G:\MMSE_CPP\docs\pdcch_chain_sdk_api_reference.md)
- [PDCCH Chain SDK 版本策略](G:\MMSE_CPP\docs\pdcch_chain_sdk_versioning_policy.md)
- [PDCCH Module API 集成示例](G:\MMSE_CPP\docs\pdcch_module_api_example.md)
- [PDCCH→PDSCH 交接 SDK V1](G:\MMSE_CPP\docs\pdcch_pdsch_handoff_sdk_v1.md)
  包含 Grant 字段、所有权、状态码、支持矩阵、CMake 集成与联调检查清单。

## 其他设计说明

- [MMSE Equalizer TDD 设计说明](G:\MMSE_CPP\docs\mmse_equalizer_tdd.md)
- [MMSE_CPP 算法原理与优化方法详解](G:\MMSE_CPP\docs\mmse_algorithm_and_optimization_guide.md)

## 性能与预算报告

- [PDCCH CPU/GPU 性能对比分析（2026-07-14）](G:\MMSE_CPP\docs\pdcch_cpu_gpu_performance_analysis_2026-07-14.md)
- [MMSE CUDA Profiling 报告（2026-07-03）](G:\MMSE_CPP\docs\mmse_cuda_profile_report_2026-07-03.md)
- [LTE MMSE 预算报告 (2026-07-01)](G:\MMSE_CPP\docs\lte_mmse_budget_report_2026-07-01.md)
