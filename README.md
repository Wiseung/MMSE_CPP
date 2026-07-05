# MMSE_CPP

## 本地提交门禁

仓库克隆完成后先执行一次 `npm install`。本仓库使用 `husky + lint-staged` 在本地提交时做门禁：
如果暂存文件格式不符合要求，或者原生代码改动导致轻量级 `mmse_tests` 构建/测试冒烟检查失败，则阻止提交。

原生门禁规则如下：

- 已暂存的 `*.cpp/*.h` 文件会在本地通过 `clang-format` 格式化
- 已暂存的 `*.md/*.json/*.yml/*.yaml` 文件会通过 `prettier` 格式化
- 如果暂存改动触及 `src/`、`include/`、`tests/`、`bench/` 或 `CMakeLists.txt` 等原生构建/测试面，`pre-commit`
  钩子会执行本地 `cmake -> build mmse_tests -> ctest` 冒烟检查；若失败则拒绝提交

## CI / CD

当前仓库使用两条工作流：

- `ci`：针对 `main`、`codex/**` 分支以及 Pull Request 在 Windows 上构建并运行测试
- `cd`：当 `main` 分支 CI 成功后，自动生成发布包并部署到 `staging`；`production` 仍保留为
  `workflow_dispatch` 手工触发

在启用真实部署前，需要先在 GitHub 环境 / 仓库变量中配置：

- `production` 环境下的 `MMSE_PRODUCTION_DEPLOY_DIR`

推荐但非必须的 `staging` 配置：

- `staging` 环境下的 `MMSE_STAGING_DEPLOY_DIR`

当前部署行为是基于文件拷贝：CD 工作流会生成 `dist/mmse_cpp-release.zip`，然后把它拷贝到目标目录，
同时保留带时间戳的文件名和一个 `latest` 副本。如果 `MMSE_STAGING_DEPLOY_DIR` 还没有配置，
`staging` 会退回到 runner 本地目录，以保证 CD 流程仍然能完整跑通。

## CUDA 运行时策略

GPU 传输路径现在通过 `MmseEqualizerGpuConfig` 暴露了两类明确的运行时策略开关：

- `sigma2_ownership`
  - `kHostOwnedIir`：由 host 持有 IIR 平滑后的 `sigma2` 状态，并在 `equalize` 前把标量写回 device
  - `kDeviceOwnedState`：由 device 持有平滑后的 `sigma2` 状态，host 只读取一个摘要值用于 sanity / 观测
- `validation_policy`
  - `kReleaseSanity`：发布路径只保留轻量级的有限值 / 正值 sanity 检查
  - `kTestDeepTrace`：保留 CPU-vs-GPU trace 对齐能力，仅用于 debug / test 场景

这样既能让生产路径保持精简，也能在测试场景下保留更严格的验证模式。

在 `kReleaseSanity` 下，CUDA scratch 被压缩为一个 4-float 头部，用于轻量级状态检查：
`output_slot`、`symbol`、`subcarrier` 和运行时 `sigma2`。
逐采样的 equalizer trace 负载只会在 `kTestDeepTrace` 下分配和回传。

当前采样验证被拆成两层：

- 发布态 `spot_check_sample_count`：只用于轻量级输出检查，例如
  `xhat` 有限、`sinr` 有限且为正、以及头部级别的 `sigma2` sanity
- 调试态 `trace_sample_count`：在采样 RE 上额外输出逐采样的 equalizer trace 负载，
  供 CPU-vs-GPU 对齐分析

## LTE PDCCH 适配

均衡路径现在通过 `ExtractDescriptor::channel_type` 支持第二种 LTE 下行提取模式：

- `MmseChannelType::kPdsch`：现有数据区流程
- `MmseChannelType::kPdcch`：LTE PDCCH 控制区 RE 提取流程

对于 `kPdcch`，调用方必须提供：

- `control_symbol_count`：由 `PCFICH/CFI` 推导出的 LTE 控制区大小，在 normal-CP 下取值 `1..3`
- `control_re_exclusion_masks`：按控制符号、按 PRB 的 12-bit RE mask，用于在均衡前排除控制区里
  不属于 PDCCH 的 RE，例如被 `PCFICH/PHICH` 占用的 RE

对于希望停留在 DTO 层做链路集成的场景，PDCCH helper 层还提供了当前 LTE 支持边界下的可加式保留 RE 生成器：

- `mmse::pdcch::append_pcfich_reserved_control_re_list(...)`
- `mmse::pdcch::append_phich_reserved_control_re_list(...)`
- `mmse::pdcch::append_fdd_phich_reserved_control_re_list(...)`

这些 helper 会在 `make_pdcch_mmse_input(...)` 把列表转换成 `control_re_exclusion_masks` 之前，
自动填充当前 20 MHz normal-CP LTE 边界下 `FrontendPdcchIndication::reserved_control_res`。

当前 helper 契约围绕一个共享的 LTE 控制子帧上下文展开：

- `FrontendPdcchIndication::control_subframe`
- `PdcchMmseInput::control_subframe`
- `mmse::pdcch::LteControlSubframeContext`

当前 DTO 层的集成路径是：

1. 填充 `FrontendPdcchIndication::control_subframe`
2. 调用 `append_pcfich_reserved_control_re_list(frontend)`
3. 调用 `append_phich_reserved_control_re_list(frontend, ...)`
4. 调用 `make_pdcch_mmse_input(...)`

`run_pdcch(...)` 现在会先通过一条集中式校验路径验证低层 `PdcchMmseInput`，
然后再构建内部 descriptor。

当前支持边界：

- 仅支持 LTE，与仓库现有基于 CRS 的 20 MHz normal-CP 设计保持一致
- 传统 `run_pdcch(...)` 的 per-RE 契约支持 `1 Tx port`
- 新增 `run_pdcch_td(...)` 契约支持 `2 Tx port` LTE PDCCH 发射分集去映射，
  并为每个 RE 对输出两个软符号
- 基于 helper 的自动 `PHICH` 保留仍限制在相同的 LTE 20 MHz normal-CP helper 契约内，
  不会扩展 equalizer 运行时契约

### 模块集成接口面

对于链路集成，优先使用专门的 PDCCH module API，而不是手工构造一个通用 `ExtractDescriptor`。

推荐的单头文件 SDK 入口如下：

- 仅 PDCCH 接口面：`#include "mmse/pdcch_chain_sdk.h"`
- 面向 `PBCH/PDCCH/PCFICH` 以及 `PDSCH` 下游 helper 的统一 LTE 接口面：
  `#include "mmse/lte_chain_sdk.h"`
- LTE 总体信道流程概览：`docs/lte_pdcch_pdsch_channel_decode_overview.md`
- 文档索引：`docs/README.md`
- LTE SDK 首页：`docs/lte_equalized_channel_sdk_interface.md`（`LTE Equalized Channel SDK v1`）
- PBCH 快速开始：`docs/pbch_chain_sdk_quick_start.md`
- PCFICH 快速开始 / API 参考：`docs/pcfich_chain_sdk_quick_start_api_reference.md`
- PDSCH 下游 LLR / 解扰接口：`docs/pdsch_llr_downstream_quick_start_api_reference.md`
- LTE MMSE 预算报告：`docs/lte_mmse_budget_report_2026-07-01.md`
- PDCCH 文档首页：`docs/pdcch_chain_sdk_interface.md`（`PDCCH Chain SDK v1`）
- 快速开始：`docs/pdcch_chain_sdk_quick_start.md`
- API 参考：`docs/pdcch_chain_sdk_api_reference.md`
- 版本策略：`docs/pdcch_chain_sdk_versioning_policy.md`

面向上游的输入：

- `PdcchMmseInput`
  - FFT 网格：`grid`
  - 小区 / 时序上下文：`sfn_subframe`、`cell_id`
  - PDCCH 区域描述：`control_symbol_count`、`n_prb`、`prb_bitmap`
  - 非 PDCCH 控制 RE 保留：`control_re_exclusion_masks`
  - 用于后续级联的元数据透传：`PdcchChainMetadata`

面向下游的输出：

- 传统 per-RE 接口面：
  - 均衡后的软符号输入：`x_hat_re`、`x_hat_im`、`sinr`
  - 每个输出 RE 的来源映射：`re_grid_indices`
  - 运行元数据和透传字段：`PdcchMmseResult`
- 新增 2Tx TD 接口面：
  - 均衡后的软符号输入：`x_hat_re`、`x_hat_im`、`sinr`
  - 每个软符号对应的 RE 对映射：`re_grid_indices0`、`re_grid_indices1`
  - 运行元数据和透传字段：`PdcchTdMmseResult`

这样能让当前模块继续聚焦在信道估计和均衡，同时又保留 PDCCH 下游阶段需要的资源位置与候选元数据。

### LTE 统一 DTO 接口面

如果集成方希望只包含一个 LTE 总入口头文件，而不是一个 PDCCH 专用头文件，则使用：

- `#include "mmse/lte_chain_sdk.h"`

这个统一头文件会导出：

- `mmse::pbch::*`
  - `FrontendPbchIndication`
  - `make_pbch_mmse_input(...)`
  - `MmseEqualizerCpuContext::run_pbch(...)`
  - `MmseEqualizerGpuContext::run_pbch(...)`
  - `make_backend_pbch_equalized_indication(...)`
- `mmse::pdcch::*`
  - 现有 DTO 与 helper 接口面
- `mmse::pdsch::*`
  - `PdschDescramblingPlanCache`
  - `PdschDescrambledLlrOutputView`
  - `PdschDescrambledLlrResult`
  - `BackendPdschDescrambledLlrIndication`
  - `prepare_pdsch_descrambling_plan(...)`
  - `build_backend_pdsch_descrambled_llr_result(...)`
  - `make_backend_pdsch_descrambled_llr_indication(...)`
- `mmse::pcfich::*`
  - `FrontendPcfichIndication`
  - `make_pcfich_mmse_input(...)`
  - `MmseEqualizerCpuContext::run_pcfich(...)`
  - `MmseEqualizerGpuContext::run_pcfich(...)`
  - `make_backend_pcfich_equalized_indication(...)`

### PDSCH 下游 helper 现状

当前仓库已经提供 `PDSCH` 下游 `LLR / 解扰` helper，但仍然没有仓库内真实存在的
`PDSCH` downstream context。推荐集成方式是：

1. 上游先调用 `MmseEqualizerCpuContext::run(...)` 或 `MmseEqualizerGpuContext::run(...)`
2. 下游再通过 `mmse::pdsch::*` helper 生成解扰后的 `LLR`
3. 若调用方存在真实的 grant / item / worker context，则由调用方持有
   `PdschDescramblingPlanCache` 和 caller-owned `LLR` buffer
