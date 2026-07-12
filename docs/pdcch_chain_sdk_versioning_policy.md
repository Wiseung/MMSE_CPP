# PDCCH Chain SDK 版本策略

本页定义 `PDCCH Chain SDK v1` 的兼容性边界。

本仓库还额外提供了一套并行的可加式 TD 接口，用于 `2 Tx port` LTE PDCCH：

- `MmseEqualizerCpuContext::run_pdcch_td`
- `MmseEqualizerGpuContext::run_pdcch_td`
- `PdcchTdMmseOutputView`
- `PdcchTdMmseResult`
- `BackendPdcchTdEqualizedIndication`

以及一组可加式的 blind-decode helper：

- `build_pdcch_control_region(...)`
- `build_pdcch_common_search_candidate_llrs(...)`
- `recover_pdcch_convolutional_rate_matched_llrs(...)`
- `decode_pdcch_dci_format1a_with_adapter(...)`
- `run_pdcch_cpu_common_search_decode(...)`
- `run_pdcch_cpu_si_rnti_search(...)`
- `build_pdcch_ue_specific_search_candidates(...)`
- `decode_pdcch_ue_specific_search_dci_format1a(...)`
- `run_pdcch_cpu_ue_specific_search(...)`
- `run_pdcch_cpu_si_rnti_geometry_search(...)`

这些符号不属于下面定义的冻结 `v1` 单 RE 契约。`v1` 边界仍然专门约束
`run_pdcch(...)` 及其 per-RE DTO 语义；新增 helper 维持向后兼容，但不改变冻结 core 的解释方式。

相关页面：

- [LTE Equalized Channel SDK 文档](/G:/MMSE_CPP/docs/lte_equalized_channel_sdk_interface.md)
- [文档首页](/G:/MMSE_CPP/docs/pdcch_chain_sdk_interface.md)
- [快速开始](/G:/MMSE_CPP/docs/pdcch_chain_sdk_quick_start.md)
- [API 参考](/G:/MMSE_CPP/docs/pdcch_chain_sdk_api_reference.md)

## 本页覆盖的版本

- `PDCCH Chain SDK v1`

## v1 中冻结的接口面

下面这些接口面被视为稳定接口，在 `v1` 内不得做不兼容变更：

- 对外总头文件路径：
  - `mmse/pdcch_chain_sdk.h`
- 对外 DTO 名称：
  - `mmse::pdcch::FrontendPdcchIndication`
  - `mmse::pdcch::ReservedControlRe`
  - `mmse::pdcch::BackendPdcchEqualizedIndication`
  - `mmse::PdcchMmseInput`
  - `mmse::PdcchMmseOutputView`
  - `mmse::PdcchMmseResult`
  - `mmse::PdcchChainMetadata`
- 对外 helper 名称：
  - `mmse::pdcch::make_pdcch_mmse_input`
  - `mmse::pdcch::make_backend_pdcch_equalized_indication`
  - `mmse::pdcch::decode_re_grid_index`
- 运行时入口：
  - `MmseEqualizerCpuContext::run_pdcch`
  - `MmseEqualizerGpuContext::run_pdcch`
- API 参考中定义的字段含义、单位和取值语义
- 输出顺序契约：
  - `x_hat_re[i]`、`x_hat_im[i]`、`sinr[i]` 和 `re_grid_indices[i]` 描述的是同一个 RE
- API 参考中记录的 LTE 支持边界
- API 参考中记录的 `MmseStatus` 含义

## v1 内不允许的变更

下面这些都属于不兼容变更，不能在 `v1` 下进行：

- 重命名或移除上面列出的任意对外 DTO、helper 或入口
- 修改任意已文档化字段的类型
- 修改已文档化单位
  - 例如把 `sinr` 从线性值改成 dB
  - 或者把 `sigma2` 从线性功率改成其他量纲
- 修改 `re_grid_indices` 的 LTE 网格索引语义
- 修改 `decode_re_grid_index` 的映射公式
- 修改 `PdcchChainMetadata` 透传字段的含义
- 修改 `control_re_exclusion_masks` 的 bit 布局
- 修改文档中已有 `MmseStatus` 的成功/失败语义
- 用会改变当前已验证行为的方式静默扩展支持边界
  - 例如没有版本升级和新契约说明，就宣称支持 `2 Tx port` PDCCH

## v1 内允许的变更

下面这些变更在不升主版本的情况下允许进行：

- 内部实现变更
  - 性能优化
  - 库内部的内存布局调整
  - CPU/GPU 传输路径调整
- 更严格的校验，只要文档化的合法输入面不变
- 增量文档澄清
- 增量 helper 函数
- 以向后兼容方式追加 DTO 字段，但前提是：
  - 现有字段的名称、类型、顺序含义和单位都不变
  - 老调用方忽略新字段后仍然能工作
- 在文档中补充 `MmseStatus` 的处理建议，但已有枚举值的语义必须保持不变

## 必须引入新主版本的变更

下面这些变更必须引入新的主版本，例如 `v2`：

- 修改单头文件 SDK 入口路径
- 修改任意冻结字段的类型或语义
- 修改 helper 返回语义
- 修改输出索引语义
- 修改支持的 PHY 范围
  - 例如把 NR 语义直接塞进同一套 DTO 契约
- 引入一种对输入要求或输出解释不同的 `2 Tx port` PDCCH 支持方式

## 推荐迁移规则

当确实需要做不兼容变更时：

1. 保持 `v1` 行为不变
2. 并行引入新的 `v2` 契约
3. 明确记录迁移差异
4. 只在下游全部迁移完成后再考虑退役 `v1`
