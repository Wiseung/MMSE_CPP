# MMSE_CPP

[![CI](https://github.com/Wiseung/MMSE_CPP/actions/workflows/ci.yml/badge.svg)](https://github.com/Wiseung/MMSE_CPP/actions/workflows/ci.yml)
[![CD](https://github.com/Wiseung/MMSE_CPP/actions/workflows/cd.yml/badge.svg)](https://github.com/Wiseung/MMSE_CPP/actions/workflows/cd.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Security Policy](https://img.shields.io/badge/Security-Policy-blue)](./SECURITY.md)

高性能 LTE PHY 基带处理仓库，聚焦以下能力：

- 从 LTE FFT 后频域资源网格中提取目标 `RE`
- 基于 `CRS` 做信道估计与 `MMSE` 均衡
- 为 `PBCH / PCFICH / PDCCH / PDSCH` 提供统一或专用的 SDK 调用面
- 为部分下游链路提供 `LLR / descrambling` helper
- 提供 `AVX2` 与 `CUDA` 加速路径，以及对应测试、基准和质量门禁

这个仓库的核心输出是 **equalized symbols + SINR + 必要的资源网格元数据**。  
对 PDCCH，它还提供可选的 CPU `DCI 1A` 解码链路，覆盖 common-search、已知 RNTI 的
UE-specific search，以及 `6/15/25/50/75/100 RB / FDD` 范围内的 SI-RNTI 几何搜索。GPU PDCCH
路径保持 `100 RB`。它仍然**不是**完整 LTE
接收机，不负责 `MIB / CFI` 硬判决、非 `DCI 1A` 格式、`TB` 或 `MAC PDU` 的最终解码输出。

## 目录

- [项目定位](#项目定位)
- [当前能力与边界](#当前能力与边界)
- [仓库结构](#仓库结构)
- [环境与依赖](#环境与依赖)
- [快速开始](#快速开始)
- [如何快速调用](#如何快速调用)
- [文档导航](#文档导航)
- [GitHub 协作入口](#github-协作入口)
- [License](#license)
- [Security and Quality](#security-and-quality)

## 项目定位

如果你第一次接触这个仓库，可以先把它理解为：

> 一个面向 LTE 下行物理层的 **equalized-channel runtime / SDK**，输入是 FFT 后的 LTE 频域资源网格，输出是可供下游继续处理的均衡软符号、`SINR` 和资源网格位置信息。

它适合做的事：

- 为 `PDSCH` 或控制信道构造高性能 `RE` 提取与均衡链路
- 为 `PBCH / PCFICH / PDCCH` 提供可集成的 LTE SDK 入口
- 为下游解码器提供 `PDCCH / PDSCH` 的 `LLR` 或 descrambled `LLR`
- 为 `PDCCH common search + DCI 1A` 及 `UE-specific + DCI 1A` 提供内建尾咬卷积码译码的 CPU 可验收链路
- 做 CPU/GPU 一致性验证、性能分析和预算评估

它当前不做的事：

- `PBCH -> MIB` 最终译码
- `PCFICH -> CFI` 最终判决
- 非 `DCI 1A` 的通用最终 `DCI` 输出
- `PDSCH` 的速率恢复、`HARQ` 软合并、`Turbo` 译码和 `MAC PDU` 解析

## 当前能力与边界

| 能力                                      | 当前状态   | 主要入口                                                                  |
| ----------------------------------------- | ---------- | ------------------------------------------------------------------------- |
| 通用 LTE `RE` 提取 + `MMSE` 均衡          | 已支持     | `MmseEqualizerCpuContext::run(...)` / `MmseEqualizerGpuContext::run(...)` |
| `PBCH` equalized `RE` 输出                | 已支持     | `run_pbch(...)`                                                           |
| `PCFICH` equalized `RE` 输出              | 已支持     | `run_pcfich(...)`                                                         |
| `PDCCH 1Tx` equalized `RE` 输出           | 已支持     | `run_pdcch(...)`                                                          |
| `PDCCH 2Tx` transmit-diversity 去映射输出 | 已支持     | `run_pdcch_td(...)`                                                       |
| `PDSCH` descrambled `LLR` helper          | 已支持     | `mmse::pdsch::*`                                                          |
| `PDCCH` `QPSK LLR + descrambling` helper  | 已支持     | `mmse::pdcch::make_backend_pdcch_descrambled_llr_indication(...)`         |
| `PDCCH` `REG/CCE` 重组 helper             | 已支持     | `mmse::pdcch::build_pdcch_control_region(...)`                            |
| `PDCCH` 分阶段 CPU 基准                   | 已支持     | `pdcch_decode_bench`                                                      |
| `PDCCH UE-specific + DCI 1A` CPU 盲检索   | 已支持     | `run_pdcch_cpu_ue_specific_search(...)`，目标 RNTI 列表与 `L=1/2/4/8`     |
| `PBCH -> MIB` 最终译码                    | 不在仓库内 | 下游外部模块                                                              |
| `PCFICH -> CFI` 最终译码                  | 不在仓库内 | 下游外部模块                                                              |
| `PDCCH` 通用盲检索与通用最终 `DCI` 译码   | 部分支持   | 覆盖 common search 与 UE-specific 的 `DCI 1A`；不覆盖其它 DCI format      |
| `PDSCH` `Turbo/HARQ/MAC` 解码             | 不在仓库内 | 下游外部模块                                                              |

CPU `PDCCH SI-RNTI` 未知几何搜索通过
`run_pdcch_cpu_si_rnti_geometry_search(...)` 提供，支持 `6/15/25/50/75/100 RB`、FDD 与
normal CP；GPU PDCCH 保持 `100 RB`。

额外的 `PDCCH` helper / 正式入口：

- `build_pdcch_common_search_candidate_llrs(...)` 与 `recover_pdcch_convolutional_rate_matched_llrs(...)`
- `check_pdcch_crc_rnti(...)` 与 `decode_pdcch_dci_format1a_with_adapter(...)`
- `run_pdcch_cpu_common_search_decode(...)`
- `run_pdcch_cpu_si_rnti_search(...)`
- `run_pdcch_cpu_ue_specific_search(...)`
- `run_pdcch_cpu_si_rnti_geometry_search(...)`

对 `PDCCH` 来说，当前最容易误解的边界是：

- 仓库已经支持控制区大小约束、`PCFICH / PHICH` 保留资源排除、`CRS` 信道估计、`MMSE` 均衡、`2Tx` 去映射
- CPU PDCCH 支持 `6/15/25/50/75/100 RB` 的连续 FFT 后频域网格；`6 RB` 支持最多四个实际控制符号
- 仓库现在还支持 `REG / CCE` helper、common search、UE-specific 候选构造、`L=1/2/4/8` 速率恢复、内建尾咬卷积码译码、`CRC-RNTI` 与 `DCI 1A` 解析
- 仓库仍不支持 `PCFICH -> CFI` 硬判决、非 `DCI 1A` 的通用最终 `DCI` 解码；GPU PDCCH 仍固定为 `100 RB`，不提供其它带宽的 GPU 解码阶段

如果你需要完整背景，先读：

- [LTE 下行信道译码总览](./docs/lte_pdcch_pdsch_channel_decode_overview.md)
- [LTE PDCCH 完整流程说明](./docs/lte_pdcch_complete_flow.md)

## 仓库结构

```text
MMSE_CPP/
├─ include/mmse/          # 公共头文件与 SDK 入口
├─ src/                   # CPU / GPU / CUDA 实现
├─ tests/                 # 单元测试
├─ bench/                 # demo、基准与 profiling 工具
├─ docs/                  # 中文技术文档、API 参考与性能报告
├─ .github/workflows/     # CI / CD 工作流
├─ LICENSE                # Apache-2.0 许可
├─ SECURITY.md            # 安全策略
└─ CMakeLists.txt         # 构建入口
```

建议优先关注：

- `include/mmse/mmse_equalizer.h`：通用运行时入口
- `include/mmse/lte_chain_sdk.h`：统一 LTE SDK 头文件
- `include/mmse/pdcch_module_api.h`：`PDCCH` DTO/helper 接口
- `bench/pdcch_module_demo.cpp`：最直接的可运行链路示例

## 环境与依赖

### 必需依赖

- `CMake >= 3.31`
- 支持 `C++20` 的编译器

### 已验证环境

- Windows 2022
- MSVC
- Ninja

### 可选依赖

- CUDA Toolkit  
  用于启用 GPU/CUDA 后端；如果不可用，仓库仍可编译 CPU 路径。
- Node.js / npm  
  仅在你希望启用本地 `husky + lint-staged` 提交门禁时需要。
- Python  
  仅用于发布包和部署脚本，例如 `scripts/create_release_bundle.py`。

## 快速开始

### 1. 克隆仓库

```powershell
git clone https://github.com/Wiseung/MMSE_CPP.git
cd MMSE_CPP
```

### 2. 可选：安装本地 Git 提交门禁

如果你只想构建和运行仓库，本步骤不是必需的。  
如果你希望本地提交时自动执行格式化和轻量测试，请执行：

```powershell
npm install
```

### 3. 配置构建

推荐使用当前 CI 同类的单配置 Ninja 路径：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

如果你明确只想构建 CPU 路径：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMMSE_ENABLE_CUDA=OFF
```

### 4. 构建

```powershell
cmake --build build --parallel
```

### 5. 运行测试

```powershell
ctest --test-dir build --output-on-failure
```

如果你使用的是 Visual Studio 这类多配置生成器，而不是上面的 Ninja 单配置路径，请带上配置名：

```powershell
ctest --test-dir build -C Release --output-on-failure
```

### 6. 运行第一个示例

如果你想最快看到 SDK 调用路径，先运行 `PDCCH` demo：

```powershell
cmake --build build --target pdcch_module_demo --parallel
.\build\pdcch_module_demo.exe
```

## 如何快速调用

### 先选对入口

| 你的目标                            | 推荐头文件               | 主入口                                                                                        |
| ----------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------- |
| 通用 `PDSCH` / 自定义 `RE` 均衡     | `mmse/mmse_equalizer.h`  | `run(...)`                                                                                    |
| 统一 LTE SDK 入口                   | `mmse/lte_chain_sdk.h`   | `run_pbch / run_pcfich / run_pdcch / run_pdcch_td`                                            |
| `PDCCH 1Tx` 控制区链路              | `mmse/pdcch_chain_sdk.h` | `FrontendPdcchIndication -> make_pdcch_mmse_input(...) -> run_pdcch(...)`                     |
| `PDCCH 2Tx` transmit-diversity 链路 | `mmse/pdcch_chain_sdk.h` | `run_pdcch_td(...)`                                                                           |
| `PDSCH` descrambled `LLR` helper    | `mmse/lte_chain_sdk.h`   | `prepare_pdsch_descrambling_plan(...)` / `make_backend_pdsch_descrambled_llr_indication(...)` |

另外，`mmse/pdcch_chain_sdk.h` 还导出正式 CPU 盲检索入口：

- `run_pdcch_cpu_common_search_decode(...)`
- `run_pdcch_cpu_si_rnti_search(...)`

### 最常见调用路径

#### 1. 通用均衡路径

适合你已经自己知道要提取哪些 `RE`，并且只需要统一的 equalizer runtime：

1. 构造 `PlanarGridViewF32`
2. 构造 `ExtractDescriptor`
3. 初始化 `MmseEqualizerCpuContext` 或 `MmseEqualizerGpuContext`
4. 调用 `run(...)`
5. 读取 `EqualizerOutputView`

#### 2. `PDCCH 1Tx` SDK 路径

适合你已经知道控制区大小，并希望仓库帮你完成控制区 `RE` 提取、`CRS` 信道估计和 `MMSE`：

```cpp
#include "mmse/pdcch_chain_sdk.h"

mmse::MmseEqualizerCpuContext ctx;
ctx.init({});

mmse::pdcch::FrontendPdcchIndication frontend = ...;
mmse::pdcch::append_pcfich_reserved_control_re_list(frontend);
mmse::pdcch::append_phich_reserved_control_re_list(frontend, ...);

mmse::PdcchMmseInput in = mmse::pdcch::make_pdcch_mmse_input(grid, frontend);
mmse::PdcchMmseOutputView out = ...;
mmse::PdcchMmseResult meta{};

ctx.run_pdcch(in, out, meta);
auto backend = mmse::pdcch::make_backend_pdcch_equalized_indication(meta, out);
```

这条路径的工作终点是：

- 得到 `equalized RE`
- 得到 `sinr`
- 得到 `re_grid_indices`

它**不是**最终 `DCI` 解码结束点。

完整可运行示例见：

- [`bench/pdcch_module_demo.cpp`](./bench/pdcch_module_demo.cpp)

#### 3. `PDCCH 2Tx` 路径

如果控制信道是 `2 Tx port transmit diversity`，不要再走 `run_pdcch(...)`。  
应直接切换到：

- `run_pdcch_td(...)`

它会输出：

- 软符号
- `sinr`
- `re_grid_indices0 / re_grid_indices1`

也就是“每个软符号对应哪一对来源 `RE`”。

#### 4. `PDCCH` CPU blind-decode 路径

可以直接调用：

- `run_pdcch_cpu_common_search_decode(...)`
- `run_pdcch_cpu_si_rnti_search(...)`
- `run_pdcch_cpu_ue_specific_search(...)`
- `run_pdcch_cpu_si_rnti_geometry_search(...)`

其中 `run_pdcch_cpu_si_rnti_search(...)` 固定使用 `SI-RNTI` 语义。
`run_pdcch_cpu_ue_specific_search(...)` 接收目标 RNTI 列表，按 LTE `Y_k` 搜索空间
枚举 `L=1/2/4/8`。`run_pdcch_cpu_si_rnti_geometry_search(...)` 则在当前 20 MHz/FDD
边界内枚举 CFI 与 PHICH 资源几何，利用唯一 `SI-RNTI + DCI 1A` 命中锁定缓存。

这条路径会在仓库内顺序完成：

1. `run_pdcch(...)` 或 `run_pdcch_td(...)`
2. `REG / CCE` 恢复与 common-search 或 UE-specific 候选构造
3. `QPSK LLR + descrambling`
4. 速率恢复
5. 内建尾咬卷积码译码；也可提供外部回调覆盖
6. `CRC-RNTI` 校验与 `DCI 1A` 解析

输出是命中的候选列表：通用入口返回 `PdcchCommonSearchDecodeResult::hits`，SI-RNTI 专用入口返回 `PdcchSiRntiSearchResult::hits`。

## 文档导航

### 如果你是第一次看这个项目

- [文档总索引](./docs/README.md)
- [LTE Equalized Channel SDK 文档首页](./docs/lte_equalized_channel_sdk_interface.md)
- [LTE 下行信道译码总览](./docs/lte_pdcch_pdsch_channel_decode_overview.md)

### 如果你想看 `PDCCH`

- [PDCCH 完整流程说明](./docs/lte_pdcch_complete_flow.md)
- [PDCCH Chain SDK 文档首页](./docs/pdcch_chain_sdk_interface.md)
- [PDCCH Chain SDK 快速开始](./docs/pdcch_chain_sdk_quick_start.md)
- [PDCCH Chain SDK API 参考](./docs/pdcch_chain_sdk_api_reference.md)
- [PDCCH Module API 集成示例](./docs/pdcch_module_api_example.md)
- [LTE DCI 输出语义与 CE/MMSE 接口说明](./docs/lte_dci_and_ce_mmse_reference.md)

### 如果你想看 `PBCH / PCFICH / PDSCH`

- [PBCH 快速开始](./docs/pbch_chain_sdk_quick_start.md)
- [PCFICH 快速开始与 API 参考](./docs/pcfich_chain_sdk_quick_start_api_reference.md)
- [PDSCH 下游 LLR / 解扰接口面快速开始与 API 参考](./docs/pdsch_llr_downstream_quick_start_api_reference.md)

### 如果你想看算法、性能和预算

- [MMSE_CPP 算法原理与优化方法详解](./docs/mmse_algorithm_and_optimization_guide.md)
- [MMSE CUDA Profiling 报告（2026-07-03）](./docs/mmse_cuda_profile_report_2026-07-03.md)
- [LTE MMSE 预算报告（2026-07-01）](./docs/lte_mmse_budget_report_2026-07-01.md)

## GitHub 协作入口

如果你在 GitHub 上使用这个项目，建议按目的选择入口，而不是全部都走 `Issues`。

| 你的诉求                           | 应该去哪里  | 链接                                                                               |
| ---------------------------------- | ----------- | ---------------------------------------------------------------------------------- |
| 报告 bug、回归、构建失败、测试失败 | Issues      | [Issues](https://github.com/Wiseung/MMSE_CPP/issues)                               |
| 提问、讨论设计取舍、确认集成方式   | Discussions | [Discussions](https://github.com/Wiseung/MMSE_CPP/discussions)                     |
| 看路线图、依赖顺序、执行状态       | Projects    | [MMSE_CPP Project](https://github.com/users/Wiseung/projects/1)                    |
| 查看长期说明、FAQ、操作知识沉淀    | Wiki        | [Wiki](https://github.com/Wiseung/MMSE_CPP/wiki)                                   |
| 私下报告安全漏洞                   | Security    | [Security Advisories](https://github.com/Wiseung/MMSE_CPP/security/advisories/new) |

使用建议：

- `Issues`：只放可执行的问题、缺陷和明确需求
- `Discussions`：适合“先讨论再落 issue/PR”的问题
- `Projects`：适合看当前 roadmap，而不是追 PR 历史
- `Wiki`：如果某个主题需要长期维护、比 `README` 更长、但又不属于 API 参考，放到 Wiki 更合适  
  如果当前 Wiki 页面为空，请以 [`docs/`](./docs/) 为准

## License

本项目采用 [Apache License 2.0](./LICENSE)。

这意味着你可以在遵守许可条款的前提下：

- 使用、复制和分发本项目
- 修改源码并分发修改版本
- 在商业项目中集成本项目

你仍需要遵守 `Apache-2.0` 的保留许可、变更说明和免责声明要求。若你计划对外发布
基于本仓库的派生版本，请直接阅读 [LICENSE](./LICENSE) 正文。

## Security and Quality

### Security

- 安全策略文件：[`SECURITY.md`](./SECURITY.md)
- 私密漏洞报告入口：<https://github.com/Wiseung/MMSE_CPP/security/advisories/new>
- 不要通过公开 `Issues` 报告安全漏洞

### Quality

本仓库的质量保障分为本地门禁、CI、CD、测试和 profiling 五层：

#### 1. 本地提交门禁

执行 `npm install` 后，会启用 `husky + lint-staged`：

- `*.cpp / *.h` 走 `clang-format`
- `*.md / *.json / *.yml / *.yaml` 走 `prettier`
- 如果改动触及 `src/`、`include/`、`tests/`、`bench/` 或 `CMakeLists.txt`，会自动触发本地 `cmake -> build mmse_tests -> ctest` 冒烟检查

#### 2. CI

CI 工作流：

- 入口：<https://github.com/Wiseung/MMSE_CPP/actions/workflows/ci.yml>
- 触发：`main`、`codex/**`、Pull Request
- 内容：Windows 2022 上执行 `configure -> build -> test`

#### 3. CD

CD 工作流：

- 入口：<https://github.com/Wiseung/MMSE_CPP/actions/workflows/cd.yml>
- `main` 分支 CI 成功后自动打包并部署到 `staging`
- `production` 保持手工触发

#### 4. 单元测试与示例

- 主测试目标：`mmse_tests`
- 关键示例目标：`pdcch_module_demo`
- 其他常用目标：
  - `mmse_bench`
  - `mmse_cuda_profile`
  - `mmse_channel_budget`

#### 5. 性能与预算文档

如果你需要判断一个改动是否只是“功能正确”还是“真正达到工程要求”，请同时参考：

- [MMSE CUDA Profiling 报告（2026-07-03）](./docs/mmse_cuda_profile_report_2026-07-03.md)
- [LTE MMSE 预算报告（2026-07-01）](./docs/lte_mmse_budget_report_2026-07-01.md)

这两份文档比单次控制台输出更接近项目级质量判断标准。
