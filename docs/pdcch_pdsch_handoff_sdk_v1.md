# PDCCH→PDSCH 交接 SDK V1

## 1. 接口定位

`PdcchPdschHandoffConfigV1`、`PdschGrantV1` 和 `make_pdsch_grant_v1()` 将已通过 CRC-RNTI 校验并完成语义解析的 localized DCI 1A 命中转换为 PDSCH 可消费的 host-owned Grant。

公共入口为：

```cpp
#include "mmse/lte_chain_sdk.h"
```

V1 只负责调度信息交接，不执行 PDSCH 速率恢复、HARQ soft combining、Turbo 解码或 MAC 解析。

## 2. 最小调用流程

```cpp
mmse::pdcch::PdcchDciFormat1ADecodeResult hit = get_matched_dci_1a();

mmse::handoff::PdcchPdschHandoffConfigV1 config{};
config.start_symbol = control_symbol_count;
config.n_tx_ports = 1;
config.n_layers = 1;
config.transmission_mode = 1;
config.pmi = -1;
config.codeword = 0;
config.codeword_count = 1;

mmse::handoff::PdschGrantV1 grant{};
const mmse::MmseStatus status =
    mmse::handoff::make_pdsch_grant_v1(hit, config, grant);
if (status != mmse::MmseStatus::kOk) {
    return;
}

mmse::ExtractDescriptor descriptor{};
if (mmse::handoff::make_pdsch_extract_descriptor_v1(grant, n_rx_ant, descriptor) !=
    mmse::MmseStatus::kOk) {
    return;
}

// TM1 使用 run(...)；2Tx/TM2/单层使用 run_pdsch_td(...).
consume_pdsch_grant(grant, descriptor);
```

完整可运行示例为 `pdcch_pdsch_handoff_demo`，源码位于 `bench/pdcch_pdsch_handoff_demo.cpp`。

## 3. 输入配置

| 字段                | 含义                   | V1 约束                      |
| ------------------- | ---------------------- | ---------------------------- |
| `start_symbol`      | PDSCH 首个 OFDM symbol | `1..3`，通常等于已确认的 CFI |
| `n_tx_ports`        | CRS 发射端口数         | `1` 或 `2`                   |
| `n_layers`          | PDSCH 层数             | 固定为 `1`                   |
| `transmission_mode` | LTE transmission mode  | `1Tx/TM1` 或 `2Tx/TM2`       |
| `pmi`               | 预编码矩阵索引         | 固定为 `-1`                  |
| `codeword`          | 码字索引               | 固定为 `0`                   |
| `codeword_count`    | 码字数                 | 固定为 `1`                   |

这些字段不在 DCI 1A 中，必须来自已确认的小区配置、PCFICH/CFI 和传输模式上下文。调用方不得从 DCI payload 猜测这些值。

## 4. Grant 字段

| 字段组     | 字段                                                                                                                     | 说明                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| 版本       | `schema_version`                                                                                                         | V1 固定为 `1`                                                                   |
| 时频上下文 | `sfn_subframe`、`sfn`、`subframe`、`physical_cell_id`                                                                    | 原始时序标识、拆分后的 SFN/子帧和 PCI                                           |
| DCI 身份   | `rnti`、`rnti_type`、`dci_format`                                                                                        | SI-RNTI/C-RNTI 与 DCI 1A 标识                                                   |
| CRC/候选   | `crc`、`pdcch_chain`                                                                                                     | 传输 CRC、计算 CRC、恢复 RNTI、request/candidate/CCE/聚合级别                   |
| 原始数据   | `raw_payload_bits`                                                                                                       | Grant 自有的 DCI payload 副本                                                   |
| 资源       | `downlink_bandwidth_prb`、`start_prb`、`n_prb`                                                                           | 载波带宽与 localized 连续分配                                                   |
| 物理资源   | `physical_prb_bitmap`                                                                                                    | bit `prb % 16` 位于 word `prb / 16`，可直接填入 `ExtractDescriptor::prb_bitmap` |
| 符号       | `start_symbol`                                                                                                           | PDSCH 首个 symbol                                                               |
| 调制/TBS   | `mcs`、`mcs_valid`、`modulation_order`、`transport_block_size_index`、`transport_block_size_bits`                        | C-RNTI 使用 MCS 映射；SI-RNTI 使用 DCI 中的 `I_TBS`、`N_PRB^1A` 和固定 QPSK     |
| HARQ       | `harq_process`、`new_data_indicator`、`new_data_indicator_valid`、`redundancy_version`                                   | C-RNTI 暴露 NDI；SI-RNTI 的 NDI 无效                                            |
| 功控/DAI   | `transmit_power_control`、`transmit_power_control_valid`、`downlink_assignment_index`、`downlink_assignment_index_valid` | C-RNTI 暴露 TPC；仅 TDD DCI 的 DAI 有效                                         |
| PHY        | `n_tx_ports`、`n_layers`、`transmission_mode`、`pmi`                                                                     | 来自 handoff config                                                             |
| 码字       | `codeword`、`codeword_count`                                                                                             | V1 固定为单码字 0                                                               |

`mcs` 保留 DCI 的原始 5 位字段。对 SI-RNTI，该字段语义是 `I_TBS`，因此 `mcs_valid == false`；对 C-RNTI，该字段才是 MCS，`mcs_valid == true`。

## 5. 所有权与生命周期

- `PdschGrantV1` 完全由调用方持有，不引用 PDCCH 结果内部存储。
- `raw_payload_bits` 在成功转换时复制到 Grant；PDCCH 结果销毁或复用后仍可读取。
- `physical_prb_bitmap`、CRC 和候选元数据均按值存储。
- `make_pdsch_grant_v1()` 和 `make_pdsch_extract_descriptor_v1()` 不保留输入指针，不分配设备内存，不创建异步任务。
- 任一失败返回前都会把输出对象重置为默认零值，调用方不得消费失败调用留下的输出。

## 6. 状态码

| 状态码               | 含义                    | 常见原因                                                                                    |
| -------------------- | ----------------------- | ------------------------------------------------------------------------------------------- |
| `kOk`                | Grant/descriptor 已生成 | 输入命中且配置在 V1 支持范围内                                                              |
| `kInvalidArgument`   | 输入契约不一致          | 未命中、CRC-RNTI 不一致、字段有效性冲突、越界 PRB、无效 RX 数                               |
| `kUnsupportedConfig` | 输入合法但 V1 不支持    | PDCCH order、distributed VRB、CIF、保留 MCS/I_TBS、非 100 PRB PHY、非 TM1/TM2、非单层单码字 |

V1 不使用 `kBufferTooSmall`，因为 Grant 使用自有 `std::vector` 保存 payload。

## 7. 支持矩阵

| 能力                            | V1 状态                                                   |
| ------------------------------- | --------------------------------------------------------- |
| SI-RNTI localized DCI 1A        | 支持                                                      |
| C-RNTI localized DCI 1A         | 支持；调用方必须从 UE-specific RNTI 上下文提供真实 C-RNTI |
| FDD DCI 1A                      | 支持                                                      |
| TDD DCI 1A/DAI 透传             | 支持解析和交接；下游仍须提供正确 TDD PHY 上下文           |
| 物理 PRB 位图                   | 支持 localized 连续分配                                   |
| MCS→Qm/I_TBS、TBS               | 支持 LTE 基础下行 MCS `0..28` 和 `I_TBS 0..26`            |
| SI-RNTI TBS                     | 支持 `N_PRB^1A = 2/3`                                     |
| 1Tx/TM1/单层/单码字             | 支持                                                      |
| 2Tx/TM2/单层/单码字             | 支持；均衡时调用 `run_pdsch_td(...)`                      |
| distributed VRB                 | 不支持，明确拒绝                                          |
| PDCCH order                     | 不支持，明确拒绝                                          |
| CIF/cross-carrier scheduling    | 不支持，明确拒绝                                          |
| MCS `29..31` 或 `I_TBS > 26`    | 不支持，明确拒绝                                          |
| 多层、多码字、其它 TM、PMI      | 不支持，明确拒绝                                          |
| HARQ soft combining、Turbo、MAC | 不在本接口范围                                            |

当前 PDSCH equalizer 使用 20 MHz/1200 子载波网格，因此 handoff V1 明确拒绝非 100 PRB 载波。DCI 解析器本身仍可解析已有测试覆盖的标准 LTE 带宽。

## 8. CMake 安装与下游集成

安装：

```powershell
cmake -S . -B build -DMMSE_BUILD_TESTS=OFF -DMMSE_BUILD_BENCH=OFF
cmake --build build --config Release
cmake --install build --config Release --prefix C:\sdk\MMSE_CPP
```

下游 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.31)
project(pdsch_consumer LANGUAGES CXX)

find_package(MMSE_CPP CONFIG REQUIRED)

add_executable(pdsch_consumer main.cpp)
target_link_libraries(pdsch_consumer PRIVATE MMSE_CPP::mmse_cpp)
```

配置下游时把安装前缀加入搜索路径：

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH=C:\sdk\MMSE_CPP
cmake --build build --config Release
```

导出的 target 会传递 C++20 和安装后的 include 路径；下游不应手工拼接库文件路径。

## 9. 联调检查清单

- [ ] PDCCH 结果 `matched == true` 且 `crc.matches_expected_rnti == true`
- [ ] RNTI 上下文明确为 SI-RNTI 或真实 C-RNTI
- [ ] DCI format 为 localized 1A，且不是 PDCCH order
- [ ] `downlink_bandwidth_prb == 100`
- [ ] `start_symbol` 来自已确认 CFI，并与 PDSCH 提取配置一致
- [ ] 端口数、TM、层数、PMI 和码字配置来自同一小区/UE 上下文
- [ ] `physical_prb_bitmap` 置位数等于 `n_prb`
- [ ] C-RNTI 场景检查 `mcs_valid`、`new_data_indicator_valid` 和 `transmit_power_control_valid`
- [ ] SI-RNTI 场景确认固定 QPSK，并使用 `I_TBS + N_PRB^1A` 的 TBS
- [ ] TM1 调用通用 `run(...)`；2Tx/TM2/单层调用 `run_pdsch_td(...)`
- [ ] 任一非 `kOk` 状态都丢弃 Grant，不进入 PDSCH 解调
