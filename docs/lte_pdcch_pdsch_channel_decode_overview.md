# LTE 下行信道译码流程总览

## 文档目的

这份文档从两个层面梳理 LTE 下行信道：

1. **标准完整链路**  
   说明 `PBCH`、`PDCCH`、`PDSCH` 在一个完整 LTE 接收机中的端到端译码流程，以及各自最终输出信息。
2. **`MMSE_CPP` 工程代码链路**  
   说明当前仓库实际实现到了哪一层，哪些模块已经覆盖，哪些模块仍然需要在仓库外部完成。

换句话说，这份文档既回答：

- `PBCH -> MIB` 怎么来
- `PDCCH -> DCI` 怎么来
- `PDSCH -> 传输块 / MAC PDU / 更高层载荷` 怎么来

也明确说明当前 `MMSE_CPP` 工程在整个接收链中的位置。

---

## 一、PBCH / PDCCH / PDSCH 的最终输出

### 1. PBCH 的最终输出

`PBCH` 的最终业务输出是：

- `MIB`（`MasterInformationBlock`）

典型字段包括：

- 下行带宽
- `PHICH` 配置
- `SFN` 高位
- 为后续控制信道与数据信道译码提供的系统基础参数

### 2. PDCCH 的最终输出

`PDCCH` 的最终业务输出是：

- 一个或多个通过 CRC-RNTI 校验的 `DCI`

典型字段包括：

- `RNTI` 假设
- `DCI` 格式
- 聚合级别
- `CCE` 位置
- 上下行调度授权
- `HARQ` 进程信息
- `MCS / RV / NDI`
- 与 `PDSCH` / `PUSCH` 调度相关的资源字段

### 3. PDSCH 的最终输出

`PDSCH` 的最终业务输出是：

- 一个或两个传输块（`Transport Block`）

中间层和最终层输出通常包括：

- 均衡后的复数软符号
- 每个符号对应的后验 `SINR`
- 解扰后的软比特 / `LLR`
- 速率恢复后的比特流
- `Turbo` 译码后的码块
- 码块 CRC / 传输块 CRC 结果
- `MAC PDU`
- `MAC` 子头 / 控制单元 / `SDU`
- 更高层 `RRC / NAS / IP` 负载

---

## 二、LTE 标准完整译码流程

### 1. PBCH 完整流程

1. 小区搜索与同步
2. 通过 `PSS/SSS` 获得 `PCI` 与帧边界
3. 提取 `PBCH` 对应资源单元
4. 信道估计与均衡
5. `QPSK` 软解调得到 `LLR`
6. 解扰
7. 速率恢复
8. 尾咬卷积码译码
9. CRC 检查与天线端口假设判决
10. 输出 `MIB`

### 2. PDCCH 完整流程

1. 获得控制区大小（来自 `PCFICH/CFI` 或可信外部输入）
2. 确定控制区中被 `PCFICH / PHICH` 占用的非 PDCCH 资源
3. 提取有效 `PDCCH RE`
4. 基于 `CRS` 的信道估计
5. `MMSE` 均衡
6. 若是 `2Tx`，做发射分集去映射
7. `QPSK` 软比特生成
8. `REG` 重组
9. `CCE` 重组
10. 在搜索空间内做盲检索
11. 对每个候选做解扰、速率恢复、卷积译码
12. 做 CRC + `RNTI masking`
13. 输出通过校验的 `DCI`

### 3. PDSCH 完整流程

1. 用 `DCI + MIB + 小区上下文` 构建 `PDSCH` 调度
2. 推导 `PRB` 分配、起始符号、层数、传输模式、HARQ 状态
3. 提取被调度的 `PDSCH RE`
4. 参考信号信道估计
5. `MMSE` 均衡
6. 做层处理 / 分集处理 / 码字路径恢复
7. 星座软解调
8. 解扰
9. 速率恢复
10. 码块拼接 / 分段恢复
11. `Turbo` 译码
12. CRC24A / CRC24B 检查
13. 输出传输块
14. 解析 `MAC PDU`
15. 向更高层交付消息

---

## 三、当前 `MMSE_CPP` 工程代码实际覆盖范围

### 1. 已覆盖的能力

当前仓库实现了：

- LTE 下行资源提取
  - `MmseChannelType::kPdsch`
  - `MmseChannelType::kPdcch`
- 基于 `CRS` 的信道估计
- `MMSE` 均衡
- 输出均衡后的 `x_hat` 与 `sinr`
- 在 `PDSCH` equalized 输出之后提供可选的下游 helper：
  - max-log `LLR` 生成
  - `PDSCH` 解扰
  - 显式 `PdschDescramblingPlanCache`
  - caller-owned `LLR` 输出面
- `PDCCH` 场景下对 `PCFICH / PHICH` 占用资源的排除
- `PDCCH 2Tx` 发射分集去映射

### 2. 未覆盖的能力

当前仓库**没有**实现：

- `PBCH` 译码
- `MIB` 解析
- `PCFICH` 译码
- `PHICH` 译码
- `PDCCH` 盲检索
- `DCI` 解码 / CRC-RNTI 验证
- 仓库内真实存在的 `PDSCH` 下游 context / grant context / decode context
- `PDSCH` 速率恢复 / `HARQ` 软合并 / `Turbo` 译码
- `MAC PDU` 解码
- `RRC / ASN.1` 解码

也就是说，本工程当前主要处于：

**资源网格 -> 有效 RE 提取 -> 信道估计 -> 均衡 -> 软符号 / SINR 输出**

这一段。

---

## 四、当前工程代码的公开输出面

### 1. 通用 PDSCH 均衡输出

输入：

- `PlanarGridViewF32`
- `ExtractDescriptor`，其中 `channel_type = kPdsch`

输出：

- `EqualizerOutputView`
- `x_hat_re`
- `x_hat_im`
- `sinr`
- `n_re_per_layer`
- `n_layers`
- `mod_order`

### 1.1 PDSCH 下游 LLR / 解扰 helper 输出

输入：

- `ExtractDescriptor`
- `EqualizerOutputView`
- `RNTI`

输出：

- `PdschDescrambledLlrOutputView`
- `PdschDescrambledLlrResult`
- 或者直接返回 `BackendPdschDescrambledLlrIndication`

可选辅助对象：

- `PdschDescramblingPlanCache`

这一层的作用是把 equalized `PDSCH` 输出转换成：

- 解扰后的 `LLR`

但它仍然不等于完整 `PDSCH` 译码链，也不代表仓库内已经存在更高层的 `PDSCH`
downstream context。

### 2. 旧版 PDCCH 单 RE 输出面

输入：

- `PdcchMmseInput`

输出：

- `PdcchMmseOutputView`
- `PdcchMmseResult`
- `x_hat_re`
- `x_hat_im`
- `sinr`
- `re_grid_indices`

这个输出面保持 `1Tx`、逐 `RE` 语义。

### 3. 新增 PDCCH 2Tx 发射分集输出面

输入：

- `PdcchMmseInput`

输出：

- `PdcchTdMmseOutputView`
- `PdcchTdMmseResult`
- `x_hat_re`
- `x_hat_im`
- `sinr`
- `re_grid_indices0`
- `re_grid_indices1`

这个输出面中，每个软符号都带有一对来源 `RE` 索引，用来表示发射分集去映射后的来源关系。

---

## 五、工程代码文件定位

### 1. 公开类型与接口

- [include/mmse/types.h](G:\MMSE_CPP\include\mmse\types.h)
- [include/mmse/mmse_equalizer.h](G:\MMSE_CPP\include\mmse\mmse_equalizer.h)
- [include/mmse/pdsch_chain_dto.h](G:\MMSE_CPP\include\mmse\pdsch_chain_dto.h)
- [include/mmse/pdsch_module_api.h](G:\MMSE_CPP\include\mmse\pdsch_module_api.h)
- [include/mmse/pdcch_chain_dto.h](G:\MMSE_CPP\include\mmse\pdcch_chain_dto.h)
- [include/mmse/pdcch_module_api.h](G:\MMSE_CPP\include\mmse\pdcch_module_api.h)

### 2. CPU 路径

- [src/mmse_equalizer_cpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_cpu.cpp)

关键函数：

- `descriptor_supported(...)`
- `build_data_re_layout(...)`
- `build_pdcch_re_layout(...)`
- `estimate_channel(...)`
- `pack_equalizer_inputs(...)`
- `equalize_1x2_scalar(...)`
- `equalize_2x2_scalar(...)`
- `demap_pdcch_transmit_diversity_scalar(...)`
- `run(...)`
- `run_pdcch(...)`
- `run_pdcch_td(...)`

### 3. GPU 路径

- [src/mmse_equalizer_gpu.cpp](G:\MMSE_CPP\src\mmse_equalizer_gpu.cpp)
- [src/mmse_cuda_runtime.cu](G:\MMSE_CPP\src\mmse_cuda_runtime.cu)
- [src/internal/mmse_cuda_runtime.h](G:\MMSE_CPP\src\internal\mmse_cuda_runtime.h)
- [src/internal/mmse_internal.h](G:\MMSE_CPP\src\internal\mmse_internal.h)

关键步骤：

- Host pinned slot staging
- `RE layout` 构建
- `CudaGridMeta` 打包
- H2D 传输
- CUDA 信道估计 kernel
- CUDA 均衡 kernel
- `PDCCH + 2Tx` strict CUDA TD 分支
- D2H 输出回传
- 可选 CPU/GPU 结果一致性验证

---

## 六、LaTeX 流程图

### 1. LTE 标准完整链路图

- LaTeX 源文件：
  - [figures/lte-downlink-full-chain.tex](G:\MMSE_CPP\figures\lte-downlink-full-chain.tex)
- 生成图片：
  - [figures/lte-downlink-full-chain.png](G:\MMSE_CPP\figures\lte-downlink-full-chain.png)

![LTE 下行完整链路](G:\MMSE_CPP\figures\lte-downlink-full-chain.png)

### 2. `MMSE_CPP` 工程代码流程图

- LaTeX 源文件：
  - [figures/mmse-cpp-pdcch-pdsch-code-flow.tex](G:\MMSE_CPP\figures\mmse-cpp-pdcch-pdsch-code-flow.tex)
- 生成图片：
  - [figures/mmse-cpp-pdcch-pdsch-code-flow.png](G:\MMSE_CPP\figures\mmse-cpp-pdcch-pdsch-code-flow.png)

![MMSE_CPP 工程代码流程](G:\MMSE_CPP\figures\mmse-cpp-pdcch-pdsch-code-flow.png)

### 3. PDCCH 专用译码流程图

- LaTeX 源文件：
  - [figures/pdcch-decode-flow.tex](G:\MMSE_CPP\figures\pdcch-decode-flow.tex)
- 生成图片：
  - [figures/pdcch-decode-flow.png](G:\MMSE_CPP\figures\pdcch-decode-flow.png)

![PDCCH 专用译码流程](G:\MMSE_CPP\figures\pdcch-decode-flow.png)

这张图只关注 `PDCCH` 本身，适合单独讲：

- 控制区大小来源
- `PCFICH / PHICH` 占用资源排除
- `CRS` 信道估计与 `MMSE` 均衡
- `2Tx` 发射分集去映射
- `REG / CCE` 重组
- 候选盲检索
- `DCI` 译码与输出

---

## 七、如何理解这三张图

### 1. 如果从标准接收机角度看

- `PBCH` 负责给出 `MIB`
- `PDCCH` 负责给出 `DCI`
- `DCI + MIB + 小区上下文` 决定 `PDSCH` 怎么解
- `PDSCH` 最终给出传输块，再往上到 `MAC PDU` 和更高层数据

### 2. 如果从当前仓库角度看

当前仓库主要做的是：

- 从 `FFT` 网格中提取有效资源
- 做 `CRS` 信道估计
- 做 `MMSE` 均衡
- 对 `PDCCH 2Tx` 做发射分集去映射
- 把软符号、`SINR`、以及 `RE` 来源信息交给外部下游

而完整的：

- `PBCH -> MIB`
- `PDCCH -> DCI`
- `PDSCH -> TB / MAC PDU`

这些业务级译码链路，目前还不在仓库内部。
