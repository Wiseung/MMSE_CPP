# 4G LTE 下行 MMSE 信道均衡模块 — 技术设计文档 (TDD)

> 适用范围：FDD LTE / 20 MHz（2048-FFT, 1200 子载波, 100 PRB, Normal CP）/ 2×2 MIMO / 1 ms TTI 硬实时。
> 设计目标：在 x86 CPU 与 NVIDIA GPU 上实现「FFT 输出 → 软解调 LLR 输入」之间的 DSP 段（CRS 插值 + MMSE 均衡），输出星座点 X̂ 与后验 SINR。
> 对齐标准：3GPP TS 36.211（物理信道与调制）、3GPP TS 36.212（复用与信道编码）。

在动手前先说明一个贯穿全文的关键结论，它决定了很多设计取舍：**真正喂给下游 LLR 的不是 X̂ 本身，而是「去偏后的 X̂ + 每 RE 每层的后验 SINR」这一对量。** 任何丢掉 SINR 的均衡器对软解码都是不完整的。下文 §2.6 会给出严格推导。

---

## 0. 系统定位与数据流总览

```
            ┌─────────────────────────── 本模块 (PHY DSP 段) ───────────────────────────┐
 FFT 输出     │  ① CRS 提取 → ② LS 估计 → ③ 2D 插值(得 H) → ④ σ² 估计                      │
 资源栅格 Y ─►│                                          ↓                                │──► LLR 生成
 (int16 cplx) │                          ⑤ MMSE 权重 W = (HᴴH+σ²I)⁻¹Hᴴ                    │   (软解调)
             │                                          ↓                                │
 ARM 描述符 ─►│                  ⑥ 均衡 X̂ = W·Y + 后验 SINR γ 计算                        │──► (X̂, γ)
             └────────────────────────────────────────────────────────────────────────┘
```

关键量级（单子帧、单天线）：1200 × 14 RE ≈ 1.68×10⁴ RE，2 RX 天线下复数 float32 输入约 **268 KB/子帧/天线**。这个数据量本身不大，**瓶颈不在算力而在「确定性时延」**——必须在 < 1 ms 内端到端完成，且抖动极小。这一点会反复影响 GPU 流水线设计。

---

## 1. 接口定义

### 1.1 输入 (Inputs)

| 名称 | 类型 | 布局 | 说明 |
|---|---|---|---|
| `grid` | 复数 `int16_t` 或 `float` | **SoA**: `re[]`, `im[]` 分离 | 频域资源栅格，按 `[ant][symbol][subcarrier]` 排列 |
| `desc` | `extract_descriptor_t` | 结构体 | ARM 下发的提取描述符 |

前级 FFT 通常给出 **block-floating-point 的 int16**（带共享指数）。建议输入侧保留 int16 以省 PCIe/内存带宽（见 §5.1），在均衡核内部转 float32。

**资源栅格的复数布局**：强制 **planar / SoA**（`re` 与 `im` 两个独立数组），不要用 `std::complex` 的交织布局。理由见 §3.1，对 SIMD 与 CUDA coalescing 都是硬要求。

**提取描述符**（一份 PDSCH 分配对应一份描述符；被动监测系统中由 ARM 盲解 PCFICH/PDCCH 后产出）：

```c
typedef struct {
    uint32_t sfn_subframe;   // SFN*10 + subframe，时间戳/对齐用
    uint16_t cell_id;        // 物理小区 ID → 决定 v_shift 与 CRS 序列
    uint8_t  n_tx_ports;     // CRS 端口数 {1,2,4}
    uint8_t  n_rx_ant;       // 接收天线数 (本设计=2)
    uint8_t  n_layers;       // 传输层数 (1 或 2)
    uint8_t  tx_mode;        // 传输模式 TM2/3/4 (决定是否需解预编码)
    uint8_t  start_symbol;   // PDSCH 起始 OFDM 符号 (跨过控制区, 1~4)
    uint8_t  mod_order;      // 2=QPSK 4=16QAM 6=64QAM 8=256QAM (传给 LLR)
    uint16_t n_prb;          // 分配的 PRB 数
    uint16_t prb_bitmap[7];  // 100-bit RBG/PRB 位图 (比 list 更省, 利于向量化掩码)
    int8_t   pmi;            // 预编码矩阵索引 (TM4); -1 表示无
} extract_descriptor_t;
```

> 设计要点：用 **位图** 而非 `prb_list[]`。位图天然映射成 SIMD/CUDA 的 RE 有效性掩码，避免分支化的「这个 RE 要不要处理」判断。

### 1.2 输出 (Outputs) — 含 SINR 的传递契约

```c
typedef struct {
    float    re, im;         // 去偏后的均衡符号 X̂ (单位功率归一)
} cplx_f;

typedef struct {
    cplx_f*  x_hat;          // [layer][re_idx] 均衡星座点 (SoA)
    float*   sinr;           // [layer][re_idx] 后验 SINR γ_k  (线性值, 非 dB)
    // 或等价地传 noise_var = 1/γ_k；二选一并写入契约文档
    uint32_t n_re_per_layer;
    uint8_t  n_layers;
    uint8_t  mod_order;      // 透传给 LLR
} equalizer_output_t;
```

**为什么必须输出 SINR / 噪声方差**：MMSE 均衡是有偏估计。下游 LLR 的高斯近似为

```
LLR(b) ≈ (1/σ²_eff,k) · ( min_{x∈X⁰_b} |x̂_k - x|²  -  min_{x∈X¹_b} |x̂_k - x|² )
其中  σ²_eff,k = 1 / γ_k
```

若只给 X̂ 而不给 γ_k，LLR 的尺度就是错的，会直接拖垮 Turbo/LDPC 译码增益。因此 **(X̂, γ) 必须成对、逐 RE 逐层传递**，且 `x_hat` 必须是「去偏」版本（见 §2.6）。

---

## 2. 算法中间过程详细拆解

### 2.1 CRS 提取（参 TS 36.211 §6.10.1）

CRS 频域位置：`k = 6m + (v + v_shift) mod 6`，其中 `v_shift = N_ID_cell mod 6`。

- Normal CP、端口 0/1：CRS 落在子帧符号 **{0, 4, 7, 11}**。
- 同一符号内端口 0 与端口 1 在频率上错开 3 个子载波（v：端口0 在 l=0 用 v=0、l=4 用 v=3；端口1 相反）。

参考序列（Gold 序列 → QPSK）：

```
r_{l,ns}(m) = (1/√2)(1 - 2c(2m)) + j(1/√2)(1 - 2c(2m+1))
c_init = 2^10 · (7(ns+1) + l + 1)(2·N_ID_cell + 1) + 2·N_ID_cell + N_CP
```

> 工程做法：CRS 序列只与 `cell_id` 有关，**离线/初始化时预生成整帧 CRS 表**（10 子帧 × 符号 × 端口），运行时查表，绝不在热路径里跑 Gold 序列。提取即按掩码 gather 出导频 RE。

### 2.2 LS 信道估计

导频处的最小二乘估计（CRS 已归一化 |X_p|=1）：

```
Ĥ_LS(p) = Y(p) · X_p*  =  Y(p) · conj(r(p))
```

即一次复乘。得到稀疏（频率每 6 个、时间 4 个符号）的信道样点 Ĥ_LS，对每个 (tx_port, rx_ant) 对各一份，2×2 共 4 组。

### 2.3 时频二维信道插值引擎

采用 **可分离（separable）2D 插值**，对延迟与吞吐都友好：

**Step A — 频率插值**（在 4 个 CRS 符号上各自做一次，把每 6 个子载波的样点补满 1200 个）：
- 默认：**线性 / 三次样条**（低时延、无需信道统计）。
- 高性能可选：**Wiener / MMSE 插值** `Ĥ = R_hp · R_pp⁻¹ · Ĥ_LS`，其中 `R_pp = R_hh + σ²I`。按假设的延迟扩展/多普勒**离线预算滤波抽头**，运行时只做 FIR 卷积。

**Step B — 时间插值**（对每个子载波，在符号 {0,4,7,11} 之间线性内插、两端外插补满 14 个符号）。

```text
estimate_channel(grid, crs_tab, desc) -> H[2][2][nsym][nsc]:
  for (tx,rx) in 2x2:
      for l in {0,4,7,11}:                      # CRS 符号
          H_ls[l] = gather(grid[rx], crs_pos(tx,l)) * conj(crs_tab[tx,l])   # §2.2
          H_freq[l][:] = freq_interp(H_ls[l])   # 每6→满1200  (Step A)
      for k in 0..1199:
          H[tx][rx][:][k] = time_interp(H_freq[{0,4,7,11}][k])              # Step B
```

> 端口 2/3（4 端口场景）导频更稀疏（每子帧 2 个符号），时间插值精度下降，需在描述符里据 `n_tx_ports` 切换抽头集。

### 2.4 噪声方差 σ² 实时估算

主用「**LS 残差法**」：插值得到的平滑信道与原始 LS 样点之差即近似噪声：

```text
σ̂² = (1/Np) · Σ_p | Ĥ_LS(p) - Ĥ_smooth(p) |²
```

备选「**相邻导频差分法**」（去除信号、保留噪声，对慢变信道稳健）：

```text
σ̂² ≈ (1/2Np) · Σ_p | Ĥ_LS(p) - Ĥ_LS(p') |²
```

实现上每个 (tx,rx) 对算一份，再对 4 个分量平均，并做 **时间平滑（一阶 IIR）** 抑制抖动。务必设下限 `σ² = max(σ̂², σ²_min)`，这是后面防奇异的第一道闸（§5.2）。

### 2.5 2×2 MMSE 权重的解析计算

模型 `y = H·x + n`，H 为 2×2。MMSE 接收滤波器：

```text
W = (Hᴴ·H + σ²I)⁻¹ · Hᴴ  ≡  Hᴴ·(H·Hᴴ + σ²I)⁻¹
```

（两式由矩阵求逆引理等价；2×2 下都只需一次 2×2 求逆。）记 `A = Hᴴ·H + σ²I`，它是 **Hermitian 正定** 矩阵：

```text
A = [ a11   a12 ]      a11, a22 ∈ ℝ⁺
    [ a12*  a22 ]
```

闭式逆（无需通用求逆例程）：

```text
det A = a11·a22 - |a12|²   (∈ ℝ)
                1     [  a22   -a12 ]
A⁻¹ = ─────────────── [             ]
              det A     [ -a12*   a11 ]
```

**因为加了 σ²I（对角加载），只要 σ² > 0，A 必正定、det A > 0，结构上不会奇异**——这是 MMSE 相对 ZF 的天然鲁棒性。然后 `W = A⁻¹·Hᴴ`。

计算流（每 RE）：
1. 由 H 算 `A = Hᴴ·H + σ²I`（4 个复乘累加 + 对角加 σ²）。
2. det、1/det。
3. A⁻¹，再 `W = A⁻¹·Hᴴ`。

### 2.6 均衡、后验 SINR 与去偏（喂给 LLR 的核心）

```text
z = W·y
G = W·H = (Hᴴ·H + σ²I)⁻¹·Hᴴ·H = I - σ²·(Hᴴ·H + σ²I)⁻¹
```

设 `g_k ≡ G_kk`（实数，0 < g_k < 1）。经典 MMSE 后验 SINR：

```text
        g_k
γ_k = ─────────         g_k = [ (Hᴴ·H + σ²I)⁻¹·Hᴴ·H ]_kk
       1 - g_k
```

**去偏输出**（MMSE 的 z_k 含偏置因子 g_k，需还原到单位增益）：

```text
x̂_k = z_k / g_k        σ²_eff,k = 1 / γ_k
```

向下游传 (x̂_k, γ_k)。注意 g_k 是 G 的对角元，2×2 下顺带即可算出，**几乎零额外开销**，但对 LLR 正确性是决定性的。这是整个模块最容易被实现者漏掉的一步。

```text
equalize_re(H[2][2], y[2], sigma2) -> (xhat[2], gamma[2]):
  A = H^H·H + sigma2·I                      # Hermitian 2x2
  det = A11*A22 - |A12|^2
  inv_det = 1.0 / max(det, DET_FLOOR)        # §5.2
  Ainv = inv_det * [[A22,-A12],[-conj(A12),A11]]
  W   = Ainv · H^H                           # 2x2
  z   = W · y                                # 均衡(有偏)
  G   = W · H                                # 取对角 g0,g1 (实部)
  for k in {0,1}:
      g = clamp(Re(G[k][k]), GMIN, 1-GMIN)   # 防 0/1
      xhat[k]  = z[k] / g                    # 去偏
      gamma[k] = g / (1 - g)                 # 后验 SINR
```

---

## 3. CPU 实现专场（C++ / SIMD）

### 3.1 数据结构：SoA 胜出

**用 SoA（Structure of Arrays），不要 AoS。** 复数也要拆 `re[]` / `im[]`。

- AoS（`std::complex<float>[]`）会让 SIMD 在做复乘时不停 shuffle 交织/解交织，且 cache line 利用率低。
- SoA 下「一条 AVX 指令处理 8 个 RE 的同一个矩阵元」，访存连续、`vfmadd` 直接用，是教科书式的数据并行。

矩阵元也 SoA 化：为 H 的 4 个复元各开 `re/im` 数组，共 8 个 float 流。**「一个 RE 一个矩阵」改写成「一个矩阵元的 N 个 RE 一个向量」**（跨 RE 向量化，而非矩阵内部向量化）。

### 3.2 AVX2 / AVX-512：跨 RE 打包求逆

核心思想：把 N 个 RE 的 2×2 求逆做成 **N-lane SIMD**。AVX2 = 8×float32（8 RE/批），AVX-512 = 16 RE/批。复乘用分离寄存器 + FMA：

```text
(a_r + j·a_i)(b_r + j·b_i):
   p_r = a_r·b_r - a_i·b_i
   p_i = a_r·b_i + a_i·b_r
```

```cpp
// 跨 8 个 RE 同时求 A^-1 (AVX2)。所有变量是 8-lane __m256
inline void mmse_inv_8re(const __m256 a11, const __m256 a22,
                         const __m256 a12r, const __m256 a12i,
                         /*out*/ __m256& d_inv) {
    __m256 a12sq = _mm256_fmadd_ps(a12i, a12i,
                       _mm256_mul_ps(a12r, a12r));        // |a12|^2
    __m256 det   = _mm256_fnmadd_ps(a12sq, _mm256_set1_ps(1.f),
                       _mm256_mul_ps(a11, a22));          // a11*a22 - |a12|^2
    det          = _mm256_max_ps(det, _mm256_set1_ps(DET_FLOOR)); // §5.2
    // 牛顿迭代精修倒数: x1 = x0*(2 - det*x0)
    __m256 x0 = _mm256_rcp_ps(det);
    d_inv = _mm256_mul_ps(x0,
              _mm256_fnmadd_ps(det, x0, _mm256_set1_ps(2.f)));
}
```

要点：

- `_mm256_rcp_ps` 是 ~12bit 近似，**必须接一步牛顿迭代** 才够 LLR 精度；或直接 `_mm256_div_ps`（更准但更慢，按精度需求选）。
- 用 **位图 → mask**（`_mm256_blendv_ps`）处理「无效 RE / 保护带」，避免分支。
- 复乘统一封装成 `cmul_re/cmul_im` inline，编译器（`-O3 -mavx2 -mfma`）会展开。
- 数据按 16/32 字节对齐（`alignas(64)`），按 cache line 分块。

> AVX-512 注意频率降档（heavy-512 在部分 x86 上触发降频）。建议同时编 AVX2 / AVX-512 两套，运行时按 CPUID 派发，实测择优——20 MHz 这种中等数据量，AVX2 往往就够且更省功耗。

### 3.3 多线程模型

- **并行粒度：PRB（或 PRB 组）。** 各 PRB 的 RE 完全独立 → 天然 data-parallel。`#pragma omp parallel for schedule(static)`，按 PRB 切块、块大小取 SIMD 批的整数倍（如 8 RE 对齐）。
- **跨子帧流水线**：子帧之间独立，用线程池让 N 与 N+1 子帧的不同阶段重叠，平滑负载。
- **NUMA / 亲和性**：硬实时下绑核（`pthread_setaffinity`），隔离中断，关闭超线程争用；用 **预分配的内存池**，热路径零 `malloc`。
- 信道估计（§2.3 插值）与均衡（§2.5/2.6）可拆成两个并行段；插值有跨子载波依赖（FIR），均衡完全独立。

线程模型推荐：**固定线程池 + 任务队列**，而非每子帧 `omp` 起停（避免 fork-join 开销与抖动）。1 ms TTI 下，确定性比峰值吞吐更重要。

## 4. GPU 实现专场（CUDA）

### 4.1 内存流转：Pinned Ring Buffer

```text
        ┌────────── Host Pinned Ring Buffer (N 槽, 三缓冲) ──────────┐
 FFT ──►│ slot0 │ slot1 │ slot2 │ ...   (cudaHostAlloc, 锁页)        │
        └───┬────────────────────────────────────────────────────┘
            │ cudaMemcpyAsync H2D (stream)            ▲ cudaMemcpyAsync D2H
            ▼                                         │
        ┌──────────────── Device Buffers ─────────────┘
        │ d_grid → [估计核] → d_H → [均衡核] → d_xhat,d_sinr │
        └─────────────────────────────────────────────────┘
```

- **必须用锁页内存**（`cudaHostAlloc`/`cudaMallocHost`），否则 `cudaMemcpyAsync` 退化为同步、无法与计算重叠。
- **Ring Buffer + 三缓冲（triple buffering）**：FFT 写槽 N+1、PCIe 搬槽 N、GPU 算槽 N-1 互不阻塞。
- 输入用 **int16** 过 PCIe（带宽减半，§5.1），上 GPU 后再转 float32。268 KB/子帧的 H2D 在 PCIe Gen3 x16（~13 GB/s）下约 20 µs，**配合流水线后被完全隐藏**。

### 4.2 并行策略：Grid/Block 划分

**一个线程负责一个 RE 的完整 2×2 MMSE**（thread-per-RE）。这是最优粒度：

- 不要「按天线端口切分」——2×2 矩阵太小，跨线程协作的同步开销远大于收益。
- Block = 256 线程；Grid = `ceil(n_valid_re / 256)`。
- 每线程把 H 的 4 个复元、y 的 2 个复元 **load 进寄存器**，全程寄存器内完成求逆/均衡/SINR，无需 shared memory（2×2 没有数据复用，shared 反而浪费）。
- **访存合并（coalescing）**：靠 §1.1 的 SoA planar 布局，相邻线程读相邻地址，达成 128B 合并访问。
- 信道估计（插值 FIR）单独成核，其内部有跨子载波复用 → 可用 shared memory 缓存导频。

### 4.3 小矩阵求逆：手写 Kernel 解析解 vs cuBLAS Batched

**强烈推荐手写 Kernel 解析解，不要用 cuBLAS Batched。** 权衡如下：

| 维度 | 手写解析解 (thread/RE) | cuBLAS `getrfBatched`+`getriBatched` |
|---|---|---|
| 单矩阵规模 | 2×2，闭式（§2.5），~20 FLOP | 通用 LU，对 2×2 是杀鸡用牛刀 |
| 启动/调度开销 | 一次 kernel launch | 需指针数组、多次 launch，per-batch 开销高 |
| 数据布局 | 寄存器内，无中间写回 | 需把每个矩阵搬成 cuBLAS 要求的布局 |
| 鲁棒性 | 可内联 det floor / NaN 守卫 | 奇异时返回 info 码，需额外分支处理 |
| 结论 | **几十万个 2×2 → 完胜** | 仅当矩阵 ≥ 8×8 且批量大时才划算 |

对 ~1.7×10⁴ 个 2×2 这种「海量小矩阵」，cuBLAS Batched 的固定开销和布局转换会吃掉所有优势。闭式解直接在寄存器里跑完。

```cuda
__global__ void mmse_equalize_kernel(
        const float2* __restrict__ H,   // [4][n_re] planar: H00,H01,H10,H11
        const float2* __restrict__ Y,   // [2][n_re]
        const float*  __restrict__ sigma2, // 每 PRB/子帧一个
        const uint32_t* __restrict__ mask, // 有效 RE 位图
        float2* __restrict__ Xhat, float* __restrict__ Sinr, int n_re) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_re || !test_bit(mask, i)) return;

    float2 h00=H[i], h01=H[n_re+i], h10=H[2*n_re+i], h11=H[3*n_re+i];
    float  s2 = sigma2[prb_of(i)];

    // A = H^H H + s2 I  (Hermitian)
    float a11 = h00.x*h00.x+h00.y*h00.y + h10.x*h10.x+h10.y*h10.y + s2;
    float a22 = h01.x*h01.x+h01.y*h01.y + h11.x*h11.x+h11.y*h11.y + s2;
    float2 a12 = cadd(cmul(conj(h00),h01), cmul(conj(h10),h11)); // 复

    float det = fmaxf(a11*a22 - (a12.x*a12.x+a12.y*a12.y), DET_FLOOR);
    float idet = __frcp_rn(det);
    // A^-1, W=A^-1 H^H, z=W y, G=W H 的对角 g0/g1, 去偏 + SINR
    // ... (闭式展开, 全寄存器)  见 §2.6 伪代码
    Xhat[i] = xhat0;  Sinr[i] = gamma0;        // layer0
    Xhat[n_re+i] = xhat1; Sinr[n_re+i] = gamma1; // layer1
}
```

### 4.4 异步流水线：CUDA Streams 隐藏时延

- **每子帧一个 stream**（或固定数量 stream 轮转），3 级流水：H2D(N) ∥ Compute(N-1) ∥ D2H(N-2)。
- 用 **CUDA Events** 做依赖与计时；用 **Graph（cudaGraph）** 把「H2D→估计核→均衡核→D2H」固化成图，**消除逐 kernel launch 的 CPU 开销与抖动**——对 1 ms 硬实时极有价值。
- 估计核与均衡核在同一 stream 内串行（有数据依赖），跨子帧的不同 stream 并行。
- 关键指标是 **p99 端到端时延** 而非平均吞吐；用 Nsight Systems 看时间线，确认 PCIe 与 compute 真的重叠了。

> 现实建议：20 MHz / 单子帧负载对现代 GPU 很轻，GPU 路线的价值在于 **同时处理多小区/多用户的批量场景**。单链路低时延，调好的 AVX2 CPU 往往时延更稳、功耗更低。架构上建议把均衡核做成「可批量（多子帧/多小区拼成大 grid）」以摊薄 launch 开销。

---

## 5. 工程注意事项与避坑指南

### 5.1 浮点 vs 定点

- **内存/PCIe 带宽**：int16 复数 = 4 B/RE，float32 复数 = 8 B/RE，**比值 2:1**。在 I/O 与跨 PCIe 链路用 int16 直接省一半带宽与一半 cache 占用。
- **计算精度**：矩阵求逆里 det 的动态范围大（小信号时 det 极小），定点极易溢出/下溢。**推荐混合策略**：I/O & 存储用 int16（配合 block-floating-point 共享指数），**求逆与 SINR 计算在核内用 float32**。这是带宽与数值稳健性的最佳折中。
- 不要全程 int16 硬算 MMSE：1/det 的定点实现需要大量 guard bit 和饱和逻辑，得不偿失。

### 5.2 奇异 / NaN 防护（务必逐条落地）

1. **对角加载是第一道防线**：σ²I 使 A 结构正定，理论上 det > 0。但要给 σ² 设下限 σ²_min（§2.4），防止高 SNR 误估为 0 退化成 ZF。
2. **det floor**：`det = max(det, DET_FLOOR)` 再取倒数，杜绝 `1/0 = Inf`。
3. **g_k 钳位**：`g = clamp(g, GMIN, 1-GMIN)`，防去偏除零和 γ=∞（§2.6）。
4. **SINR 上下限**：`gamma = clamp(gamma, GAMMA_MIN, GAMMA_MAX)`，给 LLR 一个有界、可饱和的可靠度。
5. **NaN/Inf 兜底**：CPU 端可在 debug 构建开 `feenableexcept` 抓浮点异常；release 路径用 `isfinite` 检查或直接靠上面的钳位保证不产 NaN（钳位后路径无法产生 NaN，优于事后检测）。
6. **GPU 端**：用 `__frcp_rn` / `fmaxf` 内建，避免 `-use_fast_math` 把钳位优化掉（fast-math 会假设无 NaN/Inf，与防护语义冲突，**热路径慎用 fast-math 或局部关闭**）。

### 5.3 其他易踩点

- **CRS 序列查表**：热路径绝不跑 Gold 序列；初始化按 `cell_id` 预生成整帧。
- **保护带 / DC 子载波**：1200 有效 ≠ 2048 FFT，DC 与边带要按掩码剔除，否则边缘插值会污染。
- **时间插值外推**：子帧首尾符号（在 CRS 符号 0 之前 / 11 之后）只能外推，误差大；高多普勒下考虑用相邻子帧 CRS 辅助。
- **σ² 的作用域**：是逐 PRB 还是逐子帧给一个值，要在接口契约里写死，CPU/GPU 两端保持一致。

---

## 附：模块验证与对齐建议

- **参考实现**：用 MATLAB LTE Toolbox 或 srsRAN/OAI 的 PDSCH 均衡作 golden，逐 RE 比对 X̂ 与 γ（误差阈值如相对 1e-3）。
- **CPU/GPU 一致性**：同一输入两端跑，比对输出；浮点差异容忍 ULP 级，但 SINR 趋势必须一致。
- **时延门限**：用合成满负载子帧测 p99 端到端时延，确认 < 1 ms 且抖动可控；GPU 路线重点验证流水线重叠率。

