export const KERNEL_QUESTIONS = [
  {
    id: 'kn-01',
    track: 'kernel',
    title: 'FlashAttention-1/2/3 原理演进与 Hopper 架构硬件协同 (TMA/WGMMA)',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['FlashAttention', 'Online Softmax', 'Tiling', 'TMA', 'WGMMA', 'Warp Specialization'],
    summary: '从 IO-Aware 显存层级出发，串联 FA1 Tiling/Online Softmax、FA2 序列并行与 Warp 重排，再到 FA3 Hopper 异步流水线。',
    contextSpecs: ['SRAM 20MB vs HBM 80GB', 'Online Softmax', 'Hopper TMA + WGMMA', 'FP8 Incoherent'],
    prompt: 'FlashAttention 是现代 LLM 训练和推理最核心的算子。请讲清楚：(1) FlashAttention-1 是如何通过 Tiling + Online Softmax + 反向重算消除 $O(N^2)$ HBM 读写的？(2) FlashAttention-2 相比 FA1 做了哪三项关键改进把利用率从 ~35% 拉到 ~70%？(3) 到了 Hopper H100 上，FlashAttention-3 又如何结合 TMA、WGMMA 和 Warp Specialization 实现极致异步流水？',
    rubric: [
      {
        id: 'fa1-io',
        label: 'FA1 核心：IO-Aware 分块 (Tiling) 把 Q/K/V 载入片上 SRAM，用 Online Softmax 维护局部最大值 m 和归一化分母 l，避免把 N×N Attention Matrix 写回 HBM；反向仅存 logsumexp 重算',
        weight: 26,
        keywords: ['sram', 'hbm', 'tiling', '分块', 'online softmax', 'n^2', 'o(n)', 'logsumexp', '重算', 'recompute'],
        probe: 'Softmax 需要对一整行求 max 和 sum，分块计算怎么保证算完最后一个 K 块时结果和全局 Softmax 数学完全一致？反向传播不存 N×N 矩阵那要存什么？',
      },
      {
        id: 'fa2-improvements',
        label: 'FA2 三大改进：(1) 减少非矩阵乘 FLOPs (推迟 rescale 到循环末尾)；(2) 外循环换 Q 内循环换 KV 并沿序列长度并行提高低 Batch Occupancy；(3) Warp 切分从 Split-K 改为 Split-Q 消除 Shared Memory 归约通信',
        weight: 26,
        keywords: ['fa2', '非矩阵', 'rescale', '序列维度', '并行', 'occupancy', 'split-q', 'split-k', 'warp', 'shared memory'],
        probe: 'FA1 在长序列小 Batch（比如 B=1, S=16K）时很多 SM 都在闲置，而且 Warp 之间要频繁通过 Shared Memory 归约。FA2 是怎么调换内外循环和 Warp 切分方式的？',
      },
      {
        id: 'fa3-hopper',
        label: 'FA3 Hopper 硬件协同：Warp Specialization (Producer Warp 调 TMA 异步搬数据，Consumer Warp 调异步 WGMMA) + Ping-Pong 调度重叠 GEMM 与 Softmax',
        weight: 24,
        keywords: ['fa3', 'hopper', 'tma', 'wgmma', 'warp specialization', 'producer', 'consumer', '异步', 'ping-pong', 'softmax', '重叠'],
        probe: 'H100 引入了 TMA 和异步 WGMMA。FA3 为什么要把一个 Threadblock 里的 Warp 拆成 Producer 和 Consumer？又是怎么把低速的 CUDA Core Softmax 藏在 Tensor Core GEMM 后面的？',
      },
      {
        id: 'fa3-fp8',
        label: 'FA3 FP8 精度优化：Block Quantization + Incoherent Processing (随机正交/Hadamard 旋转打散离群值) 压低 FP8 Attention 误差',
        weight: 12,
        keywords: ['fp8', 'hadamard', 'incoherent', '旋转', '离群值', 'block quantization', '精度'],
        probe: 'FA3 直接把 QKV 量化到 FP8 跑 Attention 时遇到了离群值精度问题，论文里用了什么数学技巧来平滑离群值？',
      },
      {
        id: 'io-complexity',
        label: 'HBM IO 复杂度对比：标准 Attention 为 O(N² + Nd)，FA 降为 O(N²d²/M)（M 为 SRAM 容量），大幅降低读写量',
        weight: 12,
        keywords: ['复杂度', 'io', 'm', '带宽', 'memory-bound', '访存'],
        probe: '定性或定量说一下，为什么标准 Attention 在长序列下是严重的 Memory-Bound 而不是 Compute-Bound？',
      },
    ],
    hints: [
      'L1 方向提示：记住 FA 的演进主线——**FA1 省 HBM 访存 (IO-Aware)** → **FA2 榨干 Ampere Tensor Core (调循环/提并行/去规约)** → **FA3 榨干 Hopper 异步硬件 (TMA/WGMMA/Warp 特化)**。',
      'L2 FA2 循环对调：FA1 外循环是 K/V、内循环是 Q（每次内循环都要写回并读出部分输出 O）；FA2 改为**外循环 Q、内循环 K/V**，单个输出块 $O_i$ 全程驻留在高速寄存器里累加，只写回 HBM 一次！',
      'L3 FA3 Ping-Pong：因为 H100 上 Tensor Core (GEMM) 极快而 Special Function Unit (`exp2` 做 Softmax) 很慢，FA3 让两个 Warp Group 交替执行——WG1 在跑 Softmax (CUDA Core/SFU) 时，WG2 正在满载跑 WGMMA (Tensor Core)！',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Decode 瓶颈与 FlashDecoding）：FA2 虽然沿序列维并行了 Query，但在推理 Decode 阶段 $S_q = 1$（每次只生成 1 个 Token），Query 维度根本没法切分，如果 Batch=1 且 Head 少，H100 的 132 个 SM 只用到十几个！FlashDecoding / Split-KV 是怎么解决的？',
      },
      {
        stage: 'corner',
        question: '追问 2（Causal Mask 空块跳过）：在自回归 Causal Attention 中，矩阵右上角一半全是被 Mask 掉的 0。FA2 在调度 Threadblock 时如果按顺序映射，依然会遇到不同 SM 负载不均（Wave Quantization），工程上怎么优化 Block 调度顺序？',
      },
    ],
    referenceAnswer: `### Staff 级 FlashAttention 演进精讲

#### 1. FlashAttention-1：打破 $O(N^2)$ HBM 瓶颈
- **痛点**：标准 Attention 必须把 $S = QK^T \\in \\mathbb{R}^{N\\times N}$ 和 $P = \\text{softmax}(S)$ 完整物化写回 HBM 再读出，长序列下 HBM IO 成为绝对瓶颈且显存 $O(N^2)$ 爆炸。
- **解法**：
  - **Tiling + Online Softmax**：将 $Q, K, V$ 分块加载到片上 SRAM，逐块计算 $S_{ij}$ 并动态维护行最大值 $m_i$ 与指数和 $l_i$，借助 rescale 修正因子迭代更新输出 $O_i$，全程不把 $N\\times N$ 矩阵落盘 HBM；
  - **反向重算**：前向仅保存输出 $O$ 和 $O(N)$ 大小的 \`logsumexp\` ($L_i = m_i + \\log l_i$)，反向按块在 SRAM 内即时重算 $P_{ij}$，换取显存降至 $O(N)$ 且端到端更快。

#### 2. FlashAttention-2：把 Tensor Core 利用率翻倍到 ~70%
1. **减少非 MatMul 开销**：Tensor Core 吞吐是 CUDA Core 标量算术的 16×；FA2 推迟每步对 $O_i$ 的完整除 $l_i$ 缩放，统一留到内循环结束一次做完。
2. **外循环 Q / 内循环 KV + 序列维并行**：除了 Batch 和 Head 维，增加沿 **Query Sequence 维** 切分给不同 Threadblock，同时让 $O_i$ 留在寄存器累加，只写回 HBM 一次。
3. **Warp 切分改 Split-Q**：FA1 切分 K/V（Split-K）导致多 Warp 算同一行 $O_i$ 必须经过 Shared Memory 同步归约；FA2 改为各 Warp 独立负责不同的 Q 行，零跨 Warp 归约！

#### 3. FlashAttention-3：Hopper 架构深度异步协同
- **Warp Specialization**：把一个 CTA 内的 Warp 拆为 **Producer**（只调 TMA 异步把 Q/K/V 搬进 SMEM，寄存器配额降到极低）与 **Consumer**（拿走省下的高寄存器配额专心跑异步 WGMMA + Softmax）。
- **Ping-Pong Scheduling & Intra-Warp Overlap**：两个 Consumer Warp Group 错相交替执行——一组占着 Tensor Core 跑 WGMMA 时，另一组占着 SFU/CUDA Core 算 \`exp2\` Softmax，把慢速 Softmax 完全藏在矩阵乘后面。
- **FP8 Incoherent Processing**：对 Q/K 乘随机正交 Hadamard 矩阵打散通道离群值（因 $Q H (KH)^T = Q K^T$ 数学恒等！），大幅压低 FP8 量化误差。

#### 4. Decode 追问：FlashDecoding (Split-KV)
- 当 Decode $S_q=1$ 且 Batch 很小时，Query 维无法切分；**FlashDecoding** 改为沿 **KV 序列长度切分 (Split-KV)** 给大量 SM 并行算局部 Attention 和 \`logsumexp\`，最后用一个极轻量 Reduction Kernel 合并，打满全部 132 个 SM 的 HBM 带宽！`,
  },
  {
    id: 'kn-02',
    track: 'kernel',
    title: 'NCCL 集合通信与大规模集群网络：Ring vs Tree、Rail 拓扑与拥塞控制',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['NCCL', 'AllReduce', 'Ring vs Tree', 'Rail-Optimized', 'RoCEv2/IB', 'SHARP'],
    summary: '定量推导 Ring/Tree AllReduce 带宽与延迟公式，深入解析万卡集群 Rail-Optimized 拓扑与 RoCEv2 ECN/PFC 调优。',
    contextSpecs: ['Ring vs Tree', '2(N-1)/N · S', 'Rail-Optimized IB/RoCE', 'NVLS / SHARP'],
    prompt: '大规模分布式训练和推理高度依赖 NCCL 集合通信。请你：(1) 定量对比 Ring AllReduce 与 Tree AllReduce 的通信量、带宽效率以及延迟复杂度，NCCL 什么场景选 Ring、什么场景选 Tree？(2) 什么是 Rail-Optimized 网络拓扑，它为什么特别契合 8 卡服务器的大模型训练？(3) 如果集群用的是 RoCEv2 以太网而非 InfiniBand，最容易踩哪些拥塞控制坑（ECN/PFC/ECMP Hash 极化）？',
    rubric: [
      {
        id: 'ring-vs-tree',
        label: 'Ring vs Tree 精确对比：Ring 为 ReduceScatter+AllGather，单卡收发 2(N-1)/N·S (带宽最优但延迟 O(N))；Tree/Double-Binary-Tree 延迟 O(log N) 适合小消息或超大规模节点',
        weight: 26,
        keywords: ['ring', 'tree', '2(n-1)/n', 'reducescatter', 'allgather', 'o(n)', 'o(log', '小消息', '大消息', '带宽最优', '延迟最优'],
        probe: 'Ring AllReduce 在 $N$ 张卡上同步大小为 $S$ 的张量，每张卡的发送字节量是多少？为什么卡数 $N$ 到几千时，哪怕是大张量 NCCL 也会切到 Tree 或混合算法？',
      },
      {
        id: 'nvls-sharp',
        label: '硬件网内计算加速：NVLink Switch SHARP (NVLS) / IB SHARP 把归约下沉到交换机芯片，通信量减半（每卡仅发 S 收 S）且不占额外 SM 算力',
        weight: 16,
        keywords: ['sharp', 'nvls', '网内计算', 'in-network', 'switch', '交换机', '减半', 'nvswitch'],
        probe: 'H100 / B200 的 NVSwitch 和 InfiniBand 交换机支持 SHARP / NVLS 网内计算，这相比传统 Ring AllReduce 把网卡流量砍掉了多少？',
      },
      {
        id: 'rail-topo',
        label: 'Rail-Optimized 拓扑：单机 8 张 GPU 的第 i 张网卡直连第 i 个 Leaf (Rail) 交换机，使同号 GPU 跨机 DP/PP 通信仅走 1 跳 Leaf 极少上 Spine，且匹配 PXN 机内转发',
        weight: 22,
        keywords: ['rail', 'leaf', 'spine', '同号', '一跳', '1跳', 'dp', 'pxn', '网卡', 'pci', 'numa'],
        probe: '为什么 AI 集群不把一台机器的 8 张网卡全插到同一台接入交换机，而是刻意按 GPU 编号 0~7 分别连到 8 个不同的 Rail Leaf 交换机？',
      },
      {
        id: 'roce-congestion',
        label: 'RoCEv2 痛点：ECMP 静态五元组哈希冲突（大象流哈希极化导致链路不均）、PFC 死锁/队头阻塞与暂停帧风暴，需精细配置 ECN + DCQCN + 负载均衡（Packet Spraying/Adaptive Routing）',
        weight: 22,
        keywords: ['roce', 'ecmp', '哈希', 'hash', '极化', 'pfc', 'ecn', 'dcqcn', '死锁', '风暴', '队头阻塞', 'adaptive routing', 'spray'],
        probe: '相比原生带信用流控和自适应路由的 InfiniBand，RoCEv2 在跑高负载 All-to-All 或 DP AllReduce 时为什么特别容易掉速甚至触发雪崩？',
      },
      {
        id: 'pxn-topo',
        label: '机内 PCIe/NVLink 亲和与 NCCL 调优：GPU-NIC 1:1 绑定同一 PCIe Switch 避开跨 CPU UPI/QPI，合理控制 NCCL Channel/SM 占用',
        weight: 14,
        keywords: ['pcie', 'nic', '亲和', 'upi', 'qpi', 'gpudirect', 'channel', 'sm', '拓扑'],
        probe: '单机里如果 GPU 0 跨 CPU Socket 去用挂载在另一个 NUMA 节点的 NIC 4 做 GPUDirect RDMA，会发生什么性能退化？',
      },
    ],
    hints: [
      'L1 方向提示：Ring 看“吞吐接近理论峰值 $B$ 但步数 $2(N-1)$”，Tree 看“步数 $2\\log_2 N$ 延迟极低（Double Binary Tree 同样吃满双向带宽！）”。',
      'L2 Rail 拓扑直觉：当单机 8 卡跑机内并行（如 TP/HSDP/EP=8）时，跨机通信往往发生在**所有机器的同号 GPU 之间（GPU 0 ↔ GPU 0 ...）**！按 Rail 连线后，同一 DP/PP 组里多台机器的同号 GPU 全挂在同一 Leaf 交换机下，1 跳直达不经 Spine！',
      'L3 RoCEv2 三大坑：ECMP 哈希冲突（两条大象流撞同一上行口）+ ECN 阈值没配好导致过早触发 PFC 暂停帧 + PFC 蔓延成 Pause Storm / 死锁。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（All-to-All vs Rail）：Rail-Optimized 拓扑对 DP AllReduce 简直完美（全在 Rail 内一跳完成），但在 MoE **EP All-to-All** 场景下（GPU 0 要发给另一台机器的 GPU 3），流量必须穿过 Spine，此时 NCCL 的 **PXN (PCIe × NVLink)** 技术是怎么巧妙绕开跨 Rail 的？',
      },
      {
        stage: 'corner',
        question: '追问 2（BusBW vs AlgBW）：大家测 `nccl-tests` 时输出表里有 `algbw`（算法带宽）和 `busbw`（总线带宽）两列，对于 AllReduce 它们之间的换算系数为什么是 $2\\frac{N-1}{N}$？验收集群网络时该看哪一个？',
      },
    ],
    referenceAnswer: `### Staff 级 NCCL 与集群网络深度解析

#### 1. Ring vs Tree AllReduce 定量对比（卡数 $N$，消息大小 $S$，单向链路带宽 $B$，单跳延迟 $\\alpha$）
- **Ring AllReduce**（1 次 ReduceScatter + 1 次 AllGather）：
  - 每卡总发送量：$2\\frac{N-1}{N} S$（约 $2S$），**总线带宽 \`BusBW = AlgBW × 2(N-1)/N\`**；
  - 耗时：$T_{\\text{ring}} = 2(N-1)\\alpha + 2\\frac{N-1}{N}\\frac{S}{B}$；
  - **特点**：中等 $N$ + **大消息**时带宽利用率近 100%；但延迟 $O(N)$ 随卡数线性恶化。
- **Double Binary Tree AllReduce**：
  - 耗时：$T_{\\text{tree}} = O(\\log N)\\cdot \\alpha + 2\\frac{S}{B}$（双互补树同时打满叶节点上下行带宽）；
  - **特点**：**小消息**或**超大卡数 ($N \\ge 64\\sim 1024$)** 时延迟优势碾压 Ring。
- **NVLS / IB SHARP（网内计算）**：把规约加法下沉到 NVSwitch/IB 交换机 ASIC 中，每卡只需发 $1S$、收 $1S$，有效总线流量直降 **50%**！

#### 2. Rail-Optimized 拓扑与 PXN 协同
- 机内 8 张 GPU 各自配一张就近同一 PCIe Switch 下的 NIC（保证 GPUDirect RDMA 不跨 CPU QPI/UPI），每台主机的 NIC $k$ 统一接入第 $k$ 号 **Rail Leaf 交换机**。
- 当单机 8 卡运行机内并行域（TP / HSDP / EP）时，跨机 **DP** 和 **PP** 通信大量发生在**同号 GPU 之间**，全部在同一 Rail Leaf 内 **1 跳完成**，极大减轻 Spine 拥塞。
- **PXN 妙用（针对 All-to-All 跨 Rail 通信）**：当主机 A 的 GPU 0 要发数据给主机 B 的 GPU 3 时，不再跨 Spine，而是先在主机 A 内经超高速 **NVLink** 转给同机 GPU 3，再由 GPU 3 通过本 Rail 网卡 1 跳直达主机 B 的 GPU 3！

#### 3. RoCEv2 核心拥塞痛点与治理
- **ECMP 哈希极化**：AI 训练流数量极少但单流极大（Elephant Flows），静态五元组 ECMP 极易把多条大流哈希到同一条 Spine 链路造成局部丢包 → 开启 **QP 多流拆分 (\`NCCL_IB_QPS_PER_CONNECTION=4/8\`)** 或硬件 **Packet Spraying / Adaptive Routing**。
- **PFC 风暴与 ECN 配合**：RoCEv2 依赖无损以太网 PFC，若 ECN (DCQCN) 降速阈值高于 PFC 暂停阈值，队列来不及降速就会反压触发级联 PFC Pause Storm 拖垮整个集群 → 严格保证 **ECN Mark 阈值 < PFC XOFF 阈值** 并开启 PFC Watchdog 防死锁。`,
  },
  {
    id: 'kn-03',
    track: 'kernel',
    title: 'CUDA Kernel 性能调优方法论：Nsight 分析、Bank Conflict、Occupancy 与 Triton',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['CUDA', 'Triton', 'Nsight', 'Bank Conflict', 'CUDA Graph', 'Occupancy'],
    summary: '建立自顶向下的 GPU Kernel 调优闭环：nsys 定位系统瓶颈 -> ncu Roofline 拆解算子 -> 解决访存合并、Bank Conflict 与 Launch Overhead。',
    contextSpecs: ['nsys / ncu', 'Coalesced Access', '32-Bank Swizzle', 'CUDA Graphs / Triton'],
    prompt: '给你一个运行缓慢的 PyTorch 自定义推理或训练管线，请你讲述完整的 GPU 性能分析与 CUDA/Triton Kernel 调优方法论：怎么区分是 Host Launch Bound、Memory Bound 还是 Compute Bound？在手写 CUDA/Triton Kernel 时，如何排查和解决 Global Memory 非合并访存、Shared Memory Bank Conflict 以及 Occupancy 过低问题？',
    rubric: [
      {
        id: 'nsys-ncu',
        label: '两级 Profiling 工具链：先用 Nsight Systems (nsys) 看全局时间线查 Host Gap / 通信阻塞 / 小 Kernel Launch；再用 Nsight Compute (ncu) 深入单 Kernel Roofline、SOL (Speed of Light) 和 Warp Stall 原因',
        weight: 26,
        keywords: ['nsys', 'ncu', 'nsight', 'timeline', 'gap', 'host', 'roofline', 'sol', 'stall', 'profiling'],
        probe: '拿到一个慢程序，你是直接上 `ncu` 还是先跑 `nsys`？在 `nsys` 视图里看到什么特征说明瓶颈根本不在 GPU 算子本身，而在 CPU Host 侧？',
      },
      {
        id: 'cuda-graph',
        label: '消除 Host Launch Overhead：对大量微秒级小算子（如 Decode 阶段）使用 CUDA Graphs 整图捕获发射，消除 CPU Launch 间隙',
        weight: 18,
        keywords: ['cuda graph', 'launch', 'overhead', '捕获', '小算子', 'cpu bound', 'torch.compile'],
        probe: 'Decode Batch=1 时每个算子在 GPU 上只跑 5 微秒，但 Python 发射一个 CUDA Kernel 就要 10 微秒，GPU 时间线全是锯齿空洞，怎么解决？',
      },
      {
        id: 'coalesced-mem',
        label: 'Global Memory 合并访存 (Coalesced Access) 与向量化加载 (`float4` / 128-bit LDG)：保证同一 Warp 32 线程访问连续对齐的 128B Cache Line',
        weight: 20,
        keywords: ['合并访存', 'coalesced', '128', 'float4', '向量化', 'cache line', 'warp', '对齐', 'stride'],
        probe: '同一个 Warp 里的 32 个线程按列访问一个行主序矩阵，为什么实测 HBM 带宽会直接暴跌几倍？怎么改？',
      },
      {
        id: 'bank-conflict',
        label: 'Shared Memory Bank Conflict：32 个 4-Byte Bank，同 Warp 多线程访问同 Bank 不同地址会串行化；通过 Padding (+1/+8) 或 XOR Swizzle 消除',
        weight: 20,
        keywords: ['bank conflict', '32', 'padding', 'swizzle', 'xor', 'shared memory', 'smem', '冲突', '串行'],
        probe: 'Shared Memory 为什么会发生 Bank Conflict？工程上除了简单加 Padding 浪费空间，CUTLASS / FlashAttention 普遍用的零额外显存方案是什么？',
      },
      {
        id: 'occupancy-triton',
        label: 'Occupancy 与资源权衡（Registers / SMEM / Block Size，并非越高越好，有时高寄存器+ILP 更胜高 Occupancy）及 Triton Block 级编程自动管理合并访存与软流水',
        weight: 16,
        keywords: ['occupancy', '寄存器', 'register', 'smem', 'ilp', 'triton', 'num_stages', 'num_warps', '流水'],
        probe: '是不是 Occupancy 越接近 100% Kernel 就一定越快？以及在写 Triton Kernel 时，哪几个核心 Meta 参数 (`BLOCK_SIZE`, `num_warps`, `num_stages`) 直接决定性能？',
      },
    ],
    hints: [
      'L1 方向提示：牢记黄金顺序——**先宏观 `nsys` 抓系统空洞与通信，再微观 `ncu` 看 SOL (Compute% vs Memory%) 和 Warp Stall Breakdown**。',
      'L2 Swizzle 原理：把行号的高位和列号做异或 \`col_swizzled = col ^ row\`，既保证行读取无冲突，又让列读取（如转置或 MMA 加载）分散到不同 Bank，而且**零额外 Shared Memory 浪费**！',
      'L3 Occupancy 误区：在 GEMM / FlashAttention 等 Compute-bound 算子里，往往为了用满每线程 128~255 个寄存器存更大矩阵 Tile（提高数据复用与指令级并行 ILP），Occupancy 故意压在 25%~50% 反而最快！',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Warp Stall 诊断）：你在 `ncu` 里看到一个 Memory-Bound 融合算子的主要 Warp Stall 原因是 `Stall Long Scoreboard`（高达 65%），这代表 GPU 流水线正在干等什么？应该怎么优化？',
      },
      {
        stage: 'corner',
        question: '追问 2（Triton vs CUTLASS/CUDA）：既然 Triton 写融合算子（如 RMSNorm/RoPE/CrossEntropy）又快又好维护，为什么像 FlashMLA、DeepGEMM、Marlin 这类顶级核心算子大家还是回归手写 CUDA / CUTLASS / PTX？Triton 在 Hopper/Blackwell 上的主要局限是什么？',
      },
    ],
    referenceAnswer: `### Staff 级 GPU 调优闭环

#### 1. 自顶向下 Profiling 方法论
1. **宏观体检 (\`nsys\` / PyTorch Profiler)**：先看 GPU Stream 覆盖率。若时间线上满是零散小算子和大量空白 Gap，说明是 **CPU Host Launch Bound**（上 **CUDA Graphs** 或 \`torch.compile\` / 算子融合）或 DataLoader/同步 \`cudaStreamSynchronize\` 卡顿。
2. **微观解剖 (\`ncu\`)**：查看 **SOL (Speed of Light)** 指标判断 Compute vs Memory 瓶颈，并看 **Warp State Statistics**：
   - \`Stall Long Scoreboard\`：正在干等 Global Memory / L2 数据返回 → 提高访存合并度、128-bit 向量化加载 (\`LDG.128\`)、引入 \`cp.async\` / TMA **双缓冲软件流水 (Software Pipelining)** 预取下一块数据；
   - \`Stall Short Scoreboard\` / \`Mio Throttle\`：Shared Memory 带宽打满或严重 Bank Conflict。

#### 2. 经典 CUDA / Triton 底层优化手段
- **Global Memory 合并访存**：确保同 Warp 32 线程访问连续地址（合并为最少 128B 事务），并用 \`float4\` / \`int4\` 单指令加载 16 Bytes 减少指令发射数。
- **消除 Shared Memory Bank Conflict**：Shared Memory 分为 32 个 4-Byte Bank。避免按 32 步长跨行访问撞同 Bank；经典方法是对行宽做 **Padding**，工业界 SOTA（CUTLASS/FA）采用 **XOR Swizzling**（无额外 SMEM 开销且完美兼容 \`ldmatrix\`）。
- **理性看待 Occupancy**：Memory-bound 算子需要较高 Occupancy（如 50%+）靠多 Warp 切换隐藏长访存延迟；Compute-bound 算子（GEMM/FA）更看重单线程寄存器 Tile 复用与异步流水线，Occupancy 在 25%~50% 往往已是全局最优。
- **Triton 调优与边界**：用 Triton 写 RMSNorm/SwiGLU/Chunked-CE 极高效，核心调优 \`BLOCK_M/N/K\`、\`num_warps\` 与 \`num_stages\`（软件流水深度）；但在需要极致压榨 Hopper **Warp Specialization、PTX 细粒度寄存器分配、非规则位运算反量化 (Marlin) 或 IBGDA 通信融合 (DeepEP)** 时，仍需手写 CUTLASS/CUDA/PTX。`,
  },
  {
    id: 'kn-04',
    track: 'kernel',
    title: 'GPU 架构代际演进：A100 -> H100 -> H200 -> GB200 NVL72 对 LLM 系统的颠覆',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['A100', 'H100', 'H200', 'B200', 'GB200 NVL72', 'FP4', 'NVLink Domain'],
    summary: '站在硬件第一性原理，对比四代核心 GPU 的算力、HBM 带宽/容量与 NVLink 域扩张如何重塑训练并行度与 MoE 推理形态。',
    contextSpecs: ['A100 80G', 'H100/H200 141G', 'B200 192G 8TB/s', 'GB200 NVL72 (72-GPU NVLink)'],
    prompt: '作为 AI Infra 架构师，硬件选型和提前适配下一代架构是核心职责。请对比 A100、H100、H200 到 Blackwell (B200 / GB200 NVL72) 四代 GPU 的关键硬件跃迁（算力格式、HBM 容量与带宽、互联域规模），并重点分析：**GB200 NVL72（72 张 GPU 在同一个 NVLink 全互联域内，总计 130 TB/s 聚合带宽、~13.8TB 统一 HBM3e）** 对大规模 MoE 推理和超长上下文训练带来了哪些架构级的颠覆？',
    rubric: [
      {
        id: 'specs-evolution',
        label: '精准掌握四代核心参数：A100 (312TF BF16, 2TB/s, 80G) -> H100 (989TF BF16 / 1979TF FP8, 3.35TB/s, 80G) -> H200 (141G, 4.8TB/s) -> B200 (2.25PF BF16 / 4.5PF FP8 / 9PF FP4, 192G, 8TB/s)',
        weight: 24,
        keywords: ['a100', 'h100', 'h200', 'b200', '141', '192', '4.8', '8tb', 'fp8', 'fp4', '3.35', 'hbm3e'],
        probe: '对比 H100 和 H200，计算核心完全没变，为什么很多团队在推理 70B/MoE 时愿意溢价买 H200？HBM 容量和带宽具体涨了多少？',
      },
      {
        id: 'nvl72-domain',
        label: 'GB200 NVL72 颠覆点 1：NVLink 互联域从单机 8 卡暴涨 9× 到整机柜 72 卡（每卡 1.8 TB/s 双向），打破了过去“TP/EP 尽量限制在 8 卡内”的铁律！',
        weight: 26,
        keywords: ['nvl72', '72卡', '1.8', 'nvlink域', '打破', 'tp', 'ep', '单机8卡', '全互联', '130tb'],
        probe: '过去十年我们背的八股都是“TP ≤ 8 限制在机内 NVLink，跨 8 卡就得走慢 10 倍的 IB”。到了 GB200 NVL72 上，这个架构前提发生了什么质变？',
      },
      {
        id: 'nvl72-moe',
        label: '对超大规模 MoE 推理的颠覆：671B/万亿级 MoE 可以在单个 NVL72 机柜内跑 EP72 全 NVLink 极速 All-to-All（彻底消除跨机 IB 瓶颈），同时压低 TPOT 与拉爆吞吐',
        weight: 22,
        keywords: ['moe', 'ep72', 'all-to-all', 'ib', 'rdma', 'tpot', '万亿', '极低延迟', 'wide-ep'],
        probe: '如果把 DeepSeek-R1 671B 或未来 2T 参数的 MoE 部署在 GB200 NVL72 上跑 Decode，EP72 全在 NVLink 域内执行有什么惊人收益？',
      },
      {
        id: 'fp4-ridge',
        label: '算力增长快于内存带宽（Ridge Point 持续右移）：B200 原生 NVFP4/MXFP8 成为必选项，否则 Decode 严重吃不满暴涨的 Tensor Core',
        weight: 16,
        keywords: ['fp4', 'nvfp4', 'mxfp8', 'ridge', '内存墙', 'memory wall', '算力带宽比', '量化'],
        probe: 'A100 -> H100 -> B200，算力涨了十几倍，但 HBM 带宽只涨了 4 倍（2TB/s -> 8TB/s）。算力带宽比越来越悬殊，这逼着 AI Infra 必须往哪些技术方向演进？',
      },
      {
        id: 'reliability-nvlink',
        label: '新挑战：NVL72 单机柜液冷功耗高达 ~120kW、故障爆炸半径扩大到 72 卡域，需更强的机柜级容错、NVLink 子域隔离与液冷运维能力',
        weight: 12,
        keywords: ['液冷', '功耗', '120kw', '容错', '爆炸半径', '可靠性', 'mtbf', '隔离'],
        probe: 'NVL72 这么强，它给机房基础设施和分布式集群容错带来了什么新的棘手挑战？',
      },
    ],
    hints: [
      'L1 方向提示：从“单卡算力与精度跃迁 (BF16->FP8->FP4)” + “HBM 墙与 H200/B200 扩容” + “NVLink 互联域从 8 卡跃迁到 72 卡”三条主线拆解。',
      'L2 H200 性价比真相：H100 80G 跑 70B BF16 单卡放不下，跑 8 卡又有 TP 通信开销；H200 141G + 4.8TB/s (+43% 带宽, +76% 容量) 让推理并发 KV 暴涨、Decode 速度直接快 1.4× 以上！',
      'L3 NVL72 质变：单机柜就是一台拥有 **13.8 TB 统一高速显存 + 72 卡 1.8TB/s 全 NVLink 互联** 的巨型“单机”！EP72 全走 NVLink/NVLS，彻底消除跨机 IB 瓶颈；甚至可以轻松做超大 TP/CP 支撑百万级 Context 实时交互。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（NVFP4 vs W4A16）：Blackwell 推出了带微块缩放的 NVFP4（每 16 个元素共享一个 FP8 E4M3 Scale，外加二级 Tensor Scale）。相比传统 W4A16，NVFP4 硬件原生指令如何同时加速 Prefill 和 Decode？',
      },
      {
        stage: 'corner',
        question: '追问 2（异构成本测算）：如果你们公司要采购一批推理算力跑在线 72B / DeepSeek-R1 服务，给你 H200、B200 和 H20 三种选项，你会从哪几个定量指标（$/M-Token 成本模型）做 TCO 选型决策？',
      },
    ],
    referenceAnswer: `### Principal 级 GPU 硬件代际演进与系统影响

#### 1. 核心规格演进速查（SXM / 稠密算力）
| 架构 | 关键算力 (Dense) | HBM 容量 & 带宽 | NVLink 域 & 单卡双向带宽 | 核心架构特性 |
|---|---|---|---|---|
| **A100** | 312 TF (BF16) | 80GB / 2.0 TB/s | 8 GPU / 600 GB/s | TF32, 基础 Async Copy |
| **H100** | 989 TF (BF16) / 1979 TF (FP8) | 80GB / 3.35 TB/s | 8 GPU / 900 GB/s | TMA, WGMMA, Transformer Engine FP8 |
| **H200** | 同 H100 | **141GB / 4.8 TB/s** | 8 GPU / 900 GB/s | Decode 带宽+43%、KV 显存暴涨，推理甜点卡 |
| **B200 / GB200** | ~2.25 PF (BF16) / 4.5 PF (FP8) / **9 PF (FP4)** | **180GB (B200) / 192GB (GB200) · 8.0 TB/s** | **NVL72: 72 GPU / 1.8 TB/s** | 硬件原生 MXFP8 / NVFP4, 5th Gen NVLink Switch |

#### 2. GB200 NVL72 对 LLM 架构的三大颠覆
1. **打破“单机 8 卡 NVLink 边界”**：72 张 GPU 处于同一个 1.8 TB/s 全互联 NVLink 域内（13.8 TB 显存池），彻底改写过去 3D 并行“TP/EP 不得超 8”的铁律！
2. **MoE 推理终极形态（单机柜 EP72 全 NVLink）**：DeepSeek-R1 671B 及未来 1T~2T 万亿级 MoE 可以在单个 NVL72 内展开 **DP-Attention + EP72**，所有 Dispatch/Combine 全走 1.8TB/s NVLink 而零跨机 IB 延迟，兼得超高单卡吞吐与极低 TPOT (<10ms)。
3. **百万 Context 与超高 Ridge Point 应对**：
   - 超长上下文训练/Prefill 可在 72 卡 NVLink 域内做高速 Ulysses/Ring CP；
   - 由于 B200 算力增长远快于 HBM 带宽，Ridge Point 进一步拉高（FP4 下高达 ~1125 FLOPs/Byte），**NVFP4 量化 + 投机采样 (MTP/EAGLE) + 大 Batch Wide-EP** 成为吃满算力的必选项。

#### 3. 落地新挑战与 TCO 选型
- **液冷与爆炸半径**：NVL72 整柜功耗 ~120kW（强依赖直接液冷），且单卡故障可能波及 72 卡 NVLink 域，需配合 NVLink 分区隔离与更激进的在线容错架构。
- **推理 TCO 核心公式**：决策不看单卡价格，而看 **\`$/Million-Tokens @ Target SLA (TTFT & TPOT)\`**——低延迟高并发复杂推理（如 R1）H200/GB200 凭借超高单卡带宽和 Wide-EP 占优；极致低成本离线批推或低并发长上下文场景，大显存高性价比卡（如 H20）有时 TCO 更低。`,
  },
  {
    id: 'kn-05',
    track: 'kernel',
    title: '高吞吐融合算子实战：Chunked Linear Cross-Entropy 与 Liger-Kernel',
    level: 'P6',
    frequency: '高频',
    incident: false,
    tags: ['Kernel Fusion', 'Liger-Kernel', 'Cross-Entropy', '128K Vocab', '显存优化'],
    summary: '定量揭示大词表 (128K~256K) 训练中 LM Head Logits 显存爆炸的惊人真相，并拆解 Fused Linear Cross-Entropy 分块融合实现。',
    contextSpecs: ['Vocab V=128K~256K', 'Logits [B·S, V] FP32', 'Liger-Kernel / Triton', 'Memory-Bound Fusion'],
    prompt: '现代 LLM 词表越来越大（Llama-3 是 128K，Qwen/Gemma 达到 152K~256K）。很多工程师在 SFT / 预训练微调长文本（如 Batch Tokens = 32K）时发现：明明模型不大，但一到最后算 `LM Head + CrossEntropyLoss` 就瞬间多吃掉 **30~40 GB 显存**直接 OOM！请你定量算一下这几十 GB 显存是怎么冒出来的，并详细设计一个 **Fused Chunked Linear Cross-Entropy** 融合算子（以及 RMSNorm/RoPE/SwiGLU 融合）来彻底消除它。',
    rubric: [
      {
        id: 'logits-oom-math',
        label: '定量推导显存爆炸：Token数 T=32K × 词表 V=152K × FP32 (4B) ≈ 19.5 GB，加上反向梯度 dLogits 再翻倍 ≈ 39 GB！',
        weight: 26,
        keywords: ['32k', '128k', '152k', '256k', 'fp32', '4字节', 'logits', '16gb', '20gb', '32gb', '40gb', '物化', '梯度'],
        probe: '假设单卡一次前向有 32,768 个 Token，词表大小 128K，PyTorch 默认 `CrossEntropyLoss` 内部还要转成 FP32。单单前向 Logits 和反向 dLogits 就会吃掉多少 GB 显存？',
      },
      {
        id: 'chunked-ce',
        label: 'Chunked Linear Cross-Entropy 核心思想：按 Token 维切小块 (如 chunk=4096)，逐块执行 Linear 算局部 Logits -> 立刻算 CE Loss 和梯度 dLogits -> 直接反传累加 dH 和 dW，绝不物化完整 [T, V] 张量！',
        weight: 28,
        keywords: ['chunk', '分块', '不物化', 'fused', '融合', '逐块', '原地', 'in-place', 'dh', 'dw', '累加', 'online'],
        probe: '既然前向不保存完整的 `[T, V]` Logits，那反向传播怎么算 `dH` (对隐藏态的梯度) 和 `dW` (对 LM Head 权重的梯度) 呢？',
      },
      {
        id: 'inplace-grad',
        label: '原地覆盖梯度技巧：在每个 Chunk 内算完 CE 梯度 `softmax(z) - y` 直接原地覆写该小块 Logits Buffer，零额外峰值显存；甚至直接在前向阶段就顺手算出 Chunk 的梯度（若只求标量 Loss）',
        weight: 18,
        keywords: ['原地', 'in-place', '覆写', 'softmax', 'liger', 'cut cross-entropy', 'apple', 'cce', '峰值'],
        probe: 'Liger-Kernel 和 Apple 的 Cut Cross-Entropy (CCE) 在前向和反向中如何进一步把临时显存压到几乎 0 MB？',
      },
      {
        id: 'elementwise-fusion',
        label: '其他高频融合算子：RMSNorm、RoPE、SwiGLU 全是纯 Memory-Bound，通过 Triton 融合将多次 HBM 读写合并为单次读+单次写（消除中间张量物化）',
        weight: 16,
        keywords: ['rmsnorm', 'rope', 'swiglu', 'triton', 'memory-bound', '读写', 'hbm', '中间张量', '融合'],
        probe: '除了 Cross-Entropy，Transformer 里还有哪些算子在 Liger-Kernel 里融合收益最高？它们的共同特征是什么？',
      },
      {
        id: 'tp-vocab',
        label: '分布式扩展：在 TP 下结合 VocabParallelCrossEntropy，仅需 AllReduce 极小的标量 max 和 sum_exp (O(T))，无需 AllGather 完整 Logits',
        weight: 12,
        keywords: ['vocab parallel', 'tp', 'allreduce', 'max', 'sum', 'online', 'megatron'],
        probe: '如果开启了 TP=8，LM Head 权重按词表维切分在 8 张卡上，算 CrossEntropy 时需要把 8 张卡的 Logits AllGather 拼起来吗？',
      },
    ],
    hints: [
      'L1 方向提示：先算显存账 \`T × V × 4B\`（32K × 128K × 4B = **16 GB**，再加 Softmax 中间结果和反向梯度直接飙到 **32~48 GB**！）。',
      'L2 Chunked 诀窍：把 \`[T, H]\` 按行切成 \`T/C\` 个小块（如 \`C = 4096\`）。每块临时算出的小 Logits 只有 \`4096 × 128K × 4B ≈ 2 GB\`（甚至切 1024 仅 500MB），在 Chunk 内就地算出 Loss 并顺带完成 \`dH_chunk = dLogits_chunk @ W\` 和 \`dW += H_chunk.T @ dLogits_chunk\` 累加，然后直接释放这块临时 Logits！',
      'L3 CCE 更狠：Apple Cut Cross-Entropy 连小块 Logits 都不落 HBM，直接用类似 FlashAttention 的自定义矩阵乘 Kernel 在片上 SRAM 算出 \`logsumexp\`，把显存降到几 MB 级别！',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（前向直接算反向？）：在 Liger-Kernel 的 `FusedLinearCrossEntropy` 实现里，你会发现它在 `forward()` 阶段就已经把 `dH` 和 `dW` 算好存起来了，`backward()` 时只是简单乘上一个上游 `grad_output` 标量！为什么这种设计可行，又有什么局限（比如高阶导数或非标量 Loss）？',
      },
      {
        stage: 'corner',
        question: '追问 2（`torch.compile` 够不够用）：既然 PyTorch 2 已经有了 `torch.compile` (Inductor) 自动算子融合，为什么我们还是经常要单独引入 Liger-Kernel 或手写 Triton 算子？Inductor 融合不了什么？',
      },
    ],
    referenceAnswer: `### Staff 级高吞吐融合算子设计

#### 1. 为什么大词表 \`LM Head + CrossEntropy\` 瞬间吃掉 30~48 GB？
- 设单卡 Token 总数 \`T = 32,768\`，词表 \`V = 152,064\`（Qwen），精度为 FP32（PyTorch CE 为防数值溢出需升 FP32）：
  - 前向 Logits 张量：\`32,768 × 152,064 × 4B ≈ 19.9 GB\`；
  - Softmax / LogSoftmax 缓存与反向 \`dLogits\` 再占 \`19.9 GB\` → **仅最后一层就吃掉 ~40 GB 显存**，直接击穿 80GB H100！

#### 2. Fused Chunked Linear Cross-Entropy 设计（Liger-Kernel / CCE 原理）
1. **沿 Token 维分块 (Chunking)**：将输入隐藏态 $H \\in \\mathbb{R}^{T \\times d}$ 切为小块 $H_c \\in \\mathbb{R}^{C \\times d}$（例如 $C=4096$ 或 $2048$）。
2. **逐块融合计算 + 原地梯度**：
   - 计算局部 Logits：$Z_c = H_c W^T$（仅占 ~1~2 GB）；
   - 调用 Triton **Fused Online Cross-Entropy Kernel**，在 SRAM 内一次遍历完成 \`max -> logsumexp -> loss\`，并**原地覆写** $Z_c \\leftarrow \\text{softmax}(Z_c) - y_c$ 得到梯度 $dZ_c$（零额外显存！）；
   - 趁 $dZ_c$ 还在，立刻算出该块梯度：$dH_c = dZ_c W$，$dW \\mathrel{+}= dZ_c^T H_c$；
   - 丢弃 $Z_c$，循环下一块！
3. **效果**：峰值显存从 **40 GB 暴降到 < 2 GB（直降 95%+）**，且减少海量 HBM 写回，端到端吞吐反而提升 15%~20%！
4. **更进一步（Apple Cut Cross-Entropy）**：借鉴 FlashAttention Tiling 思想，连分块 Logits 也不写入 HBM，在片上 SRAM 内完成矩阵乘点积与 \`logsumexp\` 规约，显存逼近 0 MB。

#### 3. 为什么还需要手写 Triton / Liger-Kernel 而非全靠 \`torch.compile\`？
- \`torch.compile\` (Inductor) 擅长 Pointwise/Reduction 局部融合（如 RMSNorm/SwiGLU），但难以自动推导出需要改变算法数学结构的**跨算子分块重排**（如 FlashAttention、Chunked Linear CE 的前向顺手算梯度）以及需要手工控制异步硬件原语的复杂 Kernel，且长变长动态 Shape 下有编译开销与 Recompile 抖动。`,
  },
];
