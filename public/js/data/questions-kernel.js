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
    contextSpecs: ['每个 SM 的片上存储与 HBM', 'Online Softmax', 'Hopper TMA + WGMMA', 'FP8 分块量化'],
    prompt: '以 dense attention 为例，解释 FA1 如何避免在 HBM 中物化 N×N 分数和概率矩阵，说明算术复杂度、工作显存和 HBM 访问量的区别。FA2 如何提高并行度和减少非矩阵运算？FA3 在 Hopper 上如何重叠 TMA、WGMMA 和 softmax？',
    rubric: [
      {
        id: 'fa1-io',
        label: '说明分块、在线 softmax 的重缩放及反向所需 Q/K/V、输出与行统计量',
        weight: 26,
        keywords: ['sram', 'hbm', 'tiling', '分块', 'online softmax', 'n^2', 'o(n)', 'logsumexp', '重算', 'recompute'],
        probe: 'Softmax 需要对一整行求 max 和 sum，分块计算怎么保证算完最后一个 K 块时结果和全局 Softmax 数学完全一致？反向传播不存 N×N 矩阵那要存什么？',
      },
      {
        id: 'fa2-improvements',
        label: 'FA2 减少非矩阵乘运算、沿 Q 块并行、warp 采用 sliced-Q 降低跨 warp 规约',
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
        label: '区分 dense FLOPs、分数矩阵的物化空间和 HBM IO 复杂度，注明片上容量 M 的单位',
        weight: 12,
        keywords: ['复杂度', 'hbm', '片上', 'io', '工作显存', 'nd', 'n²', 'n^2', '访存'],
        probe: '给定序长 N、head_dim d 和能容纳 M 个元素的片上空间，FA 减少了哪部分 IO？为何计算量仍随 N² 增长？',
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
        question: '追问 1（Decode）　当 S_q=1、batch 和 head 数也较小时，Flash-Decoding/Split-KV 如何为不同 SM 分工？局部输出怎样合并？',
      },
      {
        stage: 'corner',
        question: '追问 2（Causal）　哪些 K 块可以整块跳过？不同 Q 块所需工作量不同时，怎样减少最后一波 threadblock 的空闲？',
      },
    ],
    referenceAnswer: `### FA1 保存行统计量，反向按块重算

对一行 query 逐块计算 scores。设旧最大值、指数和与未归一化输出分别为 m、ℓ、u。新块 scores 为 sⱼ，取 m′=max(m,max sⱼ)，则 ℓ′=exp(m−m′)ℓ+Σexp(sⱼ−m′)，u′=exp(m−m′)u+Σexp(sⱼ−m′)vⱼ。最后 O=u/ℓ。块顺序在精确算术下不改变结果，浮点实现有舍入差异。反向用 Q/K/V、O 及行归一化统计量（如 logsumexp）重算局部概率，再求梯度。使用 dropout 时还需复现相同 mask。

**三种量**。消除的是全局物化的 N×N 中间矩阵，输入输出仍占 O(Nd)，行统计量 O(N)。dense 计算量仍为 Θ(N²d)。FA1 论文的两级存储模型中，M 按片上可容纳的元素计，在 d≤M≤Nd 的范围 HBM 访问量为 O(N²d²/M)，对照物化实现 Θ(Nd+N²)。tile 受单个 SM 可用寄存器与共享内存限制，不能使用整卡片上容量之和来定尺寸。

### FA2 与 FA3 利用什么并行性

- FA2 让不同 threadblock 持有不同 Q 行块并遍历 KV，改善小 batch、长 Q 的并行度。warp 之间划分 Q，减少对同一输出的共享内存归约。推迟最终归一化，减少非矩阵乘 FLOPs。最大值变化时仍要重缩放旧的累积量。
- FA3 在 Hopper 上用 producer/consumer warp specialization 和 TMA 搬运，用异步 WGMMA 及错相调度将矩阵乘与 softmax 重叠。FP8 路径采用块缩放等办法。对 Q、K 施加同一正交旋转保留 QKᵀ，能分散部分通道离群值。收益随精度和形状而变。
- **两项追问**。单 query 可按 KV 长度切给多组并行计算，各自保存局部输出和 logsumexp，再按其指数权重归一合并。切得过细将增加启动与规约成本。Causal prefill 可跳过全部位于未来的 K 块，只在对角边界做细粒度 mask。可先调度工作量大的 Q 块，按 batch/head 交错以缩短最后一波。

> 依据　FlashAttention（arXiv:2205.14135）、FlashAttention-2（2307.08691）和 FlashAttention-3（2407.08608）。`,
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
        label: 'Ring 每 rank 发送与接收各 2(N−1)S/N，延迟轮数 O(N)。Tree 的深度 O(log N)，说明模型与规模',
        weight: 26,
        keywords: ['ring', 'tree', '2(n-1)/n', 'reducescatter', 'allgather', 'o(n)', 'o(log', '小消息', '大消息', '带宽最优', '延迟最优'],
        probe: 'N 个 rank、每 rank 有 S 字节输入时，Ring 每 rank 发送多少？哪些延迟、带宽或拓扑条件会使 Tree 更有利？',
      },
      {
        id: 'nvls-sharp',
        label: '区分 NVSwitch NVLS 和 IB SHARP 的归约范围、端点字节与物理网络流量',
        weight: 16,
        keywords: ['sharp', 'nvls', '网内计算', 'in-network', 'switch', '交换机', '减半', 'nvswitch'],
        probe: '网内归约在哪个域执行？理想端点的发送量与链路上所有字节有什么不同？',
      },
      {
        id: 'rail-topo',
        label: 'Rail 对齐跨节点 NIC 与流量映射，实际跳数取决于 fabric。解释 PXN 机内中转',
        weight: 22,
        keywords: ['rail', 'leaf', 'spine', '同号', '一跳', '1跳', 'dp', 'pxn', '网卡', 'pci', 'numa'],
        probe: '8 张 NIC 分散在多个 rail 有什么好处？跨机同号 GPU 的路径在什么条件下仍会经过 spine？',
      },
      {
        id: 'roce-congestion',
        label: '用队列、ECN/CNP、PFC pause 与链路利用率诊断 RoCEv2，结合设备选择负载均衡',
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
      'L1 按每 rank 单向发送，Ring 有 2(N−1) 轮，每轮约 S/N 字节。树关注最深路径与分块流水。',
      'L2 同号 NIC 使用同一 rail 可避免跨 rail。每个 rail 自身也可能包含多层交换机。',
      'L3 分别检查 ECMP 偏斜、ECN 标记及 PFC 的队头阻塞。联查实际阈值、headroom 和端到端反馈延迟。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（All-to-All 与 Rail）　源 GPU 与目标 GPU 的本地 NIC 分属不同 rail 时，PXN 如何利用 NVLink 和同机另一张 NIC？',
      },
      {
        stage: 'corner',
        question: '追问 2（BusBW 与 AlgBW）　nccl-tests 的 AllReduce 为何以 2(N−1)/N 归一化？验收时怎样将这两个指标与 NIC 计数器对照？',
      },
    ],
    referenceAnswer: `### 固定字节口径后再比较算法

N 个 rank，每个输入 S 字节，每段启动延迟 α、单向有效带宽 B。标准 Ring 做 ReduceScatter 和 AllGather，各 N−1 轮、每轮每 rank 发送 S/N。每 rank **发送与接收各** 2(N−1)S/N 字节。忽略拥塞、规约计算与协议开销，T≈2(N−1)α+2(N−1)S/(NB)。nccl-tests 的 algbw=S/T，AllReduce 的 busbw=algbw×2(N−1)/N。它是归一化的有效带宽，硬件计数器才给出每条链路的实际流量。

二叉树 reduce+broadcast 的关键路径约 2⌈log₂N⌉ 跳。朴素无分块树每跳传 S，整棵树的每条边各传一份上行和下行。内部节点比叶节点忙。双互补树分割载荷、交错内外节点，分块流水后可兼顾带宽与启动延迟。NCCL 的选择依赖消息、互连、协议和版本，可用消息大小扫描加算法日志核实。

### 拓扑与拥塞追问

- **网内归约**。NVLS 在支持的 NVLink/NVSwitch 域工作，IB SHARP 利用 IB 网络的归约能力。理想端点各提交 S、取回 S，和大 N Ring 相比端点发送可接近减半。机内搬运、交换树流量、GPU 协调与跨域成本另计。
- **Rail 与 PXN**。把各节点的同号 NIC 放在同一 rail，按 GPU–NIC 亲和性放置通信组。一个 rail 自身可能跨多个 leaf/spine。PXN 可先经 NVLink 把数据交给更适合目标 rail 的同机 GPU，再从它附近的 NIC 发送，也可聚合消息。应同时看跨 NUMA 路径、机内流量和 NIC 计数。
- **RoCEv2**。ECMP 可能让大流撞在少数上行链路。用端口计数、队列水位、ECN/CNP、PFC 暂停时间和重传定位。ECN 配端侧拥塞控制应有足够时间在触发 PFC 前降速。PFC 在配置不当时会扩散拥塞。结合交换机/NIC 对乱序与自适应路由的支持选择多 QP 或负载均衡，按业务消息规模复测。

> 依据　NVIDIA nccl-tests《Performance》及 NVIDIA 关于 NCCL 2.4 双树与 NCCL 2.12 PXN 的技术说明。`,
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
        label: '由 CPU/GPU 相关时间线确认 launch 开销，再比较 CUDA Graph、融合和批量发射',
        weight: 18,
        keywords: ['cuda graph', 'launch', '发射', '融合', '捕获', '小算子', 'cpu bound', 'torch.compile'],
        probe: 'Decode Batch=1 时每个算子在 GPU 上只跑 5 微秒，但 Python 发射一个 CUDA Kernel 就要 10 微秒，GPU 时间线全是锯齿空洞，怎么解决？',
      },
      {
        id: 'coalesced-mem',
        label: '按 warp 的地址和实际传输的 32 B sector 分析合并访存，区分线程向量化和跨线程合并',
        weight: 20,
        keywords: ['合并访存', 'coalesced', '128', 'float4', '向量化', 'cache line', 'warp', '对齐', 'stride'],
        probe: '按列访问行主序矩阵时，每个 warp 要发起多少内存事务？怎样改变 lane 映射并验证实际有效带宽？',
      },
      {
        id: 'bank-conflict',
        label: '在常见 32 个 4 B bank 的模型下分析单次事务的访问与广播，针对 tile 选择 padding 或 swizzle',
        weight: 20,
        keywords: ['bank conflict', '32', 'padding', 'swizzle', 'xor', 'shared memory', 'smem', '冲突', '串行'],
        probe: 'Shared Memory 为什么会发生 Bank Conflict？工程上除了简单加 Padding 浪费空间，CUTLASS / FlashAttention 普遍用的零额外显存方案是什么？',
      },
      {
        id: 'occupancy-triton',
        label: '结合延迟隐藏、寄存器、共享内存和 spill 调 tile、num_warps、num_stages，以实际时间验证',
        weight: 16,
        keywords: ['occupancy', '寄存器', 'register', 'smem', 'ilp', 'triton', 'num_stages', 'num_warps', '流水'],
        probe: '是不是 Occupancy 越接近 100% Kernel 就一定越快？以及在写 Triton Kernel 时，哪几个核心 Meta 参数 (`BLOCK_SIZE`, `num_warps`, `num_stages`) 直接决定性能？',
      },
    ],
    hints: [
      'L1 方向提示：牢记黄金顺序——**先宏观 `nsys` 抓系统空洞与通信，再微观 `ncu` 看 SOL (Compute% vs Memory%) 和 Warp Stall Breakdown**。',
      'L2 对 float tile 写出 bank=(字地址 mod 32)，检查同一 warp 的不同 lane 到同一 bank 的不同字。再设计 padding 或局部位 XOR。',
      'L3 占用率、spill、可发射 warp 与每线程数据复用要一起看。同一输入比较不同 tile 的正确性和总耗时。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Warp Stall 诊断）：你在 `ncu` 里看到一个 Memory-Bound 融合算子的主要 Warp Stall 原因是 `Stall Long Scoreboard`（高达 65%），这代表 GPU 流水线正在干等什么？应该怎么优化？',
      },
      {
        stage: 'corner',
        question: '追问 2（Triton 与 CUDA/CUTLASS）　同一 shape、dtype 下如何判断 Triton 生成代码是否满足 tile、流水和指令需求？哪些实测差距值得改成手写实现？',
      },
    ],
    referenceAnswer: `### 从端到端步时找到限制

先固定输入形状、精度、正确性和 warm-up，记录稳态端到端时间。用 Nsight Systems 对齐 CPU 线程、CUDA API、stream/event、memcpy、collective 与 kernel。时间线有空档时，查谁在等谁：若 kernel 已就绪但 CPU 迟迟未发射，可比较 CUDA Graph 或融合。若在等输入、同步或另一条 stream，要处理相应依赖。

对关键 kernel 用 Nsight Compute 查看实际 FLOPs、DRAM/L2 bytes、吞吐、eligible warps 和 stall。Long Scoreboard 表示在等 L1TEX 路径的依赖，可能是 global、local/spill 或 texture 访问。若带宽未满，还要查相关加载的延迟与并行性。Short Scoreboard 和 MIO Throttle 分别提示较短的依赖或 MIO 指令队列压力，需进一步查看冲突与指令指标。

### 两个能在白板上算清的例子

- **全局内存**。在常见 CC≥6.0 的合并访问模型下，32 个 lane 各取相邻的一个 float，总计 128 B。起点对齐 32 B 时覆盖四个 32 B sector。跨步、错位可增加 sector。相同线程改用 float4 要保证对齐和边界。它可减少指令数，却不会自动修复 lane 间的 stride。
- **共享内存**。对 32×32 float tile，lane i 读第 i 行的同一列，行跨度 32 个字会落在同一 bank 的不同地址。把行宽改为 33 可分散到 32 个 bank。多个 lane 读同一字可以广播。对向量指令或矩阵加载按实际拆分的事务分析，可选与 tile 形状匹配的 XOR swizzle。

**Occupancy 与工具选择**。寄存器和 SMEM 限制驻留 CTA，但较大 tile 可能提高数据复用。扫 BLOCK_M/N/K、num_warps、num_stages，对照耗时、寄存器 spill、内存事务和实测吞吐。Triton 是否已生成需要的指令和流水以当前后端代码及 profile 为准。可与 CUTLASS 或手写 kernel 在相同条件下比较。

> 依据　NVIDIA《CUDA C++ Best Practices Guide》合并访存与共享内存章节，及 Nsight Compute《Profiling Guide》。`,
  },
  {
    id: 'kn-04',
    track: 'kernel',
    title: 'GPU 规格与系统选型：A100、H100、H200、B200 与 GB200 NVL72',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['A100', 'H100', 'H200', 'B200', 'GB200 NVL72', 'FP4', 'NVLink Domain'],
    summary: '区分单 GPU、8 卡系统和 72 卡机柜规格，分析算力/HBM 配比、互连与 SLO 成本。',
    contextSpecs: ['A100 SXM 80 GB', 'H100 SXM 80 GB / H200 SXM 141 GB', 'DGX B200：8 GPU', 'GB200 NVL72：72 GPU'],
    prompt: '比较 A100 80 GB SXM、H100 SXM、H200 SXM、DGX B200 与 GB200 NVL72。请注明单卡还是整机柜、稠密还是稀疏峰值，以及 NVLink 带宽的统计口径。72 GPU 同一 NVLink 域对 MoE 和长上下文训练有什么收益和限制？如何按线上 SLO 比较成本？',
    rubric: [
      {
        id: 'specs-evolution',
        label: '分清 A100/H100/H200 SXM、DGX B200 和 GB200 NVL72，统一精度、dense/sparse 和设备计数',
        weight: 24,
        keywords: ['a100', 'h100', 'h200', 'b200', '141', '180', '13.4', '4.8', '8 tb', 'fp8', 'fp4', 'hbm3e'],
        probe: '按同一 SXM 口径比较 H100 与 H200 的 HBM 容量、带宽和 dense 算力。对哪一类负载有帮助？',
      },
      {
        id: 'nvl72-domain',
        label: '72 GPU scale-up 域、单 GPU 1.8 TB/s 双向聚合与系统 130 TB/s 的区别',
        weight: 26,
        keywords: ['nvl72', '72卡', '1.8', 'nvlink域', '打破', 'tp', 'ep', '单机8卡', '全互联', '130tb'],
        probe: '两 GPU 间可用带宽、每卡聚合带宽和全机柜聚合值分别怎样解释？',
      },
      {
        id: 'nvl72-moe',
        label: '说明 rack 内 EP/CP 的可行映射和限制，包括专家数整除、负载、消息大小及跨柜路径',
        weight: 22,
        keywords: ['moe', 'ep', '专家', '整除', '副本', 'all-to-all', 'nvlink', '跨柜', 'tpot', 'wide-ep'],
        probe: '若每层有 256 个路由专家，打算让 72 个 rank 各放同样数目的整专家，需要怎样映射？',
      },
      {
        id: 'fp4-ridge',
        label: '同精度计算 ridge，区分 W4A16、原生低精度与实际业务瓶颈',
        weight: 16,
        keywords: ['fp4', 'nvfp4', 'mxfp8', 'ridge', '内存墙', 'memory wall', '算力带宽比', '量化'],
        probe: '分别计算 A100、H100、B200 的 BF16 ridge。更换到 FP4 时，哪些输入条件与 kernel 成本也发生了变化？',
      },
      {
        id: 'reliability-nvlink',
        label: '评估液冷供电、故障域、可用率和达标 goodput 对成本的影响',
        weight: 12,
        keywords: ['液冷', '供电', '功耗', '容错', '故障域', '可用率', '可靠性', 'mtbf', '隔离', '备用容量'],
        probe: 'NVL72 这么强，它给机房基础设施和分布式集群容错带来了什么新的棘手挑战？',
      },
    ],
    hints: [
      'L1 比较同精度、同稀疏性、同设备计数的峰值。再按 HBM、NVLink 与网络各自的路径看带宽。',
      'L2 H200 SXM 的 141 GB、4.8 TB/s 相对 H100 SXM 的 80 GB、3.35 TB/s 提高了容量和理论带宽。端到端收益要看受限部分。',
      'L3 GB200 NVL72 的 13.4 TB 是 72 张 GPU 的 HBM 总量。仍需显式放置参数、KV 和并行组。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（NVFP4 与 W4A16）　每 16 个值共享块 scale 的 NVFP4 与 INT4 解包后用 FP16 算的路径有何不同？怎样评估对 prefill 与 decode 的收益？',
      },
      {
        stage: 'corner',
        question: '追问 2（异构成本测算）：如果你们公司要采购一批推理算力跑在线 72B / DeepSeek-R1 服务，给你 H200、B200 和 H20 三种选项，你会从哪几个定量指标（$/M-Token 成本模型）做 TCO 选型决策？',
      },
    ],
    referenceAnswer: `### 规格先按同一口径读

下表按 NVIDIA 对应系统规格。前三行为单卡 SXM，Tensor Core 算力为**未启用结构化稀疏加速的 dense 峰值**。NVLink 数字为单 GPU 全部链路的双向聚合值。

| GPU / 系统 | dense Tensor Core 峰值 | HBM 容量与本地带宽 | NVLink |
|---|---|---|---|
| A100 SXM 80 GB | BF16 312 TFLOP/s | 80 GB，约 2.04 TB/s | 600 GB/s |
| H100 SXM 80 GB | BF16 989，FP8 1979 TFLOP/s | 80 GB，3.35 TB/s | 900 GB/s |
| H200 SXM | BF16 989，FP8 1979 TFLOP/s | 141 GB，4.8 TB/s | 900 GB/s |
| DGX B200 中的一张 B200 | BF16 约 2.25、FP8 4.5、FP4 9 PFLOP/s | 整机 8 卡共 1,440 GB、64 TB/s，即每卡 180 GB、8 TB/s | 1.8 TB/s |
| GB200 NVL72 整柜 | 36 个 Grace Blackwell Superchip，共 72 GPU，算力应按整柜另查 | 全柜 **13.4 TB HBM3e、576 TB/s 聚合**，每 GPU 访问本地 HBM | 约 **130 TB/s** 全柜聚合 |

远端 HBM 通过互连访问，延迟和带宽不同于本地 HBM。单次点到点传输也不能代入全柜 130 TB/s。

- **同精度算拐点**。A100、H100、H200、B200 的 BF16 理想 ridge 分别约 153、295、206、281 FLOPs/byte，随 SKU 并非单调。H200 的容量与带宽适合权重或 KV 读取占主导的负载，需在实际 decode 中测 KV 压力与 HBM 流量。NVFP4 为原生低精度矩阵乘，需计块 scale、量化和质量。W4A16 通常解包到 16-bit MMA。
- **拓扑设计**。把 EP dispatch/combine 或 CP 的 KV 交换放在 rack 内可减少 scale-out 通信。256 个专家无法在 72 rank 上等数放置整专家，可选支持的 EP 子组、不均匀映射或部署 32 份冗余副本凑为 288 个槽位。要核算热点、all-to-all 延迟和故障后恢复。长上下文另需核算按序列切分的激活、head 整除和 attention 的二次计算量。
- **TCO 追问**。在相同模型质量、流量、上下文分布、TTFT/TPOT SLO 下记录有效输出 token/s。若整套服务每小时完整成本为 C，达标输出速率为 g，则每百万达标输出 token 成本为 C×10⁶/(3600g)。NVL72 的液冷、机柜供电和备用容量要计入 C。并行组是故障域，测节点故障后的恢复时间、剩余服务容量和可用率，再代入 g，分别实测各 SKU。

> 依据　NVIDIA A100、H100、H200 产品规格，HGX / DGX B200 和 GB200 NVL72 系统规格表及其 sparse/dense 脚注。`,
  },
  {
    id: 'kn-05',
    track: 'kernel',
    title: '高吞吐融合算子实战：Chunked Linear Cross-Entropy 与 Liger-Kernel',
    level: 'P6',
    frequency: '高频',
    incident: false,
    tags: ['Kernel Fusion', 'Liger-Kernel', 'Cross-Entropy', '128K Vocab', '显存优化'],
    summary: '核算大词表投影的显存，比较 Liger 的 token 分块、Apple CCE 的片上规约和词表并行 CE。',
    contextSpecs: ['T=32,768 个 token', 'V=131,072 或 152,064', 'FP32 物化算例', 'LM head W[V,H]'],
    prompt: '单卡有 32,768 个 token，词表大小为 131,072（另可用 152,064 对照）。若 LM head 物化 FP32 logits，算出一份张量的大小，并说明何时反向还会有额外峰值。请设计按 token 分块的 Linear + Cross-Entropy，推导 dH、dW，比较 Apple CCE 与词表并行的实现。',
    rubric: [
      {
        id: 'logits-oom-math',
        label: 'T=32768、V=131072 时 FP32 logits 为 16 GiB。区分 dtype、缓冲寿命和真实峰值',
        weight: 26,
        keywords: ['32k', '128k', '152', 'fp32', '4字节', 'logits', '16 gib', '18.5625', '32 gib', '物化', '梯度'],
        probe: 'T=32,768、V=131,072 时，FP32 和 BF16 的一份 logits 各占多少 GiB？反向峰值还取决于什么？',
      },
      {
        id: 'chunked-ce',
        label: '沿 token 分块，逐块求 CE 与 dZ，从 dZ 求 dH 并累计 dW，避免物化全量 logits',
        weight: 28,
        keywords: ['chunk', '分块', '不物化', 'fused', '融合', '逐块', '原地', 'in-place', 'dh', 'dw', '累加', 'online'],
        probe: '既然前向不保存完整的 `[T, V]` Logits，那反向传播怎么算 `dH` (对隐藏态的梯度) 和 `dW` (对 LM Head 权重的梯度) 呢？',
      },
      {
        id: 'inplace-grad',
        label: '区分 Liger 分块复用缓冲、标量反向的预计算与 CCE 片上计算，说明剩余工作区',
        weight: 18,
        keywords: ['原地', 'in-place', '覆写', 'softmax', 'liger', 'cut cross-entropy', 'apple', 'cce', '峰值'],
        probe: 'Liger 的 chunk logits、CCE 的片上 tile 和两者都需保留的梯度各占什么空间？',
      },
      {
        id: 'elementwise-fusion',
        label: '融合可减少 RMSNorm、RoPE、SwiGLU 等的中间张量和启动次数，保留规约等必要读写',
        weight: 16,
        keywords: ['rmsnorm', 'rope', 'swiglu', 'triton', 'memory-bound', '读写', 'hbm', '中间张量', '融合'],
        probe: '除了 Cross-Entropy，Transformer 里还有哪些算子在 Liger-Kernel 里融合收益最高？它们的共同特征是什么？',
      },
      {
        id: 'tp-vocab',
        label: '词表并行按 token 求全局 max、指数和与目标 logit，可合并 SUM，无需全词表 AllGather',
        weight: 12,
        keywords: ['vocab parallel', 'tp', 'allreduce', 'max', 'sum', 'online', 'megatron'],
        probe: '如果开启了 TP=8，LM Head 权重按词表维切分在 8 张卡上，算 CrossEntropy 时需要把 8 张卡的 Logits AllGather 拼起来吗？',
      },
    ],
    hints: [
      'L1 一道题统一口径：32,768 × 131,072 × 4 B = 2³⁴ 字节 = 16 GiB，BF16 为 8 GiB。',
      'L2 W 为 [V,H] 时，Z_c=H_c Wᵀ，dH_c=dZ_c W，dW 累加 dZ_cᵀ H_c。4096 行、128 Ki 个词的 FP32 临时 logits 是 2 GiB。',
      'L3 CCE 在片上计算 tile 的 logits，只向全局内存输出需要的统计，比较各自的重算、工作区和梯度。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（前向预计算梯度）　对标量 sum/mean loss，Liger 的一类实现能在 forward 分块算好梯度。若 reduction=none 或需二阶导，为什么需要不同的实现？',
      },
      {
        stage: 'corner',
        question: '追问 2（编译器融合）　对指定 PyTorch 版本与输入形状，如何检查 torch.compile 是否已经消除了完整 logits 物化？',
      },
    ],
    referenceAnswer: `### 先算一份张量，再讨论峰值

T=32,768，V=131,072，FP32 的 [T,V] 为 2³⁴ 字节，即 **16 GiB**。BF16 为 8 GiB。V=152,064 时 FP32 为 **19,931,332,608 字节，即 18.5625 GiB**。若两份 FP32 大张量在同一时刻存活，则分别为 32 GiB 或 37.125 GiB。LM head 输出、CE 内部累加、返回梯度可以是不同 dtype，实际峰值还要看 autocast、后端、复用与释放时间。

### 分块反传要对全局有效 token 正确归一化

令 W 为 [V,H]，对 C 行隐藏态形成 Z_c=H_c Wᵀ。以无类别权重、无 label smoothing 的标准 CE 为例，按行稳定计算 LSE−z_target。sum reduction 时 dZ_c=softmax(Z_c)−onehot(y)。mean 时除以**全批有效 token 数**，ignore_index 的行置零。可把 dZ_c 写回临时 Z_c 缓冲，随后算 dH_c=dZ_c W、累加 dW+=dZ_cᵀ H_c。C=4096、V=131,072 时 FP32 临时 logits 已是 2 GiB，整体训练还要加 dW、dH、累加和算子工作区。

- **Liger 与 CCE**。对标量 loss，可在 forward 分块计算单位上游梯度对应的 dH、dW，backward 再乘同一个 grad_output。若上游是每 token 不同的权重，必须先逐行加权再汇总 dW，可在 backward 重算。高阶导数要有相应的可微反向。Apple CCE 在片上计算词表 tile，逐步归约 logsumexp 与目标 logit，避免将 chunk logits 写回 HBM。
- **词表并行**。各 rank 持有部分词表。对每 token 全局 MAX，再 SUM 各分片 exp(z−max) 以及目标所在分片的目标 logit（其余 rank 贡献 0）。两个 SUM 可按算法合并。由此可求 loss 与本地 dZ，通信量级 O(T)。若 dH 要复制到各 rank，LM head 的反向还有独立的梯度归约。
- **融合选型**。RMSNorm 的归约、RoPE 或 SwiGLU 的逐点运算都可能受益于少写一次中间张量。检查编译生成的 kernel、峰值快照和总步时，对照 eager/compile/专用实现，同时比较 loss、dH、dW 和掩码下的误差。

> 依据　Liger Kernel（arXiv:2410.10989）、Cut Your Losses（2411.09009）及 Megatron Core 的 VocabParallelCrossEntropy 实现。`,
  },
  {
    id: 'kn-06',
    track: 'kernel',
    title: '算子与通信微观 Profiling：ncu Roofline/Stall 诊断与 nccl-tests 硬件验收',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['Profiling', 'ncu', 'Warp Stall', 'Sector Efficiency', 'nccl-tests', '硬件计数器'],
    summary: '深入 Nsight Compute 指标体系（SOL、Warp Stall、Sector/Wavefront）以及 nccl-tests 对照 NIC/NVLink 硬件计数器的集群验收方法。',
    contextSpecs: ['Nsight Compute (ncu)', 'SOL / Warp Stall', '32B Sector / Wavefront', 'nccl-tests + IB Counters'],
    prompt: '单算子调优和集群网络验收是 AI Infra 最底层的 Profiling 功底。请回答：(1) 用 Nsight Compute (`ncu`) 剖析一个跑不满的 GPU kernel 时，如何结合 Speed-of-Light (SOL)、Warp Stall Breakdown 以及 Global Sector / Shared Wavefront 指标精准定位瓶颈？(2) 在新交付的 64 节点 H100 集群上，如何用 `nccl-tests` 配合底层 NIC/NVLink 硬件计数器完成通信性能验收与异常链路定位？',
    rubric: [
      {
        id: 'ncu-sol',
        label: '用 `ncu` Speed-of-Light (Compute% vs Memory%) 做三类归因：Compute-bound (>60~70% 计算管线)、Memory-bound (>60~70% DRAM/L2)、以及两者都低的 Latency/Launch/Occupancy-bound',
        weight: 22,
        keywords: ['sol', 'speed of light', 'compute', 'memory', 'latency-bound', 'dram', 'l2', 'tensor', 'ncu'],
        probe: '如果 `ncu` 报告一个 kernel 的 Compute SOL 只有 18%、Memory SOL 也只有 22%，这说明它既不卡算力也不卡 HBM 带宽，那瓶颈到底在哪里？',
      },
      {
        id: 'warp-stall',
        label: '准确解读核心 Warp Stall：`Long Scoreboard` (L1TEX/DRAM/local spill)、`Short Scoreboard` (SMEM/固定延迟)、`MIO Throttle` (SMEM/SFU 发射满)、`Wait`/`Barrier` (同步)、`Not Selected` (健康竞争)',
        weight: 26,
        keywords: ['long scoreboard', 'short scoreboard', 'mio throttle', 'barrier', 'not selected', 'stall', 'eligible', 'l1tex', 'spill'],
        probe: '在 `ncu` 的 Warp State Statistics 里，`Stall Not Selected` 占比很高是坏事吗？`Long Scoreboard` 和 `MIO Throttle` 分别对应哪类优化手段？',
      },
      {
        id: 'mem-efficiency',
        label: '微观访存效率指标：Global Memory 按实际传输 32B Sector 与理想需求比看非合并浪费；Shared Memory 看 `Wavefronts` 与理想比诊断 Bank Conflict；检查 Register Spill (Local Memory 流量)',
        weight: 20,
        keywords: ['sector', '32b', 'wavefront', 'bank conflict', 'local memory', 'spill', 'coalesced', '合并', '事务'],
        probe: '怎样在 `ncu` 里一眼看出你的 Triton/CUDA kernel 到底有没有发生 Register Spill、非合并访存或 Shared Memory Bank Conflict？具体看哪些指标？',
      },
      {
        id: 'nccl-tests-sweep',
        label: '`nccl-tests` 验收方法：扫消息大小（小消息几 KB 看延迟 α 与树/协议，大消息百 MB~GB 看峰值 `busbw`）、区分 AllReduce/AllGather/ReduceScatter/AlltoAll 各自换算系数、注意 `in-place` 与多轮抖动',
        weight: 18,
        keywords: ['nccl-tests', 'busbw', 'algbw', '消息大小', '小消息', '大消息', 'alltoall', '延迟', '带宽', '抖动'],
        probe: '做集群验收时只跑一条 1 GB AllReduce 够吗？为什么一定要把小消息（如 8 KB~256 KB）和 AlltoAll 也一起扫一遍？',
      },
      {
        id: 'hw-counters',
        label: '结合底层硬件计数器闭环验证：IB/RoCE 端口计数器 (`port_xmit/rcv_data`, `symbol_error`, `link_downed`, `ecn`/`pfc`, 重传) 与 NVLink CRC/重放错误 + 二分隔离定位坏链路/坏节点',
        weight: 14,
        keywords: ['计数器', 'counter', 'ib', 'crc', '重传', 'pfc', 'ecn', '二分', 'bisection', '拓扑', '光模块'],
        probe: '如果 64 节点全网 AllReduce 的 `busbw` 比标称低了 30%，怎么在 10 分钟内用测试和硬件计数器把拖后腿的那根线或那台机器揪出来？',
      },
    ],
    hints: [
      'L1 `ncu` 先看 SOL 象限（双低=Latency/低并行/小 Grid），再看 Eligible Warps 与 Top Warp Stall，最后落到 Memory Workload Analysis 的 Sector/Wavefront 与指令 Mix。',
      'L2 `Stall Not Selected` 表示该 Warp 其实已就绪（Eligible），只是同一周期其他就绪 Warp 先被调度器挑中执行——说明延迟隐藏非常健康，反而是好信号！',
      'L3 集群验收用二分法（Bisection）：把 64 节点分成两组 32 节点、再拆 16/8/单机机内 + Pairwise 两两对打，同时对照 IB 端口误码/降速/重传计数器。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（`ncu` 测量失真与锁频）　为什么同一份 kernel 连续跑两次 `ncu` 测出来的耗时和 SOL 会差 10%？而且 `ncu` 报告里的“Kernel Duration”为什么有时和 `nsys` 看到的真实线上耗时不一致？',
      },
      {
        stage: 'corner',
        question: '追问 2（AlltoAll 的 `busbw` 口径）　`nccl-tests` 中 AllReduce 的 `busbw = algbw × 2(N−1)/N`，那 AlltoAll、AllGather 和 ReduceScatter 的换算系数分别是多少？为什么 MoE 集群验收必须单独测 AlltoAll？',
      },
    ],
    referenceAnswer: `### 一、\`ncu\` 单算子微观诊断三步法

1. **Speed-of-Light (SOL) 定象限**：
   - **Compute / Tensor Pipe 高 (>65%)**：Compute-bound，优化聚焦 Tensor Core 形状对齐、WGMMA/低精度指令、减少非矩阵辅助指令与循环开销。
   - **DRAM / L2 高 (>65%)**：Memory-bound，优化聚焦算子融合（省 HBM 往返）、向量化加载、提高 L2/SMEM 复用与压缩数据类型。
   - **双低（Compute 和 Memory 均 <40%）**：典型 **Latency / Occupancy / 小 Grid Bound**——例如 CTA 总数少于 SM 数（Wave Quantization）、寄存器/SMEM 爆表导致每 SM 仅驻留 1~2 个 Warp 无法隐藏延迟、或大量同步屏障。
2. **Warp Stall Breakdown 看卡在哪条管线**：
   - \`Long Scoreboard\`：等 L1TEX 长延迟数据（Global Memory、Local Memory spill 或 Texture）→ 查合并访存、异步预取/多级软件流水、消除寄存器溢出；
   - \`Short Scoreboard\`：等 Shared Memory 或固定延迟指令结果 → 增加指令级并行（ILP）或重排无关指令掩盖；
   - \`MIO Throttle\`：Shared Memory / SFU / 特殊指令发射队列打满 → 减少高频 SMEM Bank 冲突、减少重复 \`exp\`/标量运算；
   - \`Barrier\` / \`Wait\`：线程块内 \`__syncthreads()\` 负载不均；
   - \`Not Selected\`：Warp 已就绪只是排队让位给同周期另一就绪 Warp，通常说明**隐藏延迟充裕（良性信号）**。
3. **访存效率与溢出实证**：
   - **Global 合并度**：看 \`L1/TEX Global Load/Store\` 的 **Actual Sectors vs Ideal Sectors**（32B/sector）；比值 > 1 说明跨步或未对齐造成带宽浪费；
   - **Shared Bank Conflict**：看 **Shared Wavefronts vs Ideal Wavefronts**；理想无冲突为 1:1，若 32-way 冲突则高达 32:1；
   - **Register Spill**：检查 \`Local Memory\` 读写字节数与 \`lmem\` 指令，若非零则需调小 Tile 或控制寄存器压力。
   - *测量纪律*：\`ncu\` 默认会**锁基础频率（Clock Control = base）**且在多 pass replay 前后自动刷新 L2 Cache，因此其耗时绝对值与线上 boost 频率、热 L2 下的 \`nsys\` 耗时不同，分析时应保持统一锁频/缓存策略对照前后版本。

### 二、\`nccl-tests\` 与硬件计数器集群验收

- **全维度扫测**：
  - **算子覆盖**：AllReduce（\`busbw = algbw × 2(N−1)/N\`）、AllGather / ReduceScatter / AlltoAll（三者换算系数均为 **\`(N−1)/N\`**）。MoE/Ulysses 集群必须单独测 AlltoAll，因其同时打满全网多对多双向链路，最易暴露 ECMP 极化、跨 Spine 拥塞与 PFC 问题，而 Ring AllReduce 往往测不出来；
  - **消息跨度**：从 8 KB 扫到 1~4 GB——小消息验延迟 $\\alpha$（协议切换 LL/LL128/Simple 与 Tree 算法），大消息验单向总线带宽 $B$（对照单机 NVLink 与跨机 NIC 理论线速）及多轮方差抖动。
- **异常链路 10 分钟二分定位**：
  1. 先单机跑机内 8 卡 AllReduce 排除单机 NVLink/PCIe 故障；
  2. 对 64 节点做 **Bisection（32 vs 32 → 16 → 8 → Pairwise 两两对打）** 快速锁定拖慢的子集与节点；
  3. 同步采集全网 **IB/RoCE 端口计数器**（实际链路协商速率是否从 NDR 400G 降级、\`symbol_error\` / \`port_rcv_errors\` / 重传 / ECN / PFC pause duration）与 \`nvidia-smi nvlink -e\`（NVLink CRC/replay 计数），结合物理拓扑表直接定位坏光模块、坏线缆或交换机端口。

> 依据　NVIDIA Nsight Compute《Kernel Profiling Guide》、nccl-tests《Performance》及 InfiniBand/RoCE 运维诊断实践。`,
  },
];
