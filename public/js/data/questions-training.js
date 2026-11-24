export const TRAINING_QUESTIONS = [
  {
    id: 'tr-01',
    track: 'training',
    title: '3D/5D 并行策略设计与通信瓶颈定量分析',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['TP/PP/DP/CP/EP', 'Megatron-LM', 'NVLink vs IB', '通信量推导'],
    summary: '在千卡 H100 集群上预训练 70B Dense / 671B MoE 模型，如何选择 TP/PP/DP/CP/EP 拓扑并定量推导通信开销？',
    contextSpecs: ['1024×H100 80G', '8×GPU NVLink 900GB/s', '8×400Gbps IB', 'SeqLen 8K~128K'],
    prompt: '假设现在要在 1024 张 H100 (80GB, 单机 8 卡 NVLink 900GB/s 双向，机间 8×400Gb/s InfiniBand) 上从零预训练一个 70B Dense LLM（80 层, H=8192, GQA 64Q/8KV；8K 上下文 GBS=4M tokens，后续扩到 128K）。请你从显存与 GBS 约束出发，给出合理的并行切分方案（FSDP/HSDP vs TP/PP/VPP/CP），说明为什么“显存放得下时应尽量克制 TP/PP/VPP”，并定量估算各维度通信原语与通信字节量。',
    rubric: [
      {
        id: 'minimize-tp-pp',
        label: '第一性原则：显存放得下且 GBS 允许时，优先 DP/FSDP2(HSDP)，尽量最小化 TP、能不开 PP/VPP 就不开（避免关键路径通信、GEMM 变窄与气泡）',
        weight: 24,
        keywords: ['显存', '放得下', 'fsdp', 'hsdp', 'tp=1', 'tp=2', 'tp=4', '不开pp', 'pp=1', '关键路径', 'gemm', '小矩阵', '最小化', '克制'],
        probe: '70B 在 1024 卡 H100 上用 FSDP2/HSDP 分片后静态显存并不大，为什么很多资深工程师反对一上来就盲目开 TP=8 或 PP=4/8 + VPP？',
      },
      {
        id: 'gbs-tp-nvlink',
        label: '结合 GBS=4M (8K 仅 512 序列) 约束推导并行度（DP≤512 ⇒ 至少需 TP/CP=2），且 TP 严控在机内 NVLink (TP≤8) 并配 SP',
        weight: 20,
        keywords: ['gbs', '512', 'nvlink', '机内', 'sp', 'sequence parallel', 'allgather', 'reducescatter', 'allreduce', '张量并行'],
        probe: 'GBS=4M tokens、seq=8K 时单步只有 512 条序列，1024 卡纯 DP 跑不满。此时选 TP=2/4 还是 PP=2？如果开了 TP+SP，通信原语怎么变？',
      },
      {
        id: 'pp-vpp-tradeoff',
        label: 'PP / Interleaved VPP 适用边界：仅当显存/通信域受限时才引入 PP；VPP 需满足层数整除 (p·v | L) 且权衡 v× P2P 通信与气泡 (p-1)/(v·m+p-1)',
        weight: 18,
        keywords: ['pp', '流水线', 'p2p', 'bubble', '气泡', '1f1b', 'interleaved', 'vpp', '整除', 'micro-batch', 'stage'],
        probe: '什么情况下才真的需要引入 PP 和 Interleaved VPP？VPP 降低气泡的同时付出了什么通信、显存和层数整除约束？',
      },
      {
        id: 'cp-long',
        label: '扩到 128K 长上下文时：单步序列数骤降 16× (仅 32 条) 且激活暴涨，必须切出 CP=8~16 (机内 Ulysses 对齐 8 KV Head + 跨机 Ring)',
        weight: 20,
        keywords: ['cp', 'context parallel', 'ring attention', 'ulysses', '128k', '长上下文', '长文本', '激活显存', 'activation', '32'],
        probe: '上下文扩到 128K、GBS 还是 4M 时，单步只剩 32 条序列了！为什么此时不仅激活会爆，DP 也直接被锁死 ≤32，怎么靠 CP 破局？',
      },
      {
        id: 'quant-math',
        label: '通信量精确估算（区分元素与 BF16 字节）：TP+SP 每层前向 8(T-1)/T·BSH 字节；ZeRO-1 每步 4Ψ 字节；Ring CP 前向 4(P-1)/P·S·H_kv 字节',
        weight: 18,
        keywords: ['bsh', '8bsh', '4bsh', '2(n-1)', 'bytes', '字节', '带宽', '公式', '估算', '通信量', '4ψ', '2ψ'],
        probe: '请按 BF16 (2字节) 精确算一下：开启 TP+SP 后，单层 Transformer Block 前向+反向（不含重算）每卡总通信字节数是多少个 B·S·H？',
      },
    ],
    hints: [
      'L1 第一性原则：能用 DP/FSDP2(HSDP) 扩展就别急着切 TP 和 PP——TP 通信卡在每层关键路径且切窄 GEMM；PP 有气泡，VPP 还会把 P2P 通信放大 v 倍。',
      'L2 定量边界：70B 静态显存 16Ψ=1120GB，在 HSDP (如 64 路分片) 下每卡仅 17.5GB，80GB 轻松放得下；但 GBS=4M/8K=512 序列 ⇒ DP≤512，故 1024 卡只需 TP=2 (或 TP=4 省重算) × PP=1 × DP=512 (HSDP)！',
      'L3 精确字节账（BF16=2B）：TP+SP 每子层 AG+RS = 2(T-1)/T·BSH 元素 = 4(T-1)/T·BSH 字节，每层 2 子层（Attn+MLP）故前向 8(T-1)/T·BSH 字节，反向同为 8(T-1)/T·BSH 字节。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（气泡与层数整除反证）：如果某团队套用传统 Megatron 配法，在 1024 卡上对 80 层 70B 模型设了 TP=4、PP=8、DP=32（GBS=4M, seq=8K, mbs=1）。请你算一下每 DP 组的 micro-batch 数 m、标准 1F1B 气泡率，以及如果开 Interleaved VPP，`VPP=4` 合法吗？这如何反证 70B 应尽量不开大 PP？',
      },
      {
        stage: 'corner',
        question: '追问 2（架构演进）：如果把 70B Dense 换成 671B MoE（类似 DeepSeek-V3，256 个路由专家，每次激活 8 个），并行策略会怎么重构？为什么 MoE 训练更要把 TP 压到 1，而把机内 NVLink 留给 EP？',
      },
    ],
    referenceAnswer: `### Staff 级参考架构方案

#### 1. 核心第一性原则：显存放得下时，尽量最小化 TP、PP 与 VPP
- **为什么不要无脑开大 TP（如直接 TP=8）？** TP 通信位于每一层前/反向的**同步关键路径**上，即便在机内 NVLink (450 GB/s 单向) 上也难以 100% 掩盖，且 TP=8 会把 GEMM 维度切窄（如 \`H=8192 → 1024\`），明显拉低 Tensor Core 算力效率。因此 **TP 只需开到满足显存与 GBS 约束的最小值（如 TP=1/2/4）**，并严格限制在机内 NVLink 域内。
- **为什么不要无脑开 PP / VPP？** PP 会引入流水线气泡 \`(p-1)/(m+p-1)\` 和首尾 Stage（Embedding / LM Head）不均衡；VPP (Interleaved 1F1B) 虽然压低气泡，但会令 P2P 通信量放大 \`v\` 倍、略增激活峰值，且要求模型层数能被 \`p × v\` 整除。**只要 FSDP2/HSDP + 轻量 TP 能装下显存且跑满 GBS，优先 \`PP=1\` 零气泡！**

#### 2. 70B Dense (80 层) 在 1024×H100 (80GB) 上的真实配比推导
- **8K 阶段（GBS = 4M tokens ⇒ 单步仅 512 条序列）**：
  - **显存账**：静态显存 \`16Ψ = 1120 GB\`。若采用 **FSDP2 / HSDP**（如 64 路分片 × 16 路跨组 DDP），单卡静态显存仅 \`17.5 GB\`，配合 Selective/Full Checkpointing（Full AC 8K \`mbs=1\` 仅 ~\`11 GB\`）总显存 ~\`40 GB\`，**显存远未到需要 TP=8 × PP=4 的程度！**
  - **GBS 约束**：单步只有 512 条序列，1024 张卡上 \`DP\` 上限被锁死在 \`512\`（\`mbs=1\`），因此至少需 \`TP × PP × CP ≥ 2\`。
  - **推荐方案 A（FSDP2/HSDP 路线，最优零气泡）**：**\`TP=2 × PP=1 × DP=512 (HSDP: 64-way Shard × 8-way Replicate)\`**，或为了免去大部分重算（提升 MFU）略增至 **\`TP=4 × PP=1 × DP=256\`**（此时每 DP 组有 2 个 micro-batch 顺便摊薄通信）。
  - **推荐方案 B（纯 Megatron + Distributed Optimizer / ZeRO-1，不做参数分片）**：权重+梯度共 \`4Ψ = 280 GB\`，需靠模型并行切到单卡 ~\`17.5~35 GB\`，此时仍优先较小组合如 **\`TP=4 × PP=2 (或 PP=4) × DP=128 (或 64)\`**，仅在气泡显著时再谨慎开启 VPP。
- **128K 扩长阶段（GBS = 4M tokens ⇒ 单步骤降至仅 32 条序列！）**：
  - 单条激活暴涨 16× 且 \`DP\` 上限跌至 \`≤ 32\`，必须拆出 **\`CP=8~16\`**（例如机内 \`Ulysses=8\` 精确对齐 8 个 KV Head + 跨机 \`Ring=2\` 走 Zig-Zag 均衡 Causal 计算），组合如 **\`TP=2~4 × CP=16 × PP=1~2 × DP=16~32\`**（配 FSDP2/HSDP）。

#### 3. 各维度通信原语与精确字节量（明确区分元素 vs BF16=2B）
| 维度 | 通信原语 | 单层/单步每卡通信量（BF16 = 2 字节） | Overlap 方式 |
|---|---|---|---|
| **TP + SP** | 每子层 1×AG + 1×RS | 每层 2 子层（Attn+MLP），前向 \`4(T-1)/T · BSH\` 元素 = **\`8(T-1)/T · BSH\` 字节**；反向不含重算同为 \`8(T-1)/T · BSH\` 字节 | Async TP / GEMM-Comm Overlap |
| **PP** | P2P Send/Recv | 每 Stage 边界 **\`2 · mbs · S · H\` 字节** | 1F1B 流水隐藏 P2P 延迟 |
| **DP (ZeRO-1)** | RS(Grad) + AG(Param) | 每步全量 \`2(D-1)/D · Ψ\` 元素 ≈ **\`4Ψ\` 字节**（与 micro-batch 数无关） | Bucket 化与反向逐层 Overlap |
| **CP (Ring)** | P2P 环形传递 KV | 每卡传 \`P-1\` 步 \`S/P\` 块 KV，前向共 **\`4(P-1)/P · S · H_kv\` 字节**（反向再加 dKV） | FlashAttention 掩盖下一块 KV 传输 |

#### 4. 追问 1 精确验算（为什么 \`PP=8\` 在此场景很亏？）
- 若设为 \`TP=4, PP=8, DP=32\`（GBS=4M, seq=8K, mbs=1 ⇒ 总 512 序列）：每 DP 组仅分到 \`m = 512 / 32 = 16\` 个 micro-batch！
- **标准 1F1B 气泡率**（按 \`PP=8, m=16\` 精确算）：气泡占比 \`(p-1)/(m+p-1) = 7/23 ≈ 30.4%\`（相对理想耗时多出 \`(p-1)/m = 7/16 = 43.8%\`）——**气泡极高！**
- **Interleaved VPP 合法性与精确公式**：80 层模型要求 \`p × v\` 整除 80，因此 **\`VPP=4\` 根本除不尽 (\`8×4=32 ∤ 80\`)**！合法取值只有 \`VPP=2\`（每块 5 层，气泡占比 \`(p-1)/(v·m+p-1) = 7/39 ≈ 17.9%\`）或 \`VPP=5\`（每块 2 层，气泡占比 \`7/87 ≈ 8.0%\`，但 P2P 通信放大 5×）。这正反证了：**显存放得下时，能不开 PP 就别开 PP！**
- **MoE 场景**：Attention 激活小而专家 FFN 通信重，把 TP 压到 \`TP=1\` 可消除 Attention 通信与小 GEMM 损耗，将整机 NVLink 留给 **EP All-to-All**，实现节点内 NVLink + 节点间 RDMA 两级路由（DeepEP）。`,
  },
  {
    id: 'tr-02',
    track: 'training',
    title: 'ZeRO-1/2/3 与 FSDP2 显存拆解及通信-显存权衡',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['ZeRO', 'FSDP2', '显存估算', 'Activation Checkpointing'],
    summary: '推导混合精度训练 16Ψ 显存构成，对比 ZeRO-1/2/3 通信量差异，并解释 FSDP2 相比 FSDP1 的核心改进。',
    contextSpecs: ['BF16 + FP32 AdamW', '7B / 70B 参数量 Ψ', 'ZeRO-1/2/3', 'PyTorch FSDP2 DTensor'],
    prompt: '请完整拆解标准 BF16 混合精度 + AdamW 训练中静态显存为什么是 16Ψ 字节？ZeRO Stage 1/2/3 分别切分了什么、各自额外引入了多少通信量？以及实际工程中为什么 PyTorch FSDP2 用 DTensor 替代了 FSDP1 的 FlatParameter？',
    rubric: [
      {
        id: 'mem-16psi',
        label: '准确拆解 16Ψ：BF16权重2Ψ + BF16梯度2Ψ + FP32主权重4Ψ + Adam一二阶矩8Ψ',
        weight: 25,
        keywords: ['16', '2', '4', '8', 'adam', 'master', '主权重', '动量', '方差', 'momentum', 'variance', 'fp32', 'bf16'],
        probe: '请逐项算一下 16Ψ 里的每一块分别是多少字节：参数、梯度、Master Weight、Momentum、Variance？',
      },
      {
        id: 'zero-stages',
        label: 'ZeRO-1(Opt) / ZeRO-2(Opt+Grad) / ZeRO-3(Opt+Grad+Param) 切分与内存公式',
        weight: 22,
        keywords: ['zero-1', 'zero1', 'zero-2', 'zero2', 'zero-3', 'zero3', '优化器', '梯度', '参数', 'sharding', '分片'],
        probe: 'ZeRO-1 和 ZeRO-2 各自把单卡静态显存降到了多少？为什么 Megatron 默认只开 Distributed Optimizer (类 ZeRO-1) 而不常直接套 ZeRO-3？',
      },
      {
        id: 'comm-3psi',
        label: '通信量对比：DP/ZeRO-1/2 为 2Ψ，ZeRO-3/FSDP 为 3Ψ（多 1× 前向/反向 AllGather）',
        weight: 20,
        keywords: ['3', '2', 'allgather', 'reducescatter', 'allreduce', '通信量', '1.5倍', '50%', 'prefetch'],
        probe: '从标准 DDP 到 ZeRO-3，通信总量增加了多少？这多出来的通信发生在大模型训练的哪个阶段？',
      },
      {
        id: 'fsdp2-dtensor',
        label: 'FSDP2 per-parameter DTensor 分片：消除 FlatParam 碎片/支持混合精度与子模块灵活冻结',
        weight: 18,
        keywords: ['fsdp2', 'dtensor', 'flatparameter', 'flatparam', 'per-parameter', 'dim-0', '碎片', 'lora', 'compile', 'torch.compile'],
        probe: 'FSDP1 把参数拼成一个大 FlatParameter 遇到了哪些工程痛点？FSDP2 改成 per-parameter DTensor 解决了什么？',
      },
      {
        id: 'grad-accum',
        label: 'Gradient Accumulation 下 ZeRO-3 的通信陷阱（no_sync 与 Prefetch 权衡）',
        weight: 15,
        keywords: ['梯度累积', 'gradient accumulation', 'micro-batch', 'no_sync', 'reshard_after_forward', 'zero3', '通信放大'],
        probe: '如果做 8 步 Gradient Accumulation，ZeRO-3 每一步 micro-batch 都要重新 AllGather 参数吗？怎么缓解这个通信放大？',
      },
    ],
    hints: [
      'L1 方向提示：静态显存按 Parameter / Gradient / Optimizer State (含 FP32 Master Weights) 三大块来算。',
      'L2 通信量提示：AllReduce = ReduceScatter (Ψ) + AllGather (Ψ) = 2Ψ；ZeRO-3 需要在前向和反向各 AllGather 一次参数，再 ReduceScatter 梯度，共 3Ψ。',
      'L3 FSDP2 提示：FSDP1 将所有参数展平拼接成 1D FlatParameter，导致 LoRA/冻结参数、Checkpoint 转换、torch.compile 及显存碎片都很痛苦；FSDP2 按参数张量 dim-0 独立 DTensor 切分。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（显存峰值陷阱）：很多同学以为开了 ZeRO-3 就不会 OOM，但实测在 Backward 或 Optimizer Step 时还是出现显存尖峰（Memory Spike）甚至碎片化 OOM，通常是什么原因造成的？',
      },
      {
        stage: 'corner',
        question: '追问 2（Hybrid Sharding）：在 64 节点（512 卡）集群用 FSDP2 训 32B 模型，全集群 512 路 ZeRO-3 通信效率很差，HSDP（Hybrid Sharded Data Parallel）是怎么组合域内与域间策略的？',
      },
    ],
    referenceAnswer: `### Staff 级参考解析

#### 1. 静态显存 \`16Ψ\` 完整拆解（BF16 混合精度 + FP32 AdamW）
- **BF16 参数权重**：\`2Ψ\`
- **BF16 梯度**（若用 FP32 梯度累积则为 \`4Ψ\`，总量变 \`18Ψ\`）：\`2Ψ\`
- **Optimizer States (FP32)**：
  - FP32 Master Weight：\`4Ψ\`
  - FP32 Momentum ($m_t$)：\`4Ψ\`
  - FP32 Variance ($v_t$)：\`4Ψ\`
  - 小计：\`12Ψ\`
- **合计**：\`2Ψ + 2Ψ + 12Ψ = 16Ψ\`（70B 模型静态显存 ≈ \`1.12 TB\`）。

#### 2. ZeRO-1 / 2 / 3 显存与通信量对比（数据并行度 $N_d$，显存单位：字节；通信量按元素数 / BF16 字节双标注）
| 策略 | 单卡静态显存 (字节) | 通信总量（每步每卡） | 核心权衡 |
|---|---|---|---|
| **DDP** | \`16Ψ\` | \`2Ψ\` 元素 (\`4Ψ\` 字节, AllReduce 梯度) | 零参数通信，显存最高 |
| **ZeRO-1** | \`4Ψ + 12Ψ/N_d\` | \`2Ψ\` 元素 (\`4Ψ\` 字节, RS 梯度 + AG 权重) | 几乎无额外通信，显存直降 ~4× |
| **ZeRO-2** | \`2Ψ + 14Ψ/N_d\` | \`2Ψ\` 元素 (\`4Ψ\` 字节, 逐层 RS 梯度 + AG 权重) | 通信仍为 2Ψ 元素，适合中等规模 |
| **ZeRO-3 / FSDP** | \`16Ψ/N_d\` | \`3Ψ\` 元素 (\`6Ψ\` 字节, 前向 AG + 反向 AG + 反向 RS) | 通信增 1.5×；Grad Accum 时每 micro-batch 都要 AG 参数！ |

> **Grad Accum 陷阱**：ZeRO-3 若每 micro-batch 都释放参数，GA=$k$ 时参数 AllGather 通信会被放大 $k$ 倍；可通过 \`reshard_after_forward=False\`（退化为类 ZeRO-2 保留参数）或 HSDP 局部分片缓解。

#### 3. FSDP2 相比 FSDP1 的核心工程跃迁
- **Per-Parameter DTensor (dim-0 sharding)** 替代 FSDP1 的 \`FlatParameter\`：
  1. 彻底消除大块连续 FlatParam 带来的显存分配碎片和无谓拷贝；
  2. 原生支持参数级 \`requires_grad\`（完美适配 LoRA/部分冻结）与不同参数不同精度；
  3. 完美兼容 \`torch.compile\`（无 Graph Break）与 2D/3D 并行组合（TP+FSDP2+CP）；
  4. 无通信直接异步保存分布式 Checkpoint（DCP）。

#### 4. 追问要点
- **HSDP (Hybrid Sharded Data Parallel)**：机内 8 卡（NVLink 域）做 \`FSDP/ZeRO-3\`，跨机 64 节点（IB 域）做 \`DDP/ZeRO-1\` 梯度同步，既吃满 NVLink 省显存，又避免跨机高频 AllGather 参数。`,
  },
  {
    id: 'tr-03',
    track: 'training',
    title: '超大规模 MoE 训练架构：EP All-to-All、负载均衡与 DualPipe/DeepEP',
    level: 'P8',
    frequency: '高频必考',
    incident: false,
    tags: ['MoE', 'Expert Parallelism', 'All-to-All', 'DeepEP', 'DualPipe'],
    summary: '针对 256 专家级细粒度 MoE 模型，剖析 EP All-to-All 瓶颈、Aux-loss-free 负载均衡以及计算-通信全重叠架构。',
    contextSpecs: ['671B Total / 37B Active', '256 Experts, Top-8', 'EP64 + PP16', 'NVLink + InfiniBand'],
    prompt: '以 DeepSeek-V3 (671B 总参 / 37B 激活，256 路由专家 Top-8) 为例，在大规模跨机 Expert Parallelism (如 EP=64) 训练中，最核心的性能瓶颈和稳定性风险是什么？如何从算法-系统协同（路由限制、负载均衡）和底层通信（All-to-All 重叠）两个维度做到高 MFU 且不 Drop Token？',
    rubric: [
      {
        id: 'all2all-bottleneck',
        label: '识别 EP 核心瓶颈：Dispatch/Combine 两次 All-to-All 跨机通信与计算通信比下降',
        weight: 22,
        keywords: ['all-to-all', 'alltoall', 'dispatch', 'combine', '通信', 'ep', '专家并行', 'rdma', '带宽'],
        probe: 'MoE 前向和反向分别要做几次 All-to-All？为什么细粒度 MoE (如 Top-8) 会让 All-to-All 通信量急剧放大？',
      },
      {
        id: 'node-limited',
        label: 'Node-Limited Routing（节点受限路由，如最多路由到 M=4 个节点）利用 NVLink/IB 带宽比',
        weight: 20,
        keywords: ['node-limited', '节点限制', 'm=4', '机内', 'nvlink', '两级', 'hierarchy', '分组路由', '去重', 'dedup'],
        probe: '一个 Token 选 8 个专家，如果均匀散落到 8 台不同机器上，IB 带宽直接打满。DeepSeek 是怎么通过路由约束把 IB 流量压下来的？',
      },
      {
        id: 'aux-loss-free',
        label: '负载均衡与 Token Dropping：Aux-loss-free (动态 Bias 调节) 替代强辅助损失',
        weight: 20,
        keywords: ['负载', 'load balance', 'aux-loss', 'bias', '偏置', 'token drop', 'dropping', 'capacity', '热点', '倾斜'],
        probe: '传统 GShard/Switch 用 Aux Loss 强迫负载均衡有什么副作用？Aux-loss-free 机制是怎么在不伤害模型精度的情况下解决专家倾斜和 Token Dropping 的？',
      },
      {
        id: 'overlap-dualpipe',
        label: '通信-计算完全重叠：DualPipe 双向流水线 / Micro-batch 重叠 + SM 切分 (DeepEP)',
        weight: 22,
        keywords: ['dualpipe', 'deepep', 'overlap', '重叠', 'sm', '双向', 'attention', 'mlp', 'grouped gemm', '通信sm'],
        probe: '常规 1F1B 很难把跨机 All-to-All 完全藏在算子后面。DualPipe 和 DeepEP 是怎么把通信和计算（前向+反向）交错重叠起来的？',
      },
      {
        id: 'grouped-gemm',
        label: '算子层优化：Permute/Unpermute 融合与 Grouped GEMM / FP8 稀疏计算',
        weight: 16,
        keywords: ['grouped gemm', 'permute', 'padding', 'kernel', 'fp8', '稀疏', 'cutlass', '算子'],
        probe: 'Token Dispatch 到达目标 GPU 后，每个专家收到的 Token 数不相等，在 GPU 上怎么高效执行多专家矩阵乘？',
      },
    ],
    hints: [
      'L1 方向提示：从三层拆解——(1) 算法路由约束减少跨机扇出；(2) 无辅助损失动态偏置消除热点；(3) 底层通信 Kernel 与双向流水线实现零暴露通信。',
      'L2 带宽比提示：H800 NVLink (400GB/s) 与 IB (~50GB/s/卡) 带宽比约 3.2:1~8:1，若每个 Token 最多发往 M=4 个节点，机内再通过 NVLink 转发给多专家，可做到机间流量只发一份（去重）。',
      'L3 重叠提示：DualPipe 让一个 Chunk 的前向 Attention/MLP 与另一个 Chunk 的反向 All-to-All 重叠，并把 GPU 的部分 SM (如 20/132 个 SM) 专用于 DeepEP 异步通信。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（显存代价）：DualPipe 把流水线气泡压到极低且完美隐藏了 All-to-All，但它有一个明显的代价——模型权重需要存两份（因为双向同时从两端注入）。在 671B 模型上，这个显存代价是如何消化的？',
      },
      {
        stage: 'corner',
        question: '追问 2（容错与动态扩缩）：如果训练中某张卡收到的动态 Token 数突然飙升导致临时激活 OOM（由于不 Drop Token），工程上应该如何防御？',
      },
    ],
    referenceAnswer: `### Staff/Principal 级参考架构

#### 1. MoE EP 训练三大核心痛点
1. **All-to-All 通信爆炸**：每个 Token Top-$K$ 路由（如 $K=8$），前向需 Dispatch + Combine 两次 All-to-All，反向再做两次，共 4 次；细粒度 MoE 激活计算量小，计算/通信比急剧恶化。
2. **专家负载倾斜 (Straggler)**：热门专家计算超时且显存尖峰，强制 \`Capacity Factor\` 丢弃 Token 又损害模型收敛。
3. **跨节点 IB 带宽远低于 NVLink**：裸 All-to-All 在 EP64 跨 8 节点时通信直接成为绝对瓶颈。

#### 2. 算法-系统协同四大核心方案
- **Node-Limited Routing（节点受限路由）**：限制每个 Token 选出的 Top-8 专家最多分布在 $M=4$ 个节点内。发送端按目标节点去重（每节点只通过 IB 发一次 Token），到达目标节点后再经由高带宽 NVLink 分发给该节点内的多个目标专家，精准匹配 NVLink / IB 带宽比。
- **Aux-Loss-Free 负载均衡**：去掉干扰主 Loss 梯度的强辅助损失，为每个专家维护独立偏置项 $b_i$（仅影响路由选专家，不乘入权重门控值）：若某专家本步过载则 $b_i \leftarrow b_i - \gamma$，欠载则 $b_i \leftarrow b_i + \gamma$，实现几乎零 Token Dropping 与稳定收敛。
- **DualPipe + DeepEP 极致计算通信重叠**：
  - 将每层拆分为 \`Attention -> All-to-All Dispatch -> MoE MLP -> All-to-All Combine\`。
  - DualPipe 从流水线两端双向注入 Micro-batch，使 Forward 的计算与 Backward 的 All-to-All 通信（反之亦然）在时间轴上精准交错。
  - **DeepEP**：预留少量 GPU SM（如 H800 132 SM 中划出 20 SM）跑自定义 PTX/IBGDA 异步通信内核，其余 SM 满载跑计算，无同步 Stall。
- **Grouped GEMM + FP8**：通过 Fusion Permute 将变长 Token 按专家重排，调用 CUTLASS/DeepGEMM 的 FP8 Masked/Grouped GEMM 避免为每个小专家单独 Launch Kernel。

#### 3. 追问要点
- **DualPipe 权重翻倍**：结合高 DP 下的 **ZeRO-1 / Optimizer Sharding**，参数本身只占静态显存极小部分（FP8/BF16 参数经 EP64+PP16 分片后每卡仅几 GB），翻倍完全在 80GB 预算内；此外后来的 **DualPipeV (V-shape)** 通过单端折叠进一步消除了双份权重开销。`,
  },
  {
    id: 'tr-04',
    track: 'training',
    title: 'FP8 混合精度训练稳定性、Scaling 策略与 Loss Spike 防御',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['FP8', 'E4M3/E5M2', 'Blockwise Quant', 'Loss Spike', 'Hopper'],
    summary: '解析 FP8 E4M3 vs E5M2 数值特性、Per-Tensor Delayed Scaling 的缺陷，以及细粒度 Tile/Block-wise FP8 训练如何消除溢出。',
    contextSpecs: ['H100 FP8 1979 TFLOPS', 'E4M3 / E5M2', '1×128 / 128×128 Scaling', 'WGRAD FP32 Accum'],
    prompt: '在 H100/B200 上开启 FP8 训练可以把矩阵乘理论算力翻倍，但大规模 LLM 长期用 FP8 训练很容易出现精度退化甚至 Loss Spike。请分析 E4M3 与 E5M2 的区别、Transformer Engine 传统 Delayed Scaling 为什么会翻车，以及 Fine-grained (Tile/Block-wise) FP8 训练的关键工程设计。',
    rubric: [
      {
        id: 'fp8-format',
        label: 'E4M3 (高精度、窄动态范围±448，用于前向激活/权重) vs E5M2 (宽动态范围±57344，用于梯度)',
        weight: 22,
        keywords: ['e4m3', 'e5m2', '动态范围', '尾数', '指数', '448', '梯度', '前向', '激活'],
        probe: 'E4M3 和 E5M2 各自的指数位、尾数位以及适用张量类型是什么？为什么现在很多细粒度方案连梯度也统一用 E4M3？',
      },
      {
        id: 'delayed-scaling',
        label: 'Delayed Scaling (历史 Amax 窗口) 面对 Activation Outlier 突变导致的 Underflow/Overflow',
        weight: 22,
        keywords: ['delayed', 'amax', 'outlier', '离群值', '溢出', '下溢', 'overflow', 'underflow', 'per-tensor', 'transformer engine'],
        probe: 'Transformer Engine 的 Per-Tensor Delayed Scaling 为什么在遇到激活离群值 (Outliers) 时容易造成大片小数值被截断成 0？',
      },
      {
        id: 'fine-grained',
        label: '细粒度量化：激活 1×128 Tile-wise、权重 128×128 Block-wise 隔离离群值',
        weight: 24,
        keywords: ['1x128', '128x128', 'tile', 'block', '细粒度', 'fine-grained', '分组', 'group', 'online', '在线量化'],
        probe: 'Block-wise / Tile-wise 量化是如何把 Outlier 影响局限在局部 128 个元素里的？为什么权重用 128×128 而激活是 1×128？',
      },
      {
        id: 'fp32-accum',
        label: 'Hopper WGMMA 内部累加精度陷阱（约 14-bit 尾数）与 CUDA Core 高精度 FP32 晋升累加',
        weight: 18,
        keywords: ['累加', 'accum', 'wgmma', 'cuda core', 'fp32', '精度损失', 'tensor core', 'promotion', 'master'],
        probe: '你知道 Hopper Tensor Core 在执行 FP8 WGMMA 时的内部累加精度其实并不是完整的 FP32 吗？这在长内积维度 K 上会造成什么偏差，工程上怎么修？',
      },
      {
        id: 'sensitive-ops',
        label: '敏感模块保持 BF16/FP32：Embedding、LM Head、MoE Gate、LayerNorm、Attention Softmax',
        weight: 14,
        keywords: ['embedding', 'lm head', 'gate', 'router', 'norm', 'softmax', 'bf16', '敏感', '主权重'],
        probe: '全模型都能转 FP8 吗？哪些算子和张量必须保留 BF16 或 FP32？',
      },
    ],
    hints: [
      'L1 方向提示：从“数值表示 (E4M3/E5M2) -> 缩放粒度 (Per-Tensor vs Block-wise) -> 硬件累加精度 (WGMMA 14-bit 截断) -> 敏感层白名单”四层回答。',
      'L2 细粒度原因：激活在 Channel 维存在剧烈离群值，按 1×128 分组后离群值只影响同一组 128 元素；权重按 128×128 分块刚好对齐 GEMM 的 Tile 维度，反向转置无需重算 Scale。',
      'L3 Hopper 硬件彩蛋：Hopper FP8 WGMMA 累加器实际尾数对齐精度仅约 13~14 bit，DeepSeek-V3 每隔 $N_C=128$ 个元素（每 4 次 WGMMA）把中间结果拷贝到 CUDA Core 用完整 FP32 寄存器做二次累加。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（通信与存储）：既然已经做了 FP8 训练，通信（比如 EP All-to-All 和 DP 梯度）以及 Optimizer States 能不能也压缩到 FP8 或 BF16 来进一步省显存和带宽？各有什么风险？',
      },
      {
        stage: 'corner',
        question: '追问 2（Blackwell 演进）：到 NVIDIA Blackwell (B200) 架构引入了硬件原生 MXFP8 / NVFP4 (Microscaling) 支持，这相比在 Hopper 上手写 CUDA Core 晋升累加带来了什么架构简化？',
      },
    ],
    referenceAnswer: `### Staff 级参考解析

#### 1. FP8 格式特性
- **E4M3**：1 符号 + 4 指数 + 3 尾数，最大值 \`448\`，精度更高 → 适合前向权重与激活（在细粒度 Scaling 下梯度也可全用 E4M3）。
- **E5M2**：1 符号 + 5 指数 + 2 尾数，最大值 \`57344\`，与 FP16 指数域一致 → 传统用于梯度以防溢出。

#### 2. Per-Tensor Delayed Scaling 的核心缺陷
- 整个大张量共享 1 个 FP32 Scale，一旦某个 Channel 出现 100× 级 **Activation Outlier**，Scale 被迫拉大，导致同张量里 99% 正常值发生 **Underflow 归零**；
- Delayed Scaling 依赖过去 N 步历史 Amax 预测当前步，一旦单步梯度突增即发生溢出/饱和，触发 Loss Spike。

#### 3. 细粒度在线 FP8 训练方案（DeepSeek-V3 / 现代 SOTA）
- **激活**：\`1 × 128\` Tile-wise 在线计算当前块的 Amax 与 Scale（完美隔离单 Channel 离群值）。
- **权重**：\`128 × 128\` Block-wise 分块量化（对齐矩阵乘 K/N Tile，转置对称无需重复量化），可令前向/反向统一使用高精度 **E4M3**。
- **解决 Hopper WGMMA 累加精度缺陷**：Hopper FP8 Tensor Core 内部累加实际只有约 14-bit 尾数精度（误差最高达 2%）。解决方案：在 GEMM 内层每累加 \`N_C = 128\` 个元素（4 条 WGMMA 指令）后，将结果晋升到 **CUDA Core FP32 寄存器**中做高精度累加，并将 Block Scale 乘入。
- **高精度白名单**：Embedding、Output Head、MoE Router/Gating、RMSNorm、Attention Core（部分场景）、Master Weights 保持 BF16/FP32；Optimizer States 可视情况将一阶矩存为 BF16 进一步省显存。`,
  },
  {
    id: 'tr-05',
    track: 'training',
    title: '超长上下文训练 (128K-1M)：Ring Attention vs DeepSpeed Ulysses',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['Context Parallelism', 'Ring Attention', 'Ulysses', 'Zig-Zag', 'GQA'],
    summary: '深入对比序列并行/上下文并行两大范式（Ring Attention 与 Ulysses）的通信量、Causal Mask 负载均衡与 GQA 适配限制。',
    contextSpecs: ['128K ~ 1M Context', 'GQA (64 Q / 8 KV)', 'CP=16', 'FlashAttention'],
    prompt: '要把一个 70B GQA 模型（64 个 Query Head，8 个 KV Head）的训练上下文从 8K 扩到 256K，必须引入 Context Parallelism (CP)。请深度对比 Ring Attention 和 DeepSpeed Ulysses 两种方案的实现原理、通信量公式、优缺点，以及在 GQA + Causal Mask 场景下如何组合（Hybrid CP）并解决负载不均问题？',
    rubric: [
      {
        id: 'ring-attn',
        label: 'Ring Attention：按序列切分，环形 P2P 传递 KV 块并与 FlashAttention 计算重叠',
        weight: 22,
        keywords: ['ring', '环形', 'p2p', 'kv', 'overlap', 'flashattention', 'online softmax', '序列切分'],
        probe: 'Ring Attention 是怎么结合 Online Softmax 在多卡间增量更新 Attention 输出的？它需要传 Q 还是传 KV？',
      },
      {
        id: 'ulysses',
        label: 'DeepSpeed Ulysses：All-to-All 将序列切分转为 Head 维切分，单卡算完整序列的部分 Head',
        weight: 22,
        keywords: ['ulysses', 'all-to-all', 'alltoall', 'head', '头维度', '转置', '通信量小'],
        probe: 'Ulysses 是怎么做维度转置的？它在 Attention 前后分别需要几次 All-to-All？',
      },
      {
        id: 'gqa-limit',
        label: 'GQA 限制：Ulysses 并行度受限于 KV Head 数 (8)，超限需重复 KV 破坏带宽优势',
        weight: 20,
        keywords: ['gqa', 'kv head', '8', '受限', '限制', '复制', '重复', 'mqa', 'mla'],
        probe: '题目里模型只有 8 个 KV Head，如果我们想开 CP=16，直接用 Ulysses 会遇到什么硬限制？',
      },
      {
        id: 'causal-load',
        label: 'Causal Mask 下三角负载不均与 Zig-Zag / Striped Attention 切分平衡算力',
        weight: 20,
        keywords: ['causal', '因果', '下三角', '负载不均', 'zig-zag', 'zigzag', 'striped', '对称', '首尾配对'],
        probe: '在自回归 Causal Mask 下，Ring Attention 如果按连续块切分 [0..1/4], [1/4..1/2]... 会发生什么严重的算力浪费？怎么解决？',
      },
      {
        id: 'hybrid-cp',
        label: 'USP (Unified Sequence Parallelism) 混合方案：机内 Ulysses + 机间 Ring',
        weight: 16,
        keywords: ['hybrid', 'usp', '混合', '机内', '机间', 'nvlink', 'ib', '组合'],
        probe: '如果把 Ulysses 和 Ring 组合在多机集群里，应该把哪一个放在机内 NVLink，哪一个放在跨机 IB？为什么？',
      },
    ],
    hints: [
      'L1 方向提示：Ulysses 通信量随并行度 $P$ 下降（$O(S/P)$），但受限于 Head 数量；Ring Attention 不受 Head 数限制，但通信量与 $S$ 成正比、依赖极高序列长度来掩盖通信。',
      'L2 Causal 负载均衡：把序列切成 $2P$ 块，第 $i$ 张卡拿第 $i$ 块和第 $2P-1-i$ 块（首尾配对 Zig-Zag），每张卡的下三角有效计算面积完全相等！',
      'L3 混合拓扑：机内 8 卡刚好匹配 8 个 KV Head 用 Ulysses（利用 NVLink 极速 All-to-All），跨机用 Ring Attention（P2P 点对点传递，对 IB 带宽压力小且易于与计算 Overlap）。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（通信量对比）：假设总序列长度 $S$ 很大，隐藏维 $H$，CP 并行度 $P$。请比较 Ulysses 与 Ring Attention 各自的通信总量（忽略 GQA 先按 MHA 算），在什么条件下 Ring 反而比 Ulysses 通信更少？',
      },
      {
        stage: 'corner',
        question: '追问 2（变长 Packing）：预训练数据通常会做 Document Packing（多条短文档拼成一条 256K 序列，中间用 varlen mask 隔断，避免跨文档注意）。Packing 场景下 Zig-Zag Ring Attention 会遇到什么新问题？',
      },
    ],
    referenceAnswer: `### Staff 级参考解析

#### 1. Ring Attention vs DeepSpeed Ulysses 核心对比
| 维度 | Ring Attention | DeepSpeed Ulysses |
|---|---|---|
| **切分与通信** | 保持序列维切分；环形 P2P 传递 $K, V$ 块，借助 FlashAttention Online Softmax 累积 | Attention 前 All-to-All 把 \`[S/P, H]\` 变为 \`[S, H/P]\`；算完后再 All-to-All 换回 |
| **单卡前向通信量（MHA）** | 每卡发收 \`2(P-1)/P · S·H\` 元素（BF16 为 \`4(P-1)/P · S·H\` 字节；GQA 时缩为 \`4(P-1)/P · S·H_kv\` 字节！） | 4 次 All-to-All（Q/K/V/O），每卡 \`4(P-1)/P · (SH/P)\` 元素（BF16 为 \`8(P-1)/P · SH/P\` 字节） |
| **计算通信重叠** | 天然极佳（FlashAttention 算当前块同时 P2P 预取下一块 KV） | 较难完全重叠（All-to-All 必须完成后才能进 Attention） |
| **GQA 约束** | **无限制**，且 GQA 让 KV 通信直接缩小 8×！ | 强受限：\`P ≤ n_kv_heads\`（否则需复制 KV，丢失优势） |

#### 2. Causal Mask 负载不均与 Zig-Zag (Striped) 优化
- 若直接连续切分，Rank 0 只算左下角极小面积，Rank $P-1$ 算满全三角，算力倾斜近 $2\times$。
- **Zig-Zag 切分**：将序列等分为 $2P$ 个 Chunk，Rank $i$ 持有 \`Chunk[i]\` 和 \`Chunk[2P-1-i]\`（首尾对称配对）。例如 $P=4$ 时，Rank 0 拿 \`(0,7)\`，Rank 1 拿 \`(1,6)\`，每张卡的 Causal Attention FLOPs 严格相等。

#### 3. 256K + GQA (8 KV Heads) 最佳实践：USP (Hybrid CP)
- **机内 NVLink (Ulysses=8)**：正好对齐 8 个 KV Head，单机 NVLink 毫秒级完成 All-to-All，不复制 KV。
- **机间 IB (Ring=2 或 4)**：跨机只传压缩后的 KV（已按 GQA 缩小 8×），且通过 FlashAttention 完整 Overlap 掩盖 IB 传输延迟。
- **Document Packing 追问**：当 256K 内混杂极长文档和大量短文档时，静态 Zig-Zag 仍可能负载不均，工程上需引入 **Per-Document Load Balancing / Context-aware Packing**（如 Megatron THD 格式 + 动态负载均衡调度）。`,
  },
  {
    id: 'tr-06',
    track: 'training',
    title: '万卡集群容错与稳定性排障：Loss Spike、SDC、慢节点与无感 Checkpoint',
    level: 'P8',
    frequency: '高频必考',
    incident: true,
    tags: ['容错', 'Checkpoint', 'SDC', 'NCCL Hang', 'Straggler'],
    summary: '万卡训练平均每数小时发生一次硬件故障，如何定位静默数据损坏 (SDC)、NCCL Hang、慢节点，并把 MTTR 压到分钟级？',
    contextSpecs: ['10,000+ GPUs', 'Goodput > 95%', 'MTBF ~3~6 Hours', 'Async Checkpoint'],
    prompt: '【线上实战排障】你负责维护一个 8192 卡 H100 预训练任务。过去一周集群有效训练时间（Goodput）跌到了 72%：一方面频繁出现 NCCL Timeout/Hang 和偶发 Loss Spike，另一方面每次节点故障重启加载 Checkpoint 要花 25 分钟。请系统性地设计一套排障定位与高可用容错体系。',
    rubric: [
      {
        id: 'hang-diag',
        label: 'NCCL Hang / 慢节点排查：Flight Recorder、Trace 归因、区分真 Hang vs 单点 Straggler',
        weight: 22,
        keywords: ['flight recorder', 'watchdog', 'hang', 'straggler', '慢节点', '降频', 'thermal', 'ib', '丢包', 'pfc', 'trace'],
        probe: '报 NCCL Timeout 时，日志里 8192 张卡都在报错。你怎么在 1 分钟内定位到底是哪 1 张卡或者哪条光模块拖死了整个集合通信？',
      },
      {
        id: 'sdc-spike',
        label: 'Loss Spike 与 SDC (静默数据损坏) 定位：区分数据脏样本 vs GPU 硬件静默算错',
        weight: 20,
        keywords: ['sdc', '静默', 'loss spike', 'grad norm', '数据', '脏数据', '确定性', '回放', 'checksum', '硬件'],
        probe: '遇到一个诡异的 Loss Spike，你怎么区分是某个 Batch 的数据有毒，还是某张 GPU 发生了静默计算错误 (SDC)？',
      },
      {
        id: 'async-ckpt',
        label: '异步持久化 Checkpoint：GPU -> CPU Pinned Memory 秒级快照 + 后台写 DFS，消除阻塞',
        weight: 22,
        keywords: ['async', '异步', 'checkpoint', 'pinned', 'cpu', '快照', 'snapshot', '后台', '阻塞', 'dfs'],
        probe: '传统同步写 Checkpoint 每次都要停训几分钟，异步 Checkpoint 怎么做到把训练阻塞时间降到 1~2 秒以内？要注意什么一致性问题？',
      },
      {
        id: 'fast-recovery',
        label: '分钟级快速恢复：In-Memory Peer Recovery（健康节点内存广播权重）+ 冗余热备节点 (Hot Spare)',
        weight: 20,
        keywords: ['in-memory', '内存恢复', 'peer', '热备', 'hot spare', 'dp', '广播', 'broadcast', '弹性', 'elastic'],
        probe: '重启加载 Checkpoint 要花 25 分钟通常卡在远程分布式存储读带宽和进程冷启动。既然 DP 组里其他副本节点还活着，能不能不读磁盘直接恢复？',
      },
      {
        id: 'precheck',
        label: '事前隔离与自动驱逐：开机 NCCL/GEMM 体检、XID/ECC 监控与自动 Cordon/Drain',
        weight: 16,
        keywords: ['xid', 'ecc', 'dcgm', '体检', '预检', '驱逐', 'cordon', '隔离', '健康检查', 'goodput'],
        probe: '怎样防止坏卡反复被调度进新任务继续污染集群？上线前和运行中要盯哪些核心硬件指标？',
      },
    ],
    hints: [
      'L1 方向提示：Goodput = (1 - 故障停机占比 - Checkpoint开销 - 慢节点拖尾)。按“快速定位 -> 快速止血恢复 -> 事前防呆”三段式回答。',
      'L2 内存恢复提示：因为 DP 副本间参数相同、或者仅挂掉 1 台机器，可由健康节点从内存直接通过 RDMA/NCCL 向替换进来的热备节点拷贝权重+优化器状态（如 ByteRobust / MegaScale / TorchFT），秒级续训！',
      'L3 SDC 排查提示：在怀疑节点上跑固定种子的 GEMM Checksum 自检（如已知黄金结果的 MatMul），同时对异常步保存 Data Loader Index 做隔离重放验证。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（细节深挖）：异步 Checkpoint 把权重从 GPU D2H 拷到 CPU 内存时，如果在拷贝还没完成前，GPU 就开始执行下一步 Optimizer Step 改写参数了，会产生什么问题？怎么解决？',
      },
      {
        stage: 'corner',
        question: '追问 2（弹性训练）：如果坏了一台机器但集群暂时没有热备机，有没有办法降级到 8184 卡继续跑而不影响算法超参（Global Batch Size / LR 调度）？',
      },
    ],
    referenceAnswer: `### Staff/Principal 级高可用容错架构

#### 1. 定位 NCCL Hang 与慢节点 (Straggler)
- **开启 PyTorch NCCL Flight Recorder (\`TORCH_NCCL_TRACE_BUFFER_SIZE\`)**：在环形缓冲区记录每张卡最近的 Collective 序号、开始/完成时间戳。一旦超时自动 Dump，全局比对即可定位出**唯一未进入 Collective 的卡**（算子/Dataloader 卡死）或**进入最晚的卡**（慢节点）。
- **慢节点根因**：GPU 过热降频 (Thermal Throttling)、PCIe 降速、IB 光模块误码/重传/PFC 风暴、Host CPU/内存页回收卡顿。通过每步记录 Step 内的 Stage 耗时分布，识别 P99 离群节点自动告警驱逐。

#### 2. Loss Spike 与 SDC (Silent Data Corruption) 根因分离
- **两步诊断**：(1) 记录每步 Data Index 与各 Rank 的 Local Grad Norm；(2) 若出现 Spike，先在另一组健康节点上用相同 Checkpoint + 相同 Batch 重放——若**复现**则是脏数据/学习率/数值稳定性问题（跳过该 Batch 或回滚）；若**不复现**则高度怀疑原节点发生了 **SDC**，立即在嫌疑节点执行确定性 GEMM/Attention Golden-Output 校验隔离坏卡。

#### 3. 把 Checkpoint 与恢复耗时从 25 分钟压到 < 2 分钟
- **两级异步 Checkpoint**：
  - Stage 1 (同步阻塞 < 1~2s)：使用 Pinned Host Memory，通过独立 Copy Stream 将 GPU 状态 D2H 拷贝至本机内存（必须在下一步 \`optimizer.step()\` 写脏参数前通过 CUDA Event 同步完 D2H，或采用 Copy-on-Write / 双缓冲）。
  - Stage 2 (完全后台异步)：独立进程将 Host 内存快照分片异步刷盘到并行文件系统/对象存储。
- **免冷启 & In-Memory Recovery**：
  - **热备节点池 (Hot Spare)** 预拉镜像与驱动；
  - 节点故障时无需重启全部 8192 个进程（或使用热进程池），且优先从健康 DP Peer 节点通过 RDMA 直接 **Peer-to-Peer 内存恢复**参数，彻底绕开共享存储读放大风暴；
  - 结合 XID 关键错误（如 XID 48/63/74/79/94/95）、NVLink CRC 错误、IB 丢包计数器实现分钟级自动隔离驱逐。`,
  },
  {
    id: 'tr-07',
    track: 'training',
    title: '流水线并行 Bubble 优化：1F1B、Interleaved、Zero-Bubble 与 DualPipe',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['Pipeline Parallelism', '1F1B', 'Zero-Bubble', 'B/W Split', 'DualPipe'],
    summary: '定量分析 GPipe -> 1F1B -> Interleaved 1F1B -> Zero-Bubble (ZB-H1/H2) 的气泡率、激活显存与通信开销权衡。',
    contextSpecs: ['PP=8', 'Micro-batches m=32', 'F / B / W Split', 'Activation Memory'],
    prompt: '流水线并行（PP）最大的痛点是流水线气泡（Bubble）。请按演进脉络对比 GPipe、1F1B、Interleaved 1F1B 以及 Sea AI Lab 的 Zero-Bubble (ZB-H1/ZB-H2) 方案：它们各自的气泡率公式是什么？各自付出了什么显存或通信代价？Zero-Bubble 是如何打破 1F1B 的气泡下界的？',
    rubric: [
      {
        id: 'gpipe-1f1b',
        label: 'GPipe vs 1F1B：气泡率均为 (p-1)/(m+p-1)，但 1F1B 将激活峰值从 O(m) 降为 O(p)',
        weight: 22,
        keywords: ['gpipe', '1f1b', 'p-1', '激活', '显存', 'warmup', 'cooldown', '峰值', 'o(p)', 'o(m)'],
        probe: 'GPipe 和基础 1F1B 的理论气泡率其实是一样的！那为什么工业界几乎全都用 1F1B 替代 GPipe？',
      },
      {
        id: 'interleaved',
        label: 'Interleaved 1F1B (VPP)：每卡分配 v 个不连续虚拟 Stage，气泡降为 1/v，代价是 P2P 通信增 v 倍',
        weight: 22,
        keywords: ['interleaved', 'vpp', '虚拟', '1/v', '通信增加', 'v倍', 'p2p', 'chunk'],
        probe: 'Interleaved 1F1B 把气泡缩小了 v 倍，它付出的代价是什么？（提示：看通信和激活显存）',
      },
      {
        id: 'bw-split',
        label: 'Zero-Bubble 核心洞察：将反向拆为 B (对输入求导，在关键路径) 和 W (对权重求导，无上游依赖)',
        weight: 26,
        keywords: ['zero-bubble', 'zero bubble', 'zb', '拆分', 'b', 'w', '输入梯度', '权重梯度', 'dgrad', 'wgrad', '依赖'],
        probe: '为什么反向传播拆成 B (dgrad) 和 W (wgrad) 就能填满气泡？B 和 W 的数据依赖有什么本质区别？',
      },
      {
        id: 'zb-tradeoff',
        label: 'ZB-H1 vs ZB-H2 权衡：W 延迟执行会拉长激活生命周期（ZB-H2 显存最高翻倍）及 Optimizer 同步问题',
        weight: 18,
        keywords: ['zb-h1', 'zb-h2', '显存', '生命周期', 'optimizer', 'rollback', '同步', 'grad norm', 'allreduce'],
        probe: '把 W 往后推去填 Cooldown 气泡，会带来哪两个现实问题？（显存和 Optimizer Step 全局同步）',
      },
      {
        id: 'dualpipe-diff',
        label: '对比 DualPipe 双向注入：不仅消气泡，还兼顾 MoE All-to-All 通信与计算重叠',
        weight: 12,
        keywords: ['dualpipe', '双向', 'all-to-all', 'overlap', 'moe', '权重两份'],
        probe: '既然 Zero-Bubble 已经几乎做到 0 气泡，为什么 DeepSeek-V3 还要发明 DualPipe？',
      },
    ],
    hints: [
      'L1 方向提示：任何 Linear 层 $Y = XW$ 的反向都有两步矩阵乘：算 $\\nabla X = \\nabla Y \\cdot W^T$ (记为 B) 和算 $\\nabla W = X^T \\cdot \\nabla Y$ (记为 W)。',
      'L2 关键依赖：上游 Stage 只等你的 B ($\\nabla X$)，根本不等你的 W ($\\nabla W$)！所以 W 可以被调度器自由挪到任意气泡空隙里执行。',
      'L3 显存细节：W 需要保留前向激活 $X$ 直到 W 算完才能释放，因此推迟 W 越多（如 ZB-H2），峰值激活显存越大。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Optimizer 同步坑）：在 Zero-Bubble 里，不同 PP Stage 的最后几个 W 是在不同时刻算完的，但 AdamW 之前通常要做全局 Grad Norm Clipping（需要全 PP 同步）以及检查 Inf/NaN。ZB 是如何处理这个全局同步不引入新气泡的？',
      },
      {
        stage: 'corner',
        question: '追问 2（Stage 负载不均）：在 LLM 中，Stage 0 有 Embedding，最后一个 Stage 有 LM Head + 128K 词表的 Cross-Entropy，往往导致最后一个 Stage 耗时和显存远大于中间 Stage，实际工程怎么调平？',
      },
    ],
    referenceAnswer: `### Staff 级参考解析

#### 1. 流水线调度演进对比（$p$ = PP 数，$m$ = Micro-batch 数，$v$ = 虚拟 Stage 数）
| 方案 | 理论气泡率 | 峰值激活显存（相对单 micro-batch） | P2P 通信量 |
|---|---|---|---|
| **GPipe** | $\\frac{p-1}{m+p-1}$ | $O(m)$（必须攒完所有前向才反向） | $1\\times$ |
| **1F1B** | $\\frac{p-1}{m+p-1}$ | $O(p)$（早反向早释放激活） | $1\\times$ |
| **Interleaved 1F1B** | $\\frac{p-1}{v\\cdot m + p - 1}$ | 略高于 1F1B（约 $(1 + \\frac{p-1}{p\\cdot v})\\times$） | $v\\times$ |
| **ZB-H1** | 约 $\\frac{1}{3}$ 原始气泡（假设 $T_F \\approx T_B \\approx T_W$） | 与 1F1B **相同** $O(p)$ | $1\\times$ |
| **ZB-H2** | **接近 0%** | 最高约 $2\\times$ 1F1B（推迟释放激活） | $1\\times$ |

#### 2. Zero-Bubble 的核心打破点
- 对线性层 $Y = XW$，反向分为两步：
  - **B (dgrad)**：$\\nabla X = \\nabla Y W^T$ — 位于关键路径，必须立即传给前一个 Stage；
  - **W (wgrad)**：$\\nabla W = X^T \\nabla Y$ — **无跨 Stage 依赖**，只依赖本地缓存的 $X$ 与 $\\nabla Y$，可灵活调度填入 Warmup/Cooldown 气泡中！
- **Optimizer Post-Validation 技巧**：为了避免等待全局 Grad Norm / NaN 同步产生同步气泡，ZB 让各 Stage 先基于前一步状态乐观更新（或延迟一步校验），仅在极少见的 NaN 发生时执行 Rollback。

#### 3. 首尾 Stage 不均衡（Embedding / LM Head）工程优化
- 将首个 Stage 和末尾 Stage 各少分 1~2 层 Transformer Layer（Megatron \`--decoder-first-pipeline-num-layers\` / \`--decoder-last-pipeline-num-layers\`）；
- 或者将 Embedding 和 LM Head 独立拆出为单独阶段 / 结合 Chunked CrossEntropy 压平末尾显存峰值。`,
  },
  {
    id: 'tr-08',
    track: 'training',
    title: '从零定量估算千卡训练 MFU、训练周期与真实瓶颈拆解',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['MFU', '6ND', 'FLOPs', '性能分析', '端到端估算'],
    summary: '用 6ND 公式快速心算千卡训练 70B 模型的周期，并精准剖析为什么真实 MFU 很难超过 50%~55%。',
    contextSpecs: ['70B Dense', '15T Tokens', '1024×H100 BF16 989 TFLOPS', 'MFU 估算'],
    prompt: '老板问你：用 1024 张 H100 (BF16 稠密峰值 989 TFLOPS) 从零预训练一个 70B Dense 模型，训 15T Tokens，大概需要多少天？请现场给出你的定量推导过程（列出假设的 MFU）；然后深入解释：为什么哪怕没有任何网络故障，实际 MFU 往往卡在 45%~55%，剩下 45% 的算力到底去哪儿了？',
    rubric: [
      {
        id: 'flops-6nd',
        label: '正确使用 6ND（含重算 8ND 注意 MFU 定义按模型有效 FLOPs 6ND 算）并算出总 FLOPs ≈ 6.3e24',
        weight: 25,
        keywords: ['6nd', '6*n*d', '6×', '6.3', '10^24', 'e24', '8nd', 'flops', '有效算力'],
        probe: '开启 Full Activation Checkpointing 时 GPU 实际做了 8ND 次计算，那算 MFU 时分子应该用 6ND 还是 8ND？为什么？',
      },
      {
        id: 'days-calc',
        label: '端到端天数计算准确：1024×989e12×MFU(0.45~0.5) ≈ 4.5~5e17 FLOP/s => 约 140~160 天',
        weight: 22,
        keywords: ['天', 'day', '140', '150', '160', '145', '86400', '45%', '50%', '989'],
        probe: '能把具体天数算出来吗？（提示：一天 86400 秒，1024 卡 × 989 TFLOPS ≈ 1e18 FLOP/s，按 50% MFU 算大概多少天？）',
      },
      {
        id: 'mfu-vs-hfu',
        label: '区分 MFU 与 HFU（重算引入的 2ND 算力损耗）及长序列 Attention O(S²) 项',
        weight: 18,
        keywords: ['hfu', '重算', 'recompute', 'attention', 's^2', '序列长度', '12*l*h*s'],
        probe: '除了 6ND，当序列长度拉到 32K/128K 时，Transformer 的精确 FLOPs 还要加上哪一项？',
      },
      {
        id: 'loss-reasons',
        label: '算力损耗拆解：通信气泡(PP/TP/DP)、Memory-bound 非矩阵算子、Host Overhead/小矩阵',
        weight: 22,
        keywords: ['memory-bound', '访存', 'bubble', '气泡', '通信', 'norm', 'softmax', 'swiglu', 'host', 'launch', 'power', '降频'],
        probe: '除了通信和重算，单卡纯算的时候为什么矩阵乘也跑不到标称 989 TFLOPS？',
      },
      {
        id: 'power-thermal',
        label: '硬件物理现实：标称 989 TFLOPS 实际受 700W TDP 功耗墙限制（GEMM 极限约 820~850 TFLOPS）',
        weight: 13,
        keywords: ['tdp', '功耗', '700w', '主频', '频率', '850', '800', 'cublas', '理论极限'],
        probe: '如果单卡只跑一个超大 cuBLAS BF16 GEMM（完全无通信、无其他算子），在 H100 上实测能跑到 989 TFLOPS 吗？',
      },
    ],
    hints: [
      'L1 估算捷径：总 FLOPs ≈ $6 \\times 70\\times 10^9 \\times 15\\times 10^{12} = 6.3 \\times 10^{24}$。1024 张 H100 理论峰值约 $1.01 \\times 10^{18}$ FLOP/s。',
      'L2 天数速算：$6.3\\times 10^{24} / (1.01\\times 10^{18} \\times 0.48) \\approx 1.3 \\times 10^7$ 秒 $\\approx 150$ 天（若切 FP8 有效提速 ~1.4-1.5× 可压到 ~100 天）。',
      'L3 损耗拆解：功耗墙 (~15%) + 激活重算 (若开 Full 耗 25%) + Memory-bound 算子 (Norm/RoPE/Elementwise/Optimizer ~10-15%) + 通信/PP 气泡 (~10-15%)。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（老板压工期）：150 天太久了，老板要求 45 天训完，打算把卡扩到 4096 张 H100。线性加速比（Weak/Strong Scaling）能保持吗？什么地方会率先崩掉？',
      },
      {
        stage: 'corner',
        question: '追问 2（HFU 高但 MFU 低）：你在监控面板上看到 GPU Util 一直是 100%，HFU 高达 68%，但 MFU 只有 38%，这通常说明系统配置踩了什么坑？',
      },
    ],
    referenceAnswer: `### Staff 级定量推导与损耗拆解

#### 1. 端到端训练时间速算
- **有效总计算量**（忽略超长序列二次项时）：
  $$\\text{FLOPs} \\approx 6 N D = 6 \\times (70 \\times 10^9) \\times (15 \\times 10^{12}) = 6.3 \\times 10^{24}$$
  *(若序列长度 $S$ 很大，需加上 Attention 二次项：Causal 跳过上三角后约 $6 \\cdot L \\cdot d \\cdot S$ FLOPs/token，未跳过按全矩阵计为 $12 \\cdot L \\cdot d \\cdot S$。)*
- **集群有效算力**（1024×H100 BF16 标称 989 TFLOPS，假设合理工程 MFU = **48%**）：
  $$1024 \\times 989 \\times 10^{12} \\times 0.48 \\approx 4.86 \\times 10^{17}\\text{ FLOP/s}$$
- **总耗时**：
  $$T = \\frac{6.3 \\times 10^{24}}{4.86 \\times 10^{17}} \\approx 1.296 \\times 10^7\\text{ s} \\approx \\mathbf{150\\text{ 天}}$$
  *(加上约 5% 集群宕机/Ckpt 开销，实际交付约 **158 天**；若升级 FP8 训练，MFU 相对 BF16 提速 ~1.35~1.5×，可缩至 ~105 天。)*

#### 2. 缺失的 ~50% 算力去哪了？
1. **硬件 TDP 功耗墙 (~15%)**：700W 满载跑大 GEMM 时 H100 核心频率会动态回落，单卡纯 cuBLAS 极限通常在 ~820-850 TFLOPS（标称 989 的 ~85%）。
2. **激活重算 Overhead (HFU vs MFU 剪刀差)**：MFU 分子只认有效 6ND；如果开了 Full Checkpointing，实际执行 8ND（多花 33% 前向计算），这就是为什么可能出现 **HFU=65% 但 MFU=48%**。
3. **非矩阵 Memory-Bound 算子 (~10-15%)**：RMSNorm、RoPE、SwiGLU、Cross-Entropy、AdamW 更新全是访存密集型，拉低算力利用率（靠 TorchCompile / Liger-Kernel 融合压低）。
4. **并行通信与流水线气泡 (~10-15%)**：TP 未完全重叠的 AllReduce/AllGather、PP Bubble、DP 尾部同步、以及慢节点尾延迟。
5. **Host Overhead & DataLoader**：Python 调度、小 Kernel Launch 延迟、分布式数据预处理抖动。`,
  },
];
