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
    contextSpecs: ['1024×H100 SXM 80 GB', '每 GPU NVLink 900 GB/s 双向聚合', '每节点 8×400 Gb/s IB', 'GBS=4,194,304 tokens'],
    prompt: '在 1024 张 H100 SXM 80 GB（每机 8 卡，机间每机 8×400 Gb/s）上训练 70B Dense 模型，80 层、H=8192、64 个 Q head 和 8 个 KV head。每步固定 4,194,304 tokens，等长序列从 8,192 扩到 131,072。给出两阶段各一组可行的 TP/PP/DP/CP，解释显存、GBS、通信和流水线气泡之间的取舍，再说明换成 MoE 后怎样安排 EP。',
    rubric: [
      {
        id: 'minimize-tp-pp',
        label: '结合显存峰值、FSDP 参数通信和 TP/PP 开销比较可行方案，允许不同拓扑',
        weight: 24,
        keywords: ['显存', '放得下', 'fsdp', 'hsdp', 'tp=1', 'tp=2', 'tp=4', '不开pp', 'pp=1', '关键路径', 'gemm', '小矩阵', '最小化', '克制'],
        probe: '如何在同一 GBS 下比较 HSDP 参数 AllGather、机内 TP 和跨机 PP 的成本？',
      },
      {
        id: 'gbs-tp-nvlink',
        label: '8K 时每步 512 条序列，mbs=1 时 DP≤512，TP×PP×CP 至少为 2，结合拓扑选择 TP/SP',
        weight: 20,
        keywords: ['gbs', '512', 'nvlink', '机内', 'sp', 'sequence parallel', 'allgather', 'reducescatter', 'allreduce', '张量并行'],
        probe: '每步 512 条序列如何限制 DP？在满足总卡数的几种 TP/PP/CP 组合中怎样选择？TP+SP 的通信是什么？',
      },
      {
        id: 'pp-vpp-tradeoff',
        label: '对比 1F1B 和 Interleaved VPP 的理想气泡率、阶段边界数、层数与调度约束',
        weight: 18,
        keywords: ['pp', '流水线', 'p2p', 'bubble', '气泡', '1f1b', 'interleaved', 'vpp', '整除', 'micro-batch', 'stage'],
        probe: '在均匀切层和自定义切层两种实现里，VPP 分别要满足哪些约束？',
      },
      {
        id: 'cp-long',
        label: '128K 时每步 32 条序列，DP≤32。给出卡数乘积、激活和本地 head 切分都成立的方案',
        weight: 20,
        keywords: ['cp', 'context parallel', 'ring attention', 'ulysses', '128k', '长上下文', '长文本', '激活显存', 'activation', '32'],
        probe: '每步 32 条长序列如何限制 DP？比较切分序列、参数和流水阶段对激活峰值与步时的影响。',
      },
      {
        id: 'quant-math',
        label: '通信量注明单 rank 发送口径、精度、组大小，以及 TP/PP 分片后的本地参数量',
        weight: 18,
        keywords: ['bsh', '8bsh', '4bsh', '2(n-1)', 'bytes', '字节', '带宽', '公式', '估算', '通信量', '4ψ', '2ψ'],
        probe: '请按 BF16 (2字节) 精确算一下：开启 TP+SP 后，单层 Transformer Block 前向+反向（不含重算）每卡总通信字节数是多少个 B·S·H？',
      },
    ],
    hints: [
      'L1 先列卡数恒等式和每步序列数，再比较 FSDP 通信、机内 TP 的 GEMM 形状与 PP 的空闲时间。',
      'L2 8K、mbs=1、TP=2、PP=CP=1 时，DP=512，每个 DP rank 处理 1 个 micro-batch。HSDP 的分片维度在 DP 内。',
      'L3 BF16 的 TP+SP 中，每子层前向有一次 AllGather 和一次 ReduceScatter。按每 rank 单向发送计，先算一个子层，再计 Attention 和 MLP。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（流水线气泡）　在 8K 阶段取 TP=4、PP=8、DP=32、mbs=1。算出每个 DP 副本的 micro-batch 数和标准 1F1B 气泡率。若各虚拟阶段层数相同，80 层允许 VPP=4 吗？',
      },
      {
        stage: 'corner',
        question: '追问 2（MoE）　换成 256 个路由专家、Top-8 的 671B MoE 后，Attention 和专家层各自如何分组？哪些实测条件支持 TP=1？',
      },
    ],
    referenceAnswer: `### 从每步样本数确定可行拓扑

等长无填充，设每个 DP 副本每次处理 b 条序列、累积 m 次。每步 token 数 = DP × b × m × S，物理卡数 = TP × PP × CP × DP。8K 时每步 512 条，128K 时只有 32 条。HSDP 的分片和复制都在这个 DP 维度内。

- **8K 示例**。取 TP=2、PP=CP=1、DP=512，b=m=1。以 BF16 权重和梯度、FP32 主权重和 Adam 两个矩共 16 字节/参数估算，若 DP 内 64 路全分片、8 路复制，每卡常驻模型状态约 70×10⁹×16/(2×64) = **8.75 GB**。还要为层级 AllGather、激活、预取和工作区留空间。64 路分片跨机，要和较小分片组或更高机内 TP 的步时、峰值显存对照。
- **128K 示例**。取 TP=2、CP=16、PP=1、DP=32，b=m=1。标准 head 切分后每个 TP rank 有 4 个 KV head。若后端支持混合 CP，同机 8 卡可排 TP=2 × Ulysses=4，再用 Ring=4 跨机。也可用纯 Ring。算例约束是 TP×PP×CP 至少为 32，CP 的最优值取决于本地激活和链路。
- **PP 反例**。8K 取 TP=4、PP=8、DP=32，则 m=16。均衡阶段、忽略 P2P 时，1F1B 的空闲时间占总时间 7/23 = 30.4%。等层切分要求 8v 整除 80，v=4 不满足，v=2 时理想占比 7/39 = 17.9%。自定义不等长布局须按最慢阶段重算。VPP 增加跨阶段消息，显存紧或 FSDP 跨机传参过重时，PP 仍有价值。

### 每 rank 发送字节数

以下采用 ring 账本，接收量另计，BF16 为 2 字节。S 为 TP 组内参与该运算的序列长度，H 为未做 TP 切分的隐藏维。
- **TP+SP**。典型 Megatron 稠密层的 Attention 和 MLP 各做一次 AG 和一次 RS，前向每层发送 8(T−1)/T × bSH 字节。反向同量，不含重算。
- **PP**。若边界传完整 BF16 隐状态，每 micro-batch 前向传 2bSH 字节，反向同量。若边界传序列分片，改用本地张量形状。
- **DP**。设单个 TP/PP 分片有 Ψ_local 个参数，D 路 RS 梯度和 AG 更新权重且二者均为 BF16，每次同步发送 4(D−1)/D × Ψ_local 字节。归约精度和同步频率影响结果。
- **Ring CP**。S 为整个 CP 组序列长度、Hkv 为本 TP rank 的 KV 宽度，朴素满环前向每层发送 4(C−1)/C × bS Hkv 字节。

**MoE 追问**。按 attention 和专家分别画通信组。EP 可重用稠密层的 DP 设备维度。比较每专家 token 数、跨节点 dispatch/combine、dense 参数复制和激活峰值，再决定 TP 与 EP。DeepSeek-V3 在其训练拓扑上使用 TP=1。

> 依据　Megatron-LM 并行训练论文（arXiv:2104.04473）及 DeepSeek-V3 Technical Report 第 3 节。`,
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
    prompt: '假设 BF16 权重、BF16 梯度、FP32 主权重和 FP32 Adam 一二阶矩均常驻。拆解每参数 16 字节的来源，并比较 ZeRO-1/2/3 在 D 个 DP rank 上的静态显存和一次前反向的通信。PyTorch FSDP2 的 per-parameter DTensor 改变了哪些分片、冻结和保存权重的操作？',
    rubric: [
      {
        id: 'mem-16psi',
        label: '准确拆解 16Ψ：BF16权重2Ψ + BF16梯度2Ψ + FP32主权重4Ψ + Adam一二阶矩8Ψ',
        weight: 25,
        keywords: ['16ψ', 'bf16', 'fp32', 'adam', 'master weight', '主权重', '一阶矩', '二阶矩', '动量', '方差'],
        probe: '请逐项算一下 16Ψ 里的每一块分别是多少字节：参数、梯度、Master Weight、Momentum、Variance？',
      },
      {
        id: 'zero-stages',
        label: 'ZeRO-1(Opt) / ZeRO-2(Opt+Grad) / ZeRO-3(Opt+Grad+Param) 切分与内存公式',
        weight: 22,
        keywords: ['zero-1', 'zero1', 'zero-2', 'zero2', 'zero-3', 'zero3', '优化器', '梯度', '参数', 'sharding', '分片'],
        probe: '采用相同权重和梯度精度时，ZeRO-1/2 各节省哪些常驻状态？什么条件下愿意用参数 AllGather 再换显存？',
      },
      {
        id: 'comm-3psi',
        label: '区分一次前反向与整步。理想 Ring 下每 rank 发送量含 (D−1)/D，按通信 dtype 换算字节',
        weight: 20,
        keywords: ['allgather', 'allreduce', 'reducescatter', 'reduce-scatter', 'all-gather', 'ring', '发送量', '权重重分片', '反向 ag', '6fψ', '4fψ'],
        probe: '设一次前反向且前向后重分片，ZeRO-3 比 DDP 多传什么？若改变 reshard 策略或累积频率，通信会怎样变化？',
      },
      {
        id: 'fsdp2-dtensor',
        label: 'FSDP2 的参数表示、通信分组、冻结参数和分片 state dict',
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
      'L2 一次前反向按每卡发送计，Ring 的 RS 或 AG 各是 (D−1)/D 个完整张量。参数重分片策略决定反向是否还需 AG。',
      'L3 FSDP1 通常按包装单元展平。FSDP2 保留参数身份，多个参数仍可合并通信。',
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
    referenceAnswer: `### 常驻状态和通信分别算

设 Ψ 为一个 DP 副本中的参数个数，D 为分片数。题设包含权重 2Ψ、梯度 2Ψ、主权重 4Ψ、Adam 一阶矩和二阶矩各 4Ψ 字节，共 16Ψ。若梯度改存 FP32，合计为 18Ψ。不同优化器实现应按实际张量核算。

| 策略 | 每卡常驻字节 | 一次前反向、一次更新的每卡发送量 |
|---|---|---|
| DDP | 16Ψ | 梯度 AllReduce，4fΨ 字节 |
| ZeRO-1 | 4Ψ + 12Ψ/D | RS 梯度、AG 更新后的权重，4fΨ |
| ZeRO-2 | 2Ψ + 14Ψ/D | RS 梯度、AG 更新后的权重，4fΨ |
| ZeRO-3 | 16Ψ/D | 前反向各 AG 一次权重，RS 一次梯度，6fΨ |

f=(D−1)/D，以 Ring 每 rank 的发送量计，接收量相同。表中通信张量均为 BF16，ZeRO-3 在前向后重分片。全量参数临时展开、并发预取、激活和 optimizer 临时张量都可能造成峰值。可用逐阶段显存快照区分活跃张量与分配器碎片。

### FSDP2、累积与跨节点

- FSDP2 默认用沿 dim-0 分片的每参数 DTensor 表示模型，保留参数名和身份，便于分别冻结参数和处理分片 state dict。每次 fully_shard 调用决定一组参数的集合通信粒度，组太大可能抬高峰值。编译和 checkpoint 仍需按模型验证。
- 累积 k 个 micro-batch 时，应逐次计 AG/RS。\`reshard_after_forward=False\` 可以省同一 micro-batch 反向前的 AG。PyTorch 2.9 的 \`set_reshard_after_backward(False)\` 可保留参数到下个 micro-batch，省去下一次前向 AG。是否延迟梯度同步另行设置。保留完整参数或梯度会增加峰值，关闭梯度同步本身不会取消参数 AG。
- 512 卡 HSDP 示例可在机内 8 卡分片、64 台间复制同号分片。机内做参数 AG 和梯度 RS，跨机对相应梯度分片做 AllReduce。换来的代价是常驻状态按 8 路分摊，需核算 32B 模型的本地容量。

> 依据　ZeRO（arXiv:1910.02054）及 PyTorch 2.9 的 fully_shard、FSDPModule 文档。`,
  },
  {
    id: 'tr-03',
    track: 'training',
    title: '超大规模 MoE 训练架构：EP All-to-All、负载均衡与 DualPipe/DeepEP',
    level: 'P8',
    frequency: '高频必考',
    incident: false,
    tags: ['MoE', 'Expert Parallelism', 'All-to-All', 'DeepEP', 'DualPipe'],
    summary: '以 DeepSeek-V3 为例，分析 Top-8 路由的跨机流量、热点专家及 DualPipe 可隐藏的通信。',
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
        label: '节点受限路由控制跨机扇出，按目的节点去重后由机内互连分发',
        weight: 20,
        keywords: ['node-limited', '节点受限', '目的节点', '跨 ib', '去重', '机内', 'nvlink', '两级', '层次', 'dedup'],
        probe: 'Top-8 的逻辑分发次数、目的节点数和实际跨 IB 发送的激活份数有何区别？',
      },
      {
        id: 'aux-loss-free',
        label: '负载均衡与 Token Dropping：Aux-loss-free (动态 Bias 调节) 替代强辅助损失',
        weight: 20,
        keywords: ['负载', 'load balance', 'aux-loss', 'bias', '偏置', 'token drop', 'dropping', 'capacity', '热点', '倾斜'],
        probe: '动态路由偏置怎样调整负载、怎样影响专家输出权重？剩余倾斜如何处理？',
      },
      {
        id: 'overlap-dualpipe',
        label: 'DualPipe 的双向流水、跨 micro-batch 重叠与通信 SM 预算',
        weight: 22,
        keywords: ['dualpipe', 'deepep', 'overlap', '重叠', 'sm', '双向', 'attention', 'mlp', 'grouped gemm', '通信sm'],
        probe: '常规 1F1B 很难把跨机 All-to-All 完全藏在算子后面。DualPipe 和 DeepEP 是怎么把通信和计算（前向+反向）交错重叠起来的？',
      },
      {
        id: 'grouped-gemm',
        label: '按专家重排 token，以 Grouped GEMM 处理变长的稠密矩阵乘法',
        weight: 16,
        keywords: ['grouped gemm', 'permute', 'padding', 'kernel', 'fp8', '稀疏', 'cutlass', '算子'],
        probe: 'Token Dispatch 到达目标 GPU 后，每个专家收到的 Token 数不相等，在 GPU 上怎么高效执行多专家矩阵乘？',
      },
    ],
    hints: [
      'L1 分别画出 dispatch 和 combine，比较最忙 rank 的 token 数、IB 流量和专家计算。',
      'L2 V3 限制一个 token 最多到 4 个节点。按目的节点去重后，每个远端节点各收一份激活，再做机内分发。',
      'L3 跨 micro-batch 的计算要足够长，通信流和计算流之间也必须满足数据依赖，才能隐藏通信。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（显存代价）　原版 DualPipe 需要两份参数。哪些参数由 EP/PP 分摊，哪些仍复制？如何估算可用的激活空间？',
      },
      {
        stage: 'corner',
        question: '追问 2（容错与动态扩缩）：如果训练中某张卡收到的动态 Token 数突然飙升导致临时激活 OOM（由于不 Drop Token），工程上应该如何防御？',
      },
    ],
    referenceAnswer: `### 跟踪一个 token 的前向与反向

Top-8 为每个 token 产生 8 条逻辑专家分配。前向 dispatch 将隐藏向量交给专家，combine 将输出按门权重加总。反向分别传送对应梯度，常见实现前反向各两次逻辑 All-to-All。物理流量要扣除本地路由并考虑分层传输。V3 限制最多 4 个目的节点，同一目的节点的多个专家可共享一份跨 IB 的输入，再经 NVLink 转发。

- **负载**。选专家时对原始分数加动态 bias，过载减、欠载增。组合专家结果使用原始亲和分数。V3 还保留了很小的序列级均衡辅助项。路由均衡改善长期分布，单个 micro-batch 仍可能集中在少数专家。
- **算子与重叠**。按专家整理 token，用 Grouped GEMM 批量运行大小不同的稠密 FFN。DualPipe 交错不同 micro-batch 的前反向，让已就绪的计算覆盖 dispatch/combine。通信核占用部分 SM，还会与 GEMM 竞争内存带宽。用最慢 rank 的时间线和暴露在关键路径上的通信检验效果。
- **容量与显存追问**。原版 DualPipe 保存两份参数，按各 stage 本地 dense/共享专家、EP 分片、优化器和在途激活分别预算。Dropless 路径按实际路由计数准备缓冲，对热点预留余量，必要时将同一批专家 token 分轮计算并按原路由合并。限制每专家容量后直接丢弃 token 会改变训练语义。

> 依据　DeepSeek-V3 Technical Report（arXiv:2412.19437），第 2.1.2、3.2 节。`,
  },
  {
    id: 'tr-04',
    track: 'training',
    title: 'FP8 混合精度训练稳定性、Scaling 策略与 Loss Spike 防御',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['FP8', 'E4M3/E5M2', 'Blockwise Quant', 'Loss Spike', 'Hopper'],
    summary: '比较 FP8 格式、历史 amax 和当前块缩放，分析离群值、累加误差及 Loss Spike 的定位方法。',
    contextSpecs: ['H100 SXM dense FP8 1,979 TFLOP/s', 'E4M3 / E5M2', '1×128 / 128×128 scaling', 'FP32 accumulation'],
    prompt: '以 Hopper 上的 FP8 训练为例，解释 E4M3 和 E5M2 的表示范围，比较 per-tensor delayed scaling 与细粒度当前块缩放。若某个 batch 出现 Loss Spike，如何区分 scale、累加精度、模型敏感层和数据的影响？',
    rubric: [
      {
        id: 'fp8-format',
        label: 'E4M3/E5M2 的尾数与指数位、最大有限值和不同配方中的用途',
        weight: 22,
        keywords: ['e4m3', 'e5m2', '动态范围', '尾数', '指数', '448', '梯度', '前向', '激活'],
        probe: 'E4M3 和 E5M2 各自的指数位、尾数位以及适用张量类型是什么？为什么现在很多细粒度方案连梯度也统一用 E4M3？',
      },
      {
        id: 'delayed-scaling',
        label: 'Delayed Scaling (历史 Amax 窗口) 面对 Activation Outlier 突变导致的 Underflow/Overflow',
        weight: 22,
        keywords: ['delayed', 'amax', 'outlier', '离群值', '溢出', '下溢', 'overflow', 'underflow', 'per-tensor', 'transformer engine'],
        probe: '历史 amax 偏大或偏小时分别有哪些风险？同一张量的不同通道共用 scale 会怎样影响舍入？',
      },
      {
        id: 'fine-grained',
        label: 'DeepSeek-V3 的激活 1×128 和权重 128×128 scale 粒度，以及每块的数值取舍',
        weight: 24,
        keywords: ['1×128', '128×128', '1x128', '128x128', 'tile', 'block', '细粒度', '分块', '局部 scale', 'fine-grained', '分组'],
        probe: '激活 1×128 和权重 128×128 分组的离群值影响范围各有多大？转置如何影响 scale 布局？',
      },
      {
        id: 'fp32-accum',
        label: '区分 Hopper FP8 指令输出类型与长 K 内积的中间累加误差，解释分段 FP32 累加',
        weight: 18,
        keywords: ['累加', 'accum', 'wgmma', 'cuda core', 'fp32', '精度损失', 'tensor core', 'promotion', 'master'],
        probe: '你知道 Hopper Tensor Core 在执行 FP8 WGMMA 时的内部累加精度其实并不是完整的 FP32 吗？这在长内积维度 K 上会造成什么偏差，工程上怎么修？',
      },
      {
        id: 'sensitive-ops',
        label: '敏感模块保持 BF16/FP32：Embedding、LM Head、MoE Gate、LayerNorm、Attention Softmax',
        weight: 14,
        keywords: ['embedding', 'lm head', 'gate', 'router', 'norm', 'softmax', 'bf16', '敏感', '主权重'],
        probe: '哪些运算或状态对低精度敏感？如何验证适合的保留精度？',
      },
    ],
    hints: [
      'L1 分别检查表示范围、scale 估计、累加和敏感算子。',
      'L2 分块将离群值的影响局限在该块。转置与重分组时要检查量化方向。',
      'L3 V3 在 H800 上观察到长 K 内积的累加误差，以 K=128 分段进行高精度累加。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（通信与存储）：既然已经做了 FP8 训练，通信（比如 EP All-to-All 和 DP 梯度）以及 Optimizer States 能不能也压缩到 FP8 或 BF16 来进一步省显存和带宽？各有什么风险？',
      },
      {
        stage: 'corner',
        question: '追问 2（Blackwell）　MXFP8 和 NVFP4 如何在硬件中处理块 scale？它们与 Hopper 上控制长内积累加误差解决的是同一个问题吗？',
      },
    ],
    referenceAnswer: `### 表示、缩放和累加是三个独立选择

E4M3 有 4 位指数、3 位尾数，常用有限值格式最大 448。E5M2 为 5 位指数、2 位尾数，最大 57,344。TE 的 HYBRID 配方常在前向用 E4M3、反向梯度用 E5M2。缩放粒度足够细时，也有梯度用 E4M3 的配方。

- **缩放**。设反量化因子 s，量化为 FP8(x/s)。历史 amax 估得太小会使新峰值饱和，估得太大会让小量的有效精度下降乃至舍入为零。Per-tensor 控制影响范围，delayed/current 控制何时取得 amax，这两项应分别讨论。
- **V3 的实现**。激活按 1×128，权重按 128×128 计算局部 scale。单个异常值仍影响同块元素。报告在 H800 的 FP8 GEMM 上观测到部分和仅保留约 14 位精度。其内核对长 K 的乘加分段，每 128 个元素将部分和提升到 CUDA Core 做 FP32 累加并处理 scale。可将输出同 BF16/FP32 基线逐层比较，观察误差随 K 的变化。
- **遇到 Loss Spike**。在相同 checkpoint 和 batch 上比较 BF16 与 FP8 的首个异常层，记录 amax、scale、饱和/零值比例、grad norm。优先检查 router、归一化、softmax、输出层及优化器状态。逐项回到高精度可以区分缩放、累加、通信和数据问题。
- **扩展追问**。通信可量化 dispatch 的激活，必须传 scale 或约定格式。梯度归约涉及跨 rank 求和，要单独选累加精度并验证收敛。Adam 矩的低精度存储也需独立验证。Blackwell 的 MXFP8 可在点积内使用每 32 个元素的块 scale。NVFP4 使用更细的 4-bit 格式和块缩放。原生块缩放减少软件处理，累加误差仍取决于实际指令、K 和数据。

> 依据　DeepSeek-V3 Technical Report 第 3.3 节及 NVIDIA Transformer Engine《Using FP8 and FP4》。`,
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
    prompt: '一个 64 个 Q head、8 个 KV head 的 70B GQA 模型要训练 256K 上下文。考虑用 16 卡做 Context Parallelism。比较 Ring Attention 与 DeepSpeed Ulysses 的数据布局、通信量和因果计算负载，给出一种不复制 KV head 的可行组合。',
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
        label: '无 KV 复制的 head 切分要求本地 Q/KV head 能被 Ulysses 度数整除，解释超限选项',
        weight: 20,
        keywords: ['gqa', 'kv head', '受限', '整除', '复制', '重复 kv', 'mqa'],
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
        label: '结合本地 head 数与互连选择 Ulysses×Ring 的分解，并估算通信与重叠',
        weight: 16,
        keywords: ['hybrid', 'usp', '混合', '机内', '跨机', 'nvlink', 'ulysses', 'ring', '跨节点'],
        probe: '给定每机 8 卡、TP=1，怎样将 Ulysses 和 Ring 映射到两个节点？若 TP 已切分 KV head，方案如何变化？',
      },
    ],
    hints: [
      'L1 Ring 留 Q 在本地并传 KV，Ulysses 在序列与 head 的切分之间做转置。先固定每卡发送的口径。',
      'L2 单条等长 causal 序列可将 2P 块首尾配对，比较每个 rank 实际需算的有效 Q–K 对数。',
      'L3 若 TP=1，每机 8 卡且有 8 个 KV head，可以在机内做 8 路 Ulysses、两台之间做 2 路 Ring。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（通信量）　设 MHA，无 KV 复制，按每 rank 前向发送元素数比较完整 Ring 和 Ulysses。改变 P 或改为 GQA 后，二者的相对大小怎样变化？',
      },
      {
        stage: 'corner',
        question: '追问 2（变长 Packing）：预训练数据通常会做 Document Packing（多条短文档拼成一条 256K 序列，中间用 varlen mask 隔断，避免跨文档注意）。Packing 场景下 Zig-Zag Ring Attention 会遇到什么新问题？',
      },
    ],
    referenceAnswer: `### 两种布局与可复算的通信账

设 batch=1，全长 S，P 个 rank，Q 的总宽度为 Hq，K、V 各为 Hkv。以下只计一次前向、每 rank 发给其他 rank 的元素，BF16 乘 2 得字节。排除反向、TP 和自发自收。

| 方案 | 数据布局 | 每 rank 发送元素 |
|---|---|---|
| 完整 Ring | 本地保留 S/P 个 Q，轮转 P−1 块 K/V，按行维护在线 softmax 的最大值、归一化和输出分子 | 2(P−1) S Hkv / P |
| Ulysses | Attention 前交换 Q/K/V 使每 rank 拥有全序列的部分 head，之后逆交换 O | (P−1) S (2Hq+2Hkv) / P² |

Q/K/V 可打包成一次 All-to-All，也可分别发。MHA 时两者之比 Ring/Ulysses = P/2，P=2 相等，P>2 时完整 Ring 发得更多。GQA 下比值为 P Hkv/(Hq+Hkv)，较窄的 KV 有利于 Ring。减枝的 causal 通信、KV 复制或不同网络拓扑要重新计数。通信量和暴露延迟是两种指标。

### 16 卡、8 个 KV head

在 TP=1、每机 8 卡时，机内 NVLink 做 Ulysses=8、跨机做 Ring=2 可令每个 Ulysses rank 分得 8 个 Q head、1 个 KV head。纯 16 路按 head 均分需要复制 KV 或换算法。Ring 对齐当前 KV 块和下一块传输，实际重叠取决于块计算时间和链路。

单条 causal 序列的尾端 Q 能看到更多 K，连续均分 token 会使后部 rank 更忙。将 2P 个等长块首尾配对可平衡有效 Q–K 对数。若 packed 文档使用文档间隔离 mask，还要按各文档的真实边界累计这些对数，按 token 均分可能失衡。若允许跨文档 attention，则用连续序列的 mask 来算。

> 依据　DeepSpeed-Ulysses（arXiv:2309.14509）和 Ring Attention（arXiv:2310.01889）。`,
  },
  {
    id: 'tr-06',
    track: 'training',
    title: '万卡集群容错与稳定性排障：Loss Spike、SDC、慢节点与无感 Checkpoint',
    level: 'P8',
    frequency: '高频必考',
    incident: true,
    tags: ['容错', 'Checkpoint', 'SDC', 'NCCL Hang', 'Straggler'],
    summary: '从 8192 卡训练作业的超时、Loss Spike 和重启耗时入手，定位故障并验证 checkpoint 的可恢复性。',
    contextSpecs: ['8192×H100', '题设 Goodput 72%', '重启 25 分钟', 'Async Checkpoint'],
    prompt: '【线上实战排障】你负责维护一个 8192 卡 H100 预训练任务。过去一周集群有效训练时间（Goodput）跌到了 72%：一方面频繁出现 NCCL Timeout/Hang 和偶发 Loss Spike，另一方面每次节点故障重启加载 Checkpoint 要花 25 分钟。请系统性地设计一套排障定位与高可用容错体系。',
    rubric: [
      {
        id: 'hang-diag',
        label: 'NCCL Hang / 慢节点排查：Flight Recorder、Trace 归因、区分真 Hang vs 单点 Straggler',
        weight: 22,
        keywords: ['flight recorder', 'watchdog', 'hang', 'straggler', '慢节点', '降频', 'thermal', 'ib', '丢包', 'pfc', 'trace'],
        probe: '超时后怎样比较各 rank 的 collective 顺序和时间，把嫌疑范围缩小到代码、设备或链路？',
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
        label: '一致版本快照、训练侧阻塞和异步持久化分别计时。完整提交后才能恢复',
        weight: 22,
        keywords: ['async', '异步', 'checkpoint', 'pinned', 'cpu', '快照', 'snapshot', '后台', '阻塞', 'dfs'],
        probe: '异步 checkpoint 哪一阶段仍会阻塞训练？何时可安全改写源状态，何时可宣告该 checkpoint 可恢复？',
      },
      {
        id: 'fast-recovery',
        label: '验证 DP peer 是否覆盖全部状态分片，再选择内存副本、持久化 checkpoint 或热备',
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
      'L1 明确 Goodput 的分母，用已提交的有效 token 或未丢失的进度识别慢步、阻塞、停机和重放。',
      'L2 同号权重副本可能缺失优化器或 RNG 分片。恢复前先验证同一步的全状态覆盖。',
      'L3 在健康与嫌疑节点做相同输入的确定性重放，对照数值容差、版本与硬件错误记录。',
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
    referenceAnswer: `### 从第一个分歧定位

- 题设的 72% 应先核对分母。净新增且保留的训练 token 数除以墙钟时间得到 token/s，再除以同配置健康基线速率可得到百分比。若平台按有效 GPU·秒占比分摊，则沿用平台定义。对齐超时前各 rank 的 step、process group、collective 序号/形状、进入和完成时间。PyTorch NCCL Flight Recorder 的环形缓冲和超时 dump 需事先开启。缺席 rank 可能卡在上一算子、数据加载或不同的代码分支。所有 rank 都进入同一操作时，再看链路重试、GPU Xid、ECC、降频和通信拓扑。超时报错的 rank 未必是故障源。
- Loss Spike 时保存权重、优化器、RNG、数据游标和局部梯度统计。相同软件、checkpoint、样本在健康机和嫌疑机交叉重放，控制非确定性规约。单次未复现不足以诊断 SDC。跟随设备重复出现的数值异常和独立硬件校验能增强归因。

### 快照和恢复的两个完成时刻

1. 在同一逻辑步冻结全部可变状态。让 D2H 等待生产者 stream，再通过 event 或独立不可变副本，保证任何写入都晚于对应读取。Pinned host 缓冲在后台写出前保持有效。限制在途快照数并测量 staging 阻塞。
2. 各 rank 的分片和元数据持久化、完整性检查通过后再提交 manifest。故障时选择最近一次完整提交。恢复模型、optimizer、scheduler、RNG 和数据状态。机内副本或热备能缩短恢复，前提是同一步的每个必要分片都有幸存冗余，并能重建通信组。
3. **8192→8184**。先确认 TP/PP/CP 和专家组能否在新 world size 上整除，再核对新 DP、mbs、梯度累积能否精确保持 GBS，以及分片能否重排。若整数条件不成立，可另选可行的参与卡数或并行布局。按已消费 token 恢复学习率和数据游标，验证一次故障演练的丢失进度与恢复墙钟耗时。

> 依据　PyTorch《Flight Recorder》及《Asynchronous Saving with Distributed Checkpoint》。`,
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
        probe: '在同样的理想气泡率下，1F1B 与先做完所有前向的 GPipe 在激活生命周期上有何区别？',
      },
      {
        id: 'interleaved',
        label: '在阶段均衡、通信可忽略时区分 VPP 的气泡时间与占比。解释新增的 P2P 边界',
        weight: 22,
        keywords: ['interleaved', 'vpp', '虚拟', '1/v', '通信增加', 'v倍', 'p2p', 'chunk'],
        probe: 'VPP 的气泡时间和气泡占总时间的比例分别怎样变化？通信次数与激活峰值如何估计？',
      },
      {
        id: 'bw-split',
        label: 'Zero-Bubble 拆分输入梯度和权重梯度，使 dX 优先跨 stage 传回，dW 在更新前完成',
        weight: 26,
        keywords: ['dgrad', 'wgrad', 'dx', 'dw', '输入梯度', '权重梯度', '依赖', '延后', '关键路径'],
        probe: '为什么反向传播拆成 B (dgrad) 和 W (wgrad) 就能填满气泡？B 和 W 的数据依赖有什么本质区别？',
      },
      {
        id: 'zb-tradeoff',
        label: 'ZB-H1/H2 延后 W 的激活预算和 optimizer 全局校验代价',
        weight: 18,
        keywords: ['zb-h1', 'zb-h2', '显存', '生命周期', 'optimizer', 'rollback', '同步', 'grad norm', 'allreduce'],
        probe: '把 W 往后推去填 Cooldown 气泡，会带来哪两个现实问题？（显存和 Optimizer Step 全局同步）',
      },
      {
        id: 'dualpipe-diff',
        label: '对比 DualPipe 双向注入：不仅消气泡，还兼顾 MoE All-to-All 通信与计算重叠',
        weight: 12,
        keywords: ['dualpipe', '双向', 'all-to-all', 'overlap', 'moe', '权重两份'],
        probe: '在跨机 MoE 场景，DualPipe 的调度还要覆盖哪些通信？',
      },
    ],
    hints: [
      'L1 方向提示：任何 Linear 层 $Y = XW$ 的反向都有两步矩阵乘：算 $\\nabla X = \\nabla Y \\cdot W^T$ (记为 B) 和算 $\\nabla W = X^T \\cdot \\nabla Y$ (记为 W)。',
      'L2 上一 stage 要等待 dX。dW 在本地输入和输出梯度就绪后可延后，但必须赶在参数更新之前，且中间值要保留至使用结束。',
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
    referenceAnswer: `### 先定义气泡率

p 为物理 stage 数，m 为每个副本每步的 micro-batch 数。气泡率是单 stage 空闲时间占迭代总时间的比例。阶段均衡、消息开销可忽略时，GPipe 和同步 1F1B 都是 (p−1)/(m+p−1)。p=8、m=32 时为 7/39≈17.9%。GPipe 的首段可积攒 m 份激活，1F1B 的常规调度最多约 min(m,p) 份，具体看 stage。

- 每个 rank 交错执行 v 个等时虚拟 stage 时，理想气泡**时间**降至 1/v，**占比**为 (p−1)/(vm+p−1)。v=2 时是 7/71≈9.9%。线性链经过 pv−1 个逻辑边界，跨卡消息取决于映射。额外通信与负载不均会侵蚀收益。
- 线性层 Y=XW 的反向含 dX=dY Wᵀ 和 dW=Xᵀ dY。前一 stage 要等 dX 才能继续。dW 只需在本次更新前完成，把它移入空档可缩短关键路径，代价是 X、dY 活得更久。
- **ZB-H1/H2**。论文在 F、B(dX)、W(dW) 各耗 t 且通信可忽略的模型中，基准气泡时间为 3(p−1)t，H1 约为 (p−1)t，激活预算与 1F1B 相当，其理想气泡占比为 (p−1)/(3m+p−1)。H2 可达零调度气泡，激活预算最高约为两倍。实测时应代入各段耗时、通信和峰值显存。
- **更新边界**。全局 grad norm 和非有限值检查可能阻塞下一步。论文的 optimizer post-validation 乐观执行，若全局结果需要裁剪或跳步，恢复一致状态并重做受影响计算。回滚频繁时收益下降。DualPipe 还把 MoE 的 dispatch/combine 与另一批计算错开，目标包括缩短暴露通信。
- **调平首尾**。测每个 stage 的前反向时间和激活峰值，按 embedding、LM head、词表 CE 的开销调层数。分块 CE 侧重降低显存，是否缩短末段须重新计时。

> 依据　Megatron-LM（arXiv:2104.04473）与 Zero Bubble Pipeline Parallelism（arXiv:2401.10241）。`,
  },
  {
    id: 'tr-08',
    track: 'training',
    title: '从零定量估算千卡训练 MFU、训练周期与真实瓶颈拆解',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['MFU', '6ND', 'FLOPs', '性能分析', '端到端估算'],
    summary: '用 6ND 估算千卡训练周期，区分有效模型 FLOPs、实际执行 FLOPs 和集群停机时间。',
    contextSpecs: ['70B Dense', '15T Tokens', '1024×H100 BF16 989 TFLOPS', 'MFU 估算'],
    prompt: '1024 张 H100 SXM、BF16 稠密峰值每卡 989 TFLOP/s，预训练 70B Dense 模型，消费 15T tokens。忽略长序列 attention 的二次项，任选一组合理 MFU 估算天数。若实测 MFU 为 48%，用哪些证据区分重算、GPU 算子、通信和停机的影响？',
    rubric: [
      {
        id: 'flops-6nd',
        label: '说明 6ND 是参数矩阵乘的有效训练 FLOPs 近似，算出 6.3×10²⁴，并说明重算口径',
        weight: 25,
        keywords: ['6nd', '6*n*d', '6×', '6.3', '10^24', 'e24', '8nd', 'flops', '有效算力'],
        probe: '按参数矩阵乘近似，全前向重算一次增加多少 FLOPs？计算 MFU 和 HFU 时各计入哪些工作？',
      },
      {
        id: 'days-calc',
        label: '时间换算使用 10¹² FLOP/TFLOP 和 86,400 秒/天，48% 时约 150 天',
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
        label: '区分 SKU 标称 dense 峰值与对应形状的实测速率，结合时钟、功耗及 Tensor Core 指标判断',
        weight: 13,
        keywords: ['tdp', '功耗', '主频', '时钟', '频率', 'cublas', '形状', 'shape', '带宽', '峰值'],
        probe: '如果单卡只跑一个超大 cuBLAS BF16 GEMM（完全无通信、无其他算子），在 H100 上实测能跑到 989 TFLOPS 吗？',
      },
    ],
    hints: [
      'L1 估算捷径：总 FLOPs ≈ $6 \\times 70\\times 10^9 \\times 15\\times 10^{12} = 6.3 \\times 10^{24}$。1024 张 H100 理论峰值约 $1.01 \\times 10^{18}$ FLOP/s。',
      'L2 有效时间 = 总 FLOPs / (卡数 × 单卡 dense FLOP/s × MFU)。单独记录是否已经包含 checkpoint 和停机。',
      'L3 用时间线区分重算、关键路径上暴露的通信、访存受限算子与 host 空档，避免重复计算重叠时间。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（老板压工期）：150 天太久了，老板要求 45 天训完，打算把卡扩到 4096 张 H100。线性加速比（Weak/Strong Scaling）能保持吗？什么地方会率先崩掉？',
      },
      {
        stage: 'corner',
        question: '追问 2（计数口径）　GPU busy 100%、HFU 68%、MFU 38%。一次完整前向重算能否独自解释这个差距，接下来应该核对什么？',
      },
    ],
    referenceAnswer: `### 70B、15T tokens 的训练时间

按 dense Transformer 的参数矩阵乘近似，每参数每 token 约 2 FLOPs 前向、4 FLOPs 反向。总计 6ND = 6×70×10⁹×15×10¹² = **6.3×10²⁴ FLOPs**。H100 SXM 取题设 BF16 非稀疏峰值。1024 卡按 MFU=0.48 得 4.86×10¹⁷ FLOP/s，训练天数为 6.3×10²⁴/(4.86×10¹⁷×86,400) ≈ **150 天**。按 45% 和 50% 分别约 160 天和 144 天。如果 MFU 只在有效稳态步上测，且有效训练时间占墙钟的 95%，交付约 150/0.95≈158 天。若已按端到端 token/s 计算，停机已经计入。

- 6ND 忽略 attention 的序列相关项。每层全注意力的 QKᵀ 和 PV 前反向约多 12HS FLOPs/token，因果只计有效三角形时长序列近似减半。比较 MFU 时统一模型 FLOPs 定义、precision 和峰值。
- MFU 分子是有效模型 FLOPs，HFU 按实际执行（含重算）计。简单全前向重算约由 6ND 变为 8ND，二者峰值口径一致时 HFU/MFU 约 4/3。观测 68%/38% 已超过这个比值，应核查额外重算、padding、算子计数和采样窗口。GPU busy 100% 只说明采样期间有核在执行。
- 用相同形状和 dtype 的 GEMM 基准与时钟、功耗记录检查单卡效率。用 step 时间线量化未重叠的 TP/DP 通信、PP 空档、访存核和主机等待。各项可能重叠，应以关键路径归因。
- **4096 卡、45 天**。固定同一任务量属于 strong scaling，所需端到端 MFU 约 6.3×10²⁴/(4096×989×10¹²×45×86400) ≈ **40%**。核对 GBS 下每 rank 可用 micro-batch 数，再测扩容后的通信和气泡。

> 依据　PaLM（arXiv:2204.02311）关于 MFU 的定义及 NVIDIA H100 SXM 产品规格。`,
  },
  {
    id: 'tr-09',
    track: 'training',
    title: '分布式训练 Profiling 实战：Step Time 拆解、关键路径归因与 Straggler 定位',
    level: 'P7',
    frequency: '高频必考',
    incident: true,
    tags: ['Profiling', 'PyTorch Profiler', 'nsys', 'HTA', '通信重叠', 'Straggler'],
    summary: '基于 nsys / Kineto / HTA 对多卡训练做时间线拆解，避免跨流重复计数，区分真链路慢与等慢节点。',
    contextSpecs: ['512×H100 训练 MFU 偏低', 'nsys / PyTorch Kineto', 'HTA 时间拆解', 'Critical Path'],
    prompt: '512 张 H100 上跑 Megatron/FSDP2 训练，稳态 MFU 只有 32%。你手上有 PyTorch Profiler (Kineto) 和 Nsight Systems trace。请说明怎样正确拆解单步耗时（计算、暴露通信、空闲/Host 等待）、为什么不能把多个 CUDA Stream 上的核耗时直接相加，以及如何从 trace 上区分“网络链路真慢”与“某个 rank 算慢导致全员在 AllReduce 里干等（Wait-for-Straggler）”？',
    rubric: [
      {
        id: 'step-breakdown',
        label: '按 GPU 时间轴并集 (Union/Critical Path) 拆解纯计算、重叠通信、暴露通信与设备空闲，不把并发流时长直接相加',
        weight: 24,
        keywords: ['并集', 'union', 'critical path', '关键路径', '暴露通信', 'exposed', 'overlap', '重叠', '并发流', 'stream', 'hta'],
        probe: '计算流和 NCCL 通信流并发执行时，为什么把所有 kernel 的 duration 求和会超过单步墙钟时间？应该怎样定义“暴露通信时间”？',
      },
      {
        id: 'straggler-vs-net',
        label: '对齐各 rank 进入同一 Collective 的起始时间戳，区分单 rank 上游变慢（Wait-for-Straggler）与全员同时进入后的链路拥塞/降速',
        weight: 26,
        keywords: ['straggler', '慢节点', '进入时间', '对齐', 'wait', '等待', '上游', 'dataloader', 'gc', '链路', '拥塞', '同步'],
        probe: 'Trace 里看到 511 张卡的 AllReduce kernel 都跑了 40ms，凭什么判断这 40ms 是网络带宽掉速，还是第 512 张卡晚进了 35ms？',
      },
      {
        id: 'host-idle',
        label: '定位 GPU 空闲 (Idle) 归因：CPU launch 队列断流、Host 侧数据加载/Python GC、以及隐式设备同步 (`cudaStreamSynchronize`/`.item()` 触发 D2H)',
        weight: 20,
        keywords: ['idle', '空闲', 'host', 'launch', 'cpu', 'synchronize', '.item()', 'd2h', 'dataloader', 'gc', '断流'],
        probe: 'GPU 时间线上出现周期性空白 Gap 时，怎样在 nsys 里顺着 CUDA API 和 Host 线程找出是谁卡住了发射队列？',
      },
      {
        id: 'profiler-overhead',
        label: '控制采集开销与失真：跳过 warmup 仅采 1~2 个稳态 step，关闭/慎用 `with_stack` 与 `profile_memory` 避免放大 CPU overhead 制造假瓶颈',
        weight: 16,
        keywords: ['warmup', '开销', 'overhead', '失真', 'with_stack', 'record_shapes', '采样窗口', '稳态', 'cupti', 'nvtx'],
        probe: '为什么开着全量 `with_stack=True` 抓 10 个 step 的 Kineto trace，经常抓出一堆实际上线上根本不存在的 CPU Gap？怎么规范采集窗口？',
      },
      {
        id: 'actionable-verify',
        label: '基于关键路径做闭环验证：对比单机同配置基线、按 HTA/trace 量化改动前后的暴露通信和有效步时',
        weight: 14,
        keywords: ['基线', '单机', '闭环', '步时', 'step time', 'nvtx', '验证', '对比', '迭代'],
        probe: '改完 bucket 大小或 overlap 参数后，仅看“某几个通信核变快”够吗？最终应该拿哪几个 trace 指标验收？',
      },
    ],
    hints: [
      'L1 跨 stream 并发时，用区间并集把单步拆为：仅计算、计算与通信重叠、仅通信（暴露通信）、完全空闲四类互斥时段。',
      'L2 抓多卡同一 collective 的开始与结束时刻：先到者在 kernel 里空转等最后到达者，只有对齐最晚进入时刻后的耗时才反映真实传输时间。',
      'L3 采集时遵循 `wait -> warmup -> active(1~2 steps)`，并通过 NVTX 标记前向、反向、优化器与数据加载阶段。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（通信核抢占算力）　在 trace 上看到 AllReduce 和 GEMM 实现了 100% 时间重叠（暴露通信为 0），为什么单步总时间还是比单卡无通信慢了 12%？怎样验证通信核对计算核的干扰？',
      },
      {
        stage: 'corner',
        question: '追问 2（规模化采集）　512 卡同时全量落盘 nsys/Kineto trace 文件极易打爆共享存储且难以人眼比对。在千卡规模下如何低成本抓到“恰好变慢的那 1 个 rank”？',
      },
    ],
    referenceAnswer: `### 按时间轴并集拆解单步，不做跨流简单求和

多 stream 并发时，直接累加所有 kernel 时长会重复计入重叠区间。按单个稳态 step 的 GPU 活跃时间轴取并集（如 Holistic Trace Analysis 的做法），划分为四段互斥时间：
- **纯计算**：仅有计算核在执行；
- **计算-通信重叠**：计算流与通信流同时活跃；
- **暴露通信（Exposed Comm）**：仅有通信核在跑、计算流已停等；
- **设备空闲（Idle）**：没有任何 GPU 核运行，需进一步区分是 **Host 发射不及**、**数据/同步等待**还是 **PP 气泡**。

### 区分“真链路慢”与“等慢节点（Wait-for-Straggler）”

NCCL 集合通信 kernel 在各 rank 各自到达时启动，但必须等最晚的 rank 进入后才能完成传输。因此 **511 张卡上显示 40 ms 的 AllReduce，有 35 ms 可能只是在空等第 512 张卡**。排查步骤：
1. 对齐同一逻辑步、同一 ProcessGroup、同一序号 collective 在各 rank 的**入队/开始时刻**（结合主机时钟同步偏移、NCCL Flight Recorder 或 NVTX step 锚点）；
2. 若存在明显晚到的 rank，回溯该 rank 上游的前一算子、DataLoader、Python GC、CPU 抢占、GPU 降频或检查点 D2H；
3. 若所有 rank 几乎同时进入却整体拉长，再查跨机流量、ECMP 极化、重传、PFC 暂停或错误选型（如小消息走了低效拓扑）。

### 采集失真、隐性干扰与大规模定位

- **控制探针扰动**：使用 \`schedule(wait, warmup, active=1~2)\`；\`with_stack\`、\`record_shapes\` 和高频内存追踪都会抬高 CPU 发射延迟，可能把原本 GPU-bound 的负载测出虚假 CPU Gap。先抓轻量 \`nsys\` + NVTX 确认宏观步时未明显漂移，再开细节。
- **100% 重叠仍变慢的追问**：NCCL kernel 会占用若干 SM（由 channel/CTA 数决定）并争用 HBM/NVLink 带宽，拉长并发 GEMM 的实际时长。可在单机对照“关闭通信 vs 开启并行通信”的**同形状 GEMM 耗时**（或调优 \`NCCL_MAX_NCHANNELS\` / 改用 NVLS/SHARP 减 SM 占用）来量化 SM 争抢代价。
- **大规模定位**：常态只采集轻量逐 rank 阶段计时（step/fwd/bwd/opt/data）和 Flight Recorder 环形缓冲；检测到某 rank 连续偏慢后，仅对该慢节点及其同通信域的 1~2 个对照健康节点按需触发短窗口 trace。

> 依据　Meta Holistic Trace Analysis（HTA）时间分解方法、PyTorch Profiler 文档与 NVIDIA Nsight Systems 指南。`,
  },
  {
    id: 'tr-10',
    track: 'training',
    title: '训练显存 Profiling 与 OOM 排查：Memory Snapshot、Allocated vs Reserved 与碎片定位',
    level: 'P7',
    frequency: '高频',
    incident: true,
    tags: ['Profiling', 'Memory Snapshot', 'OOM', '显存碎片', 'Expandable Segments'],
    summary: '用 PyTorch Memory Snapshot 拆解训练期动态显存峰值，区分真实容量不足、分配器碎片、外部显存占用与张量泄漏。',
    contextSpecs: ['80 GB H100 训练 OOM', 'Memory Snapshot', 'Allocated vs Reserved', 'Caching Allocator'],
    prompt: '在 80 GB H100 上跑 FSDP2/Megatron 长上下文训练，静态模型状态算下来只占 28 GB，但训练跑了几个 step 后突然 OOM，报错显示 `tried to allocate 2.00 GiB (GPU 0; 79.25 GiB total; 61.40 GiB allocated; 1.10 GiB free; 74.80 GiB reserved in total by PyTorch)`。请解释报错里各数字的含义，给出用 PyTorch Memory Snapshot 定位峰值与碎片的方法，并说明怎样排查跨步缓慢涨显存的内存泄漏。',
    rubric: [
      {
        id: 'oom-numbers',
        label: '准确解读 OOM 报错各字段：allocated (活跃张量)、reserved (PyTorch Caching Allocator 持有)、total−reserved (CUDA Context / NCCL / Workspace / 外部进程占用)',
        weight: 24,
        keywords: ['allocated', 'reserved', 'free', 'caching allocator', 'context', 'nccl', 'workspace', 'nvidia-smi', '13.4', '碎片'],
        probe: '报错里 `allocated=61.4 GiB` 但 `reserved=74.8 GiB`，中间差出来的 13.4 GiB 去哪了？另外 `79.25 - 74.8 = 4.45 GiB` 又被谁占着？',
      },
      {
        id: 'memory-snapshot',
        label: '使用 `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` 查看 Active Memory Timeline 与按调用栈归因的峰值构成',
        weight: 24,
        keywords: ['snapshot', '_record_memory_history', 'timeline', '调用栈', 'stack', '峰值', 'peak', 'backward', 'allgather', '激活'],
        probe: '拿到 OOM 时的 Memory Snapshot（pickle 文件）在可视化工具里打开后，你重点看哪两个视图来判断是哪个算子撑爆了峰值？',
      },
      {
        id: 'frag-vs-cap',
        label: '区分「真活跃峰值超容」与「变长/大小块交错引发的虚拟地址碎片」；说明 `expandable_segments:True` (VMM) 治碎片但不减活跃张量总量',
        weight: 22,
        keywords: ['碎片', 'fragmentation', 'expandable_segments', 'vmm', '变长', '动态shape', '活跃', '容量', 'max_split_size'],
        probe: '开启 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 底层靠什么机制缓解碎片？什么情况下开了它依然会 OOM？',
      },
      {
        id: 'transient-spikes',
        label: '识别常见瞬时峰值来源：FSDP 参数全量展开与预取缓冲、大词表 FP32 Logits/CE、Optimizer 首次 step 懒初始化状态、以及跨 stream 异步张量延迟释放 (`record_stream`)',
        weight: 18,
        keywords: ['prefetch', '预取', 'logits', 'optimizer', '懒初始化', 'step 1', 'record_stream', '跨流', '临时'],
        probe: '为什么很多训练任务 Step 0 跑过去了，偏偏在 Step 1 的 Backward 或 `optimizer.step()` 突然 OOM？',
      },
      {
        id: 'leak-debug',
        label: '跨步缓慢涨显存的泄漏排查：保留了带 Autograd Graph 的 Tensor（日志/列表未 `.detach()`/`.item()`）、Python 循环引用、或未释放的 Hook/缓存',
        weight: 12,
        keywords: ['泄漏', 'leak', 'detach', 'item', '计算图', 'autograd', '引用', 'gc', 'hook', 'diff'],
        probe: '如果前 20 个 step 每步 `allocated` 稳定涨 200 MB 直到第 50 步 OOM，你怎么快速抓出是哪一行代码把张量挂着没释放？',
      },
    ],
    hints: [
      'L1 先算清三笔差值：`reserved - allocated = 13.4 GiB` 是分配器内已向驱动申请但未被活跃张量使用的空洞/缓存；`total - reserved - free` 主要是 CUDA Context、cuBLAS/cuDNN workspace 与 NCCL 通信缓冲。',
      'L2 变长序列、动态路由 MoE 或预取缓冲交错申请/释放不同尺寸块时，Caching Allocator 容易出现“总空闲够但找不到连续 2 GiB 块”的碎片。',
      'L3 泄漏排查对比两个稳态 step 结束后的 snapshot diff，直接看新增存活张量的 Python 分配调用栈。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（跨流张量滞留）　在手动写异步通信/计算 overlap 时，对临时张量调用了 `tensor.record_stream(comm_stream)`。如果计算流跑得比通信流快很多，可能会触发怎样的显存暴涨？如何避免？',
      },
      {
        stage: 'corner',
        question: '追问 2（`nvidia-smi` 比 PyTorch 高一大截）　`torch.cuda.max_memory_reserved()` 显示只有 58 GiB，但 `nvidia-smi` 却显示 76 GiB 甚至驱动报 OOM，多出来的 ~18 GiB 可能是谁占用的？怎么排查？',
      },
    ],
    referenceAnswer: `### 先拆解 OOM 报错里的四组数字

以报错 \`total=79.25, allocated=61.40, reserved=74.80, free=1.10 GiB\` 为例：
- **\`allocated = 61.40 GiB\`**：当前仍被 PyTorch Tensor 引用的活跃显存（静态状态 + 在途激活 + 预取/临时张量）。
- **\`reserved − allocated = 13.40 GiB\`**：PyTorch CUDA Caching Allocator 已向驱动申请、目前空闲但未归还的空间。当申请 \`2.00 GiB\` 连续块失败时，通常说明这 \`13.40 GiB\` 被零散小块切成**不连续的碎片空洞**（且驱动侧仅剩 \`1.10 GiB\` 无法再扩新段）。
- **\`total − reserved − free ≈ 3.35 GiB\`**：PyTorch 分配器之外的占用，包括 CUDA Driver Context、cuBLAS/cuDNN workspace、NCCL 通信缓冲池、自定义扩展的 \`cudaMalloc\` 或同卡其他进程。

### 用 Memory Snapshot 定位峰值、碎片与泄漏

1. **抓包**：在可复现步开启 \`torch.cuda.memory._record_memory_history(max_entries=100000)\`，在 OOM 异常钩子或 step 末调用 \`_dump_snapshot()\`，用 PyTorch Memory Visualizer 查看：
   - **Active Memory Timeline**：区分“稳定台阶”（模型/优化器状态）与“尖峰”（某层激活、大词表 logits、FSDP unshard/prefetch 叠加）；点击尖峰处可直接看到分配该张量的完整 Python 调用栈；
   - **Allocator State History**：直观看出 \`reserved\` 与 \`allocated\` 之间的碎片空洞分布。
2. **区分碎片 vs 真超容**：
   - 若 \`reserved − allocated\` 很大且输入长度/MoE token 数剧烈变化，开启 \`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\`（底层基于 CUDA VMM 将分散物理页映射为连续虚拟段）常可消除大部分碎片；但注意它在**活跃张量峰值本身超过可用容量**时无效，且与部分依赖静态地址的旧分配路径需兼容验证。
   - **Step 1 才 OOM 的常见元凶**：AdamW 优化器的一二阶矩往往在**第一次 \`optimizer.step()\` 才懒分配**（额外暴涨 8Ψ 字节），或首次 backward 才同时出现完整梯度 + 激活峰值。
3. **两项追问**：
   - **\`record_stream\` 滞留**：Caching Allocator 在关联 stream 完成前不会复用该块，且异步事件回收存在滞后；若生产流远快于消费流，多步临时张量堆积会撑爆显存。应改用显式 \`Event.wait_stream\` 同步或有界双缓冲复用固定内存，限制在途数量。
   - **\`nvidia-smi\` 远高于 \`reserved\`**：常见于多进程/同卡残留进程、超大 NCCL buffer（过多 process group / channel）、cuBLAS workspace、或第三方推理/C++ 算子绕过 PyTorch Allocator 直接调用 \`cudaMalloc\`。可用 \`nvidia-smi --query-compute-apps\`、逐一排查 ProcessGroup 数量与 Nsight Systems 的 CUDA Memory Allocation 事件归因。
   - **跨步泄漏**：比较 Step $k$ 与 Step $k+1$ 结束后存活张量的调用栈 diff，高频祸首是把带 \`grad_fn\` 的 loss/metrics Tensor 直接 \`append\` 进 Python list 而未调用 \`.detach()\`/\`.item()\`。

> 依据　PyTorch《Understanding CUDA Memory Usage》及 CUDA Caching Allocator 文档。`,
  },
];
