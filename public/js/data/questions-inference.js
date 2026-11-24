export const INFERENCE_QUESTIONS = [
  {
    id: 'inf-01',
    track: 'inference',
    title: 'vLLM PagedAttention、Continuous Batching 与 Chunked Prefill 核心机制',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['vLLM', 'PagedAttention', 'Continuous Batching', 'Chunked Prefill'],
    summary: '剖析传统静态 Batching 的显存碎片和空泡问题，深入 PagedAttention 页表映射、Iteration-level 调度及 Chunked Prefill 消抖原理。',
    contextSpecs: ['vLLM v1', 'Block Size 16', 'TTFT vs TPOT', 'Prefix Caching'],
    prompt: '请系统讲清楚现代 LLM 推理引擎（如 vLLM）中三大核心基石——PagedAttention、Continuous Batching 和 Chunked Prefill 各自解决了什么痛点？特别是：开启 Continuous Batching 后，为什么还需要 Chunked Prefill，它是怎么平衡 Prefill 吞吐与 Decode TPOT 抖动的？',
    rubric: [
      {
        id: 'paged-attn',
        label: 'PagedAttention：将 KV Cache 切分为固定物理 Block，逻辑-物理 Block Table 映射消除外部碎片与预留浪费',
        weight: 24,
        keywords: ['pagedattention', 'block', '块', '页表', 'block table', '碎片', '虚拟内存', '非连续', 'copy-on-write', 'cow'],
        probe: '传统推理按 max_seq_len 预分配连续显存，内存浪费率往往超过 60%。PagedAttention 借鉴操作系统虚拟内存做了哪些映射？支持哪些共享优化？',
      },
      {
        id: 'cont-batch',
        label: 'Continuous Batching (Iteration-level Scheduling)：每步 Decode 后立即释放已完成请求并插入新请求',
        weight: 22,
        keywords: ['continuous', 'iteration', '迭代级', '动态', 'orca', '插入', '完成', 'padding', '长尾'],
        probe: 'Static Batching 为什么吞吐极低？Iteration-level 调度下不同长度的序列怎么做 Attention 和 FFN 批处理？',
      },
      {
        id: 'chunked-prefill',
        label: 'Chunked Prefill：将长 Prompt 拆成固定 Token Budget 块，与 Decode 请求混批 (Piggyback) 消除 TPOT 毛刺',
        weight: 24,
        keywords: ['chunked prefill', '分块', 'tpot', '毛刺', '抖动', '混批', 'piggyback', 'token budget', 'sarathi', 'stall'],
        probe: '如果突然来了一个 32K Prompt 的新请求，没有 Chunked Prefill 时正在 Decode 的几十个在线请求会发生什么？Chunked Prefill 为什么不仅消抖，还能白嫖 Decode 的权重搬运？',
      },
      {
        id: 'overhead-tradeoff',
        label: '开销权衡：Chunked Prefill 会增加多次读取同一前缀 KV Cache 的重复访存；Block 过小增加元数据和访存离散度',
        weight: 16,
        keywords: ['重复读取', 'kv', '开销', 'block size', '权衡', 'trade-off', 'ttft', '访存'],
        probe: '把一个 32K Prompt 切成 16 个 2K 的 Chunk，Attention 阶段会引入什么额外的 HBM 访存开销？',
      },
      {
        id: 'preemption',
        label: '显存满载抢占机制：KV Cache 耗尽时触发 Preemption（Recompute vs Swap-out 到 CPU）',
        weight: 14,
        keywords: ['preempt', '抢占', 'recompute', 'swap', '驱逐', 'oom', '重算'],
        probe: '当并发太高导致所有物理 KV Block 被占满、已有请求又要生成新 Token 时，vLLM 会怎么处理？',
      },
    ],
    hints: [
      'L1 方向提示：PagedAttention 管“显存利用率”，Continuous Batching 管“请求进出不空等”，Chunked Prefill 管“长 Prefill 不阻塞 Decode 且拉高计算密度”。',
      'L2 Piggyback 妙处：Decode 是 Memory-Bound（读一次模型权重只算几个 Token），把一个 Chunked Prefill 塞进同一个 Batch 后，Linear 层读一次权重同时服务 Prefill 和 Decode，算力利用率大涨且 Decode 延迟几乎不增！',
      'L3 重复读代价：第 $k$ 个 Chunk 计算 Self-Attention 时必须重读前 $k-1$ 个 Chunk 的 KV Cache，因此 Chunk Size 不能切得太小（通常 512~4096 tokens）。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（调度细节）：vLLM V1 架构重构中取消了原先的 Swap 机制而主要采用 Recompute 抢占，同时做了零开销 Async Scheduling 和 Persistent Batch，为什么工业界普遍认为 Recompute 比 Swap-out 到 CPU 更好？',
      },
      {
        stage: 'corner',
        question: '追问 2（极端场景）：在 128K 超长上下文场景下，Chunked Prefill 已经很难同时兼顾 TTFT 和 TPOT 了，为什么这时候要走向 Prefill-Decode (PD) 分离架构？',
      },
    ],
    referenceAnswer: `### Staff 级参考解析

#### 1. PagedAttention：消除 KV Cache 显存碎片
- 传统引擎按最大长度预分配连续显存，存在**内部碎片（预留未用）**和**外部碎片（内存空洞）**，显存浪费常达 60%~80%。
- **PagedAttention** 仿照 OS 虚拟分页：将 KV Cache 划为固定大小的物理块（如 \`Block Size = 16\` tokens），每个请求维护逻辑块到物理块的 \`Block Table\`，按需动态分配，显存浪费降至 \`< 4%\`（仅尾块半满）；同时天然支持 **Beam Search / Parallel Sampling 的 Copy-on-Write** 及 **Automatic Prefix Caching (APC)**。

#### 2. Continuous Batching (Iteration-Level Scheduling)
- 不必等整个 Batch 里最长的请求吐出 EOS，在**每一次 Forward 迭代结束**时立刻踢出已完结请求、塞入等待队列中的新请求。
- FFN/Linear 层将所有 Token 展平拼接为 2D 张量 \`[total_tokens, H]\` 彻底消除 Padding；Attention 层使用 FlashAttention Varlen / PagedAttention Kernel 处理变长序列。

#### 3. Chunked Prefill（Sarathi-Serve / vLLM 标配）
- **痛点**：一个 16K Prompt 的 Prefill 耗时可能达数百 ms~数秒，直接卡死同机所有正在进行的 Decode 请求（Generation Stall），导致 TPOT P99 飙升。
- **机制**：为每次 Iteration 设定 \`max_num_batched_tokens\`（如 2048），长 Prompt 被切块与当前所有 Decode Token 拼入同一 Batch 执行。
- **双重收益**：(1) 消除 TPOT 尖峰；(2) **Piggybacking**：Decode 原本读取全部权重的 HBM 带宽开销顺带摊薄到 Prefill Chunk 的大量 Token 上，Linear 层直接从 Memory-bound 拉向 Compute-bound。
- **代价**：后序 Chunk 需重复从 HBM 读取前面 Chunk 的 KV Cache（$O(C^2)$ 次块读取），故 Chunk 不宜过小。

#### 4. 追问要点
- **Recompute vs Swap**：PCIe 带宽（~32-64 GB/s）远低于 HBM，且长文本重算 Prefill 在 Compute-bound GPU 上往往比 PCIe 来回搬运更快，同时架构大幅简化（vLLM V1 默认优先 Recompute）。`,
  },
  {
    id: 'inf-02',
    track: 'inference',
    title: 'Prefill-Decode (PD) 分离架构设计与 KV Cache 传输优化',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['PD分离', 'DistServe', 'Mooncake', 'RDMA', 'KV Cache传输'],
    summary: '解析 PD 分离为何是大规模推理标配，深入探讨 KV Cache 跨节点传输瓶颈、Layer-wise 流水传输及 PD 弹性配比调度。',
    contextSpecs: ['Prefill Pool + Decode Pool', 'RDMA / NIXL / Mooncake', 'TTFT / TPOT SLA', 'P:D 配比'],
    prompt: '在超大规模在线推理服务（如高并发长输入 Chat / Coding 助手）中，DistServe、Mooncake、SGLang/vLLM Disaggregated Serving 都采用了 Prefill-Decode (PD) 分离架构。请说明为什么 Chunked Prefill 仍不能完全替代 PD 分离？PD 分离引入的最大新瓶颈是什么？工程上如何把 KV 传输延迟隐藏掉，并设计异构并行度与动态 P:D 配比？',
    rubric: [
      {
        id: 'pd-motivation',
        label: '本质动机：Prefill (Compute-bound, 要高算力/低TP) 与 Decode (Memory-bound, 要高带宽/大Batch/大EP) 硬件需求与并行度解耦',
        weight: 24,
        keywords: ['compute-bound', 'memory-bound', '解耦', '干涉', '干扰', '并行度', 'tp', 'dp', 'ep', 'sla', 'ttft', 'tpot'],
        probe: '除了消除 Prefill 对 Decode 的延迟干扰，PD 分离在“并行策略（如 TP/DP/EP）和硬件选型”上还带来了什么混部做不到的红利？',
      },
      {
        id: 'kv-transfer',
        label: '核心新瓶颈：P 到 D 跨机传输海量 KV Cache（如 70B 32K 序列可达数 GB），依赖 RDMA GPUDirect / NIXL / Mooncake',
        weight: 22,
        keywords: ['kv', '传输', 'rdma', 'gpudirect', 'nixl', 'mooncake', '带宽', '网络', '零拷贝'],
        probe: '算一下一个 70B GQA 模型 32K Prompt 的 KV Cache 有多大？如果走 400Gbps IB RDMA 要传多久？',
      },
      {
        id: 'layerwise-overlap',
        label: '传输隐藏：Layer-wise 逐层异步传输（算第 i+1 层时传第 i 层 KV）+ 预分配 Decode KV 显存防死锁',
        weight: 22,
        keywords: ['layer-wise', '逐层', 'overlap', '异步', '流水', '掩盖', '预分配', '死锁', 'handshake'],
        probe: '怎么避免 Prefill 全部算完才开始传 KV 导致 TTFT 翻倍？以及传之前如果 Decode 节点没显存了怎么办？',
      },
      {
        id: 'kv-cache-centric',
        label: '以 KV Cache 为中心的全局缓存池（如 Mooncake KVCache Pool / CPU+SSD 分层缓存）跳过重复 Prefill',
        weight: 16,
        keywords: ['cache pool', 'prefix', '缓存池', '分层', 'cpu', 'ssd', '命中', '路由', 'kvc'],
        probe: '多轮对话或 RAG 场景下，每次都重新做 Prefill 太贵了，Mooncake 这种 KVCache-centric 架构是怎么结合分布式缓存池进一步省 P 节点算力的？',
      },
      {
        id: 'pd-ratio',
        label: 'P:D 节点配比随业务输入/输出长度比动态变化（如长输入短输出重 P，长 CoT 推理重 D）及动态角色切换',
        weight: 16,
        keywords: ['配比', '比例', '长输入', '长输出', 'cot', '弹性', '潮汐', '动态', '切换'],
        probe: '假设白天是长文档摘要（输入 16K 输出 200），晚上是 R1 深度推理（输入 500 输出 8K），固定的 P:D 配比会出什么问题？怎么动态调优？',
      },
    ],
    hints: [
      'L1 方向提示：PD 分离不只是“不抢占”，更是让 P 集群用小 TP/高算力卡吃满 TTFT，让 D 集群用大 Batch/高 HBM 带宽/Wide-EP 吃满吞吐与 TPOT！',
      'L2 定量算例：70B GQA (L=80, n_kv=8, d=128, BF16=2B) 每 Token KV = `2×80×8×128×2 = 320 KiB`；32K 序列正好 `10 GiB`！400Gbps IB (50GB/s) 单向也要传 `~215ms`；若换成 DeepSeek MLA，每 Token 仅 ~70 KiB，传输直接降 ~4.5×！',
      'L3 逐层传输：P 节点算完 Layer $l$ 立刻通过异步 RDMA Write 推送给 D 节点预留的物理 Block，实现计算与跨机传输近乎完全重叠。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（并行度不匹配）：如果 Prefill 实例是 TP=4，而 Decode 实例是 TP=8（或者 DP Attention + EP），同一个 KV Cache 张量在 P 和 D 上的切分方式不一样，RDMA 传输怎么处理 Layout 转换与零拷贝？',
      },
      {
        stage: 'corner',
        question: '追问 2（异构硬件）：很多团队想用 H100/H800 做 Prefill、用 H20（算力低但 HBM 大、带宽高）做 Decode，这种异构 PD 方案在什么业务模型下 ROI 最高？有哪些落地坑？',
      },
    ],
    referenceAnswer: `### Staff 级参考架构

#### 1. 为什么 Chunked Prefill 无法完全替代 PD 分离？
- **强干扰与 SLO 耦合**：超长上下文（32K~128K）下，Chunked Prefill 要么切大块破坏 Decode TPOT，要么切小块导致重复读 KV 严重拖慢 TTFT。
- **最优并行度与硬件画像完全不同**：
  - **Prefill (Compute-bound)**：适合小 TP（减少通信）、高 FLOPS 卡（H100/B200），按 TTFT 扩展。
  - **Decode (Memory-bound)**：适合大 Batch、大 KV 容量、高 HBM 带宽卡（H200/H20）以及 MoE 下的 **DP-Attn + Wide-EP**。混部被迫共用同一种并行度，两头妥协。

#### 2. KV 传输瓶颈与 Layer-wise 隐藏
- 70B GQA (80 层, 8 KV head, 128 维, BF16) 每 Token KV ≈ \`320 KB\`，32K 序列达 \`10 GB\`。
- **优化手段**：
  1. **GPUDirect RDMA (NIXL / Mooncake Transfer Engine)** 绕开 Host CPU 与内核拷贝；
  2. **Layer-wise 逐层流水**：P 算完第 $l$ 层立即异步 RDMA Write 第 $l$ 层 KV 到 D 预留的 Block，算第 $l+1$ 层同时隐藏第 $l$ 层传输，总 TTFT 额外开销压缩到最后一层通信（几毫秒）；
  3. **显存预约与背压 (Admission Control)**：P 开始算前先向 D 申请 KV 槽位（或采用 Pull 模式由 D 主动拉取），避免算完后 D OOM 造成算力白费与死锁；
  4. **架构减负**：FP8 KV Cache 减半传输量；DeepSeek MLA 低秩压缩把 KV 缩到传统 GQA 的 1/5~1/10，天然利好 PD 分离。

#### 3. 全局 KV 缓存池与动态 P:D 配比
- **KVCache-Centric 调度（Mooncake）**：构建跨机 DRAM/SSD KV Cache 池，结合 Prefix-aware Router 将请求路由到缓存命中节点，直接跳过重复 Prefill。
- **动态配比**：短输入长 CoT 业务 P:D 常为 \`1:3 ~ 1:5\`；长上下文 RAG 业务可达 \`2:1 ~ 4:1\`。采用共享模型权重的弹性实例支持低峰/潮汐时段动态切换 P/D 角色。`,
  },
  {
    id: 'inf-03',
    track: 'inference',
    title: 'KV Cache 压缩演进：MHA -> GQA -> MLA 矩阵吸收与 RadixAttention',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['KV Cache', 'GQA', 'MLA', '矩阵吸收', 'RadixAttention', 'SGLang'],
    summary: '定量推导 KV Cache 公式，深刻剖析 DeepSeek MLA 如何在推理 Decode 阶段通过矩阵吸收省掉巨大 KV 显存，并对比 Prefix Caching 树结构。',
    contextSpecs: ['MHA / GQA / MLA', 'Weight Absorption', 'SGLang Radix Tree', 'KV 显存公式'],
    prompt: 'KV Cache 是长上下文和高并发推理最大的显存和 HBM 带宽杀手。请先写出标准 GQA 的单 Token KV Cache 显存公式；再深入解释 DeepSeek-V2/V3 提出的 MLA (Multi-Head Latent Attention) 在推理时为什么只需要缓存极小的压缩潜向量 c_KV？“矩阵吸收（Weight Absorption）”具体是怎么推导的，以及它和 RoPE 冲突是怎么解决的？最后对比一下 SGLang RadixAttention 如何复用前缀 KV。',
    rubric: [
      {
        id: 'kv-formula',
        label: '准确写出单 Token KV 显存公式：2 × L × n_kv × d_h × bytes（并能定量举例）',
        weight: 20,
        keywords: ['2', 'l', 'n_kv', 'd_h', 'bytes', '公式', '320kb', '显存', 'b*s'],
        probe: '请先给出 GQA 场景下单 Token KV Cache 的精确字节公式？70B 模型 128K 长度单请求光 KV 就要占多少 GB？',
      },
      {
        id: 'mla-latent',
        label: 'MLA 原理：将 KV 联合低秩投影到潜向量 c_KV (如 512 维)，仅缓存 c_KV + 解耦 RoPE 的 k_R (64 维)',
        weight: 22,
        keywords: ['mla', 'latent', '低秩', 'c_kv', '512', '压缩', '潜向量', '缓存'],
        probe: 'MLA 相比 GQA 为什么既保住了近似 MHA 的表达能力，又把 KV Cache 压到了极低？实际缓存进显存的是哪两个张量？',
      },
      {
        id: 'weight-absorb',
        label: '矩阵吸收 (Weight Absorption)：Decode 时把 W_UK 吸收进 W_Q (算 q·W_UK^T)，把 W_UV 吸收进 W_O，全程不解压展开完整 K/V',
        weight: 24,
        keywords: ['吸收', 'absorb', 'absorption', 'w_uk', 'w_uv', 'w_q', 'w_o', '结合律', '不解压', '矩阵乘'],
        probe: '如果 Decode 每步都要把 c_KV 乘上 W_UK 还原成巨大的 K 和 V，那省下的只是容量、并没有省算力或带宽。MLA 推理时是怎么用“矩阵吸收”绕开还原 KV 的？',
      },
      {
        id: 'decoupled-rope',
        label: 'Decoupled RoPE：RoPE 位置旋转矩阵位于 Q 和 K 中间破坏了矩阵吸收结合律，故拆出独立低维 k_R 施加 RoPE',
        weight: 18,
        keywords: ['rope', '解耦', 'decoupled', '位置编码', '结合律', '不可交换', 'k_r', '64'],
        probe: '为什么不能直接对解压后的 K 施加标准 RoPE？DeepSeek 为什么要专门设计 Decoupled RoPE？',
      },
      {
        id: 'radix-tree',
        label: 'SGLang RadixAttention：基于基数树 (Radix Tree) 管理共享前缀 KV 物理页 + LRU 淘汰与缓存感知调度',
        weight: 16,
        keywords: ['radix', '基数树', 'sglang', '前缀', 'prefix', 'lru', '复用', '多轮', 'agent', 'system prompt'],
        probe: '在多轮对话、Few-shot 或 Agentic 树状搜索场景下，SGLang 的 RadixAttention 相比普通 Hash-based Prefix Caching 有什么调度优势？',
      },
    ],
    hints: [
      'L1 方向提示：MLA 靠“低秩投影压缩状态”+“矩阵乘法结合律吸收上投影矩阵”+“解耦 RoPE 避开位置矩阵阻隔”三板斧。',
      'L2 吸收推导：原本 $q_t^T (W^{UK} c_j^{KV}) = (q_t^T W^{UK}) c_j^{KV}$！先把 $W^{UK}$ 乘到当前步仅 1 个 Token 的 $q_t$ 上，就永远不用把历史几万个 Token 的 $c_j^{KV}$ 解压回高维！',
      'L3 Prefill vs Decode：Prefill 阶段 $q$ 的序列也很长，吸收反而让 FLOPs 变大，因此实际引擎往往 **Prefill 不吸收（直接展开用标准 FA），Decode 阶段启用矩阵吸收（MQA-like 形态）**！',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Prefill vs Decode 差异）：矩阵吸收在 Decode 阶段简直是神技，但如果在 Prefill 阶段也盲目做矩阵吸收，计算量会发生什么变化？为什么 DeepSeek / SGLang 在 Prefill 和 Decode 会走两条不同的 Attention Kernel 路径？',
      },
      {
        stage: 'corner',
        question: '追问 2（Cache-aware Scheduling）：SGLang 在等待队列里有几十个不同 Prefix 的请求，如果简单按 FIFO 调度，可能会不断把刚算好的 Prefix Cache 给 LRU 淘汰掉（缓存抖动），应该怎么改调度顺序？',
      },
    ],
    referenceAnswer: `### Staff 级参考解析

#### 1. KV Cache 显存公式与定量对比
- **GQA 单 Token KV 显存**：
  $$\\text{Bytes/Token} = 2 \\times L \\times n_{kv} \\times d_h \\times \\text{bytes}$$
  - Llama-3 70B ($L=80, n_{kv}=8, d_h=128$, BF16)：\`320 KiB / token\` → 128K 单请求占 **40 GiB**！
- **DeepSeek-V3 MLA** ($L=61, d_c=512, d_r=64$, BF16)：每 Token 只需缓存 $c_t^{KV}$ (512 维) + 解耦 RoPE $k_t^R$ (64 维) = \`576 × 2B × 61 ≈ 68.6 KiB / token\`（仅 70B 的 ~1/4.7，且模型总参大 10×！）。

#### 2. 矩阵吸收 (Weight Absorption) 与 Decoupled RoPE
- 设潜向量 $c_j^{KV} \\in \\mathbb{R}^{d_c}$，上投影 $K_j = W^{UK} c_j^{KV}$，$V_j = W^{UV} c_j^{KV}$。
- **吸收推导**：
  $$q_t^T K_j = q_t^T (W^{UK} c_j^{KV}) = \\underbrace{(q_t^T W^{UK})}_{q_t' \\in \\mathbb{R}^{d_c}} c_j^{KV}$$
  同样输出侧把 $W^{UV}$ 吸收到输出投影 $W^O$ 中。于是整个 Decode Attention 变成所有 Query Head 共享同一个低维 $c^{KV}$ 的 **MQA 形态**，HBM 访存量暴降，且完美契合 Tensor Core (FlashMLA)！
- **为什么需要 Decoupled RoPE**：如果加了位置矩阵 $R_{t-j}$，表达式变为 $q_t^T R_{t-j} W^{UK} c_j^{KV}$，位置旋转矩阵卡在 $q_t$ 与 $W^{UK}$ 之间**破坏了结合律**无法吸收！因此 MLA 将 Query/Key 拆成 NoPE（做低秩吸收）和独立小维度 RoPE（仅 64 维共享 $k_t^R$）两路拼接。
- **Prefill vs Decode 分流**：Prefill 序列很长，若直接吸收会导致多 Head 在 $d_c=512$ 维上重复计算、FLOPs 暴涨；故 **Prefill 解压后走标准 FlashAttention，Decode 走矩阵吸收 FlashMLA**。

#### 3. SGLang RadixAttention
- 用 **Radix Tree (基数树)** 索引已计算的 KV 物理页，边为变长 Token 序列，支持任意共享前缀的零拷贝复用（System Prompt、多轮历史、Tree-of-Thought）。
- 配合 **Longest-Prefix-Match (LPM) 调度**：优先调度与当前缓存树匹配最长前缀的请求，最大化缓存命中率并配合叶节点 LRU 淘汰防缓存抖动。`,
  },
  {
    id: 'inf-04',
    track: 'inference',
    title: 'Roofline 模型定量分析 LLM 推理：Decode 为什么是 Memory-Bound？',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['Roofline', 'Arithmetic Intensity', 'Ridge Point', 'H100', 'Batch Size'],
    summary: '手算 H100 FP16/FP8 Ridge Point、Decode 与 Prefill 的计算强度 (OI)，并精确推导出 Batch Size 临界点。',
    contextSpecs: ['H100: 989 TF / 3.35 TB/s', 'Ridge Point ~295', 'OI = FLOPs / Bytes', 'TTFT / TPOT'],
    prompt: '请用 Roofline 模型定量推导：在 H100 (BF16 算力 989 TFLOPS，HBM3 带宽 3.35 TB/s) 上部署一个 70B Dense 模型 (BF16)，为什么 Batch=1 的 Decode 是严重的 Memory-Bound？随着 Batch Size 增大，大约到多大 Batch Size 时 FFN 线性层会跨过 Ridge Point 进入 Compute-Bound？请给出推导步骤。',
    rubric: [
      {
        id: 'ridge-point',
        label: '准确计算 H100 Ridge Point (拐点) = 989 TFLOPS / 3.35 TB/s ≈ 295 FLOPs/Byte',
        weight: 24,
        keywords: ['ridge', '拐点', '295', '300', '989', '3.35', 'ops/byte', 'flops/byte', '算术强度', 'arithmetic intensity'],
        probe: 'H100 在 BF16 下的硬件拐点（Ridge Point，即算力带宽比）是多少 FLOPs/Byte？',
      },
      {
        id: 'decode-oi',
        label: 'Decode 线性层 OI 推导：权重 2P 字节，Batch=B 时计算 2PB FLOPs => OI = B (FLOPs/Byte)（忽略少量激活搬运）',
        weight: 26,
        keywords: ['2p', '2b', 'oi', 'batch', 'b', '1 flop', '权重', '搬运', '2*p', 'b/2'],
        probe: '对一个矩阵乘 Y = XW（W 占 2·M·K 字节，B 个 token 共做 2·B·M·K 次 FLOPs），它的算术强度 OI 和 Batch Size B 的精确关系是什么？',
      },
      {
        id: 'batch-threshold',
        label: '临界 Batch 推导：BF16 下 OI = B => 需 B ≈ 295 跨过 Ridge；FP8 下 OI 翻倍为 2B 且 Ridge 翻倍为 591 => 临界 Batch 依然约为 295！（W4A16 提前到 ~74）',
        weight: 22,
        keywords: ['295', '300', '临界', '饱和', 'compute-bound', '计算密集', 'batch', 'fp8', '591', '590', '74'],
        probe: '照你刚才的公式，BF16 下 Batch 大约到多少才会从 Memory-Bound 变成 Compute-Bound？如果换成 FP8 呢，临界 Batch 会变大、变小还是基本不变？',
      },
      {
        id: 'kv-bound',
        label: '点出 Attention 阶段不同：每个请求读独立 KV Cache，增大 Batch 无法复用 KV，Attention 始终是 Memory-Bound',
        weight: 18,
        keywords: ['attention', 'kv cache', '无法复用', '独立', 'memory-bound', 'gqa', 'batch', '共享'],
        probe: '增大 Batch Size 可以让 Linear/FFN 层变 Compute-Bound，那 Self-Attention 层也会随着 Batch 增大变成 Compute-Bound 吗？为什么？',
      },
      {
        id: 'sla-tradeoff',
        label: '工程约束：受限于单卡 KV 显存容量与 TPOT SLA，线上往往无法开到 300 Batch（或需量化/大EP/PD分离）',
        weight: 10,
        keywords: ['tpot', 'sla', '延迟', '显存', '吞吐', 'o(1)', '上限'],
        probe: '在到达临界 Batch 之前，增大 Batch 时每 Token 延迟 (TPOT) 变化不大而系统总吞吐线性增长，一旦超过临界点会发生什么？',
      },
    ],
    hints: [
      'L1 方向提示：Roofline 核心就两个数——硬件拐点 $\\text{Ridge} = \\frac{\\text{Peak FLOPS}}{\\text{Mem BW}}$，以及算子计算强度 $\\text{OI} = \\frac{\\text{FLOPs}}{\\text{Bytes Moved}}$。',
      'L2 黄金心算公式：BF16 Linear 层，参数量 $P$，权重字节 $2P$，处理 $B$ 个 Token 计算量 $2PB$，所以 $\\text{OI} = \\frac{2PB}{2P} = B$ FLOPs/Byte（若 FP8 权重为 $1P$ 字节，则 $\\text{OI} = 2B$，但 FP8 算力也翻倍，所以临界 Batch 同样在 ~295！）。',
      'L3 Attention 陷阱：Linear 权重是所有请求共享的，但 KV Cache 是每个请求私有的，因此增大 Batch 并不能提高 Decode Attention 的算术强度（除非共用 Prefix 或用 GQA/MLA 靠多 Query Head 复用同一份 KV！）。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（量化直觉检验）：刚才说 BF16 临界 Batch 是 ~295。如果换成 W4A16 (4-bit 权重 + FP16 算力) vs W8A8 FP8 (8-bit 权重 + FP8 算力)，各自的 Ridge Point 和临界 Batch Size 分别是多少？各自适合什么流量场景？',
      },
      {
        stage: 'corner',
        question: '追问 2（GQA 的 Roofline 视角）：我们常说 GQA 省显存，其实从 Roofline 看它对 Decode Attention 算术强度也有本质提升。假设 GQA Group Size = 8（8 个 Q 共享 1 个 KV），Attention 的 OI 提升了多少倍？',
      },
    ],
    referenceAnswer: `### Staff 级定量 Roofline 推导

#### 1. 硬件 Ridge Point 计算（H100 SXM）
- **BF16 拐点**：$\\text{Ridge}_{\\text{BF16}} = \\frac{989\\text{ TFLOPS}}{3.35\\text{ TB/s}} \\approx \\mathbf{295\\text{ FLOPs/Byte}}$
- **FP8 拐点**：$\\text{Ridge}_{\\text{FP8}} = \\frac{1979\\text{ TFLOPS}}{3.35\\text{ TB/s}} \\approx \\mathbf{591\\text{ FLOPs/Byte}}$

#### 2. Decode Linear 层 Arithmetic Intensity (OI)
- 设线性层权重参数量 $P$，精度为 $b$ 字节（BF16: $b=2$, FP8: $b=1$, INT4: $b=0.5$）。
- 处理 Batch = $B$ 个 Token（Decode 每请求 1 token）：
  - **计算量**：$2 \\cdot B \\cdot P$ FLOPs
  - **访存量**（权重占主导，忽略微小激活）：$b \\cdot P$ Bytes
  - **计算强度**：$\\text{OI} = \\frac{2BP}{bP} = \\mathbf{\\frac{2B}{b}}$ FLOPs/Byte
- **结论**：
  - **BF16 ($b=2$)**：$\\text{OI} = B$。Batch=1 时 $\\text{OI}=1 \\ll 295$，算力利用率仅 **0.34%**，彻底 Memory-Bound！直到 **$B \\approx 295$** 才跨过拐点进入 Compute-Bound。
  - **W4A16 ($b=0.5$, BF16 算力)**：$\\text{OI} = 4B$，临界点提前到 **$B \\approx 74$**，中小 Batch 吞吐/延迟暴涨 4×！
  - **FP8 ($b=1$, FP8 算力翻倍)**：$\\text{OI} = 2B$，但 Ridge 也翻倍到 591，临界点依然在 **$B \\approx 295$**，大 Batch 极限吞吐翻倍！

#### 3. Self-Attention 的不同规律
- Linear 权重跨 Batch 共享，而 **KV Cache 跨请求独立**，因此加大 Batch 不会提升普通 MHA Decode Attention 的 OI（永远是 ~1 FLOP/Byte 级别）。
- **唯一能拉高 Decode Attention OI 的方法**：**GQA/MQA/MLA**（多个 Query Head 共享读取同一份 KV，将 OI 放大 $G$ 倍，如 GQA-8 放大 8×，MLA 吸收后更是直接放大 128×！）以及 **Shared Prefix Caching**（多个请求共用同一份 System Prompt KV）。`,
  },
  {
    id: 'inf-05',
    track: 'inference',
    title: '推理量化深度对比：W4A16 (AWQ/GPTQ) vs W8A8 (FP8/SmoothQuant) vs KV 量化',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['量化', 'W4A16', 'FP8', 'AWQ', 'SmoothQuant', 'Marlin'],
    summary: '从硬件 Roofline、离群值处理和融合反量化 Kernel 三个视角，系统讲透不同推理量化路线的选型矩阵。',
    contextSpecs: ['W4A16 (AWQ/GPTQ/Marlin)', 'W8A8 (FP8/INT8)', 'KV FP8', 'Activation Outliers'],
    prompt: '在线上 LLM 推理降本增效时，算法和业务经常问：到底该用 W4A16 (AWQ/GPTQ)、W8A8-INT8 (SmoothQuant)、W8A8-FP8 还是 FP8 KV Cache？请从硬件算子原理、离群值 (Activation Outliers) 处理机制、Batch Size 收益拐点（Roofline 视角）以及精度-吞吐权衡四个维度给出选型指南。',
    rubric: [
      {
        id: 'w4a16-vs-w8a8',
        label: 'Roofline 选型核心：W4A16 减 4× 权重搬运适合中小 Batch Memory-bound；W8A8/FP8 激活硬件 2× 算力适合大 Batch / Prefill',
        weight: 26,
        keywords: ['w4a16', 'w8a8', 'memory-bound', 'compute-bound', 'batch', 'prefill', 'decode', '搬运', '2倍算力', 'tensor core'],
        probe: '为什么在 Prefill 阶段或大 Batch 离线批推场景，W4A16 有时反而比 FP8 甚至 BF16 还慢？',
      },
      {
        id: 'outliers-method',
        label: '离群值机制：SmoothQuant 通过等效数学变换把激活离群值平滑迁移到权重；AWQ 根据激活幅度保护 1% 显著权重通道',
        weight: 22,
        keywords: ['outlier', '离群值', 'smoothquant', 'awq', '显著', 'salient', '通道', '平滑', 'scale', '迁移', 'gptq', 'hessian'],
        probe: '同样是处理激活值里的离群通道，SmoothQuant 和 AWQ 的核心思想分别是什么？',
      },
      {
        id: 'dequant-kernel',
        label: 'W4A16 融合反量化 Kernel (如 Marlin)：在寄存器内 On-the-fly 反量化 INT4->FP16 再喂 Tensor Core，避免写回 HBM',
        weight: 20,
        keywords: ['marlin', '反量化', 'dequant', '融合', 'register', '寄存器', 'cutlass', 'kernel', '位运算', 'lop3'],
        probe: 'W4A16 不能直接用 INT4 Tensor Core 乘 FP16，那实际 Kernel (比如 Marlin) 是怎么做到接近理论 4× 带宽提速且不拖慢算力的？',
      },
      {
        id: 'fp8-advantage',
        label: 'H100/B200 时代 W8A8-FP8 成为甜点：指数位天然容纳大动态范围、无需复杂校准集、同时加速 Prefill+Decode',
        weight: 18,
        keywords: ['fp8', '动态范围', '校准', '无损', 'e4m3', 'h100', 'b200', 'blockwise'],
        probe: '在 Hopper/Blackwell 卡上，为什么 W8A8-FP8 几乎全面取代了 W8A8-INT8 (SmoothQuant)？',
      },
      {
        id: 'kv-quant',
        label: 'KV Cache 量化 (FP8 / INT4)：突破并发和长上下文显存天花板，注意 Key (Per-Channel 离群) 与 Value (Per-Token) 分布差异',
        weight: 14,
        keywords: ['kv', 'per-channel', 'per-token', '并发', 'fp8 kv', 'kivi', '显存', '精度'],
        probe: '做 KV Cache 量化时，Key 和 Value 的数值分布特点不一样，工程上通常分别按什么维度做量化？',
      },
    ],
    hints: [
      'L1 方向提示：记住核心法则——W4A16 是“减访存不加算力”，W8A8 是“减半访存+算力翻倍”。',
      'L2 离群值对比：SmoothQuant 是 $Y = (X \\cdot \\text{diag}(s)^{-1}) (\\text{diag}(s) \\cdot W)$ 把激活难点转移给权重；AWQ 发现只需保护 1% 对应大激活的显著权重通道（按通道放大再量化）就能大幅保精度。',
      'L3 KV 分布：Key 在 Channel 维度存在显著离群值（适合 Per-Channel 量化），Value 在 Token 维度相对平稳（适合 Per-Token 量化）。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（大 Batch W4A16 退化）：在 Marlin Kernel 中，随着 Batch 超过 ~64~128，为什么 W4A16 的加速比会快速回落，甚至在 Prefill 时慢于 BF16？瓶颈转移到了哪里？',
      },
      {
        stage: 'corner',
        question: '追问 2（NVFP4 / W4A4）：Blackwell 支持硬件原生 FP4 Tensor Core (W4A4)，相比 W4A16 它解锁了什么能力？同时 W4A4 要上生产还必须解决什么精度难题（比如旋转变换 QuaRot / Hadamard）？',
      },
    ],
    referenceAnswer: `### Staff 级量化选型与原理矩阵

#### 1. 核心量化方案选型对比
| 方案 | 带宽压缩 | 峰值算力 | 最优场景 | 关键痛点 |
|---|---|---|---|---|
| **W4A16 (AWQ/GPTQ + Marlin)** | **4×** | 1× (BF16/FP16) | 端侧/小卡部署大模型、中小 Batch 低延迟 Decode | 大 Batch/Prefill 时片上反量化指令开销成为瓶颈，反而不如 FP8 |
| **W8A8-INT8 (SmoothQuant)** | 2× | 2× (INT8 TC) | A100/L40S 等无 FP8 Tensor Core 的上一代卡 | INT 线性分布对离群值极度敏感，依赖校准集与平滑因子 |
| **W8A8-FP8 (E4M3)** | 2× | 2× (FP8 TC) | **H100/H200/B200 生产首选**：Prefill+Decode 通吃 | 仅支持 Ada/Hopper+ 架构；需注意细粒度 Scale 保精度 |
| **FP8 KV Cache** | KV 2× | — | 高并发、长上下文显存受限场景 | 与 W8A8-FP8 叠加近乎无损，放大 2× 最大并发 Batch |

#### 2. 离群值处理与高效反量化 Kernel
- **SmoothQuant**：观测到激活通道有剧烈离群值而权重平坦，通过数学等价变换 $Y = (X \\cdot S^{-1})(S \\cdot W)$ 把量化难度平滑分摊给权重。
- **AWQ**：通过激活幅度识别 1% 显著权重通道，不直接混精度（对硬件不友好），而是对显著通道乘以最优缩放系数 $s > 1$ 减小其量化相对误差。
- **Marlin Kernel (W4A16)**：将 INT4 权重异步加载进 Shared Memory/Register，在寄存器中利用位运算（\`lop3\`/FP16 指令技巧）**即时融合反量化**并直接送入 Tensor Core MMA，彻底消除反量化写回 HBM 的带宽开销，在小 Batch 下可跑出接近 3.8× 的实测提速；但一旦进入 Compute-bound 大 Batch，反量化 ALU 指令会抢占发射周期，导致效率落后于硬件原生 FP8/FP4。

#### 3. KV Cache 量化特征
- **Key**：Channel 维离群明显 → 优先 Per-Channel（或配合 RoPE 前量化）；**Value**：分布平滑 → Per-Token。在 H100 上直接用 E4M3 FP8 KV Cache 配合动态 Per-Tensor/Per-Head Scale 已能做到 <0.2% 精度损失。`,
  },
  {
    id: 'inf-06',
    track: 'inference',
    title: 'Speculative Decoding (投机采样)、EAGLE 与 MTP (Multi-Token Prediction)',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['Speculative Decoding', 'EAGLE', 'MTP', 'Tree Attention', '无损验证'],
    summary: '深入 Rejection Sampling 无损验证机制、Tree Verification、MTP 训练推理一体，以及大 Batch 下投机采样失效的 Roofline 根因。',
    contextSpecs: ['Draft + Verify', 'Lossless Distribution', 'EAGLE-2/3 / DeepSeek MTP', 'Acceptance Rate α'],
    prompt: '投机采样（Speculative Decoding）能把低并发下的单请求生成速度提升 2~3 倍。请解释：(1) 它是如何在理论上严格保证输出分布与原 Target 模型**完全无损等价**的？(2) 相比早期独立的草稿小模型，EAGLE 和 DeepSeek MTP (Multi-Token Prediction) 做了哪些架构演进？(3) 为什么高并发大 Batch 时投机采样收益急剧下降甚至产生负优化？',
    rubric: [
      {
        id: 'draft-verify',
        label: '核心原理：利用 Decode Memory-Bound 空闲算力，小模型串行草拟 K 个 Token，大模型单次 Forward 并行验证 K+1 个位置',
        weight: 22,
        keywords: ['memory-bound', '空闲算力', '并行验证', 'draft', 'verify', 'k+1', '一次前向', '带宽'],
        probe: '为什么 Target 大模型一次验证 5 个 Token 的耗时几乎和只算 1 个 Token 一样？这背后利用了什么硬件特性？',
      },
      {
        id: 'rejection-sampling',
        label: '无损 Rejection Sampling：以 min(1, p(x)/q(x)) 概率接受；若拒绝则从 norm(max(0, p - q)) 残差分布重采样，严格保分布一致',
        weight: 24,
        keywords: ['rejection', '拒绝采样', '无损', 'p(x)/q(x)', 'min(1', '残差', 'p-q', '分布等价', '温度'],
        probe: '在 Temperature > 0 的随机采样下，怎么数学保证投机采样生成的分布和原模型严格一致，不会向小模型偏移？',
      },
      {
        id: 'eagle-mtp',
        label: '演进：EAGLE 复用 Target 倒数第二层隐藏态 Feature + 轻量预测头 + 动态 Tree Attention；MTP 在预训练阶段即联合训练多步预测模块',
        weight: 22,
        keywords: ['eagle', 'mtp', 'feature', '隐藏态', 'hidden', 'tree', '树状', 'medusa', '预训练', '联合训练'],
        probe: '独立小模型当 Draft 需要额外维护词表对齐和独立 KV Cache。EAGLE 和 DeepSeek-V3 的 MTP 分别是怎么突破这个限制的？',
      },
      {
        id: 'large-batch-fail',
        label: '大 Batch 失效根因：总验证 Token 变为 B×(K+1)，迅速跨过 Ridge Point 变 Compute-Bound，被拒绝的 Token 变成纯粹的算力浪费',
        weight: 20,
        keywords: ['大batch', 'ridge', 'compute-bound', '浪费', '负优化', '高并发', '吞吐', 'b*(k+1)'],
        probe: '如果在线服务 Batch Size 已经到了 128，再开 K=4 的投机采样，有效 Batch 变成了 640，此时为什么吞吐反而下降、延迟上升？',
      },
      {
        id: 'kv-rollback',
        label: '工程落地细节：KV Cache 投机写入与回滚（Rollback / Truncate）及 Tree Mask 构造',
        weight: 12,
        keywords: ['回滚', 'rollback', 'kv', 'mask', 'truncate', 'pagedattention', 'cuda graph'],
        probe: '验证完如果只接受了前 2 个 Token，后 3 个被拒绝 Token 写进 PagedAttention 的 KV Cache 怎么处理？',
      },
    ],
    hints: [
      'L1 方向提示：投机采样本质是用“过剩的 Tensor Core FLOPS”去换“稀缺的 HBM 权重带宽读取次数”。',
      'L2 无损采样：以 $\\min(1, p(x)/q(x))$ 接受；拒绝后从 $\\text{norm}(\\max(0, p(x)-q(x)))$ 重采——全概率公式展开后 $P(x) = q(x)\\min(1, p/q) + (1-\\beta)\\frac{\\max(0,p-q)}{1-\\beta} = p(x)$！',
      'L3 演进对比：Medusa 预测头之间独立（难捕捉条件依赖）；EAGLE 引入 Target Feature 级联自回归并构造动态置信度树（Tree Attention 一次并行验多个候选分支）；MTP 把轻量解码层放进预训练主流程，推理零额外训练成本。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（期望加速比）：假设单步接受率 $\\alpha = 0.8$，Draft 步数 $K = 4$，Draft 模型单步耗时是 Target 的 $c = 0.08$ 倍。大概能算一下期望每次验证产出多少个 Token，以及理论净加速比是多少吗？',
      },
      {
        stage: 'corner',
        question: '追问 2（MoE + 投机采样）：在 MoE 模型（如 DeepSeek-R1）里开 MTP 投机采样，除了常规计算量增加，对 MoE 路由专家激活数量和显存读取有什么意想不到的负面影响？',
      },
    ],
    referenceAnswer: `### Staff 级参考解析

#### 1. 严格无损的 Rejection Sampling 推导
- 设目标模型分布 $p(x)$，草稿模型分布 $q(x)$。对草稿生成的 token $x \\sim q(x)$：
  1. 以概率 $\\min\\left(1, \\frac{p(x)}{q(x)}\\right)$ **接受**；
  2. 若被**拒绝**，则从修正残差分布 $p'(x) = \\text{norm}(\\max(0, p(x) - q(x)))$ 重采样一个替换 token，并截断后续草稿。
  3. 若 $K$ 个全部接受，目标模型最后一个位置本身还会免费吐出第 $K+1$ 个 token。
- 由全概率公式可得任意 token 的最终生成概率严格等于 $p(x)$，**数学 100% 无损**。

#### 2. 架构演进：Draft Model -> EAGLE -> MTP
- **传统 Draft Model**：需另训同词表小模型，单独占 KV，特征对齐弱。
- **EAGLE / EAGLE-2/3**：直接复用目标模型倒数第二层高维 Hidden Feature + 极轻量 1 层解码器做 Feature 级自回归预测（保留上下文不确定性信息），并基于 Token 置信度动态构建 **Token Tree**，用自定义 **Tree Attention Mask** 在单次 Verify 中并行验证几十个分支，接受长度拉升至 4~6+ tokens/step。
- **DeepSeek MTP**：在 Pre-training 阶段直接挂载轻量顺序预测头（共享 Embedding 与 Output Head），既提升模型表征能力，推理期即插即用实现无损加速。

#### 3. 大 Batch 失效与 MoE 陷阱
- **大 Batch Roofline 失效**：验证有效 Token 数为 $B(K+1)$。当 $B$ 已接近 Ridge Point 时，验证从免费白嫖变成真正的 Compute-bound，未命中分支消耗的全是纯 FLOPs，吞吐反降。工程上需**根据当前 Batch Size 动态开关/缩减 $K$**。
- **MoE 额外代价**：同一请求验证 $K+1$ 个 Token 可能路由到完全不同的专家集合，导致单步需从 HBM 加载的活跃专家权重变多（放大了 Memory-bound 读权重压力），因此 MoE 场景一般选择较小的 $K=1\\sim 2$（如单步 MTP）。`,
  },
  {
    id: 'inf-07',
    track: 'inference',
    title: '超大规模 MoE 推理部署 (DeepSeek-R1 671B)：DP Attention + Wide-EP 与 TBO',
    level: 'P8',
    frequency: '高频必考',
    incident: false,
    tags: ['DeepSeek-R1', 'Wide-EP', 'DP Attention', 'Two-Batch Overlap', 'EPLB'],
    summary: '剖析 671B MoE 推理为何放弃纯 TP 而转向 DP Attention + EP32/EP64，并讲透双批次重叠 (TBO) 与冗余专家负载均衡 (EPLB)。',
    contextSpecs: ['671B MoE / 256 Experts', 'DP Attention + EP64/EP320', 'Two-Batch Overlap', 'EPLB 冗余专家'],
    prompt: '要把 DeepSeek-R1 (671B FP8，256 路由专家 + MLA) 部署成高吞吐在线服务，为什么不能像部署 70B Dense 那样简单用单机 TP=8 或跨机 TP=16？DeepSeek 官方推理架构提出的「Attention 用 DP + MoE 用超大规模 EP (如 EP32/EP144/EP320) + Two-Batch Overlap + 冗余专家 (EPLB)」分别解决了什么核心矛盾？',
    rubric: [
      {
        id: 'dp-attn',
        label: 'DP Attention 替代 TP Attention：避免 MLA 压缩后的单一潜向量 KV Cache 在 TP 间被重复存储 N 份，释放海量 KV 显存拉大 Batch',
        weight: 24,
        keywords: ['dp attention', 'mla', 'kv重复', '复制', '不切分', '显存', 'batch', 'tp'],
        probe: '在 MLA 架构里，所有 Head 共享一个 512 维的 c_KV。如果 Attention 开 TP=8，会发生什么非常浪费的事情？改用 DP Attention 带来什么收益？',
      },
      {
        id: 'wide-ep',
        label: 'Wide-EP (大专家并行，每卡只放少数专家)：大幅压缩单卡权重显存、减少每步从 HBM 加载的专家权重，并聚合多 DP 组 Token 拉高单专家 Grouped GEMM Batch Size',
        weight: 24,
        keywords: ['wide-ep', 'ep', '专家并行', '单卡权重', 'grouped gemm', '算术强度', '聚合', 'batch', 'memory-bound'],
        probe: '单机 8 卡跑 256 个专家时，每卡要放 32 个专家权重，Decode 阶段单个专家分到的 Token 寥寥无几。把 EP 拉宽到 32 甚至 144 张卡，单卡的显存和计算密度会发生什么质变？',
      },
      {
        id: 'tbo',
        label: 'Two-Batch Overlap (TBO / Micro-batch Ping-Pong)：将请求拆成两微批交替执行 Attention/MLP 与 All-to-All Dispatch/Combine 掩盖跨机 RDMA 延迟',
        weight: 20,
        keywords: ['tbo', 'two-batch', 'overlap', 'ping-pong', 'micro-batch', 'dispatch', 'combine', 'deepep', '掩盖', '隐藏'],
        probe: '跨机开 EP64 后，每次 Decode 都要做跨机 All-to-All，延迟很高。DeepSeek 是如何通过 Two-Batch Overlap (TBO) 把通信耗时藏起来的？',
      },
      {
        id: 'eplb',
        label: '冗余专家负载均衡 (EPLB)：线上统计热门专家，在空闲显存中复制多份高热点专家，消除长尾专家卡脖子',
        weight: 18,
        keywords: ['冗余', 'redundant', 'eplb', '热点', '负载均衡', '复制', 'straggler', '重排'],
        probe: '在线推理时用户请求话题集中（如大量代码题），某些专家会被疯狂路由成为热点拖垮整批 Decode，推理期又不能改模型权重，怎么解决？',
      },
      {
        id: 'pd-ep-scale',
        label: 'PD 分离配不同 EP 规模：Prefill 算力密集用较小 EP (如 EP32)，Decode 追求极限并发与带宽用超大 EP (如 EP144/EP320)',
        weight: 14,
        keywords: ['prefill', 'decode', 'pd分离', 'ep32', 'ep144', '320', '低延迟内核', 'low-latency'],
        probe: '为什么 Prefill 集群用较小 EP (如 EP32)，而 Decode 集群要开到极宽的 EP144 甚至 EP320？通信内核有什么不同？',
      },
    ],
    hints: [
      'L1 方向提示：核心矛盾是“MLA 怕 TP 复制 KV”+“256 专家单卡权重太重导致 Batch 上不去、单专家算术强度极低”。',
      'L2 Wide-EP 魔力：EP 扩到 64~320 卡后，每张卡只需存 1~4 个专家，权重显存骤减，腾出几十 GB 显存装 KV Cache，且几十个 DP 组的 Token 汇聚到同一专家上，直接把 MoE Grouped GEMM 推向高算术强度！',
      'L3 TBO 机制：把单卡的 Batch 对半拆成 Batch A 和 Batch B：当 A 在跑 Attention + MoE MLP 计算时，B 正在通过 DeepEP Low-Latency RDMA 内核做 Dispatch/Combine，形成完美双缓冲流水。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Latency vs Throughput）：把 EP 开到 144 或 320 张卡，虽然集群总吞吐（tokens/s/GPU）极高，但单请求的端到端 TPOT 延迟受限于跨机 All-to-All 和 TBO 拆分。如果业务要求极低 TPOT (<15ms/token)，架构上应该怎么调整？',
      },
      {
        stage: 'corner',
        question: '追问 2（容错爆炸半径）：Wide-EP 意味着 32~144 张卡强耦合在同一个 All-to-All 通信域里，只要其中 1 张卡挂掉整个大实例就瘫痪了。生产运维中如何降低这种超大 EP 实例的故障爆炸半径？',
      },
    ],
    referenceAnswer: `### Principal 级超大规模 MoE 推理架构

#### 1. 为什么放弃纯 TP 而采用 DP Attention + Wide-EP？
- **DP Attention 解决 MLA KV 重复问题**：MLA 的核心状态是所有 Head 共享的 512 维 \`c_KV\`，无法像 MHA 那样按 Head 切分给 TP；若开 TP=8，每卡都会存一份完全相同的 \`c_KV\`，浪费 8× 显存！改成 **DP Attention** 后，每张卡独立服务不同请求的 Attention，零重复 KV，单实例最大并发直接放大 8×。
- **Wide-EP (如 P: EP32, D: EP144/EP320) 破局内存与计算密度**：
  - 671B FP8 权重约 671GB，若单机 8 卡跑则权重就吃光显存；拉宽到 EP144/EP320 后，每张卡仅承载 1~2 个专家，腾出海量 HBM 给 KV Cache；
  - 更关键的是，Decode 时 144 个 DP 组的活跃请求全部通过 Dispatch 汇聚到对应专家，让单个专家的 Grouped GEMM Batch Size 提升十几倍，彻底摆脱极低算术强度的 Memory-bound 泥潭。

#### 2. 隐藏跨机 All-to-All：Two-Batch Overlap (TBO) + DeepEP
- **Prefill**：拆成 2 个 Micro-batch，交替执行计算与普通 High-Throughput DeepEP All-to-All。
- **Decode**：同样拆成 2 个 Micro-batch，由于每层有 5 个子阶段，利用 **DeepEP Low-Latency IBGDA (纯 RDMA 零 CPU 参与)** 内核在后台异步 Dispatch/Combine，前台 GPU SM 全速跑 FlashMLA 与 FP8 Grouped GEMM，实现通信延迟几乎全掩盖。

#### 3. 热点治理：冗余专家负载均衡 (EPLB)
- 每隔几分钟在线统计各专家的真实命中热度，在每张卡预留 1 个冗余专家槽位（例如 256 个逻辑专家部署为 288~320 个物理专家实例），将最热门的专家多部署几份副本，并在路由时均匀分摊请求，压平长尾 Straggler。`,
  },
  {
    id: 'inf-08',
    track: 'inference',
    title: '线上推理服务 SLO 排障：TTFT 飙升、TPOT 抖动与 KV 抢占雪崩',
    level: 'P7',
    frequency: '高频必考',
    incident: true,
    tags: ['SLO排障', 'TTFT', 'TPOT', 'Preemption', 'Prefix Routing'],
    summary: '真实线上告警复盘：大促高峰期 TTFT P99 飙到 8 秒、TPOT 周期性卡顿 500ms，如何逐层定位并根治？',
    contextSpecs: ['TTFT SLA < 1s', 'TPOT SLA < 40ms', 'vLLM / SGLang 集群', '监控与路由'],
    prompt: '【线上故障排障】你负责的 72B 在线推理集群（64 台 8×H100，跑 vLLM/SGLang）在业务高峰期突然告警：平均 GPU 利用率看起来只有 65%，但 TTFT P99 从 600ms 暴涨到 8 秒，TPOT P99 周期性出现 400~800ms 的毛刺，且整体输出吞吐反而在下降。请给出你的系统性排查路径、可能根因清单以及立即可落地的止血与根治方案。',
    rubric: [
      {
        id: 'metrics-triage',
        label: '核心指标分层观测：拆分 Queueing Time vs Prefill Time，查看 num_preemptions、KV Cache Usage%、Prefix Hit Rate、Running/Waiting Queue',
        weight: 24,
        keywords: ['queue', '排队', 'kv cache usage', 'preemption', '抢占', 'prefix', '命中率', 'waiting', 'running', '监控'],
        probe: '进 Prometheus / Grafana 看板的第一眼，你最先看哪 4 个推理引擎内部指标来区分到底是“排队堵了”、“Prefill 慢了”还是“显存雪崩了”？',
      },
      {
        id: 'preemption-thrash',
        label: '识别 KV 抢占雪崩 (Preemption Thrashing)：KV 使用率打满 99%+ 反复驱逐重算，导致吞吐不升反降、TTFT/TPOT 双崩',
        weight: 22,
        keywords: ['雪崩', 'thrashing', '抢占', 'recompute', '重算', '99%', '打满', '吞吐下降', 'max_num_seqs'],
        probe: '为什么流量越高、整体输出吞吐反而“不升反降”？这种反常的吞吐倒挂在 KV Cache 满载时是怎么形成的？',
      },
      {
        id: 'routing-imbalance',
        label: '网关路由失衡：普通 Round-Robin 导致 Prefix Cache 命中率骤降 + 长请求集中打爆单实例',
        weight: 20,
        keywords: ['路由', 'round-robin', 'prefix-aware', '负载均衡', '亲和', '长尾', 'kv-aware', '倾斜', '一致性哈希'],
        probe: '64 个实例整体 GPU 利用率只有 65%，却有实例在疯狂排队和抢占，网关层的负载均衡策略可能犯了什么典型错误？',
      },
      {
        id: 'chunk-host',
        label: '毛刺根因：未开 Chunked Prefill / Token Budget 过大、Python GC/Tokenizer CPU 阻塞、或 CUDA Graph 未命中回退 Eager',
        weight: 18,
        keywords: ['chunked prefill', 'budget', 'python', 'gc', 'tokenizer', 'cpu', 'cuda graph', '毛刺', '周期'],
        probe: 'TPOT 出现“周期性 400ms 卡顿”，除了 Prefill 抢占，Host CPU 和 Python 运行时层面还有哪些高频元凶？',
      },
      {
        id: 'mitigation',
        label: '止血与根治：限流/准入控制 (Token-based Admission)、切 Prefix+Load 感知路由、开 FP8 KV Cache 扩并发、异步 Tokenizer / PD 分离',
        weight: 16,
        keywords: ['限流', '准入', 'admission', 'fp8 kv', 'pd分离', '多进程', 'async', '止血'],
        probe: '线上正在告警，哪两个改动可以在 5 分钟内立刻止血？中长期架构怎么根治？',
      },
    ],
    hints: [
      'L1 方向提示：遇到“平均利用率不高但 P99 崩了+吞吐倒挂”，90% 是两大元凶叠加：(1) 网关路由不感知 KV 负载与前缀导致单点过载；(2) 单点过载触发 KV Cache Preemption 雪崩！',
      'L2 周期性卡顿检查清单：大 Prompt Prefill 阻塞、vLLM 抢占重算暂停 Decode、Python 同步 GC / Tokenizer 占满主进程 Event Loop。',
      'L3 速效止血：调低 \`max_num_seqs\` / 开启 Token-level 准入控制防止 KV 超过 90% 触发雪崩 + 网关切为 **Least-KV-Load + Prefix-Aware** 路由。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Prefix-Aware 路由的副作用）：如果网关严格按照 System Prompt / 前缀哈希做亲和路由，遇到一个超级爆款 Agent（80% 流量共享同一个 System Prompt），会引发什么新灾难？怎么设计“缓存亲和 + 负载均衡”的混合路由算法？',
      },
      {
        stage: 'corner',
        question: '追问 2（Host 瓶颈排查）：怎么用工具实证确认 400ms 的周期性毛刺到底是 GPU 算子慢了，还是 Python 主线程 Tokenizer/Detokenizer/Scheduler 卡住了？',
      },
    ],
    referenceAnswer: `### Staff 级线上排障 Playbook

#### 1. 黄金 4 指标速诊（区分根因）
1. **Queueing Time vs Pure Prefill Time**：若 Queueing 占 7s，是过载/排队/路由倾斜；若 Pure Prefill 占 7s，是 Prefix Cache 失效或超长 Prompt 冲击。
2. **\`gpu_cache_usage_perc\` & \`num_preemptions_total\`**：若部分实例 KV 使用率逼近 98%~100% 且抢占计数飙升，说明发生了 **KV 抢占雪崩 (Preemption Thrashing)**——新请求一进来就把老请求 KV 踢掉重算，GPU 算力全耗在反复重算上，导致**有效输出吞吐倒挂**！
3. **\`prefix_cache_hit_rate\`**：若命中率突降，Prefill 算力开销瞬间放大 5~10×。
4. **单实例队列深度分布（P99 vs Mean）**：解释“集群平均利用率 65% 但 P99 崩盘”的路由倾斜。

#### 2. TPOT 周期性 400~800ms 毛刺排查
- **Prefill 阻塞**：未开 Chunked Prefill 或 \`max_num_batched_tokens\` 设太大（如 16384），长 Prompt 周期性卡死 Decode。
- **Host / Python 运行时阻塞**：Tokenizer/Detokenizer 跑在主进程、Python Cyclic GC 触发 Stop-the-world、或 Batch Size 超出预设 CUDA Graph 捕获范围回退到 Eager Mode。
- **定位工具**：抓 \`py-spy\` 看主循环火焰图 + \`nsys\` 看 GPU Stream 上是否出现几百毫秒的空白 Gap（有大空白 Gap 必是 Host CPU/通信阻塞，非 GPU 算慢）。

#### 3. 止血与根治方案
- **5 分钟止血**：(1) 启用 KV 水位保护（KV > 85% 拒绝新进队列 / 网关摘流防雪崩）；(2) 调优 \`max_num_batched_tokens\`（2048~4096）保护 Decode TPOT；(3) 对超长 Prompt 做分级路由隔离。
- **中长期根治**：
  - 网关升级为 **Bounded-Load Prefix-Aware Routing**（兼顾前缀命中与实例 KV/等待队列负载：当命中节点负载超过均值 1.2× 时强制溢出到最闲节点，解决爆款 Agent 热点）；
  - 开启 **FP8 KV Cache** 翻倍并发容量，升级 vLLM V1 / SGLang 独立多进程异步 Frontend 消除 Python 阻塞，重度长文本业务上 **PD 分离**。`,
  },
];
