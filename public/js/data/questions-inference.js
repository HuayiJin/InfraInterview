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
        label: '固定大小的物理块和逻辑块表减少 KV 预留浪费，说明块内尾部碎片与共享条件',
        weight: 24,
        keywords: ['pagedattention', 'block', '块', '页表', 'block table', '碎片', '虚拟内存', '非连续', 'copy-on-write', 'cow'],
        probe: '连续预留最大长度会浪费哪些空间？块表如何支持按需分配与共享？',
      },
      {
        id: 'cont-batch',
        label: 'Continuous batching 在迭代边界更新活跃请求，按 token 与 KV 预算接纳新请求',
        weight: 22,
        keywords: ['continuous', 'iteration', '迭代级', '动态', 'orca', '插入', '完成', 'padding', '长尾'],
        probe: 'Static Batching 为什么吞吐极低？Iteration-level 调度下不同长度的序列怎么做 Attention 和 FFN 批处理？',
      },
      {
        id: 'chunked-prefill',
        label: '按迭代 token 预算分块 prefill，与 decode 混批，解释 ITL、TTFT 和吞吐的权衡',
        weight: 24,
        keywords: ['chunked prefill', '分块', '混批', '合批', 'token budget', 'token 预算', 'itl', 'tpot', 'sarathi', 'piggyback'],
        probe: '单引擎里突然插入 32K prompt 会怎样影响在途 decode？混批对权重读取和迭代耗时各有什么影响？',
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
        label: 'KV 块不足时的抢占、重算和可选换出，说明所举引擎与版本',
        weight: 14,
        keywords: ['preempt', '抢占', 'recompute', 'swap', '驱逐', 'oom', '重算'],
        probe: '当并发太高导致所有物理 KV Block 被占满、已有请求又要生成新 Token 时，vLLM 会怎么处理？',
      },
    ],
    hints: [
      'L1 分别看 KV 容量、每轮哪些请求进出和每轮可处理多少 token。',
      'L2 一轮混批能让线性层在更多 token 之间复用权重。chunk 太大同样会拉长这轮 decode 的间隔。',
      'L3 后续 chunk 仍需访问前缀 KV。用不同预算测 TTFT、逐 token 间隔和有效吞吐。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（抢占）　以 vLLM v0.10.2 的 V1 默认重算为例，什么情况下重算历史前缀比把 KV 换出再换入便宜？',
      },
      {
        stage: 'corner',
        question: '追问 2（长输入）　128K 请求与交互式 decode 混部时，如何判断继续调 chunk budget 还是拆 P/D 池？',
      },
    ],
    referenceAnswer: `### 三种机制各管什么

- **PagedAttention 管 KV 的放置**。逻辑 token 块经 block table 指向固定大小的物理块，请求增长时按块申请。同规格池中可用任一空块，减少连续大块分配的外部碎片。每个独占序列的最后一块最多有 block_size−1 个空位。共享相同前缀的只读块，分叉后再写时可用 copy-on-write。块表、元数据和尾块仍占显存。
- **Continuous batching 管请求进出**。在迭代边界释放已完成请求的槽位，按剩余 token 和 KV 预算补入请求。线性层可以合并不同请求的 token。Attention 按序列元数据保持请求间隔离。为 CUDA Graph 或内核对齐时仍可能 padding。
- **Chunked prefill 管单轮工作量**。长 prompt 分多轮处理，剩余预算内可与 decode 合批，降低单个长 prefill 独占引擎的时长。较大 chunk 有利于 GEMM 和 prompt 吞吐，但在途请求可能等得更久。较小 chunk 增加调度开销，后续 chunk 还需读历史 KV。按真实到达率画 TTFT、ITL P99 与 goodput 曲线。

**抢占追问**。vLLM v0.10.2 的 V1 默认采用 recompute。比较被逐出请求的未命中前缀重算时间，与 KV 写出、读回及排队耗时。前缀长度、缓存命中和有效链路决定选哪条路径。若调遍单池预算仍达不到两类 SLO，可用同等 GPU 比较 PD 分离，并计入 KV 传输和低负载闲置。

> 依据　PagedAttention（arXiv:2309.06180）及 vLLM v0.10.2《Optimization and Tuning》。`,
  },
  {
    id: 'inf-02',
    track: 'inference',
    title: 'Prefill-Decode (PD) 分离架构设计与 KV Cache 传输优化',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['PD分离', 'DistServe', 'Mooncake', 'RDMA', 'KV Cache传输'],
    summary: '比较混部和 PD 分离的 SLO、KV 传输和资源配比，估算长 prompt 在给定链路下的传输时间。',
    contextSpecs: ['Prefill Pool + Decode Pool', 'RDMA / NIXL / Mooncake', 'TTFT / TPOT SLA', 'P:D 配比'],
    prompt: '长输入与交互式 decode 混部时，什么条件下值得改为 PD 分离？请比较可达 SLO 与成本，估算单请求跨节点 KV 流量，并说明如何设计传输流水、异构并行度和 P:D 配比。',
    rubric: [
      {
        id: 'pd-motivation',
        label: '按实测算力、带宽、显存和 SLO 分别选择 P/D 的硬件与并行度',
        weight: 24,
        keywords: ['compute-bound', 'memory-bound', '解耦', '干涉', '干扰', '并行度', 'tp', 'dp', 'ep', 'sla', 'ttft', 'tpot'],
        probe: '分池后两侧可独立调整哪些参数？什么负载或拓扑会让混部更有利？',
      },
      {
        id: 'kv-transfer',
        label: '估算逻辑 KV 与关键链路字节，区分传输引擎、直连路径与有效带宽',
        weight: 22,
        keywords: ['kv', '传输', 'rdma', 'gpudirect', 'nixl', 'mooncake', '带宽', '网络', '零拷贝'],
        probe: '算一下一个 70B GQA 模型 32K Prompt 的 KV Cache 有多大？如果走 400Gbps IB RDMA 要传多久？',
      },
      {
        id: 'layerwise-overlap',
        label: '分块或逐层重叠仅能隐藏部分传输。预留目标页并设置背压、超时和回收',
        weight: 22,
        keywords: ['layer-wise', '逐层', 'overlap', '异步', '流水', '掩盖', '预分配', '死锁', 'handshake'],
        probe: '首 token 和第二个 token 分别在哪个节点产生？KV 传输慢时会体现在哪个指标，目标节点没空间时怎样处理？',
      },
      {
        id: 'kv-cache-centric',
        label: '按兼容前缀复用跨节点 KV，比较远程读取与重新 prefill 的代价',
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
      'L1 独立调整 P 的 prompt 服务率与 D 的输出服务率，结合端到端 SLO 选择 TP/DP/EP。',
      'L2 80 层、8 KV head、head_dim=128、BF16 时为 320 KiB/token。32,768 个 token 的逻辑 KV 是 10 GiB。',
      'L3 前一层 KV 就绪后即可尝试传输。比较生产和传输速度，并核对接收端页容量。',
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
    referenceAnswer: `### 分离的收益要足以覆盖传输与排队

共置时长 prefill 和在途 decode 争用同一执行时间。拆池后可分别按输入长度、输出并发、显存、TPOT 和 TTFT 选择硬件及 TP/DP/EP。长 prompt 通常能形成较大的 GEMM，小批 decode 常受权重或 KV 带宽限制。低流量、短输入或互连慢时，混部可能有更高的单位卡 goodput。

**算例**。80 层、8 个 KV head、每头 128 维、BF16 的 GQA，每 token 为 2×80×8×128×2=327,680 字节，即 320 KiB。32,768 个 token 共 **10 GiB**。全部字节经过一条 400 Gb/s 单向链路时，序列化下界是 10×2³⁰/(400×10⁹/8)≈**215 ms**。多 rail 若各走不重复的分片可并行。只传缺失前缀、TP 复制、协议、竞争和 layout 转换都会改变字节数或有效带宽。

- **传输路径**。先预留 D 的页与接收队列，按层或块登记完成事件，用支持的直连路径传 KV，超时或取消时释放两端资源。逐层流水只隐藏与计算重叠的部分。P 若直接返回首 token，移交变慢可表现为首到第二 token 的间隔变长。若等 D 就绪才返回，TTFT 也会增长。
- **TP4→TP8**。以 8 个 KV head 为例，P 每 rank 有 2 个 head，D 每 rank 需 1 个。按全局层、head、token 块将对应片段送往目标页，核对 dtype、scale、布局和块大小。不连续布局需打包或转换，复制 KV 的布局还需扇出。
- **缓存与配比**。Mooncake 将 KV 放入分层缓存池。路由时寻找与模型版本、token 前缀和位置配置相符的块，比较远程读取与本地重算耗时。只有已物化、仍有效的部分可跳过 prefill。设请求率 λ，需要处理的平均未命中输入和输出为 I、O，每个 P/D 实例在目标 SLO 下的服务率为 rP、rD，按平均吞吐估算，实例数下界分别为 ⌈λI/rP⌉、⌈λO/rD⌉，应再为峰值和排队 SLO 留余量。长输入短输出与长 CoT 应分别回放。异构卡按精度、KV 传输兼容性和每个达标请求的成本比较。

> 依据　DistServe（arXiv:2401.09670）与 Mooncake（FAST ’25）。`,
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
    prompt: '给出 GQA 每 token 的 KV Cache 字节公式。DeepSeek-V2/V3 的 MLA 每层缓存哪些张量？从一头的打分和加权求和推导矩阵吸收，并解释 Decoupled RoPE 的作用。最后说明 SGLang RadixAttention 在多请求之间如何复用前缀。',
    rubric: [
      {
        id: 'kv-formula',
        label: '准确写出单 Token KV 显存公式：2 × L × n_kv × d_h × bytes（并能定量举例）',
        weight: 20,
        keywords: ['n_kv', 'kv head', 'd_h', 'head_dim', 'bytes', '字节', '320 kib', '40 gib', '2×'],
        probe: '请先给出 GQA 场景下单 Token KV Cache 的精确字节公式？70B 模型 128K 长度单请求光 KV 就要占多少 GB？',
      },
      {
        id: 'mla-latent',
        label: '低秩 MLA 缓存 c_KV 与共享的旋转 k_R，给出 V3 在指定精度下的维度与字节',
        weight: 22,
        keywords: ['mla', 'latent', '低秩', 'c_kv', '512', '压缩', '潜向量', '缓存'],
        probe: 'MLA 用什么结构减少缓存？比较缓存大小时需要给出哪些模型维度？',
      },
      {
        id: 'weight-absorb',
        label: '利用结合律将 K 上投影移到当前 query，将 V 上投影移到对潜向量加权求和之后',
        weight: 24,
        keywords: ['吸收', 'absorp', 'absorb', '上投影', '加权求和', '潜空间', 'w_uk', 'w_uv', '结合律'],
        probe: '如果 Decode 每步都要把 c_KV 乘上 W_UK 还原成巨大的 K 和 V，那省下的只是容量、并没有省算力或带宽。MLA 推理时是怎么用“矩阵吸收”绕开还原 KV 的？',
      },
      {
        id: 'decoupled-rope',
        label: '相对位置旋转依赖历史位置，妨碍按 query 做一次固定吸收。RoPE 分支独立缓存',
        weight: 18,
        keywords: ['rope', '解耦', 'decoupled', '位置编码', '结合律', '不可交换', 'k_r', '64'],
        probe: '为什么不能直接对解压后的 K 施加标准 RoPE？DeepSeek 为什么要专门设计 Decoupled RoPE？',
      },
      {
        id: 'radix-tree',
        label: 'SGLang RadixAttention：基于基数树 (Radix Tree) 管理共享前缀 KV 物理页 + LRU 淘汰与缓存感知调度',
        weight: 16,
        keywords: ['radix', '基数树', 'sglang', '前缀', 'prefix', 'lru', '复用', '多轮', 'agent', 'system prompt'],
        probe: '基数树和块哈希都能复用前缀。怎样找到可复用的块并兼顾命中率与排队公平性？',
      },
    ],
    hints: [
      'L1 方向提示：MLA 靠“低秩投影压缩状态”+“矩阵乘法结合律吸收上投影矩阵”+“解耦 RoPE 避开位置矩阵阻隔”三板斧。',
      'L2 对每个 head 按矩阵乘结合律重排 qᵀ(Uₖcⱼ)=(Uₖᵀq)ᵀcⱼ，再对历史位置打分。',
      'L3 长 query 的 prefill 和单 query 的 decode 使用不同形状，比较是否展开 KV、潜空间维度和实测 kernel 性能。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Prefill 与 decode）　两种形状下，直接展开 K/V 与在潜空间计算分别读写哪些张量、做多少运算？',
      },
      {
        stage: 'corner',
        question: '追问 2（Cache-aware Scheduling）：SGLang 在等待队列里有几十个不同 Prefix 的请求，如果简单按 FIFO 调度，可能会不断把刚算好的 Prefix Cache 给 LRU 淘汰掉（缓存抖动），应该怎么改调度顺序？',
      },
    ],
    referenceAnswer: `### 比较逻辑缓存时先给出层数、维度和 dtype

等维 K/V 的 GQA 每 token 为 2L × n_kv × d_h × b 字节。L=80、n_kv=8、d_h=128、b=2 时是 320 KiB，131,072 个 token 占 **40 GiB**。DeepSeek-V3 的 61 层 MLA 每层缓存 512 维联合潜向量 c 和共享的 64 维旋转 key。BF16 时是 61×576×2=70,272 字节，即 **68.625 KiB/token**。这是两种具体配置的全模型逻辑量，单卡还取决于 TP/PP、复制、缓存精度和页布局。

### 对每个 head 移动两个线性变换

1. 设内容 key 为 Kⱼ=Uₖcⱼ。打分 qᵀKⱼ=(Uₖᵀq)ᵀcⱼ，可对当前 q 只做一次上投影的转置乘法。
2. 设 Vⱼ=Uᵥcⱼ，attention 概率为 aⱼ，则 Σ aⱼVⱼ=Uᵥ(Σ aⱼcⱼ)。先在潜空间聚合，再上投影和输出映射。避免对每个历史位置物化完整 K/V，无需预先构造巨大的融合权重。
3. 若内容分量也做 RoPE，qᵀRₜᵀRⱼUₖcⱼ 中的旋转随历史位置 j 改变，无法为所有 j 预先算一个变换后的 q。解耦后，内容打分走吸收路径，位置打分由旋转后的 qᴿ 与缓存 kᴿ 贡献，同一 softmax 合并。Decode 常受益于紧凑历史读取。prefill 的长 Q 可能更适合临时展开后使用普通 attention kernel，应按形状比较。

**RadixAttention** 以 token 前缀建基数树，复用兼容模型、位置和缓存粒度下的已算 KV。最长前缀匹配有利于复用。还应为等待时间长的请求保留调度机会。页或块哈希也能做前缀缓存。共享存储节省 prefill 和容量，decode 端实际少读多少 HBM 取决于 kernel。

> 依据　DeepSeek-V2（arXiv:2405.04434）及 SGLang（arXiv:2312.07104）。`,
  },
  {
    id: 'inf-04',
    track: 'inference',
    title: 'Roofline 模型定量分析 LLM 推理：Decode 为什么是 Memory-Bound？',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['Roofline', 'Arithmetic Intensity', 'Ridge Point', 'H100', 'Batch Size'],
    summary: '用 H100 SXM 上的理想权重流量估算 FFN 拐点，并给出 KV、激活和实际部署如何改变结论。',
    contextSpecs: ['H100: 989 TF / 3.35 TB/s', 'Ridge Point ~295', 'OI = FLOPs / Bytes', 'TTFT / TPOT'],
    prompt: '使用 H100 SXM 的 BF16 dense 峰值 989 TFLOP/s 和 3.35 TB/s HBM 带宽，分析 70B Dense 服务中的一个本地 FFN 权重分片。为何 batch=1 的 decode 线性层算术强度低？在只计权重读取和计入激活读写两种模型下，分别怎样估计临界 batch？70B BF16 部署时需要另行满足容量约束。',
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
        label: '说明只计权重读取时 OI≈2B/b_w，并给出含输入与输出流量的式子',
        weight: 26,
        keywords: ['2b', 'oi', 'batch', '算术强度', '权重', '激活', '搬运', '2p', '字节'],
        probe: '对 [B,K] × [K,N] 的矩阵乘，把权重、激活读取和结果写出都计入，OI 怎样变？',
      },
      {
        id: 'batch-threshold',
        label: '用给定 dense 峰值估算 BF16、原生 FP8 和融合 W4A16 的理想拐点，区分实际性能',
        weight: 22,
        keywords: ['295', '300', '临界', '饱和', 'compute-bound', '计算密集', 'batch', 'fp8', '591', '590', '74'],
        probe: '照你刚才的公式，BF16 下 Batch 大约到多少才会从 Memory-Bound 变成 Compute-Bound？如果换成 FP8 呢，临界 Batch 会变大、变小还是基本不变？',
      },
      {
        id: 'kv-bound',
        label: '独立序列的 KV 读取随 batch 增长，GQA 的同头组复用取决于 kernel',
        weight: 18,
        keywords: ['attention', 'kv cache', '无法复用', '独立', 'memory-bound', 'gqa', 'batch', '共享'],
        probe: '增大 Batch Size 可以让 Linear/FFN 层变 Compute-Bound，那 Self-Attention 层也会随着 Batch 增大变成 Compute-Bound 吗？为什么？',
      },
      {
        id: 'sla-tradeoff',
        label: '分析显存、共享前缀、实际带宽与 TPOT SLO 对有效 batch 和吞吐的限制',
        weight: 10,
        keywords: ['tpot', 'sla', 'slo', '延迟', '显存', '吞吐', '容量', '实际带宽', '通信时间'],
        probe: '实测 TPOT、吞吐与 KV 容量怎样限制 batch？哪项监控可验证从带宽受限到算力受限的判断？',
      },
    ],
    hints: [
      'L1 方向提示：Roofline 核心就两个数——硬件拐点 $\\text{Ridge} = \\frac{\\text{Peak FLOPS}}{\\text{Mem BW}}$，以及算子计算强度 $\\text{OI} = \\frac{\\text{FLOPs}}{\\text{Bytes Moved}}$。',
      'L2 BF16 的 B、FP8 的 2B 和 W4A16 的 4B 都是只计权重字节的近似。对比的算力峰值要取实际 MMA 指令对应的精度。',
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
    referenceAnswer: `### 本地线性层的上界

设 X 的形状为 [B,K]，W 为 [K,N]。一次乘加计 2 FLOPs。若每个输入从 HBM 读取一次、输出写回一次，OI = 2BKN/(b_w KN + b_x BK + b_y BN)，单位 FLOPs/byte。令 R=峰值 FLOP/s ÷ HBM byte/s。由此可求 B = R b_w KN / [2KN−R(b_x K+b_y N)]。只有分母为正才有有限交点。若权重远大于激活流量，退化为 OI≈2B/b_w。

| 本地 GEMM 路径 | H100 SXM 理想 R | 只计权重时的 OI | 近似交点 B |
|---|---|---|---|
| BF16 dense | 989/3.35≈295 | B | 295 |
| 原生 W8A8 FP8 dense | 1979/3.35≈591 | 2B | 295 |
| W4A16，解包后用 FP16/BF16 MMA | 约 295 | 4B | 74 |

**怎样读这些数**。B=1、BF16 时理想算力上界约为 3.35 TFLOP/s，即峰值的 0.34%。量化 scale、解包、缓存命中、实际 tile 和 kernel 效率都会改变交点。70B BF16 仅权重约 140 GB，一张 80 GB H100 装不下。分卡后应代入本地 K、N、通信时间与 batch。

**Decode attention**。长历史、每请求一个 query 时，QK 和 PV 约做 4S n_q d FLOPs，主要读取 2S n_kv d b_kv 字节。若每份 KV 在 head 组内只读一次，OI≈2(n_q/n_kv)/b_kv，BF16 MHA 约 1，8:1 GQA 约 8。独立请求随 batch 增长，算量与 KV 字节同增。扩大 batch 前先预算每请求 KV 与总显存，按目标 TPOT 测达标吞吐，再用逐算子 HBM 字节和实际 FLOP/s 验证瓶颈。共享前缀只有在内核能复用其读取时才改变这项带宽账。

> 依据　Roofline（Williams 等，2009）及 NVIDIA H100 SXM 规格，取关闭结构化稀疏的峰值。`,
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
        label: 'W4A16 降低权重流量，W8A8 还可使用相应低精度 MMA，按硬件和负载选型',
        weight: 26,
        keywords: ['w4a16', 'w8a8', 'memory-bound', 'compute-bound', 'batch', 'prefill', 'decode', '搬运', '2倍算力', 'tensor core'],
        probe: '为什么在 Prefill 阶段或大 Batch 离线批推场景，W4A16 有时反而比 FP8 甚至 BF16 还慢？',
      },
      {
        id: 'outliers-method',
        label: 'SmoothQuant 与 AWQ 的等价缩放、GPTQ 的校准误差补偿及其目标',
        weight: 22,
        keywords: ['outlier', '离群值', 'smoothquant', 'awq', '显著', 'salient', '通道', '平滑', 'scale', '迁移', 'gptq', 'hessian'],
        probe: '同样是处理激活值里的离群通道，SmoothQuant 和 AWQ 的核心思想分别是什么？',
      },
      {
        id: 'dequant-kernel',
        label: 'W4A16 融合反量化 Kernel (如 Marlin)：在寄存器内 On-the-fly 反量化 INT4->FP16 再喂 Tensor Core，避免写回 HBM',
        weight: 20,
        keywords: ['marlin', '反量化', 'dequant', '融合', 'register', '寄存器', 'cutlass', 'kernel', '位运算', 'lop3'],
        probe: 'Marlin 的解包和反量化发生在哪里？哪些额外字节或运算会缩小理想收益？',
      },
      {
        id: 'fp8-advantage',
        label: '比较原生 FP8 与 INT8 的格式、scale、校准、硬件与内核支持',
        weight: 18,
        keywords: ['fp8', 'int8', '动态范围', '校准', 'e4m3', 'scale', 'h100', 'l40s', 'blockwise'],
        probe: '哪些 GPU 和后端支持原生 FP8 GEMM？怎样判断 FP8 与 INT8 在当前模型上的精度和吞吐？',
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
      'L1 先分清 weight-only、权重与激活均量化、仅压缩 KV 三种路径各自减少的字节与运算。',
      'L2 SmoothQuant 以对角缩放在激活和权重间迁移量化难度，AWQ 依据激活统计选择保护显著权重的缩放。',
      'L3 KIVI 的低比特方案根据实测分布，对 key 按通道、value 按 token 分组。FP8 缓存的 scale 粒度还取决于后端。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（大 batch）　固定模型和硬件，逐步增加 batch。Marlin 相对 BF16 的加速比为何可能收窄，如何定位拐点？',
      },
      {
        stage: 'corner',
        question: '追问 2（NVFP4）　原生 W4A4 MMA 与 W4A16 的硬件数据通路有什么区别？如何测量激活也量化后的精度影响？',
      },
    ],
    referenceAnswer: `### 按实际压缩的对象选型

| 路径 | 省在哪里 | 主要代价和验证 |
|---|---|---|
| W4A16 | FP16/BF16 权重载荷由 2 B/元素降至约 0.5 B，加上 group scale、零点和对齐 | Marlin 类 kernel 在片上解包和反量化后用 16-bit MMA。小 batch 权重读取占主导时收益较大，prefill 须测解包和算力瓶颈 |
| W8A8 INT8 / FP8 | 8-bit 权重，支持时可用原生低精度矩阵乘 | 查 GPU、后端、累加类型和 scale 粒度。A100 支持 INT8 TC。Ada L40S 与 Hopper H100 都有 FP8 TC。存得下 FP8 文件和执行原生 FP8 GEMM 是两个条件 |
| FP8 KV | BF16 KV 主体字节减半 | 检查 scale、attention 内核、长上下文质量与每请求峰值。权重、临时工作区与元数据仍占空间，应实测可承载并发 |

**离群值处理**。SmoothQuant 利用 XW=(XS⁻¹)(SW)，将部分激活通道的范围移到权重上，便于 W8A8。AWQ 依据激活统计选权重通道的缩放，降低显著权重的量化误差，保留规则低比特存储。GPTQ 用校准激活近似二阶信息，逐步量化并补偿剩余权重误差。三者的校准目标有别。FP8 仍有饱和与舍入，要比较静态校准和动态 scale。KIVI 的低比特 KV 设计对 key 用 per-channel、对 value 用 per-token，可作为候选。

**验证与 W4A4 追问**。同一流量回放中分开测 prefill 吞吐、decode ITL、峰值显存、任务质量与长文本检索。大 batch 权重复用充分时，W4A16 仍在做高精度乘法，不能从位宽推导端到端四倍。Blackwell NVFP4 可让两侧都走低精度 MMA，需测分块 scale、激活异常值、量化开销和质量。

> 依据　SmoothQuant（arXiv:2211.10438）、AWQ（2306.00978）、GPTQ（2210.17323）、KIVI（2402.02750）及 NVIDIA L40S 规格。`,
  },
  {
    id: 'inf-06',
    track: 'inference',
    title: 'Speculative Decoding (投机采样)、EAGLE 与 MTP (Multi-Token Prediction)',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['Speculative Decoding', 'EAGLE', 'MTP', 'Tree Attention', '无损验证'],
    summary: '推导随机投机验证的分布、期望产出和净速比，区分 EAGLE 各代与 MTP。',
    contextSpecs: ['Draft + Verify', 'Lossless Distribution', 'EAGLE-2/3 / DeepSeek MTP', 'Acceptance Rate α'],
    prompt: '草稿模型先生成 K 个 token，目标模型并行验证。随机采样如何保持指定目标分布？与独立 draft 相比，EAGLE 各代和 DeepSeek-V3 MTP 提供了怎样的候选？结合接受长度、验证成本和 batch 讨论何时值得开启。',
    rubric: [
      {
        id: 'draft-verify',
        label: '草拟 K 个条件 token，在一个 target pass 得到验证及额外采样分布，明确成本假设',
        weight: 22,
        keywords: ['memory-bound', '空闲算力', '并行验证', 'draft', 'verify', 'k+1', '一次前向', '带宽'],
        probe: '为什么验证若干位置有时比逐次 decode 便宜？什么条件下验证耗时会明显增加？',
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
        label: '区分 EAGLE 特征预测、EAGLE-2 动态树、EAGLE-3 多层特征直接预测 token 和 MTP 联合预训练',
        weight: 22,
        keywords: ['eagle', 'mtp', 'feature', '隐藏态', 'hidden', 'tree', '树状', 'medusa', '预训练', '联合训练'],
        probe: 'EAGLE、EAGLE-2、EAGLE-3 和 V3 MTP 分别怎样提供候选？草稿侧还需维护哪些状态？',
      },
      {
        id: 'large-batch-fail',
        label: '随验证宽度和 batch 评估接受产出与成本，考虑 MoE 新增专家和 KV 带宽',
        weight: 20,
        keywords: ['高 batch', '高并发', 'ridge', 'compute-bound', '额外专家', '验证', '接受长度', '吞吐', 'b(k+1)', '算力'],
        probe: '给定 batch=128、K=4，如何用接受长度和验证耗时判断开启后吞吐会升还是降？',
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
      'L2 对每个 token，把 q 采到并被接受的质量 min(p,q) 与拒绝后补回的正部 (p−q)₊ 相加。',
      'L3 EAGLE-2 改变草稿树的构造。EAGLE-3 改为多层特征和直接 token 预测。MTP 可复用预训练得到的额外模块，推理时依然消耗计算。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（期望速比）　假设未到 EOS，各位置在前面全部接受的条件下接受率均为 0.8，K=4，draft 每步成本为普通 target decode 的 0.08，一次 verify 成本为 1。每轮期望产出和理想净加速比是多少？',
      },
      {
        stage: 'corner',
        question: '追问 2（MoE + 投机采样）：在 MoE 模型（如 DeepSeek-R1）里开 MTP 投机采样，除了常规计算量增加，对 MoE 路由专家激活数量和显存读取有什么意想不到的负面影响？',
      },
    ],
    referenceAnswer: `### 为什么随机采样仍服从目标分布

对同一已接受前缀，p 是要复现的目标分布，q 是草稿实际采样分布，含各自使用的温度、截断和归一化。从 q 抽到 x 时以 min(1,p(x)/q(x)) 接受。首次拒绝时按归一化的正部 (p−q)₊ 抽替换 token，并停止验证此后草稿。每个 x 的总概率为 min(p(x),q(x)) + (p(x)−q(x))₊ = p(x)。若 K 个全收，再从目标分布抽一个 bonus token。该证明针对链式、正确校正的 sampler。贪心验证比较 argmax，树式多候选需相应验证规则。同分布不要求固定随机种子生成同一串。

### 算例与实现

- **期望产出**。当每个仍在验证的位置条件接受率都是 α，忽略 EOS，E=1+α+…+αᴷ。K=4、α=0.8 时为 **3.3616 tokens/轮**。若单次 verify 相对普通 target 步为 v，draft 步成本 c，则理想加速为 E/(v+Kc)。v=1、c=0.08 得 **2.55×**。线上应测 v、实际接受前缀长度和调度成本。
- **候选从哪来**。原始 EAGLE 用目标高层特征及前移的 token 信息做特征级自回归。EAGLE-2 增加随上下文变化的草稿树。EAGLE-3 融合低、中、高层特征并直接预测 token。V3 的 MTP 在预训练时训练顺序预测模块、共享 embedding 和输出头。候选质量和验证规则共同决定收益与分布。
- **KV 和吞吐**。树 mask 只允许看到共同前缀与本分支祖先。保留已接受路径实际计算的 KV，截断无效后缀、释放或复用其页。替换或 bonus token 的 KV 在其真正前向时补齐。高 batch 时验证可能争用算力、KV 或跨机链路。MoE 一轮还可能激活更多不同专家。应随负载比较多种 K（含关闭）的端到端吞吐与 ITL。

> 依据　Fast Inference from Transformers via Speculative Decoding（arXiv:2211.17192）、EAGLE-3（2503.01840）及 DeepSeek-V3 第 2.2 节。`,
  },
  {
    id: 'inf-07',
    track: 'inference',
    title: '超大规模 MoE 推理部署 (DeepSeek-R1 671B)：DP Attention + Wide-EP 与 TBO',
    level: 'P8',
    frequency: '高频必考',
    incident: false,
    tags: ['DeepSeek-R1', 'Wide-EP', 'DP Attention', 'Two-Batch Overlap', 'EPLB'],
    summary: '以 DeepSeek 公开的 EP32/EP144 部署分析 MLA KV、专家负载、双批重叠和冗余副本。',
    contextSpecs: ['671B MoE / 256 routed experts', '示例 P: EP32 / D: EP144', 'Two-Batch Overlap', 'EPLB'],
    prompt: '为 DeepSeek-R1 671B 设计在线推理拓扑。比较纯 TP 与 DP Attention + Wide-EP 对单卡权重、MLA KV、每专家 token 数及跨机通信的影响。以 DeepSeek 公开的 EP32 prefill / EP144 decode 部署为例，解释 Two-Batch Overlap 和冗余专家的作用及边界。',
    rubric: [
      {
        id: 'dp-attn',
        label: '普通 head TP 可能复制 MLA cache。按请求做 DP Attention 改变 KV 和 attention 权重的放置',
        weight: 24,
        keywords: ['dp attention', 'mla', 'kv重复', '复制', '不切分', '显存', 'batch', 'tp'],
        probe: '常见 MLA 的 head TP 怎样存共享潜向量？把不同请求分给不同 attention rank 后，权重和 KV 的放置各怎样变化？',
      },
      {
        id: 'wide-ep',
        label: '区分每卡专家数、每专家 token 数和总 batch，比较专家复制/分片与 All-to-All',
        weight: 24,
        keywords: ['wide-ep', 'ep', '专家并行', '单卡权重', 'grouped gemm', '算术强度', '聚合', 'batch', 'memory-bound'],
        probe: '固定全组 token 数与固定每卡 token 数时，扩大 EP 对每专家批量的影响分别是什么？',
      },
      {
        id: 'tbo',
        label: 'Two-Batch Overlap (TBO / Micro-batch Ping-Pong)：将请求拆成两微批交替执行 Attention/MLP 与 All-to-All Dispatch/Combine 掩盖跨机 RDMA 延迟',
        weight: 20,
        keywords: ['tbo', 'two-batch', 'overlap', 'ping-pong', 'micro-batch', 'dispatch', 'combine', 'deepep', '掩盖', '隐藏'],
        probe: '哪些子阶段可以跨两批执行？当每个微批的专家 GEMM 太小时会有何代价？',
      },
      {
        id: 'eplb',
        label: 'EPLB 复制、重排热点逻辑专家并在等价物理副本间分流，权衡显存及迁移',
        weight: 18,
        keywords: ['冗余', 'redundant', 'eplb', '热点', '负载均衡', '复制', 'straggler', '重排'],
        probe: '在线推理时用户请求话题集中（如大量代码题），某些专家会被疯狂路由成为热点拖垮整批 Decode，推理期又不能改模型权重，怎么解决？',
      },
      {
        id: 'pd-ep-scale',
        label: '解释公开部署的 P/D 分工和通信内核，允许其他满足 SLO 的 EP 规模',
        weight: 14,
        keywords: ['prefill', 'decode', 'pd分离', 'ep32', 'ep144', '320', '低延迟内核', 'low-latency'],
        probe: '公开案例分别取 EP32/144。给定新的到达率和 SLO，怎样选择 P/D 的 EP 度数与通信 kernel？',
      },
    ],
    hints: [
      'L1 分别核算每层共享的 MLA 潜向量、attention 权重和专家权重在每卡上的放置。',
      'L2 均匀 Top-k 路由下，全 EP 组有 T 个 token、E 个逻辑专家时，每专家平均为 kT/E。',
      'L3 对准两个 micro-batch 的计算和 dispatch/combine 时间线，扣掉 GEMM 变小与未隐藏的通信。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（延迟与吞吐）　当业务要求 TPOT <15 ms/token，怎样判断应缩小 EP 组、改变 micro-batch，还是增加较小实例的副本？',
      },
      {
        stage: 'corner',
        question: '追问 2（容错爆炸半径）：Wide-EP 意味着 32~144 张卡强耦合在同一个 All-to-All 通信域里，只要其中 1 张卡挂掉整个大实例就瘫痪了。生产运维中如何降低这种超大 EP 实例的故障爆炸半径？',
      },
    ],
    referenceAnswer: `### 先画出权重和 KV 的归属

671B 的 FP8 权重主体就约 671 GB，还需 scale、非量化参数、KV 与运行空间。TP8 能否装下依 GPU SKU 而定。普通 head TP 在 MLA 下常复制共享潜向量和 RoPE key。按请求做 DP Attention 可让一个请求的 KV 归一个 attention rank，但各 rank 要容纳其 attention 权重，也要平衡请求长度。并发增益需按剩余显存计算。

**公开部署算例**。DeepSeek 2025 年披露的一个部署单元有 256 个逻辑路由专家，再放 32 个冗余专家。Prefill EP32，每 GPU 每个 MoE 层持有 288/32=9 个路由专家实例。decode EP144，每 GPU 持有 2 个。此例两侧 MLA/共享专家都按 DP 部署。EPLB 把同一个逻辑专家的负载分配给相同权重的物理副本，重排时要同步映射和权重。

- **批量与 EP**。全局 T 个 token、Top-8、均匀路由时，每逻辑专家平均 8T/256 个分配。固定 T 而增大 EP 会改变每卡权重与通信，平均每专家 token 数不变。若更大服务域可聚合更多请求，才可能使 Grouped GEMM 更饱满。热点副本可减少最慢 rank 的负担。
- **TBO**。将就绪请求拆成两批，用一批的计算覆盖另一批的 dispatch/combine。Prefill 偏重批量带宽，decode 对小消息延迟更敏感。要比较拆分后 GEMM 效率、在关键路径上暴露的通信与 P99 TPOT。
- **低延迟与故障**。相同流量回放比较较小 EP、多个实例、低延迟通信核及是否开 TBO。缩小故障域的代价可能是额外权重副本。跨机集体通信失败时可把新流量移往独立健康实例，重建组并处理已接收请求。冗余专家本身不能让失效的 collective 自动继续。

> 依据　DeepSeek《DeepSeek-V3/R1 Inference System Overview》，2025 年 2 月公开的部署配置。`,
  },
  {
    id: 'inf-08',
    track: 'inference',
    title: '线上推理服务 SLO 排障：TTFT 飙升、TPOT 抖动与 KV 抢占雪崩',
    level: 'P7',
    frequency: '高频必考',
    incident: true,
    tags: ['SLO排障', 'TTFT', 'TPOT', 'Preemption', 'Prefix Routing'],
    summary: '根据给定高峰告警，用请求级时间线区分排队、长 prefill、KV 抢占和主机暂停。',
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
        label: '由抢占率、重算工作量和有效吞吐共同诊断 thrashing，区分可回收缓存',
        weight: 22,
        keywords: ['thrashing', '抢占', 'recompute', '重算', '驱逐', '有效吞吐', 'max_num_seqs'],
        probe: '为什么流量越高、整体输出吞吐反而“不升反降”？这种反常的吞吐倒挂在 KV Cache 满载时是怎么形成的？',
      },
      {
        id: 'routing-imbalance',
        label: '检查实例间负载倾斜，比较路由亲和性、预期等待与缓存命中的收益',
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
        label: '根据证据选择可回退的准入/路由或预算调整，用 SLO goodput、拒绝率及质量复核',
        weight: 16,
        keywords: ['限流', '准入', 'admission', '拒绝率', '预算', '路由', '灰度', 'slo', '止血'],
        probe: '根据当前证据，哪些可回退的限流或摘流措施能保护在途请求？分别用什么指标判定有效？',
      },
    ],
    hints: [
      'L1 按实例、请求长度和时段拆分，先确认 P99 是每个请求平均 TPOT 还是每次 token 间隔。',
      'L2 对齐长 prompt、抢占重算、主机暂停与 GPU 时间线，区分同时出现和真正的先后依赖。',
      'L3 比较 token 预算、可回收块、排队长度和流量偏斜，用限制新增负载的实验验证抢占假设。',
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
    referenceAnswer: `### 用同一批慢请求画时间线

TTFT 从到达到首 token，包括网关、分词、排队、prefill 及返回。逐 token 毛刺看 ITL。对一个输出 n>1 个 token 的请求，平均 TPOT=(末 token 时刻−首 token 时刻)/(n−1)。请求平均值的 P99 和各次 ITL 的 P99 可能差很多。按实例、prompt/输出长度和到达率分桶，避免用集群平均 GPU util 定根因。

| 观测 | 可以验证的假设 |
|---|---|
| 部分实例等待队列与排队时长上升，其余实例有余量 | 检查路由倾斜、长请求与热点前缀的驻留位置 |
| KV 占用高，且抢占计数增量、重复 prefill 一起升，有效输出下降 | 检查容量压力造成的驱逐与重算。区分可回收 prefix 块和正在引用的块 |
| 未命中 prefill token 增多，慢迭代与 ITL 峰值对齐 | 比较请求长度、前缀命中、chunk 和每轮 token budget |
| GPU 时间线有空档或某个 kernel 变长 | 对齐 CPU 调度、分词、GC、同步、collective 和 graph/eager 路径，再判定等待来源 |

以引擎实际 /metrics 为准。vLLM V1 的例子有 \`vllm:kv_cache_usage_perc\`、\`vllm:num_preemptions_total\`、\`vllm:inter_token_latency_seconds\`。计数器看窗口增量，命中率用同一窗口 hits/queries。

**止血与复核**。过载实例可先减少新入队 token、将超长请求导向有容量的池，测量已接收请求和拒绝率。若证据指向 prefill 干扰，灰度调低预算，同时看 TTFT 代价。热门前缀可在多个实例保留，路由时比较节省的 prefill 时间和额外排队时间。只有在 KV 容量或长期 P/D 干扰确为限制时，再比较 KV 量化或分池，并在相同负载复测 SLO goodput。

> 依据　vLLM《Production Metrics》及 v0.10.2《Optimization and Tuning》。`,
  },
  {
    id: 'inf-09',
    track: 'inference',
    title: 'LLM 推理端到端 Profiling：TTFT/ITL 拆解、GPU-Util 假象与 Host/Sync 瓶颈定位',
    level: 'P7',
    frequency: '高频必考',
    incident: true,
    tags: ['Profiling', 'TTFT/ITL', 'GPU-Util', 'DCGM', 'CUDA Graph', 'Host瓶颈'],
    summary: '揭穿 nvidia-smi GPU-Util 虚高假象，用 DCGM、nsys 和 py-spy 对在线推理服务做全链路微秒级拆解。',
    contextSpecs: ['vLLM / SGLang 线上服务', 'GPU-Util 95% 但吞吐低', 'DCGM SM/DRAM Active', 'Host Sync & TP Comm'],
    prompt: '8×H100 上部署 vLLM/SGLang 推理服务，压测发现 TTFT 与 ITL 都远差于预期，但运维看 `nvidia-smi` 报告 `GPU-Util` 高达 95%，据此认为“卡已经吃满了”。请解释为什么 `GPU-Util 95%` 不能代表算力或带宽饱和，应改看哪些硬件指标？并给出从网关到 GPU kernel 的端到端推理 Profiling 拆解路径，说明如何定位 Host 调度阻塞、隐式 D2H 同步和 TP AllReduce 延迟。',
    rubric: [
      {
        id: 'gpu-util-fallacy',
        label: '拆穿 `nvidia-smi` GPU-Util 口径：仅表示采样窗口内至少有 1 个 kernel 在运行的时间占比，改用 DCGM `SM_ACTIVE`/`TENSOR_ACTIVE`/`DRAM_ACTIVE` 或 `ncu` SOL 衡量真实饱和度',
        weight: 26,
        keywords: ['gpu-util', 'nvidia-smi', '假象', 'dcgm', 'tensor_active', 'dram_active', 'sm_active', '饱和', 'sol', '占比'],
        probe: '如果一个 decode kernel 只用了 132 个 SM 里的 10 个，且 HBM 带宽只跑了 30%，`nvidia-smi` 的 `GPU-Util` 会显示多少？应该改盯哪些真实指标？',
      },
      {
        id: 'e2e-stages',
        label: '端到端延迟分段埋点：网关/HTTP → Tokenizer → Engine Queue → Scheduler & Input Prep → H2D → Model Forward (Attn / GEMM / TP AllReduce) → Sampler & D2H → Detokenizer',
        weight: 22,
        keywords: ['tokenizer', 'scheduler', 'queue', 'h2d', 'd2h', 'sampler', 'allreduce', 'detokenizer', 'nvtx', '分段'],
        probe: '把一个请求 25ms 的单步 ITL 拆开，除了 GPU Forward 本身，哪几个环节最容易悄悄吃掉 5~10ms？',
      },
      {
        id: 'host-sync-trap',
        label: '定位 Host 与同步陷阱：主循环同步等待 Sampler 输出 (`.item()` / `cudaStreamSynchronize`) 打断异步发射流水、Python 调度与 GC、以及未命中 CUDA Graph 回退 Eager',
        weight: 22,
        keywords: ['synchronize', 'd2h', '.item()', 'async', '异步调度', 'cuda graph', 'eager', 'py-spy', 'gc', 'gap'],
        probe: '为什么推理引擎（如 vLLM V1 / SGLang）都强调“Async Scheduling / Zero-Overhead Scheduler”？同步等待上一个 step 的 sampled token 会导致 GPU 时间线出现什么形态？',
      },
      {
        id: 'forward-breakdown',
        label: 'GPU Forward 内部分解（区分 Prefill 与 Decode）：核对 GEMM、Attention (KV 读)、MoE/All-to-All、TP AllReduce 和自定义量化解包各自占比与是否启用 Custom AllReduce',
        weight: 18,
        keywords: ['prefill', 'decode', 'gemm', 'attention', 'custom allreduce', 'p2p', 'nccl', 'kv', 'roofline', '占比'],
        probe: '在 8 卡 TP Decode（batch=16）的 `nsys` trace 里，为什么有时 TP AllReduce 竟然占到单步 GPU 时间的 25%~35%？怎么用 trace 验证是否走了低延迟 Custom AllReduce？',
      },
      {
        id: 'repro-benchmark',
        label: '科学的压测与 Profiling 纪律：固定模型/长度分布/前缀命中率与到达泊松流，对比纯引擎离线吞吐与含 HTTP/前端的端到端延迟',
        weight: 12,
        keywords: ['压测', '泊松', '长度分布', 'prefix', '命中率', '控制变量', '离线', '端到端', 'p99'],
        probe: '做推理 Profiling 时，如果直接用固定 512 长度的重复 prompt 压测，测出来的指标会怎样误导线上判断？',
      },
    ],
    hints: [
      'L1 `GPU-Util` 是“时间维活跃度”而非“空间/硬件资源利用率”——单核占满时间线也能 99%，必须看 DCGM Profiling 指标（`GR_ENGINE_ACTIVE`、`PIPE_TENSOR_ACTIVE`、`DRAM_ACTIVE`）。',
      'L2 用 `nsys` 对齐 CPU 线程和 GPU Stream：如果每步 Decode 尾部固定出现 1~3ms 的 GPU 空白 Gap，且 CPU 侧正卡在 `cudaStreamSynchronize` + Scheduler/Detokenizer，说明未做好异步重叠或发生了隐式 D2H 同步。',
      'L3 小消息 TP AllReduce（几百 KB）是延迟受限（Latency-bound）：检查 trace 中是否启用了 NVLink P2P Custom AllReduce / NVLS，而不是回退到了高启动开销的通用路径。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Async Scheduling 的代价）　如果为了让 GPU 零空档，在 Step $t$ 的采样结果还没回传给 CPU 之前，CPU 就提前把 Step $t+1$ 的 kernel 发射进 GPU 队列（Async Scheduling），那遇到请求恰好在 Step $t$ 吐出 EOS 或触发 stop string 怎么办？',
      },
      {
        stage: 'corner',
        question: '追问 2（Prefill vs Decode 混部 Profiling）　开启 Chunked Prefill 后，同一张 `nsys` trace 里既有 Prefill token 又有 Decode token 混在同一个 GEMM 和 Attention 核里。如何分别定位到底是谁拉慢了 ITL？',
      },
    ],
    referenceAnswer: `### 先拆穿 \`nvidia-smi\` GPU-Util 95% 的假象

\`nvidia-smi\` 的 \`GPU-Util\` 仅表示最近采样周期内“至少有一个 CUDA kernel 在 GPU 上执行”的时间比例，**完全不反映 SM 占用、Tensor Core 利用率或 HBM 带宽饱和度**。哪怕一个只占 8 个 SM 的小算子或低效通信等待核持续占着时间轴，它照样报 95%~100%。
- **正确硬件指标（DCGM / \`ncu\`）**：
  - Prefill / 大 Batch 线性层看 **\`PIPE_TENSOR_ACTIVE\`**（Tensor Core 活跃度）与实测 TFLOP/s；
  - Decode 阶段重点看 **\`DRAM_ACTIVE\`**（HBM 带宽利用率，健康 Decode 通常应靠近实测带宽上限的 60%~80%+）与 **\`SM_ACTIVE\`**；
  - 核对实际有效输出 tokens/s 与单卡 Roofline 上界的差距。

### 端到端推理延迟拆解与定位路径

用统一 Request ID 串联四层工具（网关日志 + 引擎 NVTX/Prometheus + \`nsys\` + \`py-spy\`）：
1. **引擎外与前端**：HTTP 反序列化 → Tokenizer → Waiting Queue。若离线直调引擎很快、走网关很慢，优先查多进程/异步 Frontend、Tokenizer CPU 争抢或网关路由倾斜。
2. **Host 调度与 D2H 同步气泡（高频元凶）**：在 \`nsys\` GPU Stream 上，若每步 Decode 之间有规则空白 Gap（如 1~5 ms），顺着对应 CPU 线程常能看到：
   - Sampler 结束后调用 \`.item()\`/\`.tolist()\` 触发隐式 \`cudaStreamSynchronize\`，把 GPU 队列彻底排空再由 Python 跑下一轮 Scheduler；
   - Batch Size 或序列形态未命中预捕获的 **CUDA Graph** 回退到 Eager Mode，产生上百个微秒级小 kernel launch 锯齿；
   - Python 主进程 GC 或 Detokenizer 阻塞事件循环（用 \`py-spy\` 火焰图立即可见）。
3. **GPU Forward 内部分解**：在单步稳态 Decode 中汇总四类 kernel 耗时占比——**Linear GEMM（含反量化）**、**Paged/FlashAttention（读 KV）**、**TP AllReduce / MoE All-to-All**、**Norm/RoPE/Sampler 等零散算子**。小 Batch 8 卡 TP 中，单次 AllReduce 仅几百 KB、属延迟受限；检查 trace 是否命中机内 NVLink P2P Custom AllReduce/NVLS，避免误走高延迟路径或跨机通信。

### 两项追问

- **Async Scheduling 提前发射**：CPU 提前准备并下发 Step $t+1$ 的计算（通过 GPU 端缓冲直接读取 Step $t$ 产出的 token ID 作为下一轮输入，省去 Host 等待），将当前步调度/Detokenize 完全藏在 GPU 前向后面；若 Step $t$ 实际已命中 EOS/stop token，则丢弃 Step $t+1$ 多算出的 1 个冗余 token 并在下一轮释放槽位，用极小算力换掉整段 Host 同步空档。
- **混部归因**：对每个 step 记录 \`num_prefill_tokens\`、\`num_decode_tokens\` 与 \`chunk_size\`，按是否有 prefill token 混入分组统计 ITL 分布；结合 NVTX 区分 prefill attention 与 decode attention 核时长，再扫不同 \`max_num_batched_tokens\` 画 TTFT–ITL 帕累托曲线。

> 依据　NVIDIA DCGM Profiling Metrics 文档、vLLM V1 架构设计与 SGLang Zero-Overhead Scheduler 说明。`,
  },
];
