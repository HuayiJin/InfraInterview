export const RL_QUESTIONS = [
  {
    id: 'rl-01',
    track: 'rl',
    title: 'PPO vs GRPO vs DPO 的系统显存开销、算力瓶颈与工程演进',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['PPO', 'GRPO', 'DPO', 'RLHF', 'Rollout瓶颈'],
    summary: '在给定精度与参数规模下比较 PPO、GRPO 和离线 DPO 的状态开销，并拆解采样与更新耗时。',
    contextSpecs: ['Actor / Critic / Ref / RM', '同 prompt 采样 G 条', 'BF16 / FP32 Adam 示例', '规则或模型奖励'],
    prompt: '从系统角度比较经典 RLHF-PPO、GRPO 和离线 DPO。分别列出需要训练、只做前向以及可以预计算的状态。若角色均为 70B 且独立常驻，给出一组显存算例。GRPO 去掉 critic 后，哪些工作负载会让 rollout 成为瓶颈？怎样验证？',
    rubric: [
      {
        id: 'ppo-4models',
        label: '经典四个独立等大模型的特例为 36Ψ 字节，区分各角色、精度、常驻量与单卡峰值',
        weight: 24,
        keywords: ['actor', 'critic', 'reference', 'reward', 'rm', '四模型', '36ψ', '主权重', 'adam', 'optimizer'],
        probe: '在四个独立等大的 70B 模型且采用题设精度时，状态合计多少？单机峰值还需要知道哪些布局条件？',
      },
      {
        id: 'grpo-no-critic',
        label: 'GRPO 按同一 prompt 的 G 个回答构造组内相对优势，可使用规则奖励或学习式 RM',
        weight: 24,
        keywords: ['grpo', '去掉critic', '无critic', 'group', 'g个', '组内', '相对', 'advantage', 'verifier', '规则奖励', '省显存'],
        probe: 'GRPO 怎么在没有 Critic Value Model 的情况下估计 Advantage？这为显存和系统架构省掉了哪些组件？',
      },
      {
        id: 'rollout-bottleneck',
        label: '按 G、生成长度和有效 token/s 估算 rollout 工作量，用阶段计时定位实际瓶颈',
        weight: 22,
        keywords: ['rollout', '推理', '长cot', '自回归', '瓶颈', '生成', 'kv cache', '吞吐', '长尾', 'token/s'],
        probe: '给定 G、生成长度分布和实测生成速度，怎样判断一步训练被 rollout、验证器还是更新所限制？',
      },
      {
        id: 'dpo-offline',
        label: 'DPO 离线对比：无需在线 Rollout 与 Critic/RM，仅需 Policy + Ref（Ref 甚至可离线预计算 logprob），但缺乏 On-policy 探索能力',
        weight: 16,
        keywords: ['dpo', '离线', 'offline', '无需rollout', 'ref', '预计算', 'on-policy', '探索', 'sft'],
        probe: '在已有离线偏好对与需要对当前策略新生成的回答打分两种情况下，训练流水线如何不同？',
      },
      {
        id: 'shared-prefix',
        label: '同一模型版本下可共享相同 prompt 的 KV 块，独立后缀仍需计算和存储',
        weight: 14,
        keywords: ['prefix', '共享', 'copy-on-write', 'radix', 'kv复用', 'batched', 'prefill'],
        probe: 'GRPO 对同一个 Prompt 一次采样 G=16 或 64 条回答，在推理引擎侧有什么天然的显存和算力优化机会？',
      },
    ],
    hints: [
      'L1 分别列每个角色的参数量、是否训练、精度、是否复用，再看各阶段同时驻留哪些状态。',
      'L2 256 个 prompt、G=16、每条生成 8192 tokens 时，共生成 33,554,432 tokens。墙钟占比还要知道生成与更新速率。',
      'L3 相同 prompt 的只读 KV 块可复用，收益取决于引擎 fork/前缀缓存、放置和缓存命中。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Ref 开销）　若 GRPO 目标含固定参考策略的 KL 项，671B 的 Ref 有哪些复用或分时方案？若该项系数为零呢？',
      },
      {
        stage: 'corner',
        question: '追问 2（DAPO / Dr.GRPO 变体）：GRPO 训练中，不同样本的生成长度从 500 到 32K 剧烈波动，如果直接在 Token 维度或 Sequence 维度做 Loss 归一化，加上 Packing 训练，Infra 在梯度累积和 Loss Mask 上容易踩什么坑？',
      },
    ],
    referenceAnswer: `### 用同一套假设计显存

取四个独立等大模型、参数量均为 Ψ。可训练模型按 BF16 权重/梯度、FP32 主权重与两个 Adam 矩计 16Ψ 字节，冻结模型只存 BF16 权重计 2Ψ。下表是全模型状态总量，未计 rollout 副本、激活、KV、分片临时区。

| 方法 | 角色和工作 | 示例总量 |
|---|---|---|
| 经典 RLHF-PPO | Actor 和 Critic 训练，Ref 约束策略，RM 给奖励。需要采样动作并估计价值/优势 | 16Ψ+16Ψ+2Ψ+2Ψ=36Ψ。四个 70B 共 2.52 TB |
| GRPO | 同一 prompt 采 G 个回答，用组内奖励构造优势，常见 Aᵢ=(rᵢ−组均值)/(组标准差+ε)，无需 value critic | 用规则奖励且启用冻结 Ref 时为 18Ψ。若依赖 RM 要另加其权重。组内同等奖励时该组相对奖励信号为零 |
| 离线 DPO | 对固定 chosen/rejected 对做 teacher-forcing，用策略与冻结参考的回答 logprob 差构造偏好目标 | 16Ψ+2Ψ。固定数据、模板与 Ref 后可预计算参考 logprob，更新时只驻留策略 |

PPO 可共享 actor/critic 骨干，RM 规模也可不同。优化器分片、共置/分离和分时加载决定每卡峰值。有 KL 项时，GRPO 可对新生成的轨迹批量计算冻结 Ref，再换出。仅当冻结 base 恰好等于所需 Ref 时，禁用 LoRA 才可复用它。KL 系数为零时，此项不需 Ref 前向。

### 工作量与归一化

256×16×8192=33,554,432 个生成 token。Prefill 可复用相同 prompt 的 KV，分叉后的 decode 仍逐 token 执行。应分开记录 rollout、奖励、ref、actor 更新、换权与组内长尾，生成量本身不能推出耗时占比。

Packing 必须保留样本边界、action mask 和 group ID。按样本平均会给每条回答相同总权重。DAPO 类 token 平均以全局有效回答 token 数作分母。Dr.GRPO 的一种实现按样本数乘固定长度归一化。跨 DP 和梯度累积要聚合相应分子分母，计及 DDP 对梯度的平均，避免把不等长 micro-batch 的均值再等权平均。

> 依据　DeepSeekMath（arXiv:2402.03300）、DPO（2305.18290）及 TRL v0.26.0《GRPO Trainer》。`,
  },
  {
    id: 'rl-02',
    track: 'rl',
    title: 'RL 系统架构：训练与 Rollout 共置或分池，同步或异步',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['veRL', 'OpenRLHF', 'Colocated', 'Disaggregated', 'HybridEngine', 'Ray'],
    summary: '比较角色的物理放置与策略版本的更新节奏，估算 GPU 空闲、换权、积压和样本陈旧。',
    contextSpecs: ['veRL 3D-HybridEngine', 'OpenRLHF / Ray', 'Colocated vs Disaggregated', 'GPU Bubble'],
    prompt: '训练和 rollout 可以分时共用一组 GPU，也可以放在独立 GPU 池。请独立考虑“物理共置/分池”和“同步/异步采样更新”，结合 veRL 或 OpenRLHF 的配置举例，比较显存、权重重分片、负载配比、异构卡和策略陈旧度。',
    rubric: [
      {
        id: 'colocated-mech',
        label: '共置时同卡分时运行，可按实际余量使用 reshard、offload、sleep',
        weight: 24,
        keywords: ['colocated', '共置', '分时', 'verl', 'hybridengine', 'offload', 'sleep', 'reshard', '无配比', '气泡'],
        probe: 'veRL 把训练和推理塞进同一批 GPU，切换阶段时显存不会爆吗？它是靠什么机制在一组卡上无缝切换两种引擎的？',
      },
      {
        id: 'disagg-mech',
        label: '分池时可独立选并行度和硬件，是否异步由版本/调度协议决定',
        weight: 24,
        keywords: ['disaggregated', '分离', 'openrlhf', 'ray', '独立', '异步', 'async', '流水', 'pipeline', '广播', 'broadcast'],
        probe: '严格按版本同步时，两个独立池怎样产生空闲？容许哪些重叠仍保持轨迹使用同一行为版本？',
      },
      {
        id: 'tradeoff-bubble',
        label: '同一策略版本上的等待、共置切换成本、异步吞吐与数据陈旧分别计量',
        weight: 22,
        keywords: ['同步', '异步', 'on-policy', 'off-policy', '长尾', '切换开销', '扩展性', '利用率', '配比'],
        probe: '如何从卡数、阶段耗时、换权时间和队列积压测出分池是否更划算？',
      },
      {
        id: 'topo-mismatch',
        label: '两类放置均可需要布局转换，区分本地分片重排及跨机广播',
        weight: 16,
        keywords: ['并行度', '拓扑', 'tp', 'pp', 'fsdp', 'megatron', 'ep', '重分片', 'resharding'],
        probe: '训练和推理的最佳并行配置通常不一样（比如训用 TP4+PP2+DP8，推用 TP4×16 副本），两种架构各自怎么处理权重映射？',
      },
      {
        id: 'hetero-hw',
        label: '分池便于异构资源独立配置。比较各阶段可用内核、速度与成本',
        weight: 14,
        keywords: ['异构', 'h20', 'h100', '成本', '卡型', '内核', '计算吞吐', '带宽', 'roi'],
        probe: '如果公司手里同时有大批便宜的高显存推理卡（如 H20/L40S）和少量 H100，哪种架构能把异构算力吃干抹净？',
      },
    ],
    hints: [
      'L1 先画 GPU 放置，再画策略版本时间线。二者可以独立选择。',
      'L2 共置切换时要核算同时存在的权重、优化器、KV 和临时缓冲，按需释放、换出或复用。',
      'L3 分池严格串行时忙碌 GPU·秒是各角色卡数乘各自忙碌时间之和，除以总 GPU·秒即可估算利用率。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Ray Control Plane 瓶颈）：veRL 和 OpenRLHF 底层都广泛使用 Ray 做分布式控制面（Single Controller vs Multi Controller）。当集群扩到 2048 卡、单步生成几万条 32K 轨迹数据时，Ray Object Store 和 Driver 节点很容易 OOM 或成为调度瓶颈，怎么规避？',
      },
      {
        stage: 'corner',
        question: '追问 2（超大规模同步代价）：如果采用 Disaggregated 架构，671B 模型做完一次 Train Step 后，需要把最新权重同步给 512 张 Rollout 卡，万一同步期间 Rollout 还有长尾请求没吐完怎么办？',
      },
    ],
    referenceAnswer: `### 两个独立的设计决定

共置/分池描述 GPU 放置。同步/异步描述策略何时发布、哪些样本允许训练消费。veRL 的 HybridFlow 给出共置和 3D-HybridEngine 方案，OpenRLHF 也支持 Hybrid Engine 共置与异步分池。Ray 是编排层，框架名字不足以判定架构。

| 维度 | 共置、分时 | 独立资源池 |
|---|---|---|
| 容量与换权 | 同一批卡可轮流用于生成和训练，需预算 sleep/offload、重分片和转换峰值 | 两组可各自常驻，分别选择 TP/PP/EP 和卡型。发布版本要传输到 rollout 副本 |
| 同步更新 | 一轮所有样本都可绑定同一旧策略，切换和最后几条长轨迹会占用墙钟 | 简化为两段严格串行时，忙碌占比为 (R·tᵣ+T·tₜ)/[(R+T)(tᵣ+tₜ)]。R、T 为两池卡数，各池内部先假设全忙 |
| 流水和配比 | 分时或给部分角色留出资源均可，收益受可用空间约束 | 产训可重叠，稳态受较慢阶段限制。用样本生产/消费率、队列长度、策略版本差做背压 |

**选型**。在相同训练目标和有效样本数下比较 GPU·小时、奖励/质量、换权时长及两池空闲。严格同步并且换相便宜时，共置有吸引力。长工具调用、异构硬件、负载变化或可控的策略滞后可能使分池更合适。异构时分别测训练卡的更新速度与 rollout 卡的显存、带宽、所需精度内核和达标成本。

**控制面追问**。将大对象按 worker 分片、采用对象引用或直接数据通道、限制在途对象数和引用寿命。driver 避免集中收集所有长轨迹。**换权追问**。先停止接纳旧版本新请求，排空、分版本双缓冲或按约定暂停并重建状态。全部目标 rank 就绪后再发布。严格同步的训练批检查行为版本，异步方案额外记录实际行为 logprob 并设消费边界。

> 依据　HybridFlow（arXiv:2409.19256）及 OpenRLHF《Hybrid Engine》《Async Training & Partial Rollout》。`,
  },
  {
    id: 'rl-03',
    track: 'rl',
    title: 'RL 训练-推理引擎高频权重同步 (Weight Sync) 与 3D Resharding 加速',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['Weight Sync', 'Resharding', 'CUDA IPC', 'NCCL Broadcast', 'FP8 Rollout'],
    summary: '按训练/推理分片映射设计权重发布，估算链路下界并验证布局、缓存与策略版本。',
    contextSpecs: ['Megatron/FSDP → vLLM/SGLang', 'TP4/PP2/DP8 → 16×TP4', 'CUDA IPC / NCCL', 'BF16 → FP8'],
    prompt: 'Actor 更新后，rollout 引擎要按约定版本接收训练权重。以 72B 为例，设计同卡共置及跨机分离的权重同步，说明不同 TP/PP/DP 布局如何映射、如何控制转换峰值、如何估算时间，以及 BF16 训练到 FP8 推理时如何保证可验证的一致性。',
    rubric: [
      {
        id: 'topo-reshard',
        label: '按全局层、张量分片与物理拓扑规划来源和去向，估计必要的跨机流量',
        weight: 24,
        keywords: ['reshard', '重分片', '3d-hybridengine', 'tp', 'pp', 'dp', 'nvlink', '机内', 'allgather', '拓扑'],
        probe: '训练 TP4/PP2/DP8 转成 16 个 TP4 推理副本时，哪几块参数本地已有？要从谁取其余层？',
      },
      {
        id: 'bucket-stream',
        label: '有界 bucket 流式重排 QKV/GateUp/专家布局，控制显存，完整校验后发布',
        weight: 22,
        keywords: ['bucket', '桶', '流式', '缓冲', '重排', 'qkv', 'gate/up', '转换', 'oom', '映射'],
        probe: 'FSDP/Megatron 的权重切分格式、以及 QKV/GateUp 投影矩阵拼接方式和 vLLM 不一样。如果先一次性把 72B 参数全部 AllGather 拼好再喂给 vLLM，会发生什么灾难？怎么解决？',
      },
      {
        id: 'ipc-nccl',
        label: '区分同进程拷贝、同机 CUDA IPC 与跨机 NCCL 等传输，解释同步和生命周期',
        weight: 20,
        keywords: ['cuda ipc', 'nccl', 'broadcast', '零拷贝', '指针', '同进程', 'rdma', 'ray', '显存拷贝'],
        probe: '同机共置时，如果训练和推理跑在不同进程（比如 Ray Actor），不用磁盘也不用 CPU 拷贝，怎么直接 GPU-to-GPU 传权重？',
      },
      {
        id: 'fp8-quant-sync',
        label: '比较源侧/目标侧量化的字节、scale 布局和行为策略误差',
        weight: 18,
        keywords: ['fp8', '在线量化', 'quant', '减半', 'scale', '精度', 'mismatch', '重要性采样'],
        probe: '如果训练是 BF16、推理开 FP8，应该先传 BF16 再在推理侧量化，还是先量化再传？这会引入什么算法层面的副作用？',
      },
      {
        id: 'lora-delta',
        label: '固定基座时可只发布变动的 adapter 和相关状态，按实际字节与链路估时',
        weight: 16,
        keywords: ['lora', '增量', 'delta', 'adapter', '毫秒', '轻量'],
        probe: '哪些条件下仅传 LoRA 参数即可完整恢复新策略？如何估算同步量与耗时？',
      },
    ],
    hints: [
      'L1 方向提示：记住三个关键词——**局部通信域对齐 (3D-HybridEngine)**、**按 Bucket 流式转换防 OOM**、**GPU 直连通道 (同进程指针/CUDA IPC/NCCL Broadcast)**。',
      'L2 按内存余量定 bucket，流水完成收集、布局转换、传输和写入。校验完成后再对请求开放新版本。',
      'L3 72B BF16 的逻辑权重约 144 GB。如果全部跨单条 400 Gb/s 链路，理想下界 2.88 秒。多个接收端的时间还取决于复制拓扑。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（CUDA Graph）　若 replay 引用了旧参数地址，更新权重时怎样避免在途读写竞争？更换地址后怎样恢复有效图？',
      },
      {
        stage: 'corner',
        question: '追问 2（异步流式更新）：在 Disaggregated 异步 RL 中，Rollout 池正在生成半截的 Trajectory，此时新权重同步过来覆盖了旧权重，一条 Trajectory 前半段是 $\\pi_{t-1}$ 生的、后半段是 $\\pi_t$ 生的，工程和算法上怎么处理？',
      },
    ],
    referenceAnswer: `### 为每个目标分片列出来源

以 TP4/PP2/DP8 → 16 个 TP4 副本为例。训练时一个 PP stage 只有部分层，同一 TP 号只持有对应切片。若 DP 参数也分片，先在其分片域内补齐所需范围。推理副本的每个 TP rank 都要覆盖全层。若单个训练副本的 8 卡位于同机且训练权重在 DP 间复制，两个 PP stage 可按相同 TP rank 互换缺失的层，在机内组成两个完整 TP4 rollout 副本。若 PP 横跨节点，需按实际放置选择来源。按全局层号、Q/K/V 与 GQA head 顺序、Gate/Up、专家编号和量化布局重排，逐桶验证形状、元素和版本。

- **有界传输**。桶大小由空闲显存、带宽和启动延迟决定。把接收、转换、传下一桶重叠，限制在途缓冲数。同进程且布局兼容时可复用或拷贝。同机 CUDA IPC 可共享合格分配的句柄，需保证所有者生命周期和流同步。跨机可复用 NCCL 通信组或 RDMA 传输。
- **时间下界**。72B BF16 主体 144 GB，在单条 400 Gb/s 链路上全量传一次至少 2.88 秒。若源端先量化到 FP8，载荷约 72 GB 外加 scales，下界接近 1.44 秒。多 rail 和分层广播按最忙源、目标链路与复制字节重新算。还要加转换和提交时间。若先传 BF16 可由目标按本地 kernel 生成正确的块 scale。
- **原子发布**。等待在途读取结束再写原地址，或为版本准备独立缓冲及其匹配的图。所有 rank 就绪后统一切版本。更换捕获地址时重捕获或使用后端支持的图更新。旧权重产生的 KV 要按版本隔离，策略变化后续写通常需要用新权重重算前缀。版本标签可追溯，单靠打标签不能修复半更新。
- **验证**。固定 token 做 teacher-forced logprob 对照，并保存行为策略的实际采样概率。LoRA 仅在两边 base、配置、量化格式及合并状态一致时可只发更新的 adapter（及其他已训练参数），用真实可训练参数量乘元素字节估算。

> 依据　HybridFlow（arXiv:2409.19256）及 NVIDIA CUDA Programming Guide 的 IPC / CUDA Graphs 章节。`,
  },
  {
    id: 'rl-04',
    track: 'rl',
    title: '长 CoT 强化学习最大痛点：Rollout 长尾 (Straggler) 与 GPU 空转优化',
    level: 'P8',
    frequency: '高频必考',
    incident: false,
    tags: ['长CoT', 'Straggler', 'Partial Rollout', 'Async RL', 'Staleness'],
    summary: '解决 o1/R1 类 Reasoning RL 中“95% 样本 2K 结束，5% 样本吐到 32K 拖垮全集群”的致命长尾难题。',
    contextSpecs: ['CoT 1K–32K tokens', '同步批尾部空闲', 'Partial rollout', '策略版本与行为概率'],
    prompt: '【高频真实痛点】你在训练一个类似 DeepSeek-R1 / OpenAI o1 的长思维链（最大长度 32K）GRPO 模型。监控发现 Rollout 阶段前 20 秒所有 GPU 满载跑得很欢，但后面整整 100 秒里，95% 的样本已经生成完毕，整个 512 卡集群全在干等最后寥寥几条 32K 的超长样本吐完，导致端到端 GPU 利用率跌破 25%！请给出从纯同步优化到半异步/全异步的多层系统解法。',
    rubric: [
      {
        id: 'straggler-root',
        label: '以活跃序列、逐 token 延迟和 GPU·秒识别同步屏障和自回归长尾',
        weight: 18,
        keywords: ['屏障', '同步更新', 'barrier', '批尾', '活跃序列', '尾部', '等待 gpu', 'active sequences', 'straggler', '长尾'],
        probe: '怎样区分集群有效 token/s 下降与单个长请求 ITL 上升？',
      },
      {
        id: 'partial-rollout',
        label: '暂停并保存 token/采样状态，区分同版本续写与更新后跨版本轨迹',
        weight: 26,
        keywords: ['partial rollout', '截断', 'interruption', '续写', 'pause', 'kimi', 'areal', '跨步', 'checkpoint', '挂起'],
        probe: '如果到了时间上限，最后那 5% 还没生成完，直接暴力丢弃会破坏长度分布（惩罚长思考）。Kimi k1.5 / AReaL 的 Partial Rollout 是怎么做到“既不丢弃长思考，又不空等”的？',
      },
      {
        id: 'async-pipeline',
        label: '异步 / 半异步流水线（One-step Off-policy / Fully Async）：解耦 Rollout 生产与 Train 消费，通过 Trajectory Buffer 持续填满 Inference Batch',
        weight: 22,
        keywords: ['异步', 'async', 'off-policy', 'buffer', '队列', '生产者', '消费者', 'one-step', '流式', 'continuous'],
        probe: '改成异步生产者-消费者架构后，每当一条短请求结束，推理引擎立刻补进新 Prompt 永远保持高 Batch。但这会带来什么算法风险（Staleness）？怎么控？',
      },
      {
        id: 'staleness-control',
        label: '保存行为概率和策略版本，限制陈旧、测量 ratio 分布，说明截断或过滤的偏差',
        weight: 18,
        keywords: ['staleness', '陈旧', '版本差', 'importance sampling', '重要性采样', 'clip', 'ppo clip', 'off-policy', '策略偏移'],
        probe: '滞后多个版本的轨迹应该记录什么概率？用更新后的训练引擎重算能恢复生成时的行为概率吗？',
      },
      {
        id: 'sync-tricks',
        label: '同策略动态分发和严格等价的投机验证。识别过采样择快、提前截断造成的选择偏差',
        weight: 16,
        keywords: ['过采样', 'over-sampling', 'oversampling', '重复', '死循环', 'repetition', '动态', '负载均衡', '投机采样', 'speculative'],
        probe: '若必须保留同一行为策略和原有样本分布，哪些调度优化可用，哪些停止规则会改变数据？',
      },
    ],
    hints: [
      'L1 逐实例记录 active sequences、tokens/s、ITL 和尾部等待 GPU·秒，再决定是调度不均还是单轨迹过长。',
      'L2 在预定样本集合内动态分发，或使用保持行为分布的投机验证。丢弃慢样本与额外早停都会改变数据。',
      'L3 跨步保存 token IDs、行为概率、奖励/环境状态与版本。同版本可校验 KV 后续写。换权后重算前缀或固定旧副本。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（严防长度偏置）：有同学提出“Over-sampling 1.5 倍 Prompt，只要最快的那 1.0 倍生成完了就把剩下 0.5 倍全部 Abort 丢弃”，这样确实消除了长尾，但上线跑了一天模型能力暴跌，为什么？',
      },
      {
        stage: 'corner',
        question: '追问 2（尾部投机）　如何按活跃 batch、接受长度和目标模型验证时间决策是否启用 MTP/EAGLE，同时保持原采样分布？',
      },
    ],
    referenceAnswer: `### 区分批尾空闲和慢请求自身延迟

同步更新要等指定采样组完成。批尾有些实例无工作，其余实例能共享权重读取的 token 也变少。整体 token/s 通常下降，但单请求 TPOT 是否变慢要看其历史 KV 长度和 kernel。按实例记录 active sequences、ITL、最晚完成时间与等待 GPU·秒。

1. **保持同一行为策略**。在预先选定的样本内用全局队列动态分派，长任务尽早开跑。低活跃 batch 若可用严格验证的投机采样，比较接受长度、verify 成本和 RNG/采样器语义。原样本都已分出后，动态调度无法缩短最后一条轨迹的串行依赖。只收最快的 G 条会偏向短回答，也会改变 GRPO 组内基准。额外早停需要定义奖励与截断语义。
2. **Partial rollout**。在预算边界暂停，保存 token IDs、采样/RNG 状态、已发生动作的行为 logprob、策略版本及工具/奖励状态。若同版本续写，可复用兼容 KV 或按相同前缀重算。若换了权重，应重算前缀 KV 或在旧副本完成。后续动作由新版本产生时应逐段记录。只有已满足奖励与优势定义的组才能直接用于原 GRPO 目标。
3. **异步分池**。完成的轨迹放入有界队列，按同 prompt 的组聚合并限制最大滞后。监控版本差、完成长度/奖励分布、重要性比值和有效样本量。训练可选择滞后界限与截断校正。过滤可能选择性丢掉长样本，截断引入偏差，PPO clip 也不保证无偏。比较同一奖励预算下的训练质量和端到端有效 token/s。

> 依据　Kimi k1.5（arXiv:2501.12599）、AReaL（2505.24298）和 veRL《Rollout Correction》。`,
  },
  {
    id: 'rl-05',
    track: 'rl',
    title: 'RL 共置架构显存共存：vLLM Sleep Mode、CUDA VMM (CuMem) 与 Optimizer Offload',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['Sleep Mode', 'CuMem', 'CUDA VMM', 'CPU Offload', '显存碎片'],
    summary: '区分活跃显存、分配器缓存与 CUDA 虚拟地址，核算 sleep/offload 的 GPU 峰值、主机容量和传输时间。',
    contextSpecs: ['vLLM Sleep Level 1/2', 'cuMemUnmap / VMM', 'Pinned Memory Offload', 'CUDA Graph 保留'],
    prompt: '同一张 80 GB H100 分时运行训练引擎与 vLLM。rollout 后即使调用 torch.cuda.empty_cache()，训练也可能 OOM。解释 CUDA VMM 的地址保留与物理分配、vLLM 两个 sleep level、优化器 offload 和 CUDA Graph 恢复各自需要哪些条件。',
    rubric: [
      {
        id: 'empty-cache-limit',
        label: '指出 empty_cache() 局限：只能释放空闲未引用缓存块；推理引擎的 KV Cache 池、权重和 CUDA Graph 绑定的虚拟地址仍被强引用无法释放，且跨进程/自定义分配器不可见',
        weight: 22,
        keywords: ['empty_cache', 'caching allocator', '虚拟地址', '引用', 'kv cache', 'cuda graph', '显存碎片', '无法释放'],
        probe: '为什么明明 Rollout 已经跑完了，调 `torch.cuda.empty_cache()` 显存几乎没降下来？哪些大块显存是 PyTorch Caching Allocator 根本收不走的？',
      },
      {
        id: 'cuda-vmm',
        label: 'CUDA VMM 可保留 VA，解除映射并释放物理 allocation 的引用。唤醒要重新映射、设访问权限',
        weight: 28,
        keywords: ['cumem', 'vmm', '虚拟地址', '物理显存', 'unmap', 'map', '解绑', '保留地址', 'va', 'virtual memory'],
        probe: '怎样在保留 CUDA Graph 所用虚拟地址的同时回收底层物理显存？重新映射之后还要恢复什么？',
      },
      {
        id: 'sleep-levels',
        label: 'vLLM v0.10.2 的 L1/L2 都丢 KV。L1 备份权重，L2 丢权重但可保存模型 buffers',
        weight: 22,
        keywords: ['sleep', 'level 1', 'level 2', 'offload', '丢弃', 'discard', '唤醒', 'wakeup', '覆写'],
        probe: '再次运行同一模型与准备接收一版新权重时，两个 sleep level 各有何代价？',
      },
      {
        id: 'train-offload',
        label: '区分 Adam m/v 与 FP32 主权重，按需 offload 本地分片',
        weight: 16,
        keywords: ['主权重', 'adam', 'optimizer', 'offload', '分片', 'pinned', 'pcie', '训练侧'],
        probe: '反过来，在训练完要切回 Rollout 之前，训练引擎自己把显存占满了，推理没地方放 KV Cache，训练侧需要 Offload 哪些大头？',
      },
      {
        id: 'pcie-perf',
        label: '按本机拥有的状态、暂存和共享 PCIe/NUMA 带宽估算 Host 峰值及换相时间',
        weight: 12,
        keywords: ['pcie', '主机内存', 'host', 'ram', 'pinned', 'numa', '带宽', 'oom'],
        probe: 'Offload 时很多人忽略了 Host CPU 内存也会 OOM！8 卡跑 72B 时主机内存至少要配多大，D2H/H2D 传输怎么提速？',
      },
    ],
    hints: [
      'L1 empty_cache 只能归还当前分配器可释放的缓存。查看活跃张量、图池和其他进程的占用。',
      'L2 vLLM v0.10.2 的 Level 2 丢弃模型权重和 KV，但会保留部分模型 buffers 到 CPU。新权重写入完成后才可接请求。',
      'L3 保留 VA 只是重放的一个条件，同时确认权重内容、访问权限、形状、工作区和 stream 同步。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Host 内存与 PCIe 时间估算）：单机 8×H100 (每卡 PCIe Gen5 x16 约 50 GB/s 实测)，假设每张卡需要 Offload 40GB Optimizer States 到 Host 内存。理想情况 D2H 需要多久？什么情况下会远慢于理论值（比如 NUMA / Non-pinned / PCIe Switch 争抢）？',
      },
      {
        stage: 'corner',
        question: '追问 2（显存碎片与 PyTorch Expandable Segments）：即使把大头都释放了，PyTorch 训练和推理反复交替几百步后，为什么偶尔还是会报 `CUDA out of memory (tried to allocate 2GB, 15GB reserved but unallocated)`？怎么治？',
      },
    ],
    referenceAnswer: `### 地址还在，不等于内容还在

- \`torch.cuda.empty_cache()\` 释放当前 PyTorch allocator 可归还的空闲缓存。被权重、KV 池或图引用的活跃分配及其他进程的私有池，需由各自所有者管理。先区分 allocated、reserved、驱动空闲及其他进程用量。
- CUDA VMM 可用 \`cuMemAddressReserve\` 保留 VA。先等待在途使用结束，再 unmap 并释放物理 allocation 的引用。最后一个映射和句柄释放后，物理资源才能回收。唤醒时重新分配、映射回原 VA、设置访问权限，并恢复内容。图重放还依赖已捕获的地址、形状、缓冲和同步关系。
- **vLLM v0.10.2**。Level 1 把权重备份到 CPU，丢弃 KV。Level 2 丢弃旧权重与 KV，模型 buffers（例如 RoPE scaling tensors）仍可备份到 CPU。Level 2 适合即将加载新权重的场景。恢复必要 buffers、完成全 rank 同一版本换权，重算或作废旧前缀后才继续生成。

### Host 空间和换相时间

Adam 的两个 FP32 矩为 8Ψ 字节，若还换出 FP32 master weight 则合计 12Ψ。72B 全模型合计 864 GB。单机需求取决于该机持有的分片、复制和同时在途的 pinned staging，需计主机其他进程。每卡 40 GB、单向实测带宽 50 GB/s，单卡 D2H 理想 **0.8 秒**。8 卡各有独立 50 GB/s 且主机能同时写 400 GB/s 时，整机搬 320 GB 也是 0.8 秒。共用 50 GB/s 瓶颈时则至少 **6.4 秒**。唤回还要计 H2D。

反复切换 OOM 时检查 memory snapshot、活跃引用、图池、碎片和 allocator 外分配。对可变大小造成的碎片可对照测试 expandable_segments。若峰值活跃状态已超容量，应改分片、并发行数或 offload 计划。

> 依据　vLLM v0.10.2《Sleep Mode》及 NVIDIA CUDA Driver API《Virtual Memory Management》。`,
  },
  {
    id: 'rl-06',
    track: 'rl',
    title: 'Agentic RL / 多轮工具调用 (Sandbox RL) 的高并发基础设施设计',
    level: 'P8',
    frequency: '前沿高频',
    incident: false,
    tags: ['Agentic RL', 'Code Sandbox', 'Multi-turn', 'Async Loop', 'Prefix Cache'],
    summary: '设计支撑 SWE-Bench / 代码执行 / 搜索工具多轮交互强化学习的分布式沙箱集群与异步 Rollout 调度器。',
    contextSpecs: ['10,000+ Concurrent Sandboxes', 'Multi-turn Tool Call', 'SWE / Terminal / Browser', 'Prefix Reuse'],
    prompt: '现在最火的方向是从单轮数学 GRPO 升级到 **Agentic RL（多轮工具调用、代码沙箱运行单元测试、SWE-Bench 仓库修复、联网检索）**。每条轨迹要和外部环境交互 10~50 轮，单次代码编译/测试可能耗时 2~30 秒，而且存在恶意/死循环代码风险。请你设计一套高吞吐、低延迟、安全隔离的 Agentic RL 基础设施架构。',
    rubric: [
      {
        id: 'async-env-decouple',
        label: '环境等待期间释放生成槽位，用有界异步状态机连接推理和工具池',
        weight: 24,
        keywords: ['解耦', '异步', 'async', '事件循环', '不阻塞', 'gpu空等', 'io', '并发', 'scheduler', '挂起'],
        probe: '如果按传统同步 Batch 跑多轮 Agent Rollout（等所有样本工具执行完再一起发下一轮推理），GPU 利用率会惨到什么程度？怎么设计异步调度环？',
      },
      {
        id: 'sandbox-isolation',
        label: '按威胁模型选 microVM 或沙箱运行时，预热镜像并验证隔离、清理与重置时延',
        weight: 24,
        keywords: ['firecracker', 'gvisor', 'microvm', '沙箱', 'sandbox', '隔离', '快照', 'snapshot', '预热', 'cow', 'overlayfs', 'cgroup', 'seccomp'],
        probe: '在模型生成的代码可能死循环或耗尽资源时，怎样设计隔离、配额、环境重置和预热，并测量真实启动 P99？',
      },
      {
        id: 'prefix-kv-affinity',
        label: '相同版本和 token 前缀下用会话亲和或分层缓存，并核算等待期间 KV 驻留成本',
        weight: 22,
        keywords: ['prefix', 'radix', '亲和', 'affinity', 'session', 'kv', '复用', '多轮', '重算', 'prefill'],
        probe: '一个 SWE Agent 交互 30 轮，第 30 轮上下文已达 64K，每轮只多了几百 Token 的工具返回值。如果网关随机路由到不同推理节点，会发生什么算力灾难？怎么解决？',
      },
      {
        id: 'loss-masking',
        label: '多轮数据组织与 Loss Masking：精确区分 Model Token 与 Environment Observation Token（工具输出不计入 Policy Gradient Loss，避免模型学去拟合报错日志）',
        weight: 16,
        keywords: ['mask', 'loss mask', 'observation', '环境输出', 'token', '不计算梯度', 'trajectory', '打包'],
        probe: '在训练阶段，把 30 轮交互拼成一条长序列做 GRPO Update 时，对环境返回的 Terminal 输出和模型生成的 Action Token 在 Loss 和 Advantage 上必须做什么处理？',
      },
      {
        id: 'timeout-fault',
        label: '设置 CPU/内存/PID、磁盘与网络的适当隔离与配额，区分基础设施失败、模型行为和任务超时',
        weight: 14,
        keywords: ['超时', 'timeout', '限额', '故障', '重试', '基础设施', 'reward', '污染'],
        probe: '如果某个代码测试超时是因为沙箱宿主机 CPU 负载太高（Infra 故障）而非模型代码有死循环，直接判 Reward=0 会污染训练信号！系统怎么区分并处理？',
      },
    ],
    hints: [
      'L1 方向提示：Agentic RL 架构核心是三大支柱——**异步 Token-level 推理-环境事件环**、**预热微虚机快照沙箱池**、**多轮 Session 亲和 KV 缓存复用**。',
      'L2 会话等待工具时可释放算力槽位，是否保留 KV 另行决策。比较重新 prefill、双向搬运和等待期间占用的显存。',
      'L3 保留真正采样的 token ID 与行为 logprob。工具观察参与上下文，policy loss 只覆盖动作位置。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（KV Cache 驻留 vs 驱逐）：一条 SWE Agent 轨迹执行一次编译测试要花 20 秒，这 20 秒里它的 48K 上下文 KV Cache 是该留在 GPU 显存里等它，还是踢出去？给出定量权衡思路。',
      },
      {
        stage: 'corner',
        question: '追问 2（Tokenization drift）　按轮拼接 token IDs 与重新渲染全量 chat template，何时产生不同前缀？怎样同时保证缓存匹配和训练 action logprob 对齐？',
      },
    ],
    referenceAnswer: `### 一条轨迹的状态与失败语义

1. 调度器给每条轨迹保存会话、环境快照、策略版本和请求 ID。在模型产出工具调用后归还生成槽位，异步排队到有并发上限的环境池。工具完成后追加 observation，再排入下一轮生成。用队列上限和背压控制在途会话、CPU 工作量与 GPU KV。重试按调用 ID 去重，有副作用的工具应从一致快照恢复。
2. 执行不可信代码可采用隔离的 microVM 或 gVisor 类沙箱，按威胁模型选边界。使用只读基础镜像、会话独立可写层、最小权限与受控网络。CPU、内存、PID、磁盘和网络分别限额。快照预热可省启动耗时，归还前清理进程、文件和凭据，按实际仓库测 P95/P99。区分代码测试失败、可复现的资源超限、宿主机故障，按任务奖励规则计分或有界重试。
3. **48K KV 等待 20 秒**。设实际独占 KV 大小为 K、换出/入带宽为 Bₒ/Bᵢ。换出再换入的耗时下界为 K/Bₒ+K/Bᵢ，可与重算 48K 前缀的实测耗时比较。驻留 20 秒占用 K×20 字节·秒，以这段时间挤掉的请求数或增加的排队时长衡量机会成本。低压力时可留在 GPU。高压力时选择保留共享块、换出或重算。复用键包含模型/adapter 版本、位置配置和 token 前缀。
4. **训练数据**。存每次真实输入和采样 token IDs、行为 logprob、角色区间和 action mask。冻结 tokenizer 与模板，验证新增 observation 的分隔符、特殊 token 和拼接边界是否保留历史前缀。模板重写历史时只复用相同的最长前缀。训练按真实的状态–动作条件对齐。工具输出和 prompt 仍作为上下文，policy loss 仅取模型采样动作的位置。

> 依据　Firecracker《Production Host Setup》及 veRL 的多轮 rollout / Agent Loop 文档。`,
  },
  {
    id: 'rl-07',
    track: 'rl',
    title: 'RL 系统级数值对齐：Train-Inference Mismatch (Logprob 偏移) 根因与排查',
    level: 'P7',
    frequency: '高频',
    incident: true,
    tags: ['Precision Mismatch', 'Logprob偏移', 'KL爆炸', 'FP8 Rollout', '排障'],
    summary: '为什么同一权重、同一 Token 序列，vLLM Rollout 算出的 logprob 和 Megatron 训练算出的 logprob 不一样？如何防止策略崩溃？',
    contextSpecs: ['vLLM vs Megatron/FSDP', 'KL Divergence Spike', 'BF16/FP8 Non-determinism', 'IS Correction'],
    prompt: '同一版权重下，FP8 vLLM rollout 与 BF16 Megatron teacher-forcing 对同一动作给出了不同 logprob，有些 token 的概率比超过 5。监控还出现负的 KL 指标。怎样区分采样器口径、模型实现偏差、策略更新与估计器性质，并验证修复是否有效？',
    rubric: [
      {
        id: 'precision-kernel',
        label: '区分采样变换、量化与 kernel 数值差异，检查 MoE Top-k 边界及实际路由',
        weight: 26,
        keywords: ['fp8', 'bf16', '非结合律', '累加顺序', 'kernel', 'flashattention', 'pagedattention', 'moe', '路由', 'top-k', 'router', '算子'],
        probe: '两边同为 BF16 时，哪些输入、运算顺序或路由边界仍能产生偏差？怎样定位首个出现偏差的位置？',
      },
      {
        id: 'bug-checklist',
        label: '常见工程 Bug 排查：Weight Sync 映射错误（RoPE interleaved/half、QKV 排列、LM Head tie weights）、Padding/Packing Varlen Mask 错位、Retokenization 漂移',
        weight: 24,
        keywords: ['rope', 'qkv', 'weight sync', '映射', 'padding', 'mask', 'packing', 'tokenization', 'chat template', 'bos', 'eos'],
        probe: '在怪浮点精度之前，首先得排查哪几类致命工程 Bug（可能导致不是小数点后三位差，而是直接差出天际）？',
      },
      {
        id: 'kl-neg',
        label: '区分精确 KL、有符号 k1 样本估计和逐项非负 k3。核对采样分布、mask、符号和精度',
        weight: 18,
        keywords: ['kl', 'k3', 'ratio', '重要性', 'importance', 'clip', 'logprob', '负值', '方差'],
        probe: '为什么训练中会看到 $r_t$ 飙到 5 以上（尤其在低概率/长尾 Token 上）？这对 GRPO 的策略梯度会产生什么破坏？',
      },
      {
        id: 'mitigation-sys',
        label: '区分 rollout 行为 μ、训练旧策略和更新后策略，独立核算校正比与 PPO 比',
        weight: 22,
        keywords: ['recompute', '重算', 'tis', 'truncated importance sampling', '修正', 'lm head', 'router', 'bf16', '校准', '对齐'],
        probe: 'FP8 Rollout 速度太香了实在不想关掉，那怎么在不放弃 FP8 Rollout 的前提下，让 RL 训练长期稳定不崩？',
      },
      {
        id: 'unit-verify',
        label: '同权重同 token 的 teacher-forced 分层对照，按精度、位置、概率桶和质量基线定门槛',
        weight: 10,
        keywords: ['门禁', '单测', 'step 0', 'logits', 'max diff', '对比', '验证', 'ci'],
        probe: '如何为这个模型选择有验证依据的容差，并把它纳入换权和后端升级的回归检查？',
      },
    ],
    hints: [
      'L1 方向提示：按「先查硬 Bug（权重/RoPE/Tokenize 错位）→ 再查数值源（FP8/BF16、Kernel 规约、MoE Router 翻转）→ 最后给治理方案（敏感层高精度 + TIS 重要性修正）」的排查逻辑。',
      'L2 当两个专家分数接近时，小误差可能改变 top-k 集合。直接比较每层路由分数、索引与输出，而非预设放大倍数。',
      'L3 保留行为、旧策略和当前策略三份有明确定义的 logprob。先校验采样器，再评估截断校正或路由回放的收益和偏差。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（MoE 路由回放）　保存每层每 token 的 top-k 专家索引可以排除哪种差异？哪些误差会保留，传输量怎样估算？',
      },
      {
        stage: 'corner',
        question: '追问 2（尾部概率）　若 Δlog p=2，概率比是多少？相同的绝对概率误差和相同的 logit 误差，对长尾 token 的含义有什么不同？',
      },
    ],
    referenceAnswer: `### 对齐一次动作的三种概率

固定 checkpoint、prompt 和完整 token IDs 做 teacher-forcing，按同一位置比较预测下一个 token 的 logits。记录实际行为策略 μ（含 temperature、top-k/p、logits processor 后的采样分布）、更新前训练策略 π_old、当前策略 π_θ。先确认引擎返回 raw logprob 还是采样后 logprob，再选择目标函数所需口径。核对 mask、分母归一化、BOS、position IDs、右移一位、RoPE 与权重同步。PPO 更新比是 π_θ/π_old。rollout 校正可另用 π_old/μ。重算 π_old 不能替代生成时 μ 的记录。

### 负值究竟是哪一个量

- 精确 KL(p‖q)=Σp log(p/q) 在合法分布上非负。从 p 抽样的 k1=log p−log q 单样本或小批平均可为负。令 d=log q−log p，则 k3=exp(d)−1−d 逐项非负。在两分布支持一致、样本来自 p 时，其期望等于该 KL。若 k3 明显为负，查实际记录的是惩罚项还是正值 KL、是否错减/加权、mask 与实现。接近零时可用 FP32 及 \`expm1(d)−d\` 对照。
- Δlog p=2 对应概率比 e²≈7.39，超过 5 对应 Δlog p>ln 5≈1.61。相同绝对概率差对小概率的相对影响更大。但单凭 p 小无法推出 logit 小扰动必定放大。固定支持、无截断的 log-softmax，若各 logit 变化至多 ε，则任一 logprob 变化至多 2ε。

### 找到第一处偏离再修

先跑未量化 BF16 对照，再逐项打开 FP8、不同 batch/kernel、packing 或路由。检查 QKV/GateUp 与 LM head 映射、attention 边界。按层比较激活与 MoE top-k，按位置和原概率分桶记录 Δlogprob、ratio 尾部、clip 比例、有效样本量和任务奖励。路由回放每 token 至少需记录 MoE 层数 × top-k × 索引字节，可排除专家选择差异，量化和专家输出仍可能不同。

有证据后再保留特定层的高精度或调整 scale。在相同采样支持上，截断或屏蔽高比值样本能控制方差，但会改变估计。实际行为分布为零的动作无法靠重要性采样恢复。用相同训练预算比较质量，依据 BF16 重复测量和业务指标定容差。

> 依据　veRL《Mathematical Formulations of Rollout Correction Methods》及 John Schulman《Approximating KL Divergence》。`,
  },
];
