export const RL_QUESTIONS = [
  {
    id: 'rl-01',
    track: 'rl',
    title: 'PPO vs GRPO vs DPO 的系统显存开销、算力瓶颈与工程演进',
    level: 'P6',
    frequency: '高频必考',
    incident: false,
    tags: ['PPO', 'GRPO', 'DPO', 'RLHF', 'Rollout瓶颈'],
    summary: '定量拆解 PPO 四模型显存爆炸 vs GRPO 去 Critic 化对系统架构的颠覆：为什么算力重心彻底偏向在线 Rollout 推理？',
    contextSpecs: ['Actor / Critic / Ref / RM', 'GRPO Group Size G=8~64', 'Rollout 70%~85% Time', 'Rule-based Verifier'],
    prompt: '从 AI Infra 视角对比 DPO、传统 PPO 和 DeepSeek-R1 广泛使用的 GRPO（Group Relative Policy Optimization）：它们各自在显存里需要驻留哪些模型与状态？为什么 GRPO 去掉 Critic 模型后，整个 RL 训练集群的系统瓶颈反而从“训练显存”彻底转移到了“在线 Rollout 推理吞吐与长尾延迟”？',
    rubric: [
      {
        id: 'ppo-4models',
        label: 'PPO 四模型开销：Actor(训练16Ψ) + Critic(训练16Ψ) + Reference(推理2Ψ) + Reward Model(推理2Ψ)，总静态显存高达 ~36Ψ',
        weight: 24,
        keywords: ['actor', 'critic', 'ref', 'reward', 'rm', '四模型', '4个模型', '16', '36', '显存爆炸', 'optimizer'],
        probe: '如果 Actor 和 Critic 都是 70B 模型，算一下标准 PPO 全驻留显存光静态权重+优化器就要占多少 GB？',
      },
      {
        id: 'grpo-no-critic',
        label: 'GRPO 架构减负：去掉 Critic (省掉一半训练显存 ~16Ψ)，且可结合 Rule-based Verifier (数学/代码) 省掉 RM，靠同 Prompt 采样 G 组输出算组内相对 Advantage',
        weight: 24,
        keywords: ['grpo', '去掉critic', '无critic', 'group', 'g个', '组内', '相对', 'advantage', 'verifier', '规则奖励', '省显存'],
        probe: 'GRPO 怎么在没有 Critic Value Model 的情况下估计 Advantage？这为显存和系统架构省掉了哪些组件？',
      },
      {
        id: 'rollout-bottleneck',
        label: '瓶颈转移到 Rollout：每个 Prompt 需生成 G (如 16~64) 条超长 CoT (8K~32K)，自回归推理 Memory-Bound 占总步时 70%~85%',
        weight: 22,
        keywords: ['rollout', '推理', '70%', '80%', '85%', 'memory-bound', '长cot', '自回归', '瓶颈', '生成', 'kv cache'],
        probe: '既然 GRPO 省了一半显存，为什么大规模训 R1 类推理模型时，工程师 80% 的精力都在死磕 Rollout 阶段？',
      },
      {
        id: 'dpo-offline',
        label: 'DPO 离线对比：无需在线 Rollout 与 Critic/RM，仅需 Policy + Ref（Ref 甚至可离线预计算 logprob），但缺乏 On-policy 探索能力',
        weight: 16,
        keywords: ['dpo', '离线', 'offline', '无需rollout', 'ref', '预计算', 'on-policy', '探索', 'sft'],
        probe: '从 Infra 复杂度看 DPO 几乎和 SFT 一样简单（甚至可以把 Ref logprobs 离线算好）。那为什么 Reasoning 模型非要上复杂度高 10 倍的在线 GRPO？',
      },
      {
        id: 'shared-prefix',
        label: 'GRPO Rollout 专属优化红利：同一 Prompt 的 G 条采样完全共享 Prompt KV Cache (Radix/Prefix Caching) 且可大幅加速 Prefill',
        weight: 14,
        keywords: ['prefix', '共享', 'copy-on-write', 'radix', 'kv复用', 'batched', 'prefill'],
        probe: 'GRPO 对同一个 Prompt 一次采样 G=16 或 64 条回答，在推理引擎侧有什么天然的显存和算力优化机会？',
      },
    ],
    hints: [
      'L1 方向提示：先算清静态显存账（PPO ~36Ψ vs GRPO ~18Ψ vs DPO ~16~18Ψ），再算时间账（Train 是 Compute-bound 高 MFU，Rollout 是长尾自回归 Memory-bound）。',
      'L2 GRPO 算力账：假设 Prompt Batch=256，Group G=16，每条生成 8K tokens，单步就要推理生成 **32M tokens**，而反向训练只有 1 遍！Rollout 轻松吃掉 75%~85% 的端到端墙钟时间。',
      'L3 Prefix 红利：同一组 G 条输出共享完全一样的 Prompt，开启 PagedAttention Copy-on-Write / Prefix Caching 后 Prompt 只做 1 次 Prefill、物理 KV 只存 1 份！',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（Ref 模型优化）：在 GRPO 里我们仍然需要 Reference Model 算 KL 散度惩罚（$D_{KL}(\\pi_\\theta || \\pi_{ref})$）。如果 Actor 是 671B MoE，难不成要再常驻一套 671B 的 Ref 模型吗？工程上怎么把 Ref 模型的显存和时间开销压到最低？',
      },
      {
        stage: 'corner',
        question: '追问 2（DAPO / Dr.GRPO 变体）：GRPO 训练中，不同样本的生成长度从 500 到 32K 剧烈波动，如果直接在 Token 维度或 Sequence 维度做 Loss 归一化，加上 Packing 训练，Infra 在梯度累积和 Loss Mask 上容易踩什么坑？',
      },
    ],
    referenceAnswer: `### Staff 级参考解析

#### 1. 三大对齐范式的系统显存与复杂度对比（模型参数量 $\\Psi$）
| 维度 | DPO | PPO (RLHF) | GRPO / RLOO (Reasoning RL) |
|---|---|---|---|
| **在线驻留角色** | Actor (16Ψ) + Ref (2Ψ，可离线预算) | Actor (16Ψ) + Critic (16Ψ) + Ref (2Ψ) + RM (2Ψ) | Actor (16Ψ) + Ref (2Ψ) + 轻量 Rule Verifier |
| **总静态显存** | ~16~18Ψ | **~36Ψ**（极易 OOM） | **~18Ψ**（直降 50%） |
| **在线 Rollout** | 无（纯离线 SFT-like 流水线） | 有（每 Prompt 采 1 条） | **极重**（每 Prompt 采 $G=8\\sim 64$ 条长 CoT） |
| **核心系统瓶颈** | 纯训练吞吐（Compute-bound） | 四模型显存调度 + 多阶段通信 | **长尾 Rollout 推理吞吐 (占 70%~85% 时间)** + 权重高频同步 |

#### 2. 为什么 GRPO 把集群瓶颈彻底推向推理侧？
- 训练（Actor Update + Ref Logprob）是全序列并行的 **Compute-Bound**，H100 MFU 可达 45%~55%。
- Rollout 是逐 Token 自回归的 **Memory-Bound**，且推理模型单条 CoT 动辄 8K~32K tokens，乘上 $G=16\\sim 64$，每步生成量极其庞大；加上序列长度方差极大导致的 Straggler 空等，Rollout 成为绝对主瓶颈。

#### 3. 工程关键优化点
- **Prompt KV 零冗余共享**：同一 Prompt 的 $G$ 条轨迹在 vLLM/SGLang 内共享 Prompt 物理 KV Block（CoW / Radix Tree），Prefill 算力与 Prompt KV 显存直接缩减 $G$ 倍。
- **Ref 模型零常驻技巧**：(1) 将 Ref 权重 Offload 到 CPU Pinned Memory，仅在每步计算 \`ref_logprob\` 时按层流式加载或短暂换入；(2) 若采用 LoRA RL，则禁用 LoRA Adapter 即可零显存直接复用同一份 Base 权重充当 Ref！`,
  },
  {
    id: 'rl-02',
    track: 'rl',
    title: 'RL 系统架构对决：Colocated (veRL HybridEngine) vs Disaggregated (OpenRLHF/Ray)',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['veRL', 'OpenRLHF', 'Colocated', 'Disaggregated', 'HybridEngine', 'Ray'],
    summary: '深度对比 RLHF/GRPO 两大主流系统架构——共置分时复用 vs 分离异步池——在 GPU 利用率、显存切换、通信同步和弹性扩展上的核心 Trade-off。',
    contextSpecs: ['veRL 3D-HybridEngine', 'OpenRLHF / Ray', 'Colocated vs Disaggregated', 'GPU Bubble'],
    prompt: '大规模 RL 后训练框架分化出两大架构流派：以 veRL 为代表的 **Colocated（同卡共置 + 分时复用，HybridEngine）**，以及以 OpenRLHF / StreamRL / Async-RL 为代表的 **Disaggregated（Rollout 与 Training 独立 GPU 池分离部署）**。请你从 GPU 利用率、显存切换代价、权重同步开销、同步 vs 异步算法适配以及超大规模扩展性五个维度做深度架构对比，并说明不同规模集群该如何选型。',
    rubric: [
      {
        id: 'colocated-mech',
        label: 'Colocated (veRL) 机制：同一批 GPU 分时跑 Rollout (vLLM/SGLang) 和 Train (FSDP/Megatron)，靠 Resharding + Offload + Sleep Mode 切换，避免配比失衡气泡',
        weight: 24,
        keywords: ['colocated', '共置', '分时', 'verl', 'hybridengine', 'offload', 'sleep', 'reshard', '无配比', '气泡'],
        probe: 'veRL 把训练和推理塞进同一批 GPU，切换阶段时显存不会爆吗？它是靠什么机制在一组卡上无缝切换两种引擎的？',
      },
      {
        id: 'disagg-mech',
        label: 'Disaggregated (OpenRLHF/Ray) 机制：独立 Rollout GPU 池与 Train GPU 池，各自用最优并行度，靠跨节点 NCCL/RDMA 广播权重，天然支持异步流水 Overlap',
        weight: 24,
        keywords: ['disaggregated', '分离', 'openrlhf', 'ray', '独立', '异步', 'async', '流水', 'pipeline', '广播', 'broadcast'],
        probe: '分离式架构中，如果是严格同步 On-policy RL（Train 必须等 Rollout 完，Rollout 又等 Train 更新完），会产生什么严重问题？怎么靠异步消除？',
      },
      {
        id: 'tradeoff-bubble',
        label: '核心 Trade-off：同步算法下 Colocated 利用率更高（0 跨池互等气泡），但有切换/Offload 固定开销；超大规模/长尾严重时 Disaggregated + 异步 Off-policy 更具吞吐与扩展性',
        weight: 22,
        keywords: ['同步', '异步', 'on-policy', 'off-policy', '长尾', '切换开销', '扩展性', '利用率', '配比'],
        probe: '为什么在严格同步 On-policy 下分离架构经常 GPU 闲置一半，但到了千卡级长 CoT RL 时大家又开始转向分离+异步架构？',
      },
      {
        id: 'topo-mismatch',
        label: '引擎并行拓扑差异：Train 偏重 FSDP/DP+PP+TP，Rollout 偏重高并发小 TP 或 Wide-EP，Colocated 需做 3D 重分片而 Disaggregated 需跨机点对多广播',
        weight: 16,
        keywords: ['并行度', '拓扑', 'tp', 'pp', 'fsdp', 'megatron', 'ep', '重分片', 'resharding'],
        probe: '训练和推理的最佳并行配置通常不一样（比如训用 TP4+PP2+DP8，推用 TP4×16 副本），两种架构各自怎么处理权重映射？',
      },
      {
        id: 'hetero-hw',
        label: '异构硬件红利：Disaggregated 允许混部异构卡（如 H20/高带宽卡跑 Rollout + H100/B200 跑 Train），Colocated 必须同构',
        weight: 14,
        keywords: ['异构', 'h20', 'h100', '混部', '成本', 'roi', '同构'],
        probe: '如果公司手里同时有大批便宜的高显存推理卡（如 H20/L40S）和少量 H100，哪种架构能把异构算力吃干抹净？',
      },
    ],
    hints: [
      'L1 方向提示：核心矛盾是“严格 On-policy 同步 vs 容忍 1-step Staleness 异步”！同步选 Colocated 省互等气泡；异步/超大规模/异构卡选 Disaggregated。',
      'L2 切换账：Colocated 每次从 Train 切 Rollout，要把 Optimizer States Offload 到 CPU、释放训练激活、唤醒推理引擎显存池、并把训练权重 Reshard 进推理引擎。',
      'L3 分离账：Disaggregated 在同步模式下 Train 池和 Rollout 池永远一动一静（利用率 <50%）；必须配上 **One-step Off-policy / Async Streaming Rollout** 让两池同时满转，且需精细调校 Rollout:Train 卡数配比防止队列积压或饥饿。',
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
    referenceAnswer: `### Staff 级架构选型矩阵

| 对比维度 | Colocated (veRL HybridEngine) | Disaggregated (OpenRLHF / StreamRL / Async-RL) |
|---|---|---|
| **GPU 调度方式** | 同批 GPU 分时跑 Rollout → Ref → Train | 物理拆分为 Rollout 池 + Train 池（+ 独立 Verifier 池） |
| **同步 On-policy 利用率** | **极高**（全集群齐刷刷算 Rollout 再齐刷刷算 Train，零跨池互等） | **差**（一池干活一池发呆，必须改异步才能拉满） |
| **异步 / 长尾抗性** | 弱（批内最慢那条 32K 轨迹卡住全集群进 Train） | **极强**（Rollout 流式源源不断吐轨迹给 Train，长尾自然跨步延续） |
| **显存与切换开销** | 每次切换需 CPU Offload Optimizer + CuMem Sleep + Resharding（数秒~十几秒） | 零 Offload、常驻引擎，但每步需跨机网络同步参数 |
| **异构算力与故障隔离** | 必须同构；任意单卡挂全任务受影响 | 支持 **H20/高带宽卡 Rollout + H100 Train**；Rollout 实例可独立无感重启 |

#### 选型最佳实践
- **中小规模（8~256 卡）或要求严格 On-policy 数学对齐（如算法探索期）**：首选 **Colocated (veRL)**，省去 P:D/R:T 节点配比调优，单卡 ROI 最高。
- **超大规模（512~4096 卡）、长 CoT (16K~64K) 或 Agentic 多轮环境交互 RL**：首选 **Disaggregated + 半异步 (1-step Off-policy / Partial Rollout)**，将环境 IO 等待与 Rollout 长尾彻底与昂贵的训练池解耦。
- **Ray 控制面防炸**：绝不把大张量/KV 经由 Ray Driver 或 Object Store 传回中心节点；Control Plane 只传轻量元数据和指针，Data Plane 全部走 **GPU-to-GPU NCCL / RDMA 分布式直连**。`,
  },
  {
    id: 'rl-03',
    track: 'rl',
    title: 'RL 训练-推理引擎高频权重同步 (Weight Sync) 与 3D Resharding 加速',
    level: 'P7',
    frequency: '高频必考',
    incident: false,
    tags: ['Weight Sync', 'Resharding', 'CUDA IPC', 'NCCL Broadcast', 'FP8 Rollout'],
    summary: '深入训练引擎 (FSDP/Megatron) 到推理引擎 (vLLM/SGLang) 之间百 GB 级权重秒级同步的拓扑映射、零冗余通信与量化对齐。',
    contextSpecs: ['Megatron/FSDP → vLLM/SGLang', '3D-HybridEngine', 'CUDA IPC / NCCL Bucket', 'Sub-second Sync'],
    prompt: '在 RL 迭代循环中，每完成一次 Actor 参数更新，就必须立刻把最新权重同步给推理引擎（vLLM/SGLang）开始下一轮 Rollout。对于 72B 甚至 671B 模型，如果每次同步耗时 30 秒，整个训练吞吐就毁了。请详述：(1) 训练与推理并行拓扑不一致时如何做高效 Weight Resharding；(2) 同卡共置与跨机分离场景下分别用什么底层传输机制（CUDA IPC / NCCL / Bucket 化）做到亚秒~秒级同步；(3) 如果 Train 用 BF16 而 Rollout 用 FP8，权重同步怎么处理？',
    rubric: [
      {
        id: 'topo-reshard',
        label: '3D 拓扑对齐与零冗余重分片 (3D-HybridEngine)：通过微调 TP/PP 映射尽量把重分片局限在机内 NVLink (AllGather micro-DP 组)，避免跨机搬运',
        weight: 24,
        keywords: ['reshard', '重分片', '3d-hybridengine', 'tp', 'pp', 'dp', 'nvlink', '机内', 'allgather', '拓扑'],
        probe: '假设训练用 TP=4, PP=2, DP=8 (64 卡)，而推理每实例只需 TP=4 (16 个独立副本)。如果粗暴把全模型汇总到 Rank 0 再广播，会出什么问题？3D-HybridEngine 是怎么把同步完全留在本地小通信域的？',
      },
      {
        id: 'bucket-stream',
        label: 'Bucket 化流式转换 (Streaming Bucket / Flattened Buffer)：避免一次性物化完整 72B 未分片权重导致 OOM，按层或固定 512MB~2GB 桶流水 AllGather/Broadcast + 名称映射 (QKV/GateUp 融合)',
        weight: 22,
        keywords: ['bucket', '桶', '流式', 'oom', '逐层', 'layer', 'qkv', '融合', '命名映射', 'buffer'],
        probe: 'FSDP/Megatron 的权重切分格式、以及 QKV/GateUp 投影矩阵拼接方式和 vLLM 不一样。如果先一次性把 72B 参数全部 AllGather 拼好再喂给 vLLM，会发生什么灾难？怎么解决？',
      },
      {
        id: 'ipc-nccl',
        label: '传输通道选型：同卡共置优先同进程零拷贝指针 / CUDA IPC Handle；跨进程/跨机用专用 NCCL ProcessGroup 广播（避开 Ray Object Store 与 CPU 序列化）',
        weight: 20,
        keywords: ['cuda ipc', 'nccl', 'broadcast', '零拷贝', '指针', '同进程', 'rdma', 'ray', '显存拷贝'],
        probe: '同机共置时，如果训练和推理跑在不同进程（比如 Ray Actor），不用磁盘也不用 CPU 拷贝，怎么直接 GPU-to-GPU 传权重？',
      },
      {
        id: 'fp8-quant-sync',
        label: 'BF16 Train → FP8 Rollout 在线量化同步：在发送端（或 Bucket 传输前）在线融合量化为 FP8 + Scale，通信量直接减半并加速 Rollout 2×',
        weight: 18,
        keywords: ['fp8', '在线量化', 'quant', '减半', 'scale', '精度', 'mismatch', '重要性采样'],
        probe: '如果训练是 BF16、推理开 FP8，应该先传 BF16 再在推理侧量化，还是先量化再传？这会引入什么算法层面的副作用？',
      },
      {
        id: 'lora-delta',
        label: '增量/LoRA/稀疏同步视角：若采用 LoRA RL，每步只需同步几十 MB Adapter 权重，耗时 <10ms',
        weight: 16,
        keywords: ['lora', '增量', 'delta', 'adapter', '毫秒', '轻量'],
        probe: '相比全参同步，如果业务用 LoRA 做 RL，权重同步耗时能压缩到什么量级？',
      },
    ],
    hints: [
      'L1 方向提示：记住三个关键词——**局部通信域对齐 (3D-HybridEngine)**、**按 Bucket 流式转换防 OOM**、**GPU 直连通道 (同进程指针/CUDA IPC/NCCL Broadcast)**。',
      'L2 显存防爆技巧：FSDP 参数在同步时不能全模型 \`summon_full_params\`，必须按 Layer 或按固定大小（如 1GB Bucket）逐块 AllGather -> 切分/重排为 vLLM QKV 格式 -> 写入 vLLM 权重 Buffer -> 立即释放临时缓冲！',
      'L3 速度基准：72B BF16 = 144GB。按机内 NVLink (400GB/s+) 做局部 AllGather 理论上只需 **0.4~1.5 秒**；若跨机用 400Gbps IB 广播且做 FP8 压缩（72GB），每节点 8 卡并发也在 **1~2 秒**内完成！',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（CUDA Graph 与权重指针）：vLLM/SGLang 开启了 CUDA Graph 加速 Decode。如果权重同步时你重新 `malloc` 了新的权重 Tensor 替换旧引用，CUDA Graph 会立刻失效甚至读到野指针！工程上必须遵守什么铁律？',
      },
      {
        stage: 'corner',
        question: '追问 2（异步流式更新）：在 Disaggregated 异步 RL 中，Rollout 池正在生成半截的 Trajectory，此时新权重同步过来覆盖了旧权重，一条 Trajectory 前半段是 $\\pi_{t-1}$ 生的、后半段是 $\\pi_t$ 生的，工程和算法上怎么处理？',
      },
    ],
    referenceAnswer: `### Staff 级权重同步加速方案

#### 1. 并行拓扑对齐（veRL 3D-HybridEngine 思想）
- 设训练并行度为 \`(TP_t, PP_t, DP_t)\`，推理为 \`(TP_g, DP_g)\`（通常 \`PP_g = 1\`，且 \`TP_g\` 往往较小以换高并发）。
- 若直接从全局重建再广播，会产生大量跨机拥塞。
- **最优编排**：构造推理与训练共享的微数据并行子组（Micro-DP Group，尽量落在**同一机内 NVLink 域**），每个推理 Rank 只需在极小范围内执行一次 \`AllGather\` 即可凑齐所需分片，彻底消除跨机多余通信，实现 **Zero Redundancy**。

#### 2. Bucket 化流式转换（防 OOM + 吃满带宽）
1. 绝不一次性把完整未分片 72B/671B 模型物化到 GPU 显存，也绝不走 CPU/Ray Object Store；
2. 将所有参数按 \`512MB ~ 2GB\` 分成连续 **Flat Bucket**，逐 Bucket 执行：\`训练侧 AllGather/提取 -> 算子布局映射（Megatron/FSDP interleaved QKV/GateUp 转 vLLM packed layout）-> 在线量化 (若启用 FP8 Rollout) -> 传输写入 -> 释放临时缓冲\`；
3. **In-place 写入铁律**：必须使用 \`param.data.copy_(new_weight)\` **原地覆写**推理引擎已绑定的权重显存地址，绝不更改 Tensor 指针，以确保 vLLM/SGLang 预先捕获的 **CUDA Graph 地址完全有效**！

#### 3. 传输通路选型
- **Colocated 同进程**：直接 GPU 显存原地拷贝 / 局部 NVLink AllGather（72B < 1.5s）。
- **Colocated 跨进程**：通过 **CUDA IPC (\`cudaIpcGetMemHandle\`)** 传递 GPU 显存句柄零拷贝共享。
- **Disaggregated 跨机**：训练侧 Rank 0 组与推理侧建立持久化 **NCCL Broadcast/P2P 通信组**（多 NIC 并发流水传输），FP8 压缩可再砍半网络流量；若为异步架构，可在 Request 级打 \`policy_version\` 标签或采用 **Pause-Update-Resume** 机制保证单步权重切换仅暂停几百毫秒。`,
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
    contextSpecs: ['CoT 1K ~ 64K Tokens', '95% GPU Idle Tail', 'Partial Rollout', 'Off-policy Staleness'],
    prompt: '【高频真实痛点】你在训练一个类似 DeepSeek-R1 / OpenAI o1 的长思维链（最大长度 32K）GRPO 模型。监控发现 Rollout 阶段前 20 秒所有 GPU 满载跑得很欢，但后面整整 100 秒里，95% 的样本已经生成完毕，整个 512 卡集群全在干等最后寥寥几条 32K 的超长样本吐完，导致端到端 GPU 利用率跌破 25%！请给出从纯同步优化到半异步/全异步的多层系统解法。',
    rubric: [
      {
        id: 'straggler-root',
        label: '精准诊断根因：同步 Barrier 导致长尾木桶效应（批量末尾 Batch Size 骤降为 1~4，退化为极端 Memory-Bound，算力利用率 <1%）',
        weight: 18,
        keywords: ['木桶', 'barrier', '长尾', 'straggler', 'batch size', '骤降', 'memory-bound', '空等', '同步点'],
        probe: '为什么最后那 5% 的 32K 样本不仅仅是因为长，还会显得“越到后面吐得越慢/越浪费卡”？',
      },
      {
        id: 'partial-rollout',
        label: 'Partial Rollout / Trajectory Interruption（部分生成与跨步续写，如 Kimi k1.5 / AReaL）：达到时间/步数预算强制截断保存状态，下一轮继续生成或直接进入 Replay Buffer',
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
        label: 'Off-policy Staleness 控制：限制最大版本差 (max_staleness ≤ 1~2 步)、重要性采样比率 Clip (IS Clipping / IcePop) 或重算 Old Logprob',
        weight: 18,
        keywords: ['staleness', '陈旧', '版本差', 'importance sampling', '重要性采样', 'clip', 'ppo clip', 'off-policy', '策略偏移'],
        probe: '如果一条轨迹是用 3 个版本前的旧策略生成的，直接拿来算 GRPO 梯度会导致策略崩塌，工程和算法上怎么控 Staleness？',
      },
      {
        id: 'sync-tricks',
        label: '同步框架内缓解手段：动态路由平衡各实例剩余长度、过采样 (Over-sampling) 提前凑满有效 Batch、早停/重复循环检测（Repetition Loop Killer）',
        weight: 16,
        keywords: ['过采样', 'over-sampling', 'oversampling', '重复', '死循环', 'repetition', '动态', '负载均衡', '投机采样', 'speculative'],
        probe: '假设算法团队死活不允许任何 Off-policy（必须 100% 严格 On-policy），在不改算法的前提下，工程上有哪些立竿见影的加速手段？',
      },
    ],
    hints: [
      'L1 方向提示：从“不改算法的同步优化”到“微调算法的 Partial Rollout & One-step Async”分层作答，这最能体现 Staff Infra 工程师与算法 Co-design 的视野！',
      'L2 同步绝招：(1) 检测死循环重复复读机提前截断；(2) 用长度预测做 Prompt 调度均衡各实例；(3) 对末尾低 Batch 长尾动态开启 **Speculative Decoding / MTP** 把单条长尾生成提速 2~3×！',
      'L3 终极解法 (**Partial Rollout + Async**)：Rollout 池永不停歇流式填充新 Prompt；或到固定 Token 预算后暂停未完轨迹，把已生成前缀存入 Buffer，下个 Step 权更新后直接利用 Prefix Cache 续写剩余部分（注意记录每段生成的权重版本号或用 IS 修正）。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（严防长度偏置）：有同学提出“Over-sampling 1.5 倍 Prompt，只要最快的那 1.0 倍生成完了就把剩下 0.5 倍全部 Abort 丢弃”，这样确实消除了长尾，但上线跑了一天模型能力暴跌，为什么？',
      },
      {
        stage: 'corner',
        question: '追问 2（末尾长尾开投机采样）：前面聊过推理里大 Batch 不能开投机采样，但在同步 Rollout 尾声 Batch 掉到 < 16 时，动态开启 MTP/EAGLE 投机采样能把尾巴砍短一半！实现这种“动态开关投机采样”需要引擎做什么配合？',
      },
    ],
    referenceAnswer: `### Principal 级长 CoT Rollout 长尾治理架构

#### 1. 为什么长尾会造成雪崩级算力浪费？
- 同步 RL 在 Rollout 结束处存在全局 **Synchronization Barrier**。
- 当 95% 短样本结束后，剩余 5% 的 32K 样本把推理引擎的有效 Batch 从 512 骤降到 5~10，彻底跌入 Roofline 极低利用率区；同时其他已完成节点的全部 GPU 处于 0% 算力空转，导致有效吞吐跌至 < 25%。

#### 2. 四层递进治理方案

1. **Level 1：零算法侵入（严格 On-policy 提效）**
   - **长尾负载均衡（Rollout Dispatch）**：不要静态把 Prompt 均分给各 vLLM 实例，改用**全局共享队列动态拉取**（Work Stealing），闲置实例自动认领新 Prompt；
   - **长尾阶段动态开启投机采样 (MTP / N-gram / EAGLE)**：当活跃请求数低于阈值（如 \`B < 32\`）时自动切换投机采样，把最慢尾巴的 TPOT 压缩 2×；
   - **早停复读机**：检测严重 N-gram 无限重复输出的坏样本提前熔断（需与算法对齐 Reward）。
   > ⚠️ **反模式警示**：绝对不能“Over-sample 后粗暴丢弃未完成的长样本”——这会引入严重的 **Short-Response Selection Bias（长度截断偏置）**，导致模型再也学不会深度长思考！

2. **Level 2：Partial Rollout（Kimi k1.5 / AReaL 范式）**
   - 每个 Rollout 阶段只允许生成固定窗口（例如 4K/8K tokens 或固定时间预算）；
   - 到点后尚未吐出 EOS 的长轨迹**不丢弃**，而是将已生成的 Token 序列挂起存入 Replay Buffer；下一轮权重更新后，把挂起的半截序列作为 Prompt 继续续写，直至完成再送入训练（保证长思考分布完整无偏）。

3. **Level 3：解耦式半异步 / 全异步 RL（StreamRL / AReaL / Async-GRPO）**
   - 分离 Rollout 池与 Train 池，Rollout 引擎保持 **Continuous Batching 100% 满载**永不空等，轨迹完成后流式推入异步队列，凑满一个 Global Train Batch 即触发训练，并定期异步推送最新权重。
   - **Staleness 防护**：给每条轨迹标记策略版本差 $\\Delta v = v_{train} - v_{gen}$，丢弃超过 \`max_staleness\`（如 2~4 步）的过时样本，并用 **Importance Sampling Clipping (PPO/GRPO-Clip)** 保障策略更新稳定性。`,
  },
  {
    id: 'rl-05',
    track: 'rl',
    title: 'RL 共置架构显存共存：vLLM Sleep Mode、CUDA VMM (CuMem) 与 Optimizer Offload',
    level: 'P7',
    frequency: '高频',
    incident: false,
    tags: ['Sleep Mode', 'CuMem', 'CUDA VMM', 'CPU Offload', '显存碎片'],
    summary: '揭秘同一张 GPU 上同时挂载训练引擎与推理引擎不 OOM 的底层机制：为什么 empty_cache 不够，必须上 CUDA VMM 物理页解绑？',
    contextSpecs: ['vLLM Sleep Level 1/2', 'cuMemUnmap / VMM', 'Pinned Memory Offload', 'CUDA Graph 保留'],
    prompt: '在 Colocated（共置）RL 框架中，同一张 80GB H100 上既初始化了 PyTorch FSDP/Megatron 训练引擎，又初始化了 vLLM/SGLang 推理引擎。很多工程师自己写脚本时发现：哪怕 Rollout 完调了 `torch.cuda.empty_cache()`，训练还是立刻 OOM，或者反过来切回推理时 CUDA Graph 全部报错失效。请深入 CUDA 显存分配底层，解释为什么 `empty_cache()` 不够，以及现代引擎的 **Sleep Mode (基于 CUDA VMM / CuMem)** 和 **CPU Offload** 是怎么实现零碎片无缝显存腾挪的？',
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
        label: 'CUDA Virtual Memory Management (CuMem / cuMemUnmap)：解绑物理显存页 (Physical Allocation) 归还 GPU，但保留虚拟地址空间 (VA) 映射句柄！',
        weight: 28,
        keywords: ['cumem', 'vmm', '虚拟地址', '物理显存', 'unmap', 'map', '解绑', '保留地址', 'va', 'virtual memory'],
        probe: '如果把 vLLM 的 KV Cache 和权重直接 `free` 掉，下次唤醒时重新 `malloc` 虚拟地址变了，CUDA Graph 就全废了。CUDA VMM (CuMem) 是怎么做到“释放 99% 物理显存，但虚拟指针地址纹丝不动”的？',
      },
      {
        id: 'sleep-levels',
        label: 'vLLM Sleep Mode Level 1 (权重 Offload 到 CPU，丢弃 KV 物理页) vs Level 2 (权重和 KV 物理页全丢弃，唤醒后直接从训练引擎覆写同步权重)',
        weight: 22,
        keywords: ['sleep', 'level 1', 'level 2', 'offload', '丢弃', 'discard', '唤醒', 'wakeup', '覆写'],
        probe: 'vLLM 的 Sleep Mode 分 Level 1 和 Level 2。在 RL 训练场景里，为什么更推荐用 Level 2（直接丢弃推理权重物理页，根本不往 CPU 拷）？',
      },
      {
        id: 'train-offload',
        label: '训练侧显存腾挪：Rollout 前将 FP32 Optimizer States (占 12Ψ 最大头) 和 Param/Grad 异步 Offload 到 CPU Pinned Memory',
        weight: 16,
        keywords: ['optimizer', '12', 'cpu', 'pinned', 'offload', '异步', 'pcie', '训练侧', '腾出'],
        probe: '反过来，在训练完要切回 Rollout 之前，训练引擎自己把显存占满了，推理没地方放 KV Cache，训练侧需要 Offload 哪些大头？',
      },
      {
        id: 'pcie-perf',
        label: 'PCIe 带宽瓶颈与主机内存容量估算：72B 优化器高达 ~860GB CPU RAM，需大内存宿主机 + Pinned Memory + 多卡并发 D2H/H2D',
        weight: 12,
        keywords: ['pcie', '主机内存', 'host', 'ram', 'pinned', 'numa', '带宽', 'oom'],
        probe: 'Offload 时很多人忽略了 Host CPU 内存也会 OOM！8 卡跑 72B 时主机内存至少要配多大，D2H/H2D 传输怎么提速？',
      },
    ],
    hints: [
      'L1 方向提示：核心黑科技是 **CUDA VMM (`cuMemAddressReserve` + `cuMemCreate` + `cuMemMap` / `cuMemUnmap`)**，实现了“虚拟地址”与“物理显存”的彻底解耦！',
      'L2 Sleep Level 2 洞察：RL 每一轮训练结束后权重都会变，所以 Rollout 完根本不需要费劲把旧推理权重拷贝到 CPU（Level 1）；直接 **Level 2 彻底 `cuMemUnmap` + `cuMemRelease` 物理页**，下轮唤醒时重新 `cuMemMap` 物理页并由训练引擎直接 Weight Sync 灌入新权重，既省 Host 内存又省 PCIe 拷贝时间！',
      'L3 保持 CUDA Graph：因为虚拟地址区间（VA）始终保持 Reserved 未释放，重新 Map 新物理页到同一 VA 后，之前录制的所有 CUDA Graph 指针全部 100% 合法，唤醒即可直接满速跑！',
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
    referenceAnswer: `### Staff 级底层显存共存机制

#### 1. 为什么 \`torch.cuda.empty_cache()\` 远远不够？
- \`empty_cache()\` 只能回收 PyTorch Caching Allocator 中**已经没有 Tensor 引用**的空闲缓存块。
- 推理引擎（vLLM/SGLang）预先占用的巨量 **KV Cache Pool**、模型权重张量以及绑定了固定显存指针的 **CUDA Graph** 全部处于被引用状态，\`empty_cache()\` 动不了分毫；若直接 \`del\` 销毁重建，冷启动初始化和重录 CUDA Graph 耗时动辄几十秒。

#### 2. 核心黑科技：CUDA VMM (CuMem) + Sleep Mode
- 利用 CUDA Driver API 的虚拟内存管理将 **虚拟地址 (VA)** 与 **物理显存句柄 (Physical Handle)** 解耦：
  - 初始化时用 \`cuMemAddressReserve\` 锁死一段固定 VA 区间，并挂接物理页；
  - **Sleep（休眠）**：调用 \`cuMemUnmap\` + \`cuMemRelease\` 释放物理显存页归还给 GPU（供训练引擎使用），但**保留虚拟地址映射表不动**！
  - **Wake Up（唤醒）**：重新向驱动申请物理页并用 \`cuMemMap\` 挂回**完全相同的虚拟地址**！所有 KV 池指针与 **CUDA Graph 零重录、毫秒级复活**。
- **Sleep Level 1 vs Level 2**：
  - **Level 1**：把权重拷到 CPU 备份，KV 物理页直接丢弃（适合模型权重不变的休眠场景）；
  - **Level 2（RL 首选）**：因为训练完权重反正会更新，直接将**旧推理权重和 KV 物理页全部原地丢弃**（零 D2H 拷贝、零 CPU RAM 占用），唤醒后先映射权重页接收训练引擎 Weight Sync，再映射 KV 页开始 Rollout！

#### 3. 训练侧 Offload 与防碎片实践
- 训练完成后，将最大的 **FP32 Optimizer States (12Ψ)** 和必要参数异步拷贝到 **Pinned Host Memory**（注意绑定对应 NUMA 节点，单机 8 卡跑大模型建议配 ≥ 1~2TB Host RAM）；
- 开启 \`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\`（底层同样基于 CUDA VMM 实现动态段扩展），根治多轮切换后的虚拟块碎片化 OOM。`,
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
        label: '推理与环境完全异步解耦：请求一旦生成 Tool Call 立即挂起/释放推理槽位交给环境池执行，绝不阻塞 GPU 等待 CPU/IO 沙箱',
        weight: 24,
        keywords: ['解耦', '异步', 'async', '事件循环', '不阻塞', 'gpu空等', 'io', '并发', 'scheduler', '挂起'],
        probe: '如果按传统同步 Batch 跑多轮 Agent Rollout（等所有样本工具执行完再一起发下一轮推理），GPU 利用率会惨到什么程度？怎么设计异步调度环？',
      },
      {
        id: 'sandbox-isolation',
        label: '安全与极速沙箱池：基于 Firecracker MicroVM / gVisor / Kata / Rootless Container 预热池 + Copy-on-Write 文件系统快照（如 OverlayFS/CRIU）实现百毫秒级环境重置',
        weight: 24,
        keywords: ['firecracker', 'gvisor', 'microvm', '沙箱', 'sandbox', '隔离', '快照', 'snapshot', '预热', 'cow', 'overlayfs', 'cgroup', 'seccomp'],
        probe: '模型在探索期会写出 `rm -rf /`、Fork Bomb、死循环或者打爆宿主机网络的危险代码，同时我们每秒要起几千个 SWE-Bench 仓库环境，用普通 Docker 每次 `git clone` 根本来不及，沙箱层怎么做到“强安全 + 百毫秒极速复用”？',
      },
      {
        id: 'prefix-kv-affinity',
        label: '多轮 KV Cache 复用：同一 Agent 会话的 20 轮交互高度共享前缀，配合 Session-Affinity 路由 + SGLang RadixAttention / 分层 KV 缓存避免每轮重算 Prefill',
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
        label: '容错与长尾治理：严格的单步/全轨迹超时控制 (CPU/内存/磁盘/网络 cgroup 限额)、环境失败重试区分“模型写错代码”与“沙箱基础设施故障”',
        weight: 14,
        keywords: ['超时', 'timeout', '限额', '故障', '重试', '基础设施', 'reward', '污染'],
        probe: '如果某个代码测试超时是因为沙箱宿主机 CPU 负载太高（Infra 故障）而非模型代码有死循环，直接判 Reward=0 会污染训练信号！系统怎么区分并处理？',
      },
    ],
    hints: [
      'L1 方向提示：Agentic RL 架构核心是三大支柱——**异步 Token-level 推理-环境事件环**、**预热微虚机快照沙箱池**、**多轮 Session 亲和 KV 缓存复用**。',
      'L2 显存与延迟权衡：沙箱跑 pytest 可能花 15 秒，如果 15 秒内死占着 GPU KV Cache 不放，显存瞬间爆掉；可以把休眠期会话的 KV 异步 Offload 到 Host DRAM/Mooncake KVCache Pool，工具返回后再按 Session 亲和拉回！',
      'L3 训练侧陷阱：多轮轨迹中必须对 Observation (Tool Output) 打严格的 \`loss_mask = 0\`，且如果每轮工具调用都在独立截断，需要支持 Prefix Sharing / Tree Packing 避免同一前缀重复做训练前反向。',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（KV Cache 驻留 vs 驱逐）：一条 SWE Agent 轨迹执行一次编译测试要花 20 秒，这 20 秒里它的 48K 上下文 KV Cache 是该留在 GPU 显存里等它，还是踢出去？给出定量权衡思路。',
      },
      {
        stage: 'corner',
        question: '追问 2（Tokenization Drift）：在多轮交互里，如果每轮都把历史对话转回 String 文本再重新用 Chat Template Tokenize，会导致严重的 Token 边界漂移（Retokenization Mismatch），进而破坏 Prefix Cache 和 Logprob 对齐。怎么彻底根治？',
      },
    ],
    referenceAnswer: `### Principal 级 Agentic RL 基础设施架构

#### 1. 全异步 Rollout-Environment 状态机
- 将每条 Agent Trajectory 建模为独立的异步状态机（\`LLM_GEN -> TOOL_PARSE -> ENV_EXEC -> OBS_APPEND -> LLM_GEN\`），用高并发轻量调度器维持数倍于推理槽位的 **In-flight Trajectories**（如 GPU 并发 256，系统同时维护 2000 个活跃会话）。
- 只要某会话产出 \`</tool_call>\`，立刻释放 GPU 计算槽位，后台派发给分布式沙箱池执行；GPU 零空等 IO。

#### 2. 高安全、百毫秒级代码沙箱池
- **隔离底座**：使用 **Firecracker MicroVM** 或 **gVisor/Kata Containers**（强内核隔离防逃逸），配合严格 \`cgroups\`（CPU/Memory/PIDs 防 Fork Bomb）、只读挂载与默认断网策略。
- **极速复用**：对 SWE-Bench 仓库提前构建镜像与 **OverlayFS / 内存磁盘快照 (Snapshot)**，维护按 Repo 分组的 **Warm Sandbox Pool**，重置环境 < 100ms。
- **异常信号分离**：区分 **Code Error (模型责任 → Reward=0)** 与 **Infra Error (宿主机 OOM/网络抖动/容器崩溃 → 标记 Invalid 丢弃或重试，绝不喂脏梯度)**。

#### 3. 多轮 KV Cache 复用与 Token-ID Native 链路
- **Session-Aware Prefix Routing + 分层 KV 池**：同一会话连续多轮尽量路由到同一推理节点利用 **RadixAttention**；若工具执行耗时较长（如 > 5~10s），通过 Mooncake / LMCache 将其 KV 异步卸载到节点 DRAM，工具返回后零重算极速拉回 GPU。
- **Token-ID Native (防 Retokenization Drift)**：整个 Rollout 闭环以 **Token ID 序列**为一等公民追加拼接（仅对新增 Observation 做局部 Tokenize），不把全历史转回纯文本再重编码，彻底杜绝 BPE 合并边界变化导致的 Prefix Cache 失效与 Logprob 漂移！
- **训练 Loss Mask**：对所有环境 Observation Tokens 严格置 \`loss_mask = 0\`，仅对模型生成的 Reasoning + Tool Call Tokens 算策略梯度。`,
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
    prompt: '【硬核真实排障】你们刚把 72B GRPO 切到了 vLLM (FP8 Rollout) + Megatron (BF16 Train) 的加速方案。跑了 50 步后算法同学紧急找你：同一个模型权重还没更新呢，发现 vLLM Rollout 输出的 `old_logprob` 和 Megatron Forward 重新算出的 `logprob` 竟然存在肉眼可见的偏差，导致部分 Token 的 Importance Ratio $r_t = \\pi_{train}/\\pi_{rollout}$ 飙到 5 以上，KL 散度异常甚至出现负值、训练后期 Reward 直接崩盘！请分析所有可能的系统级根因并给出排查与修复方案。',
    rubric: [
      {
        id: 'precision-kernel',
        label: '精度与算子实现差异：FP8 Rollout vs BF16 Train 量化误差、不同 GEMM/FlashAttention/PagedAttention 规约顺序 (Floating-point 非结合律) 与 MoE Top-K 边界翻转',
        weight: 26,
        keywords: ['fp8', 'bf16', '非结合律', '累加顺序', 'kernel', 'flashattention', 'pagedattention', 'moe', '路由', 'top-k', 'router', '算子'],
        probe: '哪怕两边都用 BF16 不用 FP8，为什么 vLLM 和 Megatron 对同一句话算出的 logprob 依然对不上？如果是 MoE 模型，这种微小误差为什么会被指数级放大？',
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
        label: 'KL 散度负值/爆炸原理解析：无偏估计器 (如 k3: exp(logp_ref - logp) - (logp_ref - logp) - 1) 理论上非负，若发现异常常因 logp_old 混用或跨精度大偏移导致 ratio 失控',
        weight: 18,
        keywords: ['kl', 'k3', 'ratio', '重要性', 'importance', 'clip', 'logprob', '负值', '方差'],
        probe: '为什么训练中会看到 $r_t$ 飙到 5 以上（尤其在低概率/长尾 Token 上）？这对 GRPO 的策略梯度会产生什么破坏？',
      },
      {
        id: 'mitigation-sys',
        label: '系统与算法联合修复：同步用 Training 引擎重算 `old_logprob` (基准验证)、敏感层保 BF16（Router/LM Head/Attention）、使用 Rollout Importance Sampling (TIS / Masked IS) 显式修正推理-训练偏移',
        weight: 22,
        keywords: ['recompute', '重算', 'tis', 'truncated importance sampling', '修正', 'lm head', 'router', 'bf16', '校准', '对齐'],
        probe: 'FP8 Rollout 速度太香了实在不想关掉，那怎么在不放弃 FP8 Rollout 的前提下，让 RL 训练长期稳定不崩？',
      },
      {
        id: 'unit-verify',
        label: '建立离线数值一致性门禁测试：Step 0 同权重下用相同 Prompt 对比两引擎 Logits/Top-1/Max-Abs-Diff 与 KL 散度 (< 1e-3)',
        weight: 10,
        keywords: ['门禁', '单测', 'step 0', 'logits', 'max diff', '对比', '验证', 'ci'],
        probe: '作为 Infra 负责人，怎么把这类静默精度 Bug 彻底拦截在上线跑千卡集群之前？',
      },
    ],
    hints: [
      'L1 方向提示：按「先查硬 Bug（权重/RoPE/Tokenize 错位）→ 再查数值源（FP8/BF16、Kernel 规约、MoE Router 翻转）→ 最后给治理方案（敏感层高精度 + TIS 重要性修正）」的排查逻辑。',
      'L2 MoE 蝴蝶效应：在 MoE 模型里，哪怕 BF16 浮点误差只有 \`1e-3\`，只要落在 Router Top-8 边界上，就会导致同一 Token 在推理和训练时**路由到完全不同的专家**，瞬间引爆巨大的 Logprob 偏移！',
      'L3 工业界 SOTA 治本方案：(1) **Rollout Importance Sampling (Truncated IS / IcePop)** 明确把 $\\pi_{rollout}$ 当作行为策略做截断重要性采样修正；(2) MoE 可采用 **Routing Replay**（训练时复用 Rollout 记录的专家路由索引）或把 Router/LM Head 死锁在 FP32/BF16！',
    ],
    followUps: [
      {
        stage: 'deepen',
        question: '追问 1（MoE Routing Replay）：对于 671B MoE RL，哪怕两边都是 BF16，只要 Router 选出的专家差了 1 个，logprob 就会严重偏移。听说过 **Rollout Routing Replay (R3)** 吗？它怎么做到零精度漂移，会带来什么显存/通信成本？',
      },
      {
        stage: 'corner',
        question: '追问 2（低概率 Token 放大效应）：观察发现 99% 的高概率 Token 两边 logprob 差不到 0.01，但某些采样出来的低概率探索 Token（如 $p=0.001$）误差高达 $\\Delta \\log p = 2.0$，为什么低概率 Token 对精度差异这么敏感？',
      },
    ],
    referenceAnswer: `### Staff 级 Train-Inference Mismatch 排障指南

#### 1. 第一优先级：排除致命硬 Bug（Step-0 门禁自检）
在 Step 0 刚做完一次 Weight Sync（未更新参数）时，固定同一条 Token ID 序列分别喂给 vLLM 与 Megatron，逐层比对 Hidden States 与 Logits：
- 检查 **QKV / GateUp 切分顺序**、**RoPE Interleaved vs NeoX 布局**、**LM Head / Embedding 权重共享与同步漏项**；
- 检查 **BOS/EOS 与 Retokenization 漂移**（必须直接传 Token IDs 而非重编码字符串）；
- 检查 **Varlen / Packing Attention Mask**（是否跨样本泄漏了 Attention）。

#### 2. 第二优先级：底层数值与算子差异根因
1. **FP8 Rollout vs BF16 Train 量化噪声**：低概率尾部 Token 在 Softmax 指数放大下对微小的 Logit 误差极度敏感（Logit 差 0.5 就能让概率比差 1.65×）。
2. **浮点非结合律 & Kernel 差异**：vLLM PagedAttention/FlashInfer 与 Megatron FlashAttention/cuBLAS 的分块归约顺序不同，加上 Batch Size 不同导致矩阵乘 Split-K 策略改变，产生 Batch-Invariance 破坏。
3. **MoE Router 边界翻转（蝴蝶效应）**：1e-3 的 Hidden State 差异导致 Top-K 路由换了专家，Logprob 瞬间雪崩。

#### 3. 工业级根治方案
- **精准隔离高精度模块**：Weight Sync 与 FP8 Rollout 时，强制将 **LM Head、MoE Router Gate、RMSNorm、Embedding** 保持在 BF16/FP32，并在 FP8 上采用细粒度 \`1×128 / 128×128\` Blockwise 量化而非粗糙 Per-Tensor。
- **算法-系统联合修正（Truncated Importance Sampling / IcePop）**：不再盲目假设 $\\pi_{rollout} = \\pi_{train\_old}$，显式保存推理侧 \`rollout_logprob\`，在 Loss 中引入截断重要性采样系数 $\\min(\\frac{\\pi_{train}}{\\pi_{rollout}}, C)$ 或对偏移超阈值的极端 Token 做 Mask 屏蔽。
- **MoE 终极稳定器（Routing Replay）**：Rollout 时顺带回传每个 Token 选中的 Top-K 专家索引（每 Token 每层仅几字节），训练侧 Forward 强制按同一专家路由计算，彻底消除路由分歧。`,
  },
];
