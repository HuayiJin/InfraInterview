/**
 * Offline AI Infra Selection Explainer & Context-Aware Glossary
 * Works zero-config offline and serves as graceful fallback when Live LLM is not configured or unavailable.
 */

const GLOSSARY = [
  {
    term: 'TP+SP (张量并行 + 序列并行)',
    patterns: ['tp+sp', 'sequence parallel', 'allgather', 'reducescatter', '8(t−1)/t', '8(t-1)/t', 'bsh'],
    explain: 'TP 将 Linear 权重按列/行切分到 T 张卡；开启 Megatron SP 后，LayerNorm/Dropout 等非矩阵层的激活也按序列维切分。原本每次 AllReduce 被拆为一次进入前的 AllGather 与一次出口处的 ReduceScatter，通信总量不变，但显著降低非矩阵层的重复激活显存。',
    derivation: '1 个子层（Attention 或 MLP）在 BF16（2B）下：1 次 AllGather 发送 (T−1)/T × 2bSH 字节，1 次 ReduceScatter 发送 (T−1)/T × 2bSH 字节，合计 4(T−1)/T × bSH 字节。每个标准 Transformer 层有 Attention + MLP 两个子层，故前向单层发送 8(T−1)/T × bSH 字节；反向不含重算同量。',
    example: '设 T=4、b=1、S=8192、H=8192（BF16）：bSH = 67.1M 元素。每层前向每卡发送 8 × (3/4) × 67.1M ≈ 402.6 MB；按单向 NVLink 450 GB/s 计算，理论纯传输下界约 0.9 ms/层。',
    probe: '你可以这样追问候选人：“为什么开启 SP 后通信总量没变，却能大幅省下 LayerNorm/Dropout 的激活显存？若同时开启了 Full Activation Checkpointing，反向时 TP+SP 的通信量会增加多少？”',
  },
  {
    term: 'HSDP / FSDP2 (混合分片与每参数 DTensor)',
    patterns: ['hsdp', 'fsdp2', 'dtensor', 'fully_shard', 'reshard_after_forward', 'set_reshard_after_backward', 'flatparameter'],
    explain: 'FSDP2 用沿 dim-0 切分的 per-parameter DTensor 替代 FSDP1 的 FlatParameter，保留参数名与独立冻结语义；HSDP 在小通信域（如机内 8 卡或 64 卡）内做 ZeRO-3 参数分片，在更大域间做副本梯度 AllReduce，在显存与跨机 AllGather 开销之间折中。',
    derivation: '设总卡数 = TP × PP × DP，DP 内再拆为 D_shard 路分片 × D_rep 路复制。每卡常驻静态状态 ≈ 16Ψ / (TP × PP × D_shard)。前向后不重分片（reshard_after_forward=False）可省去同一 micro-batch 反向前的参数 AllGather，但会抬高活跃参数峰值。',
    example: '70B（16Ψ ≈ 1120 GB）取 TP=2、DP=512，若 DP 内用 64 路分片 × 8 路复制，每卡静态状态 = 1120 / (2×64) = 8.75 GB，还需另留当前层 AllGather 展开缓冲、激活与工作区。',
    probe: '你可以这样追问：“如果做 8 步 Gradient Accumulation，关闭梯度同步 (`no_sync`) 会顺带省掉 ZeRO-3 的参数 AllGather 吗？什么时候该调 `reshard_after_forward=False`？”',
  },
  {
    term: '1F1B / Interleaved VPP / Zero-Bubble 气泡率',
    patterns: ['1f1b', 'vpp', 'interleaved', 'zero-bubble', 'zb-h1', 'zb-h2', '气泡率', 'bubble', 'dgrad', 'wgrad', 'post-validation'],
    explain: '流水线并行（PP）的空闲气泡来自 Warmup 和 Cooldown 阶段。1F1B 相比 GPipe 气泡率相同但把首段激活峰值从 O(m) 降到 O(p)；Interleaved VPP 让每卡交替负责 v 个虚拟段，将理想气泡时间压至 1/v 但 P2P 消息翻倍；Zero-Bubble 将反向拆成关键路径的 dX (B) 与可延后的 dW (W) 来填补空档。',
    derivation: '设 p 个阶段、每副本 m 个 micro-batch、阶段耗时相等且忽略通信：\n- 1F1B 气泡占比 = (p−1) / (m + p − 1)\n- Interleaved VPP (v 个等时 chunk)：气泡时间缩至 1/v，占比 = (p−1) / (v·m + p − 1)\n- ZB-H1（设 F=B=W=t）：原气泡 3(p−1)t 降至 (p−1)t，理想占比 = (p−1) / (3m + p − 1)。',
    example: 'p=8、m=16 时：标准 1F1B 气泡 = 7/23 ≈ 30.4%；80 层模型要求 8v 整除 80，故 v=4 不合法，v=2 时气泡降至 7/39 ≈ 17.9%。',
    probe: '你可以这样追问：“Interleaved VPP 把气泡压低了，为什么在模型层数少、或跨机 P2P 延迟高时反而可能变慢？Zero-Bubble 延迟执行 W 会给全局 Grad Norm 带来什么同步问题？”',
  },
  {
    term: 'Ring Attention vs DeepSpeed Ulysses (Context Parallelism)',
    patterns: ['ring attention', 'ulysses', 'zig-zag', 'striped', 'hkv', '2hq+2hkv', 'context parallel', 'usp'],
    explain: '两种把超长序列切到 P 张卡的并行范式：Ring 保留本地 Q 块，沿环路轮转其他卡的 K/V 并用 Online Softmax 累积，通信易与计算重叠且不受 KV head 数限制；Ulysses 在 Attention 前后通过 All-to-All 在“序列切分”与“Head 切分”间转置，总通信量随 P 增大而下降，但无复制切分要求本地 KV head 数能被 Ulysses 度数整除。',
    derivation: '单次前向、batch=1、每 rank 向其他 rank 发送元素数：\n- 完整 Ring：每步发 S/P 块 K 和 V 共 2(S/P)Hkv，发 P−1 步 ⇒ 2(P−1)S·Hkv / P。\n- Ulysses：Q/K/V/O 4 次 All-to-All，每张量本地大小 (S/P)·width，其中 (P−1)/P 发往远端 ⇒ (P−1)S(2Hq+2Hkv) / P²。\n- 两者比值：MHA 下 Ring/Ulysses = P/2；GQA 下为 P·Hkv / (Hq + Hkv)。',
    example: '70B (64Q / 8KV) 若 TP=2，每个 TP rank 仅剩 4 个 KV head；开 CP=16 时，机内 8 卡可排 TP=2 × Ulysses=4（不复制 KV），跨机再配 Ring=4。',
    probe: '你可以这样追问：“如果模型是 GQA（8 KV heads），同时开了 TP=2 和 CP=16，为什么机内不能直接开 Ulysses=8？Document Packing 隔断 mask 下，简单首尾 Zig-Zag 为什么还可能不均衡？”',
  },
  {
    term: 'DualPipe & DeepEP (MoE 双向流水与通信核)',
    patterns: ['dualpipe', 'deepep', 'node-limited', 'aux-loss-free', 'grouped gemm', 'dispatch', 'combine', 'dropless'],
    explain: '大规模 MoE 跨机 EP 训练的痛点是 Top-K All-to-All 通信重且专家热点易倾斜。DeepSeek-V3 结合节点受限路由（限每 token 最多路由到 M=4 个节点并按节点去重）、动态 Bias 负载均衡（仅改路由选专家、不改门控加权值），以及 DualPipe 双向交错不同 micro-batch 的前反向，让计算与 DeepEP 通信高度重叠。',
    derivation: 'Top-8 路由若不限制节点，单 token 跨 IB 最多发 8 份；限 M=4 节点且发送前按目标节点去重后，跨 IB 至多发 4 份，进入目标节点后再借高带宽 NVLink 分发给同节点多专家。DualPipe 需持有两份参数（显存按 EP×PP 分摊）。',
    example: '在 H800（NVLink 400 GB/s 双向 vs IB 400 Gb/s ≈ 50 GB/s 单向），机内/机间带宽比约 4:1，M=4 节点受限正好匹配该带宽比，使 IB 与 NVLink 同步饱和。',
    probe: '你可以这样追问：“DeepEP 划出一部分 SM 专跑通信核，即使时间线上通信与计算 100% 重叠，为什么 GEMM 耗时还是可能略微变长？Dropless 不丢 token 时遇到极端热点如何防 OOM？”',
  },
  {
    term: 'FP8 训练 (E4M3/E5M2、细粒度 Scaling、WGMMA 14-bit 累加)',
    patterns: ['e4m3', 'e5m2', 'delayed scaling', '1×128', '1x128', '128×128', '128x128', 'wgmma', 'mxfp8', 'nvfp4', 'amax'],
    explain: 'FP8 训练需区分三个独立维度：数值格式（E4M3 最大 448 精度高，E5M2 最大 57344 动态范围宽）、缩放粒度与时机（per-tensor vs 1×128/128×128 块缩放；delayed 历史窗口 vs current 在线缩放），以及硬件 Tensor Core 内部累加精度。',
    derivation: '设反量化因子 s，量化值 q = FP8(x/s)。Hopper FP8 WGMMA 内部部分和仅保留约 14 位尾数对齐精度，长 K 内积会产生漂移；DeepSeek-V3 每累加 N_C=128 个元素（4 条 WGMMA）便将部分和搬到 CUDA Core 做完整 FP32 累加并乘入块 scale。',
    example: 'Blackwell MXFP8 原生支持每 32 个元素共享 1 个 E8M0 块 scale，直接在点积硬件路径应用缩放，大幅减少 Hopper 上手动 CUDA Core 晋升累加的软件开销。',
    probe: '你可以这样追问：“如果把延迟缩放改成在线缩放，但依然用 Per-Tensor 粒度，能解决离群值导致小值归零吗？为什么权重选 128×128 而激活选 1×128？”',
  },
  {
    term: '6ND、MFU 与 HFU',
    patterns: ['6nd', '8nd', 'mfu', 'hfu', '6.3×10', '6.3e24', '12hs', 'strong scaling'],
    explain: 'MFU（Model FLOPs Utilization）衡量“有效模型计算吞吐”占“硬件标称 dense 峰值”的比例；HFU（Hardware FLOPs Utilization）则把激活重算等额外实际执行的计算也计入分子。比较 MFU 时必须统一 6ND 近似、是否含 Attention 二次项、是否含停机以及稠密/稀疏峰值口径。',
    derivation: 'Dense 参数矩阵乘：前向约 2ND、反向（dX + dW）约 4ND，合计 6ND。单次完整前向重算多做 2ND，实际执行约 8ND，故 HFU/MFU ≈ 4/3。若观测到 HFU/MFU 远大于 4/3（如 68%/38%），说明另有额外重算、padding 或统计口径偏差。',
    example: '70B 训 15T tokens：6ND = 6.3×10²⁴ FLOPs。1024×H100 (989 TFLOP/s) 按 48% MFU 有效算力 4.86×10¹⁷ FLOP/s，耗时 ≈ 150 天；扩到 4096 卡压进 45 天需端到端 MFU ≥ 40%。',
    probe: '你可以这样追问：“老板说 GPU-Util 100%、HFU 68%、MFU 只有 38%，单靠开一次 Full Checkpointing 能解释这 30% 的剪刀差吗？你会去核查哪几项？”',
  },
  {
    term: 'MLA (Multi-Head Latent Attention) 与矩阵吸收',
    patterns: ['mla', '矩阵吸收', 'weight absorption', 'decoupled rope', 'c_kv', 'uₖ', 'uᵥ', '68.625', 'flashmla'],
    explain: 'MLA 将每个 token 的 K、V 联合压缩为一个低维潜向量 c_KV（如 512 维）外加共享的 64 维解耦旋转向量 k_R 存入缓存。Decode 时利用矩阵乘法结合律，把 K 上投影吸收到当前 q，把 V 上投影挪到对潜向量加权求和之后，全程无需展开历史 K/V。',
    derivation: '1) 打分吸收：qᵀ(U_k c_j) = (U_kᵀ q)ᵀ c_j（只对当前 1 个 q 做变换）。\n2) 输出吸收：Σ a_j (U_v c_j) = U_v (Σ a_j c_j)（先在低维潜空间加权求和再上投影）。\n3) 为何解耦 RoPE：若直接施加 RoPE，qᵀ R_tᵀ R_j U_k c_j 中相对位置矩阵 R_{t−j} 随历史 j 变化，阻隔了 q 与 U_k 合并，故拆分独立 64 维 RoPE 分支。',
    example: 'DeepSeek-V3 (61 层, d_c=512, d_r=64, BF16)：每 token 仅缓存 61 × 576 × 2 = 70,272 B = 68.625 KiB，约为 70B GQA (320 KiB/token) 的 1/4.7。',
    probe: '你可以这样追问：“矩阵吸收在 Decode 很省带宽，为什么在长 Prompt Prefill 阶段反而常常先展开 K/V 走标准 FlashAttention？”',
  },
  {
    term: 'Roofline 模型、Ridge Point 与算术强度 (OI)',
    patterns: ['roofline', 'ridge', '算术强度', 'oi', '2b/b_w', '2bkn', '295', '591', 'memory-bound', 'compute-bound'],
    explain: 'Roofline 通过硬件拐点 Ridge = 峰值算力 / 内存带宽，以及算子算术强度 OI = 计算量(FLOPs) / 搬运字节(Bytes) 来判定算子卡在带宽还是算力：OI < Ridge 为 Memory-bound，OI > Ridge 为 Compute-bound。',
    derivation: '对 [B, K] × [K, N] 线性层：完整 OI = 2BKN / (b_w·KN + b_x·BK + b_y·BN)。忽略激活读写时退化为 OI ≈ 2B / b_w：\n- BF16 (b_w=2)：OI = B，H100 Ridge ≈ 989/3.35 ≈ 295 ⇒ 临界 B ≈ 295；\n- FP8 (b_w=1)：OI = 2B，Ridge 同步翻倍至 ~591 ⇒ 临界 B 依然 ≈ 295；\n- W4A16 (b_w=0.5, FP16 MMA)：OI = 4B，Ridge 仍 ~295 ⇒ 临界 B 提前至 ~74。',
    example: 'Decode Self-Attention（长历史）：FLOPs ≈ 4S·n_q·d，KV 读 ≈ 2S·n_kv·d·b_kv ⇒ OI ≈ 2(n_q/n_kv)/b_kv。BF16 MHA 仅 1 FLOP/Byte，8:1 GQA 提升到 8 FLOPs/Byte，但仍远低于 295。',
    probe: '你可以这样追问：“为什么从 BF16 换到原生 FP8，Decode 线性层的临界 Batch Size 基本还是 ~295 而不是 590？增大 Batch 为什么帮不了普通 MHA 的 Decode Attention？”',
  },
  {
    term: 'PagedAttention、Chunked Prefill 与 PD 分离',
    patterns: ['pagedattention', 'block table', 'chunked prefill', 'pd分离', 'pd 分离', 'distserve', 'mooncake', 'nixl', 'itl', 'tpot', 'ttft', 'radixattention'],
    explain: 'PagedAttention 用定长物理块+页表消除连续预留的外部碎片；Chunked Prefill 按每步 token budget 切分长 Prompt 与 Decode 混批以压平 ITL 毛刺；当长输入与低延迟 Decode 干扰超出单池调节空间，或两侧最优硬件/并行度分化时，演进为 Prefill-Decode 分离并用 RDMA 流水传输 KV。',
    derivation: 'PD 分离实例配比估算：设到达率 λ、未命中平均输入 I、平均输出 O，单 P/D 实例在目标 SLO 下服务率为 r_P、r_D，则平均吞吐下界实例数分别为 ⌈λI/r_P⌉ 与 ⌈λO/r_D⌉（另加峰值/排队余量）。单请求平均 TPOT = (t_last − t_first)/(n − 1)，区别于逐 token 的 ITL。',
    example: '70B GQA 32,768 tokens 逻辑 KV = 10 GiB；经单条 400 Gb/s (50 GB/s) 链路序列化下界为 10×2³⁰ / 50×10⁹ ≈ 215 ms，需靠多 rail、逐层异步流水或 FP8/MLA 压低移交延迟。',
    probe: '你可以这样追问：“在 PD 分离里，如果 P 节点算完先吐出首 token 再传 KV，KV 传输变慢会反映在 TTFT 还是首到第二 token 的间隔里？如果目标 D 节点显存满了怎么防死锁？”',
  },
  {
    term: '投机采样 (Speculative Decoding)、EAGLE 与 MTP',
    patterns: ['speculative', '投机采样', 'rejection', 'eagle', 'mtp', 'e=1+α', '3.3616', '2.55', 'bonus token'],
    explain: '利用低 Batch Decode 空闲算力，让轻量草稿先吐 K 个候选，目标模型单次并行验证 K+1 个位置。通过 Rejection Sampling 在数学上严格保持与目标模型完全相同的采样分布。',
    derivation: '对草稿采样 x ~ q(x)，以 min(1, p(x)/q(x)) 接受；首次拒绝时从归一化的残差分布 (p−q)₊ 重采样并截断后续。全概率展开：P(x) = min(p(x), q(x)) + (p(x)−q(x))₊ = p(x)，严格无损。若条件接受率均为 α 且未到 EOS，期望产出 E = 1 + α + … + αᴷ，理想加速 = E / (v + Kc)。',
    example: 'α=0.8、K=4 ⇒ E = 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.3616 tokens/轮；若 verify 成本 v=1、draft 每步 c=0.08，则理想加速 3.3616 / 1.32 ≈ 2.55×。',
    probe: '你可以这样追问：“在 MoE 模型上开投机采样，除了验证有效 batch 变成 B(K+1) 跨过 Ridge Point 外，对每步激活的专家权重读取量还会带来什么副作用？”',
  },
  {
    term: 'Wide-EP、Two-Batch Overlap (TBO) 与 EPLB',
    patterns: ['wide-ep', 'tbo', 'two-batch', 'eplb', 'ep32', 'ep144', '288/32', '冗余专家', 'dp attention'],
    explain: '部署超大规模 MLA+MoE（如 DeepSeek-R1 671B）时，用 DP Attention 避免 MLA 潜向量在 TP 间重复存多份，用大 EP 分摊专家权重并聚合多 DP 组请求；用 Two-Batch Overlap 拆成两微批交替隐藏 All-to-All；用 EPLB 复制热点逻辑专家压平长尾。',
    derivation: '均匀 Top-k 路由下，全局 T 个 token、E 个逻辑专家时，每专家平均获 kT/E 个 token。固定 T 增大 EP 并不会提高单专家 batch，只有扩大服务域聚合更多并发请求 T 才能让 Grouped GEMM 更饱满。',
    example: 'DeepSeek 公开部署：256 逻辑专家 + 32 冗余副本 = 288 物理槽位。Prefill EP32 每卡放 288/32 = 9 个专家；Decode EP144 每卡放 288/144 = 2 个专家。',
    probe: '你可以这样追问：“如果业务要求极低 TPOT (<15ms)，把 EP144 再扩到 EP320 一定会更快吗？TBO 拆分两微批后 GEMM 变小会带来什么反作用？”',
  },
  {
    term: 'GRPO / PPO / DPO 与长 CoT Rollout 治理',
    patterns: ['grpo', 'ppo', 'dpo', 'dapo', 'dr.grpo', 'partial rollout', 'staleness', 'hybridengine', 'colocated', 'disaggregated', 'r·tᵣ'],
    explain: '经典 RLHF-PPO 需维护 Actor/Critic/Ref/RM 四角色（独立等大静态状态达 36Ψ）；GRPO 去掉 Critic 靠同 Prompt 采样 G 组算组内相对优势，静态显存减半但算力瓶颈常转向长 CoT 自回归 Rollout。物理放置（共置 vs 分池）与策略更新节奏（同步 vs 异步）是两个正交决策。',
    derivation: '独立两池严格串行同步时，忙碌占比 = (R·t_r + T·t_t) / [(R+T)(t_r + t_t)]，恒小于 1；因此同步 On-policy 下共置分时往往更省闲置，而长尾 CoT/多轮沙箱场景下采用分池 + Partial Rollout / 有界异步更能保高吞吐。',
    example: '256 prompts × G=16 × 8192 tokens = 33,554,432 生成 token。Packing 做 GRPO loss 归一化时须跨 DP 与梯度累积聚合全局有效 token 分母，避免不等长 micro-batch 等权平均引入长度偏置。',
    probe: '你可以这样追问：“在 Partial Rollout 跨步续写中，如果权重已经更新了一版，能不能直接复用上一版生成的 KV Cache 继续吐后面一半 token？”',
  },
  {
    term: 'CUDA VMM (CuMem) 与 vLLM Sleep Mode',
    patterns: ['vmm', 'cumem', 'cumemaddressreserve', 'cumemunmap', 'sleep mode', 'level 1', 'level 2', 'expandable_segments', 'empty_cache'],
    explain: '`empty_cache()` 只能归还 PyTorch 分配器里无引用的空闲缓存；CUDA VMM 将虚拟地址区间 (VA) 与底层物理显存分配解耦——休眠时解除映射并释放物理页供训练使用，唤醒时重新申请物理页映射回原 VA，从而保住 CUDA Graph 绑定的指针地址。',
    derivation: 'vLLM v0.10.2 Sleep Mode：Level 1 备份权重到 CPU 并丢弃 KV；Level 2 同时丢弃旧权重与 KV（仅保留必要模型 buffers），专为 RL 下一轮即将覆写新权重设计，省去无谓的旧权重 D2H 拷贝与 Host RAM 占用。',
    example: '8 卡每卡 40 GB 优化器状态，单卡 PCIe 50 GB/s 理想 D2H 为 0.8s；若 8 卡共享同一 50 GB/s 瓶颈（PCIe Switch/NUMA 争抢），整机 320 GB 拷贝将骤慢至至少 6.4s。',
    probe: '你可以这样追问：“CUDA VMM 唤醒重新 `cuMemMap` 回原虚拟地址后，是不是立刻就能重放 CUDA Graph 了？还需要做哪两步才安全？”',
  },
  {
    term: 'Train-Inference Logprob 偏移、k1/k3 与 log-softmax 误差上界',
    patterns: ['k1', 'k3', 'expm1', 'log-softmax', '2ε', 'logprob', 'tis', 'routing replay', '路由回放', 'μ'],
    explain: 'RL 需明确区分三条概率：实际采样行为策略 μ、更新前训练引擎策略 π_old、当前更新策略 π_θ。精确 KL 非负，单样本 k1=log p−log q 可正可负，而 Schulman k3=exp(d)−1−d (d=log q−log p) 逐项非负且期望等于 KL。',
    derivation: '为何“p 小不等于 logit 微小误差会被放大”：对固定支持、无截断的 log_softmax(z)_i = z_i − logsumexp(z)，若每个 logit 扰动至多 ε，则 z_i 与 logsumexp(z) 至多各变 ε，故任一 token 的 |Δlog p_i| ≤ 2ε（不论 p_i 多小！）。会放大的是“相同绝对概率误差 Δp”对小 p 的相对比值。',
    example: 'Δlog p = 2 对应概率比 e² ≈ 7.39；比值超 5 对应 Δlog p > ln 5 ≈ 1.61。检查 k3 接近零的数值下溢/负值可用 FP32 下 `expm1(d) - d` 核对。',
    probe: '你可以这样追问：“既然 log-softmax 里每个 logit 误差 ε 最多导致 2ε 的 logprob 差，那现实中 MoE 模型为什么还会出现 Δlog p > 2 的巨大偏移？（提示：Top-k 路由翻转与截断采样支持变化）”',
  },
  {
    term: 'FlashAttention-1/2/3 与 Online Softmax',
    patterns: ['flashattention', 'online softmax', 'logsumexp', 'tma', 'warp specialization', 'ping-pong', 'incoherent', 'hadamard', 'flash-decoding', 'split-kv'],
    explain: 'FlashAttention 通过按块加载 Q/K/V 至片上 SRAM 并用 Online Softmax 动态修正最大值与分母，避免把 N×N 分数和概率矩阵写回 HBM；FA2 调换外循环 Q / 内循环 KV 并改为 Warp Split-Q；FA3 在 Hopper 上结合 Producer(TMA)/Consumer(WGMMA) 异步流水线掩盖慢速 Softmax。',
    derivation: '旧状态 (m, ℓ, u)，新块得分 s_j，m\' = max(m, max s_j)：\nℓ\' = exp(m−m\')·ℓ + Σ exp(s_j−m\')\nu\' = exp(m−m\')·u + Σ exp(s_j−m\')·v_j\n最终 O = u/ℓ。注意：dense FLOPs 仍为 Θ(N²d)，省下的是 HBM IO（由 Θ(Nd+N²) 降至 O(N²d²/M)，M 为单 SM 片上可用容量而非整卡之和）。',
    example: 'Decode S_q=1 且 Batch 小时 Q 维无并行度，Flash-Decoding (Split-KV) 改沿 KV 序列切分给多个 SM 并行算局部 (O_k, lse_k)，最后再归并。',
    probe: '你可以这样追问：“FA1 论文里 IO 复杂度 O(N²d²/M) 中的 M 能不能代入 H100 整卡 132 个 SM 的片上 SRAM 总和？为什么？”',
  },
  {
    term: 'NCCL Ring/Tree、BusBW/AlgBW、Rail 与 PXN',
    patterns: ['busbw', 'algbw', '2(n−1)/n', '2(n-1)/n', 'sharp', 'nvls', 'pxn', 'rail', 'rocev2', 'ecmp', 'pfc', 'ecn', 'dcqcn'],
    explain: 'Ring AllReduce = ReduceScatter + AllGather，大消息带宽效率高但步数 O(N)；Double Binary Tree 延迟 O(log N) 适合小消息或大卡数；NVLS/SHARP 将归约下沉到交换机芯片，使大 N 端点发收量接近减半；Rail 拓扑对齐同号 NIC 降低跨 Spine 压力，PXN 通过机内 NVLink 中转优化跨 Rail 通信。',
    derivation: 'N 个 rank 各有 S 字节输入：\n- Ring AllReduce 每 rank 发送与接收各 2(N−1)S/N 字节，耗时 T ≈ 2(N−1)α + 2(N−1)S/(NB)；\n- nccl-tests 中 algbw = S/T，AllReduce busbw = algbw × 2(N−1)/N（而 AllGather / ReduceScatter / AlltoAll 换算系数均为 (N−1)/N）。',
    example: 'RoCEv2 若 ECN 降速阈值高于 PFC 暂停阈值，队列来不及降速就会触发级联 PFC Pause Storm 拖垮全网；需确保 ECN < PFC XOFF 并配合多 QP / 自适应路由消除 ECMP 哈希极化。',
    probe: '你可以这样追问：“为什么 `nccl-tests` AllReduce 的 `busbw` 要乘 2(N−1)/N，而 AlltoAll 只乘 (N−1)/N？用它能直接替代交换机硬件端口流量计数吗？”',
  },
  {
    term: '训练 Step Time Profiling：HTA 关键路径与 Wait-for-Straggler',
    patterns: ['hta', 'exposed', '暴露通信', 'wait-for-straggler', 'straggler', 'with_stack', 'kineto', 'critical path', '并集'],
    explain: '多 CUDA Stream 并发时不能直接累加各 kernel 时长，必须按 GPU 时间轴并集拆成四类互斥时段：纯计算、计算-通信重叠、暴露通信（Exposed Comm）和设备空闲。511 张卡上很长的 AllReduce 未必是网络慢，常常是单卡上游变慢导致的 Wait-for-Straggler。',
    derivation: '1) 对齐同一步、同一 ProcessGroup 同一序号 collective 的入队/开始时刻，先到者在 kernel 里空等最晚到达者；\n2) 采集需跳过 warmup 仅抓 1~2 个稳态 step，关闭 `with_stack` 防止放大 CPU launch 制造虚假 Gap；\n3) 即使暴露通信为 0，并发通信核仍会占用 SM 并争用 HBM/NVLink 带宽，可用单机“关/开通信”对照同形状 GEMM 时长量化。',
    example: '511 张卡的 AllReduce 在 trace 上均显示 40ms，对齐开始时间戳发现第 512 张卡因 DataLoader/GC 晚进 35ms，真实网络传输仅 5ms。',
    probe: '你可以这样追问：“如果 trace 上 AllReduce 和 GEMM 实现了 100% 时间重叠（暴露通信=0），为什么单步总时间还是比单卡慢 10%？怎么验证？”',
  },
  {
    term: '训练显存 Profiling：Memory Snapshot、Allocated vs Reserved 与碎片',
    patterns: ['snapshot', 'allocated', 'reserved', 'record_stream', '_record_memory_history', 'caching allocator', '内存泄漏', '碎片'],
    explain: 'PyTorch OOM 报错中，`allocated` 是活跃张量总大小，`reserved` 是 Caching Allocator 向驱动申请的保留池，`reserved − allocated` 大且申请大块失败多为碎片；`total − reserved − free` 则是 PyTorch 分配器之外的 CUDA Context、NCCL buffer、cuBLAS workspace 或外部进程。',
    derivation: '用 `_record_memory_history()` + `_dump_snapshot()` 查看 Active Memory Timeline：区分稳定台阶（静态状态）与瞬时尖峰（大词表 logits、FSDP unshard/prefetch、Step 1 懒初始化 Adam 矩）；`expandable_segments:True` 基于 CUDA VMM 缓解碎片但不降低活跃张量总量。',
    example: '`total=79.25, allocated=61.40, reserved=74.80, free=1.10 GiB`：分配器内 13.40 GiB 碎片空洞导致无法分配连续 2 GiB；另 3.35 GiB 被 Context/NCCL/Workspace 占用。',
    probe: '你可以这样追问：“为什么很多任务 Step 0 能跑过，偏偏在 Step 1 的 `optimizer.step()` 突然 OOM？如果 `nvidia-smi` 比 `max_memory_reserved()` 高 15 GiB，多出来的是谁？”',
  },
  {
    term: '推理 Profiling：GPU-Util 假象、DCGM 指标与 Host 同步 Gap',
    patterns: ['gpu-util', 'nvidia-smi', 'dcgm', 'dram_active', 'tensor_active', 'sm_active', 'async scheduling', 'cudaStreamSynchronize', 'py-spy', 'custom allreduce'],
    explain: '`nvidia-smi` 的 `GPU-Util` 只统计采样窗口内“至少有一个 kernel 在执行”的时间比例，哪怕只占几个 SM 或 HBM 带宽只跑到 20% 也会报 95%+。衡量真实硬件饱和度需看 DCGM `PIPE_TENSOR_ACTIVE`（Prefill/大 Batch）与 `DRAM_ACTIVE`（Decode）。',
    derivation: '推理端到端排查：用 `nsys` 对齐 CPU 线程与 GPU Stream——若每步 Decode 间出现 1~5ms 空白 Gap，常因 Sampler 尾部 `.item()` 触发隐式 `cudaStreamSynchronize`、未命中 CUDA Graph 回退 Eager、或 Python GC/Detokenizer 卡死主循环（`py-spy` 火焰图直出）。',
    example: '8 卡 TP 小 Batch Decode 中单次 AllReduce 仅几百 KB，属延迟受限；检查 trace 是否走了 NVLink P2P Custom AllReduce/NVLS，并核对 Async Scheduling 提前发射机制。',
    probe: '你可以这样追问：“既然 Async Scheduling 允许在 Step t 的 sampled token 回传 CPU 前就提前发射 Step t+1，那恰好命中 EOS 或 stop string 时怎么保证输出正确？”',
  },
  {
    term: '微观算子 Profiling：ncu SOL、Warp Stall、Sector/Wavefront 与 Register Spill',
    patterns: ['ncu', 'sol', 'speed of light', 'long scoreboard', 'short scoreboard', 'mio throttle', 'not selected', 'sector', 'wavefront', 'bank conflict', 'swizzle', 'spill', 'local memory', 'lmem', 'l1tex', 'nccl-tests'],
    explain: '`ncu` 先看 Speed-of-Light (SOL) 分辨 Compute-bound、Memory-bound 还是双低的 Latency/Occupancy-bound；再看 Warp Stall 找准卡在哪条管线，并用 Global 32B Sector 比、Shared Wavefront 比和 Local Memory 字节数实证访存浪费与寄存器溢出。',
    derivation: '- `Long Scoreboard`：等 L1TEX 长延迟（Global DRAM、Local Memory register spill 或 Texture）→ 查合并访存、异步预取/软件流水、检查 `Local Memory`/`lmem` 流量减 Tile 消 spill；\n- `Short Scoreboard`：等 SMEM/固定延迟指令；`MIO Throttle`：SMEM/SFU 队列满；\n- `Not Selected`：Warp 已就绪让位给其他就绪 Warp（良性延迟隐藏信号！）；\n- Global 看 Actual/Ideal 32B Sectors；SMEM Bank Conflict 看 Actual/Ideal Wavefronts。',
    example: '32×32 float tile 按列读时 32 个 lane 全撞 Bank 0（Wavefronts 达 32:1），行宽加 Padding 到 33 或做 XOR Swizzle 后恢复 1:1；`ncu` 默认锁 base 频率并刷 L2，故绝对耗时不同于 `nsys`。',
    probe: '你可以这样追问：“在 `ncu` 里看到 `Stall Long Scoreboard` 很高，你怎么进一步判断它是在等 Global HBM 访存，还是寄存器不够发生了 Register Spill 走到了 Local Memory？”',
  },
  {
    term: 'Chunked Linear Cross-Entropy (Liger / Apple CCE / VocabParallel)',
    patterns: ['liger', 'cce', 'cut cross-entropy', 'vocabparallel', '词表并行', '18.5625', '16 gib', 'lse', 'ignore_index'],
    explain: '大词表（128K~256K）训练时物化完整 FP32 `[T, V]` logits 动辄占用 16~37 GiB。Liger 沿 token 维分块计算局部 logits 并原地覆写梯度、逐块累加 dH 和 dW；Apple CCE 更进一步在片上 SRAM 分块归约 LSE 避免写回 HBM；TP 场景用 VocabParallelCrossEntropy 仅跨卡同步 O(T) 级 max 与 sum_exp。',
    derivation: '令 W 为 [V, H]，对 C 行隐藏态算 Z_c = H_c Wᵀ，原地算 dZ_c = (softmax(Z_c) − onehot(y)) / N_valid（ignore_index 置零），随后 dH_c = dZ_c W、dW += dZ_cᵀ H_c。注意 mean reduction 的分母必须是全批有效 token 总数，不能按每块各自平均。',
    example: 'T=32,768、V=131,072 时单份 FP32 logits = 2³⁴ B = 16 GiB（V=152,064 为 18.5625 GiB）；按 C=4096 分块后单块临时 logits 仅 2 GiB，峰值下降 ~88%+。',
    probe: '你可以这样追问：“Liger 的 FusedLinearCrossEntropy 为什么在标量 sum/mean loss 时能在 forward 里提前算好 dH 和 dW，但在 `reduction=none` 时必须换路径？”',
  },
];

function findBestGlossaryMatches(selectedText = '', surroundingContext = '') {
  const selNorm = selectedText.toLowerCase();
  const ctxNorm = `${selectedText} ${surroundingContext}`.toLowerCase();

  const scored = GLOSSARY.map((entry) => {
    let score = 0;
    for (const p of entry.patterns) {
      if (selNorm.includes(p)) score += 10 + p.length * 0.2;
      else if (ctxNorm.includes(p)) score += 2;
    }
    return { entry, score };
  })
    .filter((x) => x.score >= 8)
    .sort((a, b) => b.score - a.score);

  if (scored.length > 0) return scored.slice(0, 2).map((s) => s.entry);

  // Relax to context match if selected phrase was a general formula or Chinese snippet
  const ctxScored = GLOSSARY.map((entry) => {
    let score = 0;
    for (const p of entry.patterns) {
      if (ctxNorm.includes(p)) score += 3;
    }
    return { entry, score };
  })
    .filter((x) => x.score >= 3)
    .sort((a, b) => b.score - a.score);

  return ctxScored.slice(0, 1).map((s) => s.entry);
}

export function explainSelectionLocal({
  selectedText = '',
  surroundingContext = '',
  question = null,
  mode = 'explain',
  userFollowUp = '',
}) {
  const cleanSel = String(selectedText || '').trim();
  const matches = findBestGlossaryMatches(cleanSel, `${surroundingContext} ${question?.title || ''} ${(question?.tags || []).join(' ')}`);

  // Try matching relevant rubric / hint from active question for extra grounding
  let matchedRubric = null;
  if (question?.rubric?.length) {
    const qNorm = `${cleanSel} ${userFollowUp}`.toLowerCase();
    matchedRubric =
      question.rubric.find((r) => (r.keywords || []).some((k) => qNorm.includes(k.toLowerCase()))) || null;
  }

  const primary = matches[0] || null;
  let answer = '';

  if (userFollowUp && userFollowUp.trim()) {
    const fu = userFollowUp.trim();
    const askMath = /(公式|推导|怎么算|多少|字节|算例|例子|举例|为什么)/.test(fu);
    const askProbe = /(追问|怎么问|面试|拷打|考察)/.test(fu);
    if (primary) {
      if (askProbe) {
        answer = `针对「**${cleanSel}**」与你的追问，可以直接这样口头反问候选人：\n\n${primary.probe}\n\n${matchedRubric ? `结合本题考点补充探测：${matchedRubric.probe}` : ''}`;
      } else if (askMath) {
        answer = `### ${primary.term} · 推导与算例\n\n${primary.derivation}\n\n**具体算例**：${primary.example}`;
      } else {
        answer = `### 围绕「${primary.term}」补充拆解\n\n${primary.explain}\n\n**推导口径**：${primary.derivation}`;
      }
    } else {
      answer = `围绕「**${cleanSel}**」的自由追问（“${fu}”）：本地离线词库未命中更细的子推导。${
        matchedRubric ? `\n\n- **本题关联考点**：${matchedRubric.label}\n- **推荐核验话术**：${matchedRubric.probe}` : ''
      }\n\n> 💡 点击右上角 **「AI 配置」** 接入 DeepSeek / OpenAI / Qwen / 本地 Ollama 后，即可对任意划词句子和公式进行自由多轮深度推演。`;
    }
  } else if (primary) {
    if (mode === 'derivation') {
      answer = `### ${primary.term} · 推导与口径\n\n${primary.derivation}${
        matches[1] ? `\n\n**关联概念 (${matches[1].term})**：${matches[1].derivation}` : ''
      }`;
    } else if (mode === 'example') {
      answer = `### ${primary.term} · 具体数字算例\n\n${primary.example}\n\n**核心原理速记**：${primary.explain}`;
    } else if (mode === 'probe') {
      answer = `### 如何就「${cleanSel.slice(0, 28)}${cleanSel.length > 28 ? '…' : ''}」口头追问候选人\n\n- ${primary.probe}${
        matchedRubric ? `\n- **结合本题 Rubric (${matchedRubric.weight}%)**：${matchedRubric.probe}` : ''
      }`;
    } else {
      answer = `### ${primary.term}\n\n${primary.explain}\n\n- **关键公式 / 口径**：${primary.derivation}\n- **典型算例**：${primary.example}${
        matchedRubric ? `\n- **对应本题考点**：${matchedRubric.label}` : ''
      }`;
    }
  } else {
    const snippetCtx = surroundingContext
      ? surroundingContext.slice(0, 200) + (surroundingContext.length > 200 ? '…' : '')
      : '';
    answer = `### 上下文定位解释\n\n选中片段：「\`${cleanSel}\`」${
      question ? `（位于《${question.title}》）` : ''
    }\n\n${
      matchedRubric
        ? `- **本题直接对应考点 (${matchedRubric.weight}%)**：${matchedRubric.label}\n- **建议核验问法**：${matchedRubric.probe}`
        : snippetCtx
          ? `- **所在语境**：${snippetCtx}`
          : '- 建议选中更具体的术语、公式或完整句子，以获得更精准的展开。'
    }\n\n> 💡 当前为**本地离线专家词库**模式；如需对任意细微表述做自由展开和逐步推导，可点击右上角 **「AI 配置」** 接入 DeepSeek / OpenAI / Qwen 或本地 Ollama。`;
  }

  return {
    source: 'local',
    content: answer.trim(),
    matchedTerm: primary?.term || null,
  };
}
