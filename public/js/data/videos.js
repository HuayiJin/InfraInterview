// Curated concise, vertical, visual/diagram-focused YouTube explainers for AI Infra concepts.
// Every URL is a verified permanent youtube.com/watch?v=<id> link (no long rambling lectures, no forced padding).

export const CURATED_VIDEOS = [
  // ===== Distributed Training & Parallelism =====
  {
    id: 'vpMfAsHzI2E',
    url: 'https://www.youtube.com/watch?v=vpMfAsHzI2E',
    title: 'Tensor Parallelism vs Pipeline Parallelism vs Sequence Parallelism',
    note: '动画直觉对比 TP / PP / SP 的矩阵切分与通信位置',
    channel: 'Hyperlake',
    styleTag: '动画图解',
    lang: 'EN',
    questionIds: ['tr-01'],
    keywords: [
      'tensor parallelism',
      'pipeline parallelism',
      'sequence parallelism',
      '3d并行',
      '3d 并行',
      'tp',
      'sp',
      'allgather',
      'reducescatter',
      'megatron',
    ],
  },
  {
    id: 'PloOqHY3WYM',
    url: 'https://www.youtube.com/watch?v=PloOqHY3WYM',
    title: "DualPipe from Scratch: Implementing DeepSeek's 5D Parallelism in PyTorch",
    note: '双向流水线 DualPipe 拆解：前反向 Overlap 与 MoE All-to-All 掩盖',
    channel: 'PyTorch Foundation',
    styleTag: '架构拆解',
    lang: 'EN',
    questionIds: ['tr-03', 'tr-07'],
    keywords: [
      'dualpipe',
      '5d',
      'zero-bubble',
      'zero bubble',
      '1f1b',
      'bubble',
      '气泡',
      'deepep',
      'all-to-all',
      'alltoall',
      'moe',
      'ep',
      '双向流水',
    ],
  },
  {
    id: 'yQIrBIzmeBU',
    url: 'https://www.youtube.com/watch?v=yQIrBIzmeBU',
    title: 'Five Bottlenecks, Five Fixes: Systematic Performance Diagnosis for PyTorch Distributed Training',
    note: '分布式训练 5 大典型瓶颈的 Trace 图形化诊断（通信暴露 / Straggler / Host 阻塞）',
    channel: 'PyTorch Foundation',
    styleTag: 'Trace 实战',
    lang: 'EN',
    questionIds: ['tr-09'],
    keywords: [
      'profiler',
      'pytorch profiler',
      'kineto',
      'hta',
      'straggler',
      'step time',
      'comm overlap',
      '通信重叠',
      '关键路径',
      '慢节点',
    ],
  },

  // ===== LLM Inference & Serving =====
  {
    id: 'BxdV1Hjebz4',
    url: 'https://www.youtube.com/watch?v=BxdV1Hjebz4',
    title: 'vLLM PagedAttention Explained: Why GPU Memory Sits Empty',
    note: '图示 KV Cache 碎片根因与 PagedAttention 虚拟页表映射机制',
    channel: 'Rig Commit',
    styleTag: '内存图解',
    lang: 'EN',
    questionIds: ['inf-01'],
    keywords: [
      'pagedattention',
      'vllm',
      'block table',
      'continuous batching',
      'chunked prefill',
      'kv碎片',
      'kv cache',
      '页表',
    ],
  },
  {
    id: 'nFZGrweVF2U',
    url: 'https://www.youtube.com/watch?v=nFZGrweVF2U',
    title: 'SGLang Explained: How Efficient LLM Serving Works',
    note: '动画演示 RadixAttention 前缀树复用与 SGLang 高效调度流程',
    channel: 'Forward Pass',
    styleTag: '动画精讲',
    lang: 'EN',
    questionIds: ['inf-01'],
    keywords: ['sglang', 'radixattention', 'radix', 'prefix caching', '前缀复用', 'zero-overhead'],
  },
  {
    id: 'REeYCHfZphg',
    url: 'https://www.youtube.com/watch?v=REeYCHfZphg',
    title: 'Disaggregated Inference Explained | Prefill vs Decode, KV Cache Transfer, Dynamo & llm-d',
    note: 'PD 分离端到端图示：Prefill/Decode 池化、KV 传输与路由协同',
    channel: 'Learn-AI-with-Walid',
    styleTag: '架构图解',
    lang: 'EN',
    questionIds: ['inf-02'],
    keywords: [
      'disaggregated',
      'pd分离',
      'pd 分离',
      'prefill',
      'decode',
      'distserve',
      'mooncake',
      'kv transfer',
      'kv传输',
      'dynamo',
    ],
  },
  {
    id: '6q0ZDVkn7Lw',
    url: 'https://www.youtube.com/watch?v=6q0ZDVkn7Lw',
    title: '55. Disaggregated Prefill and Decode: Separate GPU Pools',
    note: '紧凑图解为何拆池消除 Prefill/Decode 互扰与配比权衡',
    channel: 'Hitesh C.',
    styleTag: '图解精讲',
    lang: 'EN',
    questionIds: ['inf-02'],
    keywords: ['disaggregated', 'pd分离', 'ttft', 'tpot', 'itl', 'prefill', 'decode', 'interference'],
  },
  {
    id: 'DBdeFbNrIZI',
    url: 'https://www.youtube.com/watch?v=DBdeFbNrIZI',
    title: "Multi-Head Latent Attention Explained Visually: DeepSeek's Secret to 93% Less GPU Memory",
    note: '高信息密度视觉拆解 MLA 低秩压缩与矩阵吸收（Matrix Absorption）',
    channel: 'Visual AI',
    styleTag: '视觉图解',
    lang: 'EN',
    questionIds: ['inf-03'],
    keywords: ['mla', 'multi-head latent', 'latent attention', '矩阵吸收', '吸收', 'deepseek', 'kv压缩', 'rope'],
  },
  {
    id: 'JHevNMuMB8g',
    url: 'https://www.youtube.com/watch?v=JHevNMuMB8g',
    title: 'KV Cache Explained',
    note: '直观动画演示自回归解码为何缓存 K/V 及显存增长规律',
    channel: 'Forward Pass',
    styleTag: '动画精讲',
    lang: 'EN',
    questionIds: ['inf-03'],
    keywords: ['kv cache', 'gqa', 'mha', 'mqa', 'kv 缓存', '自回归'],
  },
  {
    id: 'NIntnfu_mLc',
    url: 'https://www.youtube.com/watch?v=NIntnfu_mLc',
    title: 'GPU Roofline Model Explained from Scratch | FLOPS, Memory Bandwidth & Arithmetic Intensity',
    note: '从零画懂 Roofline 曲线、Arithmetic Intensity 与硬件拐点（Ridge Point）',
    channel: 'SS CODER',
    styleTag: 'Roofline 图解',
    lang: 'EN',
    questionIds: ['inf-04'],
    keywords: [
      'roofline',
      'arithmetic intensity',
      '算术强度',
      'ridge point',
      '拐点',
      'memory-bound',
      'compute-bound',
      '访存受限',
      '计算受限',
    ],
  },
  {
    id: 'B7wL1N8kjZQ',
    url: 'https://www.youtube.com/watch?v=B7wL1N8kjZQ',
    title: '人工智能，从零搭建 · 第19部分：推断 | AI, Built: Part 19, Inference (中文)',
    note: 'Manim 风格高质量中文动画：拆解 Prefill/Decode 算力带宽瓶颈与 Batching',
    channel: 'Teo Zeng',
    styleTag: 'Manim 动画',
    lang: '中文',
    questionIds: ['inf-04'],
    keywords: ['roofline', 'prefill', 'decode', 'batch size', '推理瓶颈', 'kv cache', 'hbm'],
  },
  {
    id: 'GHuZl1FSph4',
    url: 'https://www.youtube.com/watch?v=GHuZl1FSph4',
    title: 'Why Quantization Exists: The GPU Memory Wall Behind Every LLM',
    note: '聚焦 GPU Memory Wall，直观解释为何 W4A16/FP8 能直接拉升 Decode 吞吐',
    channel: 'Nishanth Writes Code',
    styleTag: '内存墙图解',
    lang: 'EN',
    questionIds: ['inf-05'],
    keywords: [
      'quantization',
      '量化',
      'w4a16',
      'w8a8',
      'awq',
      'gptq',
      'smoothquant',
      'fp8',
      'marlin',
      'memory wall',
      '反量化',
    ],
  },
  {
    id: 'l8gWQlrVOKQ',
    url: 'https://www.youtube.com/watch?v=l8gWQlrVOKQ',
    title: 'How LLMs Get Faster Without Changing Their Outputs | Speculative Decoding',
    note: 'Manim 动画严谨推演投机采样：Draft-Verify 并发验证与无损拒绝采样',
    channel: 'Jia-Bin Huang',
    styleTag: 'Manim 动画',
    lang: 'EN',
    questionIds: ['inf-06'],
    keywords: [
      'speculative',
      '投机采样',
      'eagle',
      'mtp',
      'multi-token prediction',
      'draft',
      'rejection sampling',
      'tree attention',
      '无损验证',
    ],
  },
  {
    id: 'qI4kwcNHxpk',
    url: 'https://www.youtube.com/watch?v=qI4kwcNHxpk',
    title: 'Speculative Decoding in 8 Minutes: The Complete Visual Guide',
    note: '8 分钟紧凑图解投机解码加速比公式、接受率 α 与批大小边界',
    channel: 'Tech Explainer',
    styleTag: '8 分钟图解',
    lang: 'EN',
    questionIds: ['inf-06'],
    keywords: ['speculative', '投机采样', '接受率', 'acceptance', 'verify', 'eagle', 'mtp'],
  },

  // ===== RL / Post-Training =====
  {
    id: 'EX8-ucKOBbA',
    url: 'https://www.youtube.com/watch?v=EX8-ucKOBbA',
    title: 'Group Relative Policy Optimization(GRPO) Visualized',
    note: '图形化拆解 GRPO 去掉 Critic 后的组内采样、Advantage 归一化与显存省幅',
    channel: 'AGI Lambda',
    styleTag: '视觉图解',
    lang: 'EN',
    questionIds: ['rl-01'],
    keywords: ['grpo', 'group relative', 'critic', 'advantage', 'kl', 'deepseek-r1', 'rollout'],
  },
  {
    id: 'fg1T-hMUamk',
    url: 'https://www.youtube.com/watch?v=fg1T-hMUamk',
    title: 'POLICY GRADIENTS Animated - the equation behind PPO, GRPO RLHF',
    note: '公式动画直观看懂 PPO Clipped Surrogate、GRPO 与 RLHF 策略梯度联系',
    channel: 'learningcurve',
    styleTag: '公式动画',
    lang: 'EN',
    questionIds: ['rl-01'],
    keywords: ['ppo', 'grpo', 'rlhf', 'policy gradient', 'importance sampling', 'ratio', 'clip', 'dpo'],
  },
  {
    id: 'Vd79NmmqY3Q',
    url: 'https://www.youtube.com/watch?v=Vd79NmmqY3Q',
    title: 'verl: Flexible and Scalable Reinforcement Learning Library for LLM Reasoning and Tool-Calling',
    note: 'veRL 官方讲解 HybridFlow 控制流/数据流分离与多引擎协同',
    channel: 'PyTorch Foundation',
    styleTag: '官方架构',
    lang: 'EN',
    questionIds: ['rl-02'],
    keywords: ['verl', 'hybridengine', 'hybridflow', 'colocated', 'disaggregated', '共置', '分池', 'openrlhf', 'ray'],
  },
  {
    id: 'H-k7oYjwruM',
    url: 'https://www.youtube.com/watch?v=H-k7oYjwruM',
    title: 'How Prompt Caching Made Long-Context LLM Agents Viable',
    note: '图解多轮 Agent / Tool-Calling 场景下 Prefix Cache 命中与重复预填充成本',
    channel: 'PY',
    styleTag: '图解精讲',
    lang: 'EN',
    questionIds: ['rl-06'],
    keywords: ['prompt caching', 'prefix cache', 'agentic', 'multi-turn', '多轮', 'tool calling', 'sandbox'],
  },

  // ===== GPU Kernel, Comm & Profiling Tools =====
  {
    id: 'us1nzaQKq0Y',
    url: 'https://www.youtube.com/watch?v=us1nzaQKq0Y',
    title: '16. FlashAttention Explained: Tiling, Online Softmax, SRAM',
    note: '一步步画清 FlashAttention 分块 Tiling、SRAM 驻留与 Online Softmax 修正因子',
    channel: 'Hitesh C.',
    styleTag: '图解精讲',
    lang: 'EN',
    questionIds: ['kn-01'],
    keywords: [
      'flashattention',
      'flash attention',
      'tiling',
      'online softmax',
      'sram',
      'hbm',
      'tma',
      'wgmma',
      'warp specialization',
    ],
  },
  {
    id: 'CX1WtsdMFh8',
    url: 'https://www.youtube.com/watch?v=CX1WtsdMFh8',
    title: 'Flash Attention Runs Softmax One Tile at a Time',
    note: '紧凑动画演示 Online Softmax 逐 Tile 递推如何避免 N×N 矩阵落回 HBM',
    channel: 'DataMListic',
    styleTag: '动画可视化',
    lang: 'EN',
    questionIds: ['kn-01'],
    keywords: ['flashattention', 'online softmax', 'tile', 'tiling', 'softmax', 'sram'],
  },
  {
    id: 'Jz9E1_pNBS8',
    url: 'https://www.youtube.com/watch?v=Jz9E1_pNBS8',
    title: "NCCL Explained: How NVIDIA's GPU Communication Library Powers Distributed Deep Learning",
    note: '图示 NCCL Ring / Tree 拓扑、NVLink 与 RDMA 集合通信数据流',
    channel: 'Yi\'s Learning Notes',
    styleTag: '通信图解',
    lang: 'EN',
    questionIds: ['kn-02'],
    keywords: [
      'nccl',
      'ring',
      'tree',
      'allreduce',
      'all-reduce',
      'nvlink',
      'rdma',
      'roce',
      'infiniband',
      'rail',
      'busbw',
      'algbw',
      'nccl-tests',
    ],
  },
  {
    id: 'dUDGO66IadU',
    url: 'https://www.youtube.com/watch?v=dUDGO66IadU',
    title: 'Intro to NVIDIA Nsight Systems | CUDA Developer Tools',
    note: 'NVIDIA 官方短演示：如何读懂 nsys 系统级 CPU-GPU Timeline、同步气泡与 CUDA Graph',
    channel: 'NVIDIA Developer',
    styleTag: '官方工具演示',
    lang: 'EN',
    questionIds: ['tr-09', 'inf-09'],
    keywords: [
      'nsys',
      'nsight systems',
      'timeline',
      'cuda graph',
      'host',
      'sync',
      'gpu-util',
      'stream',
      'nvtx',
      'd2h',
      'launch',
    ],
  },
  {
    id: 'Iuy_RAvguBM',
    url: 'https://www.youtube.com/watch?v=Iuy_RAvguBM',
    title: 'Intro to NVIDIA Nsight Compute | CUDA Developer Tools',
    note: 'NVIDIA 官方短演示：ncu Speed of Light (SOL)、Roofline 与 Warp Stall 定位单算子瓶颈',
    channel: 'NVIDIA Developer',
    styleTag: '官方工具演示',
    lang: 'EN',
    questionIds: ['kn-03', 'kn-06'],
    keywords: [
      'ncu',
      'nsight compute',
      'speed of light',
      'sol',
      'warp stall',
      'long scoreboard',
      'short scoreboard',
      'mio throttle',
      'occupancy',
      'triton',
      'cuda',
    ],
  },
  {
    id: 'GCkdiHk6fUY',
    url: 'https://www.youtube.com/watch?v=GCkdiHk6fUY',
    title: 'Memory Analysis with NVIDIA Nsight Compute | CUDA Developer Tools',
    note: 'NVIDIA 官方短演示：合并访存、32B Sector 效率、L1/L2 命中与 Bank Conflict 诊断',
    channel: 'NVIDIA Developer',
    styleTag: '官方内存分析',
    lang: 'EN',
    questionIds: ['kn-06'],
    keywords: [
      'sector',
      'wavefront',
      'bank conflict',
      'shared memory',
      'coalescing',
      '合并访存',
      'register spill',
      'local memory',
      'l1tex',
      'dram',
      'ncu',
    ],
  },
];

const VIDEO_BY_Q = new Map();
for (const v of CURATED_VIDEOS) {
  for (const qId of v.questionIds) {
    if (!VIDEO_BY_Q.has(qId)) VIDEO_BY_Q.set(qId, []);
    VIDEO_BY_Q.get(qId).push(v);
  }
}

export function getVideosForQuestion(questionId) {
  if (!questionId) return [];
  return VIDEO_BY_Q.get(questionId) || [];
}

export function getVideosForSelection(selectedText, surroundingContext = '', questionId = '') {
  const sel = String(selectedText || '').toLowerCase().trim();
  const ctx = String(surroundingContext || '').toLowerCase();
  if (!sel) return [];

  const scored = [];
  for (const v of CURATED_VIDEOS) {
    let directHits = 0;
    let ctxHits = 0;

    for (const kw of v.keywords) {
      const k = kw.toLowerCase();
      if (k.length <= 2) {
        // Short acronyms like 'tp', 'sp', 'ep', 'kl' require word boundary in ASCII
        const re = new RegExp(`(^|[^a-z0-9])${k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}([^a-z0-9]|$)`, 'i');
        if (re.test(sel)) directHits += 1;
        else if (re.test(ctx)) ctxHits += 1;
      } else {
        if (sel.includes(k)) directHits += 1;
        else if (ctx.includes(k)) ctxHits += 1;
      }
    }

    const qMatch = Boolean(questionId && v.questionIds.includes(questionId));
    if (directHits > 0) {
      scored.push({ video: v, score: directHits * 15 + ctxHits * 3 + (qMatch ? 4 : 0) });
    } else if (qMatch && ctxHits > 0) {
      scored.push({ video: v, score: ctxHits * 3 + 2 });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 2).map((item) => item.video);
}

export function buildYoutubeSearchUrl(selectedText, question = null) {
  const raw = String(selectedText || '')
    .replace(/[`*_~[\]()（）「」『』<>#]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  // Keep query tight so YouTube returns focused explainers
  const core = raw.length > 56 ? raw.slice(0, 56).trim() : raw || question?.title || 'LLM systems';
  const low = core.toLowerCase();

  let suffix = 'explained visual';
  const track = question?.track;
  if (track === 'kernel' && !/(cuda|gpu|ncu|nsight|nccl|flashattention|triton)/i.test(low)) {
    suffix = 'CUDA GPU explained';
  } else if (track === 'inference' && !/(llm|vllm|sglang|inference|kv|roofline)/i.test(low)) {
    suffix = 'LLM inference visual explained';
  } else if (track === 'training' && !/(llm|megatron|fsdp|zero|pytorch|distributed)/i.test(low)) {
    suffix = 'LLM distributed training explained';
  } else if (track === 'rl' && !/(rlhf|grpo|ppo|verl|llm)/i.test(low)) {
    suffix = 'LLM RLHF visual explained';
  } else if (!/(explained|visual|llm|gpu)/i.test(low)) {
    suffix = 'LLM visual explained';
  }

  const query = `${core} ${suffix}`.trim();
  return `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`;
}
