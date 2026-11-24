import { TRAINING_QUESTIONS } from './questions-training.js';
import { INFERENCE_QUESTIONS } from './questions-inference.js';
import { RL_QUESTIONS } from './questions-rl.js';
import { KERNEL_QUESTIONS } from './questions-kernel.js';

export const TRACKS = {
  training: {
    id: 'training',
    name: '分布式训练',
    short: 'Training',
    color: 'var(--track-training)',
    badgeClass: 'badge-training',
    desc: '3D/5D并行、ZeRO/FSDP2、MoE EP/DualPipe、FP8、长上下文CP、万卡容错与MFU',
  },
  inference: {
    id: 'inference',
    name: '推理加速与引擎',
    short: 'Inference',
    color: 'var(--track-inference)',
    badgeClass: 'badge-inference',
    desc: 'PagedAttention、Chunked Prefill、PD分离、MLA、Roofline、量化、投机采样与Wide-EP',
  },
  rl: {
    id: 'rl',
    name: 'RL / 后训练系统',
    short: 'RL Infra',
    color: 'var(--track-rl)',
    badgeClass: 'badge-rl',
    desc: 'PPO/GRPO、veRL vs OpenRLHF、权重同步、长CoT长尾、CuMem Sleep、Agentic沙箱与精度对齐',
  },
  kernel: {
    id: 'kernel',
    name: 'GPU 算子与通信',
    short: 'Kernel & Comm',
    color: 'var(--track-kernel)',
    badgeClass: 'badge-kernel',
    desc: 'FlashAttention-1/2/3、NCCL Ring/Tree、Rail拓扑、Nsight/Triton、GB200与融合算子',
  },
};

export const INTERVIEW_MODES = [
  {
    id: 'full_loop',
    title: '全栈综合模拟 (4 题)',
    badge: '推荐',
    desc: '训练 + 推理 + RL + 底层硬件各抽 1 题，完整还原大厂/AI Lab 60 分钟 Staff 轮技术面。',
    questionCount: 4,
  },
  {
    id: 'track_deep',
    title: '专项方向深挖 (3 题)',
    badge: '定向突击',
    desc: '聚焦单个核心领域（训练 / 推理 / RL / 底层算子与通信），从原理到架构层层递进。',
    questionCount: 3,
  },
  {
    id: 'stress_drill',
    title: '压力深挖 5-Why (2 题)',
    badge: '硬核高压',
    desc: '精选 P7/P8 难题，严格抠公式、显存账与边界场景，连续多轮追问到底。',
    questionCount: 2,
  },
  {
    id: 'incident_debug',
    title: '线上排障实战 (3 题)',
    badge: '真实告警',
    desc: '全真线上事故：千卡 Hang/SDC、推理 TTFT/KV 雪崩、RL Train-Infer Logprob 偏移。',
    questionCount: 3,
  },
];

export const TARGET_LEVELS = [
  { id: 'P6', label: 'P6 / Senior (L4)', desc: '扎实掌握主流框架、核心公式与单机/多机调优' },
  { id: 'P7', label: 'P7 / Staff (L5)', desc: '精通系统瓶颈定位、跨层优化、并行与调度权衡' },
  { id: 'P8', label: 'P8 / Principal (L6)', desc: '超大规模集群架构、算法-硬件协同与端到端 TCO' },
];

export const PERSONAS = {
  strict: {
    id: 'strict',
    name: '硬核量化派',
    tag: 'DeepSeek / 底层极客风',
    avatar: 'DS',
    openingStyle: '我们直接进硬核问题，请尽量用定量公式、字节数、带宽和硬件约束说话，少讲空泛概念。',
    praisePrefix: '切入点很准，定量直觉在线。',
    probePrefix: '这里我追问一个关键细节，你的回答里还没有算透：',
    deepenPrefix: '前面主干逻辑成立，我们再往下深挖一层边界场景：',
    summaryTone: '偏严格，注重第一性原理与精确数量级推导。',
  },
  pragmatic: {
    id: 'pragmatic',
    name: '大厂实战 TL',
    tag: '线上稳定性 / ROI 导向',
    avatar: 'TL',
    openingStyle: '欢迎参加今天的 AI Infra 技术面。我会结合真实集群场景考察你的架构权衡、排障思路和落地经验。',
    praisePrefix: '工程思路很扎实，考虑到了线上真实约束。',
    probePrefix: '如果放到生产集群上，这里有个很容易踩坑的点想听你展开：',
    deepenPrefix: '好，顺着你刚才的设计，我再加一个真实线上会遇到的约束条件：',
    summaryTone: '关注生产可用性、长尾延迟、容错与成本 ROI。',
  },
  coach: {
    id: 'coach',
    name: '温和引导教练',
    tag: '备考训练 / 渐进提示',
    avatar: 'CH',
    openingStyle: '别紧张，我们按真实面试节奏一步步来。你可以先搭出框架，遇到定量估算随时可以打开右上角的「随身计算器」。',
    praisePrefix: '框架非常清晰，核心机制讲得很到位！',
    probePrefix: '很棒的开头！为了回答更完整，可以再补充一下这个维度吗：',
    deepenPrefix: '答得很不错，我们趁热打铁看一道常见的进阶追问：',
    summaryTone: '循循善诱，适合日常刷题查漏补缺。',
  },
};

export const TECH_STACK_TAGS = [
  'Megatron-LM',
  'DeepSpeed / ZeRO',
  'PyTorch FSDP2',
  'vLLM',
  'SGLang',
  'TensorRT-LLM',
  'veRL',
  'OpenRLHF',
  'FlashAttention / Triton',
  'DeepEP / NCCL',
  'Ray',
  'CUDA / CUTLASS',
];

export const FORMULA_SNIPPETS = [
  { label: '6ND 算力', text: '总 FLOPs ≈ 6·N·D，MFU = (6·N·D / T) / (GPU数 × 峰值TFLOPS)' },
  { label: '16Ψ 显存', text: '静态显存 16Ψ = BF16权重(2Ψ) + 梯度(2Ψ) + FP32主权重(4Ψ) + Adam m/v(8Ψ)' },
  { label: 'KV Cache', text: '单Token KV = 2 × L × n_kv × d_h × bytes；总KV = B × S × 单Token KV' },
  { label: 'Roofline OI', text: 'Ridge Point = Peak_FLOPS / HBM_BW (H100 BF16 ≈ 295 FLOPs/Byte)；Decode OI = 2B / bytes' },
  { label: 'AllReduce', text: 'Ring AllReduce 每卡收发 = 2·(N-1)/N · S (ReduceScatter + AllGather)' },
  { label: 'PP Bubble', text: '1F1B 气泡率 = (p-1)/(m+p-1)，Interleaved VPP 后降至 (p-1)/(v·m+p-1)' },
];

export const ALL_QUESTIONS = [
  ...TRAINING_QUESTIONS,
  ...INFERENCE_QUESTIONS,
  ...RL_QUESTIONS,
  ...KERNEL_QUESTIONS,
];

export function getQuestionById(id) {
  return ALL_QUESTIONS.find((q) => q.id === id) || null;
}

export function getQuestionsByTrack(track) {
  if (!track || track === 'all') return ALL_QUESTIONS;
  return ALL_QUESTIONS.filter((q) => q.track === track);
}
