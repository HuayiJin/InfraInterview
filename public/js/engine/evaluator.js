import { TRACKS, PERSONAS } from '../data/index.js';

/**
 * Strip interviewer-inserted prompt header blocks so rubric analysis focuses on candidate notes/answers.
 */
export function extractCandidateText(rawBoard = '', question = null) {
  if (!rawBoard) return '';
  let text = String(rawBoard).replace(/\r\n/g, '\n');
  // Remove explicitly demarcated prompt headers inserted via "写入题干到白板"
  text = text.replace(/^={3,}\s*\[题目\][\s\S]*?^={10,}\s*$/gm, ' ');
  text = text.replace(/^\s*\[候选人作答区\]\s*$/gm, ' ');
  if (question?.prompt) {
    const prefix = question.prompt.slice(0, 36);
    if (prefix.length >= 16) {
      text = text.replace(question.prompt, ' ');
    }
  }
  return text.trim();
}

function normalize(str = '') {
  return String(str).toLowerCase().replace(/[\u3000\s]+/g, ' ');
}

function countKeywordMatches(normText, keywords = []) {
  const matched = [];
  for (const kw of keywords) {
    const k = kw.toLowerCase();
    if (normText.includes(k)) {
      matched.push(kw);
    }
  }
  return matched;
}

/**
 * Compute score from rubricStates map ('hit' = 1.0, 'partial' = 0.55, 'miss'/'unseen' = 0)
 */
export function computeRubricScore(question, rubricStates = {}) {
  if (!question?.rubric?.length) return { score: 0, coveredCount: 0, hitCount: 0, partialCount: 0, missCount: 0, total: 0 };
  let totalWeight = 0;
  let earned = 0;
  let hitCount = 0;
  let partialCount = 0;
  let missCount = 0;

  for (const item of question.rubric) {
    const w = Number(item.weight || 20);
    totalWeight += w;
    const state = rubricStates[item.id] || 'unseen';
    if (state === 'hit') {
      earned += w;
      hitCount += 1;
    } else if (state === 'partial') {
      earned += w * 0.55;
      partialCount += 1;
    } else if (state === 'miss') {
      missCount += 1;
    }
  }

  const score = totalWeight > 0 ? Math.round((earned / totalWeight) * 100) : 0;
  return {
    score,
    coveredCount: hitCount + partialCount + missCount,
    hitCount,
    partialCount,
    missCount,
    total: question.rubric.length,
  };
}

/**
 * Build 3 actionable follow-ups from a given rubricStates map + question follow-up tree.
 */
export function buildFollowUpsForStates(question, rubricStates = {}, targetLevel = 'P7') {
  const misses = [];
  const partials = [];
  for (const r of question.rubric || []) {
    const st = rubricStates[r.id] || 'unseen';
    if (st === 'miss' || st === 'unseen') misses.push(r);
    else if (st === 'partial') partials.push(r);
  }
  misses.sort((a, b) => (b.weight || 0) - (a.weight || 0));
  partials.sort((a, b) => (b.weight || 0) - (a.weight || 0));

  const followUps = [];
  const firstGap = misses[0] || partials[0];
  if (firstGap) {
    followUps.push({
      type: '补漏追问',
      badge: 'badge-danger',
      intent: `针对未讲透的高权重考点（${firstGap.weight}%）：${firstGap.label.split('：')[0]}`,
      question: firstGap.probe,
    });
  }
  const secondGap = misses[1] || partials[1];
  if (secondGap && secondGap.id !== firstGap?.id) {
    followUps.push({
      type: '机制核验',
      badge: 'badge-rl',
      intent: `进一步确认是否真正理解：${secondGap.label.split('：')[0]}`,
      question: secondGap.probe,
    });
  }
  if (question.followUps?.[0] && followUps.length < 3) {
    followUps.push({
      type: '定量深挖',
      badge: 'badge-training',
      intent: '考察第一性原理数学估算与性能拐点',
      question: question.followUps[0].question,
    });
  }
  if (question.followUps?.[1] && followUps.length < 3) {
    followUps.push({
      type: '生产边界',
      badge: 'badge-kernel',
      intent: '压测大规模集群极端 Corner Case 与架构权衡',
      question: question.followUps[1].question,
    });
  }
  if (followUps.length < 3) {
    followUps.push({
      type: '压测上限',
      badge: 'badge-inference',
      intent: `主干考点已覆盖，压测是否具备超越 ${targetLevel} 的架构视野与 TCO 权衡`,
      question: `如果集群规模或并发流量再扩大 10 倍，并且同时要求压降 30% 单 Token 成本（TCO），你刚才这套架构最先崩溃的环节和最值得重构的软硬件协同点是什么？`,
    });
  }
  return followUps.slice(0, 3);
}

/**
 * Local Expert AI Engine: Analyze pasted candidate answer (plus optional code board) OR manual Rubric ticks.
 */
export function analyzeBoardAnswer(question, rawBoardText = '', targetLevel = 'P7', existingRubricStates = null) {
  const candidateText = extractCandidateText(rawBoardText, question);
  const norm = normalize(candidateText);
  const charLen = candidateText.replace(/\s+/g, '').length;

  // If interviewer didn't paste much text (< 12 chars) but already clicked Rubric states manually, preserve manual states!
  const hasManualStates =
    existingRubricStates &&
    Object.values(existingRubricStates).some((s) => s === 'hit' || s === 'partial' || s === 'miss');

  if (charLen < 12 && hasManualStates) {
    const stat = computeRubricScore(question, existingRubricStates);
    const levelThresholds = { P6: 62, P7: 74, P8: 84 };
    const targetBar = levelThresholds[targetLevel] || 74;
    const levelAssessment =
      stat.score >= targetBar + 10
        ? `明显超出 ${targetLevel} 预期（基于面试官手动 Rubric 评分）`
        : stat.score >= targetBar
          ? `达到 ${targetLevel} 录用门槛，建议切入定量/生产边界深挖`
          : `暂未达到 ${targetLevel} 要求，建议针对标红/标黄漏项口头追问`;

    const misses = question.rubric.filter((r) => existingRubricStates[r.id] === 'miss' || !existingRubricStates[r.id] || existingRubricStates[r.id] === 'unseen');
    const hits = question.rubric.filter((r) => existingRubricStates[r.id] === 'hit');
    return {
      source: 'local',
      score: stat.score,
      dimensions: { principle: stat.score, quantitative: stat.score, tradeoff: stat.score, engineering: stat.score },
      levelAssessment,
      summary: `基于你手动勾选的 Rubric 状态（命中 ${stat.hitCount}、一般 ${stat.partialCount}、遗漏 ${stat.missCount}），已针对未覆盖考点生成下一轮口头追问。`,
      rubricStates: { ...existingRubricStates },
      rubricDetails: question.rubric.map((r) => ({
        id: r.id,
        label: r.label,
        weight: r.weight,
        state: existingRubricStates[r.id] || 'unseen',
        matched: [],
        probe: r.probe,
      })),
      strengths: hits.length ? [`已命中核心考点：${hits.map((h) => h.label.split('：')[0]).join('、')}`] : [],
      gaps: misses.slice(0, 3).map((m) => `待追问补漏：${m.label}`),
      followUps: buildFollowUpsForStates(question, existingRubricStates, targetLevel),
      analyzedAt: Date.now(),
    };
  }

  const rubricStates = {};
  const rubricDetails = [];
  let earnedWeight = 0;
  let totalWeight = 0;

  for (const r of question.rubric) {
    const w = Number(r.weight || 20);
    totalWeight += w;
    const matched = countKeywordMatches(norm, r.keywords || []);
    let state = 'miss';
    if (matched.length >= 3 || (matched.length >= 2 && charLen >= 100)) {
      state = 'hit';
      earnedWeight += w;
    } else if (matched.length >= 1) {
      state = 'partial';
      earnedWeight += w * 0.55;
    }
    rubricStates[r.id] = state;
    rubricDetails.push({
      id: r.id,
      label: r.label,
      weight: w,
      state,
      matched,
      probe: r.probe,
    });
  }

  // Quantitative & Engineering Depth signals
  const hasNumbers = /\d+(\.\d+)?\s*(gb|tb|mb|kb|ms|s|tflops|pflops|tf|pf|flops|bytes|b|k|m|t|%|x|倍|卡|层)/i.test(candidateText);
  const hasFormula = /([=≈×*\/]|o\(|6nd|16|2\(n-1\)|bsh|oi|ridge)/i.test(candidateText);
  const hasTradeoff = /(权衡|trade-?off|代价|缺点|相比|但是|不过|瓶颈|限制|成本|收益|利弊)/i.test(candidateText);
  const hasProduction = /(线上|生产|告警|排障|监控|p99|sla|slo|容错|oom|hang|抖动|压测|nsys|ncu|profiler)/i.test(candidateText);

  const baseScore = totalWeight > 0 ? Math.round((earnedWeight / totalWeight) * 100) : 0;
  const depthBonus = (hasNumbers ? 4 : 0) + (hasFormula ? 3 : 0) + (hasTradeoff ? 3 : 0) + (hasProduction ? 3 : 0);
  const finalScore = charLen < 12 ? 0 : Math.min(100, Math.round(baseScore * 0.88 + depthBonus));

  const dimensions = {
    principle: Math.min(100, Math.round(baseScore * 0.95 + (charLen > 120 ? 5 : 0))),
    quantitative: charLen < 12 ? 0 : Math.min(100, Math.round(baseScore * 0.55 + (hasNumbers ? 25 : 0) + (hasFormula ? 20 : 0))),
    tradeoff: charLen < 12 ? 0 : Math.min(100, Math.round(baseScore * 0.65 + (hasTradeoff ? 30 : 5))),
    engineering: charLen < 12 ? 0 : Math.min(100, Math.round(baseScore * 0.65 + (hasProduction ? 30 : 8))),
  };

  // Level Calibration
  const levelThresholds = { P6: 62, P7: 74, P8: 84 };
  const targetBar = levelThresholds[targetLevel] || 74;
  let levelAssessment = '信息不足，请贴入候选人回答/速记要点，或直接点选下方 Rubric 考点后生成追问';
  if (charLen >= 15) {
    if (finalScore >= targetBar + 10) {
      levelAssessment = `明显超出 ${targetLevel} 预期（架构完整且具备定量/边界意识）`;
    } else if (finalScore >= targetBar) {
      levelAssessment = `达到 ${targetLevel} 录用门槛，可进入深水区追问区分上限`;
    } else if (finalScore >= targetBar - 16) {
      levelAssessment = `接近 ${targetLevel} 边缘（核心主干有概念，但定量推导或关键考点存在漏项）`;
    } else {
      levelAssessment = `暂未达到 ${targetLevel} 要求（基础机制或关键瓶颈未触及，建议给 L1 提示观察学习反应）`;
    }
  }

  const hits = rubricDetails.filter((d) => d.state === 'hit');
  const partials = rubricDetails.filter((d) => d.state === 'partial');
  const misses = rubricDetails.filter((d) => d.state === 'miss').sort((a, b) => b.weight - a.weight);

  const strengths = [];
  if (hits.length) strengths.push(`准确覆盖 ${hits.length}/${rubricDetails.length} 个核心考点：${hits.map((h) => h.label.split('：')[0]).join('、')}`);
  if (hasNumbers || hasFormula) strengths.push('具备定量推导/数量级估算意识，有具体公式或指标支撑');
  if (hasTradeoff) strengths.push('主动阐述了系统 Trade-off 与约束边界');
  if (hasProduction) strengths.push('结合了真实线上生产/排障视角');
  if (!strengths.length && charLen >= 15) strengths.push('给出了初步思路框架，待进一步细化关键机制');

  const gaps = [];
  for (const m of misses.slice(0, 3)) {
    gaps.push(`未触及：${m.label}`);
  }
  for (const p of partials.slice(0, 2)) {
    gaps.push(`浅尝辄止：${p.label}`);
  }
  if (charLen >= 15 && !hasNumbers && !hasFormula) {
    gaps.push('缺少定量计算（字节数 / 带宽 / 通信量 / 临界点推导），容易停留在八股概念层');
  }

  const followUps = buildFollowUpsForStates(question, rubricStates, targetLevel);

  const summary =
    charLen < 12
      ? '暂未检测到候选人回答文本。你可以粘贴候选人的回答/会议转写，或先手动勾选下方 Rubric 考点再点击生成追问。'
      : `共识别有效回答约 ${charLen} 字，命中 ${hits.length} 项、部分命中 ${partials.length} 项、遗漏 ${misses.length} 项。${
          !hasNumbers && !hasFormula ? '目前偏定性描述，建议立刻追问一个定量公式题检验真实功底。' : '定量与架构结合较好，可切入极端规模/故障边界深挖。'
        }`;

  return {
    source: 'local',
    score: finalScore,
    dimensions,
    levelAssessment,
    summary,
    rubricStates,
    rubricDetails,
    strengths,
    gaps,
    followUps,
    analyzedAt: Date.now(),
  };
}

/**
 * Local Expert Engine: Resume & Project Tech-Stack Extractor -> 5-Why Interrogation List
 */
const RESUME_SIGNALS = [
  {
    name: 'Megatron-LM / 3D并行',
    track: 'training',
    patterns: ['megatron', '3d并行', '3d parallel', 'tp', 'pp', '1f1b', 'vpp', 'sequence parallel'],
    relatedIds: ['tr-01', 'tr-07', 'tr-08'],
    probes: [
      '你在项目里配 TP/PP/DP/CP 时，具体卡数、模型规模、Global Batch Size 和最终实测 MFU 是多少？为什么选这组并行度而不是增大 PP/减小 TP？',
      'PP 的 Bubble 率和通信 Overlap 是怎么用 nsys 实测验证的？首尾 Stage (Embedding/LM Head) 显存和耗时不均衡怎么调平的？',
    ],
  },
  {
    name: 'ZeRO / DeepSpeed / FSDP',
    track: 'training',
    patterns: ['zero', 'deepspeed', 'fsdp', 'fsdp2', 'hsdp', 'dtensor', 'offload'],
    relatedIds: ['tr-02', 'tr-08'],
    probes: [
      '你在 FSDP / ZeRO-3 上遇到过 Gradient Accumulation 通信放大或显存峰值碎片 OOM 吗？是怎么配置 HSDP / Prefetch / Reshard 策略解决的？',
    ],
  },
  {
    name: 'MoE / EP / DeepEP / DualPipe',
    track: 'training',
    patterns: ['moe', 'expert parallel', 'ep', 'all-to-all', 'alltoall', 'deepep', 'dualpipe', 'deepseek'],
    relatedIds: ['tr-03', 'inf-07'],
    probes: [
      'MoE 项目里你的 Expert Parallelism 跨了多少节点？All-to-All 通信占单步耗时多少百分比，怎么做的计算通信重叠与热点专家负载均衡？',
    ],
  },
  {
    name: 'FP8 / 混合精度 / 长上下文',
    track: 'training',
    patterns: ['fp8', 'e4m3', 'transformer engine', 'ring attention', 'ulysses', '128k', '256k', 'context parallel'],
    relatedIds: ['tr-04', 'tr-05'],
    probes: [
      '开启 FP8 训练或扩 128K 长上下文后，遇到过哪些 Loss Spike 或梯度异常？怎么证明它不是数据问题而是量化 Scale/累加精度/Causal 负载问题？',
    ],
  },
  {
    name: '万卡稳定性 / Checkpoint / 容错',
    track: 'training',
    patterns: ['容错', 'checkpoint', 'goodput', 'straggler', '慢节点', 'sdc', 'hang', '千卡', '万卡', '弹性训练'],
    relatedIds: ['tr-06'],
    probes: [
      '你们集群的真实 Goodput 和 MTTR 是多少？异步 Checkpoint 阻塞训练几秒？遇到一次 NCCL Hang，你平均几分钟能精准定位到坏卡或坏光模块？',
    ],
  },
  {
    name: 'vLLM / SGLang / TRT-LLM 推理引擎',
    track: 'inference',
    patterns: ['vllm', 'sglang', 'tensorrt', 'trt-llm', 'pagedattention', 'radix', 'chunked prefill', 'continuous batching'],
    relatedIds: ['inf-01', 'inf-03', 'inf-08'],
    probes: [
      '线上部署 vLLM/SGLang 跑的是什么卡、什么模型、目标 TTFT/TPOT SLA 是多少？高峰期遇到过 KV 抢占雪崩或周期性 TPOT 毛刺吗，怎么根治的？',
    ],
  },
  {
    name: 'PD 分离 / KV Cache / MLA',
    track: 'inference',
    patterns: ['pd分离', 'disaggregated', 'distserve', 'mooncake', 'nixl', 'mla', 'prefix cache', 'kv cache'],
    relatedIds: ['inf-02', 'inf-03'],
    probes: [
      '做 PD 分离时，单请求 KV 传输多大数据量、走什么 RDMA 传输方案、怎么把传输延迟藏在 Layer-wise 流水里？P:D 节点配比是怎么随业务潮汐动态调的？',
    ],
  },
  {
    name: '量化 / 投机采样 / Roofline 优化',
    track: 'inference',
    patterns: ['awq', 'gptq', 'smoothquant', 'marlin', 'w4a16', 'w8a8', 'speculative', '投机采样', 'eagle', 'mtp', '量化'],
    relatedIds: ['inf-04', 'inf-05', 'inf-06'],
    probes: [
      '你们上线量化（W4A16/FP8）或投机采样后，在低并发和高并发大 Batch 下的吞吐/延迟曲线分别怎么变化？拐点 Batch Size 是多少？',
    ],
  },
  {
    name: 'RLHF / GRPO / veRL / OpenRLHF',
    track: 'rl',
    patterns: ['rlhf', 'ppo', 'grpo', 'dpo', 'verl', 'openrlhf', 'hybridengine', 'rollout', 'weight sync', '权重同步', 'sleep mode', 'ray'],
    relatedIds: ['rl-01', 'rl-02', 'rl-03', 'rl-04', 'rl-05', 'rl-07'],
    probes: [
      '你们 RL 系统选了 Colocated (veRL) 还是 Disaggregated (Ray)？每步 Rollout 占总时间百分之多少，权重同步花了几秒？如何处理长 CoT 尾部 Straggler 和 Train-Infer Logprob 偏移？',
    ],
  },
  {
    name: 'Agentic RL / 代码沙箱 / 多轮工具',
    track: 'rl',
    patterns: ['agent', 'sandbox', '沙箱', 'swe', 'firecracker', '多轮', 'tool call', '代码执行'],
    relatedIds: ['rl-06'],
    probes: [
      '多轮 Agentic RL 中沙箱并发规模多大？怎么避免沙箱编译/测试 IO 阻塞 GPU Rollout，以及如何防止多轮 Retokenization Drift 破坏 Prefix Cache？',
    ],
  },
  {
    name: 'CUDA / Triton / FlashAttention / NCCL',
    track: 'kernel',
    patterns: ['cuda', 'triton', 'cutlass', 'flashattention', 'liger', 'nccl', 'rdma', 'roce', 'ib', 'nsys', 'ncu', '算子', 'kernel'],
    relatedIds: ['kn-01', 'kn-02', 'kn-03', 'kn-05'],
    probes: [
      '挑一个你手写或深度调优过的最高收益 Kernel / 通信优化：用 ncu 看到它最初卡在什么 Stall 原因？优化后带宽/算力利用率从多少提到多少？',
    ],
  },
];

export function parseResumeAndSuggest(resumeText = '', targetLevel = 'P7', allQuestions = []) {
  const norm = normalize(resumeText);
  const detectedStacks = [];
  const customProbes = [];
  const recommendedIdSet = new Set();

  for (const sig of RESUME_SIGNALS) {
    const hits = sig.patterns.filter((p) => norm.includes(p));
    if (hits.length > 0) {
      detectedStacks.push({ name: sig.name, track: sig.track, hits });
      for (const probe of sig.probes) {
        customProbes.push({
          tag: sig.name,
          track: sig.track,
          question: probe,
        });
      }
      for (const id of sig.relatedIds) recommendedIdSet.add(id);
    }
  }

  // Fallback if resume has very few recognized keywords
  if (detectedStacks.length === 0) {
    customProbes.push(
      {
        tag: '项目深挖通用 5-Why',
        track: 'training',
        question: '请挑一个你最核心的 LLM Infra 项目，先给出集群规模、模型参数、核心 SLO/MFU 指标的优化前后对比，再拆解你定位最大瓶颈的实证手段。',
      },
      {
        tag: '架构权衡追问',
        track: 'inference',
        question: '在这个项目里你否决掉的备选技术方案是什么？如果流量或模型规模再扩大 10 倍，你当前架构最先崩溃的组件会是哪里？',
      }
    );
    ['tr-01', 'inf-01', 'rl-01', 'kn-03'].forEach((id) => recommendedIdSet.add(id));
  }

  const recommendedQuestions = allQuestions
    .filter((q) => recommendedIdSet.has(q.id))
    .sort((a, b) => {
      const aMatch = a.level === targetLevel ? 1 : 0;
      const bMatch = b.level === targetLevel ? 1 : 0;
      return bMatch - aMatch;
    })
    .slice(0, 6);

  return {
    source: 'local',
    detectedStacks,
    customProbes: customProbes.slice(0, 6),
    recommendedQuestionIds: recommendedQuestions.map((q) => q.id),
    summary:
      detectedStacks.length > 0
        ? `识别到 ${detectedStacks.length} 个核心 AI Infra 技术域（${detectedStacks.map((s) => s.name).join(' / ')}），已按 ${targetLevel} 标准生成定制 5-Why 深挖清单并匹配关联真题。`
        : '简历中未检测到明显 AI Infra 框架关键词，已生成通用系统架构深挖问题，建议结合全栈基础题摸底。',
    analyzedAt: Date.now(),
  };
}

export const VERDICT_LABELS = {
  strong_hire: 'Strong Hire (强烈推荐录用)',
  hire: 'Hire (推荐录用)',
  lean_hire: 'Lean Hire (倾向录用 / 需加面确认)',
  lean_no: 'Lean No Hire (倾向不录用)',
  no_hire: 'No Hire (不建议录用)',
};

export function generateLocalFeedbackReport(session, allQuestions = []) {
  const qMap = new Map(allQuestions.map((q) => [q.id, q]));
  const questions = (session.questionIds || []).map((id) => qMap.get(id)).filter(Boolean);
  const dateStr = new Date(session.updatedAt || Date.now()).toLocaleString('zh-CN');

  const trackScores = {};
  const questionBlocks = [];
  let sumScore = 0;
  let scoredCount = 0;

  questions.forEach((q, idx) => {
    const rec = session.records?.[q.id] || {};
    const stat = computeRubricScore(q, rec.rubricStates || {});
    const score = rec.aiAnalysis?.score && stat.coveredCount === 0 ? rec.aiAnalysis.score : stat.score;
    const combinedText = [rec.candidateAnswer || '', rec.txtContent || ''].filter(Boolean).join('\n');
    const hasActivity = stat.coveredCount > 0 || combinedText.trim().length > 15;
    if (hasActivity) {
      sumScore += score;
      scoredCount += 1;
      if (!trackScores[q.track]) trackScores[q.track] = [];
      trackScores[q.track].push(score);
    }

    const hitLabels = q.rubric.filter((r) => rec.rubricStates?.[r.id] === 'hit').map((r) => r.label.split('：')[0]);
    const partialLabels = q.rubric.filter((r) => rec.rubricStates?.[r.id] === 'partial').map((r) => r.label.split('：')[0]);
    const missLabels = q.rubric.filter((r) => rec.rubricStates?.[r.id] === 'miss').map((r) => r.label.split('：')[0]);

    const excerpt = extractCandidateText(combinedText, q).slice(0, 260);
    questionBlocks.push(
      [
        `### Q${idx + 1}. [${TRACKS[q.track]?.name || q.track} · ${q.level}] ${q.title}`,
        `- **本题综合评分**：${hasActivity ? `${score} / 100` : '未考察'}`,
        hitLabels.length ? `- **已掌握考点**：${hitLabels.join('；')}` : null,
        partialLabels.length ? `- **浅尝辄止**：${partialLabels.join('；')}` : null,
        missLabels.length ? `- **核心遗漏/盲区**：${missLabels.join('；')}` : null,
        rec.notes ? `- **面试官现场备注**：${rec.notes}` : null,
        excerpt ? `- **候选人关键回答摘录**：\n  > ${excerpt.replace(/\n+/g, ' / ')}${excerpt.length >= 260 ? '...' : ''}` : null,
      ]
        .filter(Boolean)
        .join('\n')
    );
  });

  const avgScore = scoredCount > 0 ? Math.round(sumScore / scoredCount) : 0;
  const autoVerdict =
    session.verdict ||
    (avgScore >= 85 ? 'strong_hire' : avgScore >= 75 ? 'hire' : avgScore >= 64 ? 'lean_hire' : avgScore >= 50 ? 'lean_no' : 'no_hire');

  const trackSummaryLines = Object.entries(trackScores).map(([trackId, scores]) => {
    const avg = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
    return `- **${TRACKS[trackId]?.name || trackId}**：平均 ${avg} 分（考察 ${scores.length} 题）`;
  });

  return [
    `# AI Infra 技术面试评估报告`,
    ``,
    `| 字段 | 内容 |`,
    `|---|---|`,
    `| **候选人** | ${session.candidateName || '未填写'} |`,
    `| **目标定级** | ${session.targetLevel || 'P7'} |`,
    `| **面试时间** | ${dateStr} |`,
    `| **综合得分** | **${avgScore} / 100**（有效评估 ${scoredCount}/${questions.length} 题） |`,
    `| **录用结论** | **${VERDICT_LABELS[autoVerdict] || autoVerdict}** |`,
    ``,
    `## 一、核心评价与定级建议`,
    `- **亮点 (Green Flags)**：${session.greenFlags?.trim() || '候选人对主干分布式/推理机制有较好理解，详见分题考点覆盖。'}`,
    `- **风险点 (Red Flags)**：${session.redFlags?.trim() || '部分定量估算或极端生产边界尚有补足空间，建议后续轮次针对性验证。'}`,
    session.overallNotes?.trim() ? `- **综合评语**：${session.overallNotes.trim()}` : null,
    ``,
    `## 二、分领域能力概览`,
    ...(trackSummaryLines.length ? trackSummaryLines : ['- 暂未完成有效分题打分']),
    ``,
    `## 三、分题考察详情与白板记录`,
    ...questionBlocks,
  ]
    .filter((line) => line !== null)
    .join('\n');
}

/**
 * Candidate Mock Mode (Secondary View): Local multi-turn interviewer responder
 */
export function evaluateCandidateMockTurn(question, turns = [], personaId = 'strict', targetLevel = 'P7') {
  const persona = PERSONAS[personaId] || PERSONAS.strict;
  const candidateTurns = turns.filter((t) => t.role === 'candidate');
  const combinedText = candidateTurns.map((t) => t.content).join('\n\n');
  const analysis = analyzeBoardAnswer(question, combinedText, targetLevel);
  const round = candidateTurns.length; // 1, 2, 3

  // Finish after round 3 or if score >= 92 on round 2
  if (round >= 3 || (round >= 2 && analysis.score >= 92)) {
    return {
      done: true,
      analysis,
      reply: `本轮考察结束。综合你的 ${round} 轮 Online Txt 作答，本题评估得分 **${analysis.score} / 100**（${analysis.levelAssessment}）。\n\n${
        analysis.strengths.length ? `**亮点**：${analysis.strengths.join('；')}\n\n` : ''
      }${analysis.gaps.length ? `**待补强盲区**：${analysis.gaps.join('；')}` : '**考点覆盖非常扎实！**'}\n\n下方已解锁 Staff 级参考架构解析与完整 Rubric 命中清单。`,
    };
  }

  // Decide next follow-up
  const missing = analysis.rubricDetails.filter((d) => d.state === 'miss').sort((a, b) => b.weight - a.weight);
  const partial = analysis.rubricDetails.filter((d) => d.state === 'partial').sort((a, b) => b.weight - a.weight);

  let nextQuestion = '';
  let lead = '';

  if (round === 1) {
    if (missing.length > 0 && analysis.score < 78) {
      lead = `${analysis.score >= 45 ? persona.praisePrefix : '收到你的初步方案。'} ${persona.probePrefix}`;
      nextQuestion = missing[0].probe;
    } else if (question.followUps?.[0]) {
      lead = `${persona.praisePrefix} ${persona.deepenPrefix}`;
      nextQuestion = question.followUps[0].question;
    } else if (partial.length > 0) {
      lead = `${persona.praisePrefix} ${persona.probePrefix}`;
      nextQuestion = partial[0].probe;
    } else {
      lead = persona.deepenPrefix;
      nextQuestion = '请补充一下这个方案在千卡/高并发极端场景下的定量计算或故障容错权衡？';
    }
  } else {
    // Round 2 -> Round 3 final stress probe
    if (question.followUps?.[1]) {
      lead = `${persona.praisePrefix} 最后一个深水区追问：`;
      nextQuestion = question.followUps[1].question;
    } else if (missing.length > 0) {
      lead = `还差最后一块拼图，再确认一个边界点：`;
      nextQuestion = missing[0].probe;
    } else {
      lead = `很好，最后补一个生产落地问题：`;
      nextQuestion = '如果让你用 3 分钟向团队做架构评审，这个方案最容易踩的 2 个线上坑是什么？怎么监控？';
    }
  }

  return {
    done: false,
    analysis,
    reply: `${lead}\n\n> ${nextQuestion}`,
  };
}
