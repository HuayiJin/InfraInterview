import test from 'node:test';
import assert from 'node:assert/strict';

import { ALL_QUESTIONS, TRACKS } from '../public/js/data/index.js';
import {
  extractCandidateText,
  computeRubricScore,
  analyzeBoardAnswer,
  parseResumeAndSuggest,
  generateLocalFeedbackReport,
  evaluateCandidateMockTurn,
} from '../public/js/engine/evaluator.js';
import { renderMarkdown } from '../public/js/engine/markdown.js';

test('Question bank has 28 balanced AI Infra questions with valid schema & 100% rubric weights', () => {
  assert.equal(ALL_QUESTIONS.length, 28);
  const byTrack = { training: 0, inference: 0, rl: 0, kernel: 0 };
  const ids = new Set();

  for (const q of ALL_QUESTIONS) {
    assert.ok(q.id && !ids.has(q.id), `Duplicate or missing id: ${q.id}`);
    ids.add(q.id);
    assert.ok(TRACKS[q.track], `Invalid track ${q.track} on ${q.id}`);
    byTrack[q.track] += 1;
    assert.ok(['P6', 'P7', 'P8'].includes(q.level), `Invalid level ${q.level} on ${q.id}`);
    assert.ok(q.prompt.length > 30, `Prompt too short on ${q.id}`);
    assert.ok(q.referenceAnswer.length > 100, `Reference answer too short on ${q.id}`);
    assert.ok(Array.isArray(q.rubric) && q.rubric.length >= 4, `Rubric too small on ${q.id}`);
    const weightSum = q.rubric.reduce((s, r) => s + r.weight, 0);
    assert.equal(weightSum, 100, `Rubric weights sum to ${weightSum} (expected 100) on ${q.id}`);
    assert.ok(Array.isArray(q.hints) && q.hints.length >= 3, `Hints missing on ${q.id}`);
    assert.ok(Array.isArray(q.followUps) && q.followUps.length >= 2, `Follow-ups missing on ${q.id}`);
  }

  assert.equal(byTrack.training, 8);
  assert.equal(byTrack.inference, 8);
  assert.equal(byTrack.rl, 7);
  assert.equal(byTrack.kernel, 5);
});

test('extractCandidateText removes inserted prompt block cleanly', () => {
  const q = ALL_QUESTIONS[0];
  const board = `=== [题目] ${q.title} (${q.level}) ===\n${q.prompt}\n参考参数：1024×H100 80G | 8×GPU NVLink 900GB/s\n====================================\n[候选人作答区]\nTP=8 限制在机内 NVLink，PP 跨机传激活。`;
  const extracted = extractCandidateText(board, q);
  assert.ok(extracted.includes('TP=8'));
  assert.ok(!extracted.includes('[题目]'));
  assert.ok(!extracted.includes('参考参数'));
  assert.ok(!extracted.includes('[候选人作答区]'));
});

test('analyzeBoardAnswer scores strong quantitative AI Infra answer higher than weak answer', () => {
  const q = ALL_QUESTIONS.find((x) => x.id === 'tr-01');
  const weak = analyzeBoardAnswer(q, '分布式训练主要就是把模型切分到多张卡上并行跑。', 'P7');
  const strong = analyzeBoardAnswer(
    q,
    `第一性原则：显存放得下时应尽量克制 TP 和 PP/VPP，优先用 FSDP2/HSDP 零气泡，避免 TP 卡在关键路径把 GEMM 切窄。
    在 1024×H100 (80GB) 上 70B 静态显存 16Ψ=1120GB，HSDP 64 路分片单卡仅 17.5GB；但 GBS=4M、seq=8K 只有 512 条序列 ⇒ DP≤512，所以最小只需 TP=2 × PP=1 × DP=512 (HSDP)，开 SP 机内 NVLink (AllGather+ReduceScatter)，每层前向通信 8BSH 字节。
    PP 与 Interleaved VPP 只有显存或通信域受限时才上，且要注意层数整除 (p*v|80) 和 Bubble 气泡率 (p-1)/(v*m+p-1)。
    扩到 128K 长上下文时，单步仅剩 32 条序列且激活显存暴涨，需要切出 CP=16（机内 Ulysses + 跨机 Ring Attention）分摊激活显存。`,
    'P7'
  );

  assert.ok(strong.score >= 75, `Expected strong score >= 75, got ${strong.score}`);
  assert.ok(strong.score > weak.score + 40, `Expected strong (${strong.score}) >> weak (${weak.score})`);
  assert.equal(strong.followUps.length, 3, 'Should always suggest 3 follow-up cards even when 5/5 hit');
  assert.equal(computeRubricScore(q, strong.rubricStates).hitCount >= 4, true);

  // Manual Rubric preservation when interviewer didn't paste long text
  const manualOnly = analyzeBoardAnswer(q, '', 'P7', {
    'minimize-tp-pp': 'hit',
    'gbs-tp-nvlink': 'hit',
    'pp-vpp-tradeoff': 'miss',
  });
  assert.equal(manualOnly.rubricStates['minimize-tp-pp'], 'hit');
  assert.equal(manualOnly.rubricStates['pp-vpp-tradeoff'], 'miss');
  assert.ok(manualOnly.score > 40, 'Should compute score from manual rubric ticks');
  assert.equal(manualOnly.followUps.length, 3);
});

test('parseResumeAndSuggest extracts Megatron, vLLM, veRL, GRPO stacks accurately', () => {
  const resume = `主导千卡 H800 集群基于 Megatron-LM 和 DeepEP 训练 671B MoE (FP8)；线上基于 vLLM/SGLang 落地 PD分离 与 MLA 优化；近期基于 veRL HybridEngine 搭建长 CoT GRPO 强化学习链路，解决权重同步与 CuMem Sleep 显存问题。`;
  const res = parseResumeAndSuggest(resume, 'P7', ALL_QUESTIONS);
  assert.ok(res.detectedStacks.length >= 4, `Expected >=4 stacks, got ${res.detectedStacks.length}`);
  assert.ok(res.customProbes.length >= 4);
  assert.ok(res.recommendedQuestionIds.length >= 3);
});

test('generateLocalFeedbackReport generates valid structured Markdown report', () => {
  const session = {
    id: 'test-1',
    candidateName: '张工',
    targetLevel: 'P7',
    verdict: 'hire',
    greenFlags: 'Roofline 与 3D 并行定量能力突出',
    redFlags: 'RL 长尾异步调度经验偏少',
    overallNotes: '推荐录用 P7',
    questionIds: ['tr-01', 'inf-04'],
    records: {
      'tr-01': {
        txtContent: '显存放得下时优先 HSDP/FSDP2，克制 TP=2、PP=1 避免气泡，128K 扩长开 CP=16。',
        rubricStates: { 'minimize-tp-pp': 'hit', 'gbs-tp-nvlink': 'hit', 'pp-vpp-tradeoff': 'hit', 'cp-long': 'partial', 'quant-math': 'miss' },
        notes: '思路清晰',
      },
    },
    updatedAt: Date.now(),
  };
  const md = generateLocalFeedbackReport(session, ALL_QUESTIONS);
  assert.ok(md.includes('# AI Infra 技术面试评估报告'));
  assert.ok(md.includes('张工'));
  assert.ok(md.includes('Hire (推荐录用)'));
  assert.ok(md.includes('Roofline 与 3D 并行定量能力突出'));
});

test('evaluateCandidateMockTurn progresses through 3 rounds and finishes cleanly', () => {
  const q = ALL_QUESTIONS.find((x) => x.id === 'rl-01');
  const turns = [{ role: 'candidate', content: 'PPO 有 Actor Critic Ref RM 四个模型，显存 36Ψ。' }];
  const r1 = evaluateCandidateMockTurn(q, turns, 'strict', 'P7');
  assert.equal(r1.done, false);
  assert.ok(r1.reply.length > 20);

  turns.push({ role: 'interviewer', content: r1.reply });
  turns.push({ role: 'candidate', content: 'GRPO 去掉 Critic 靠 Group G 采样算相对 Advantage，瓶颈转到长 CoT Rollout 占 80% 时间。' });
  const r2 = evaluateCandidateMockTurn(q, turns, 'strict', 'P7');
  assert.equal(r2.done, false);

  turns.push({ role: 'interviewer', content: r2.reply });
  turns.push({ role: 'candidate', content: 'DPO 离线无需 Rollout，GRPO 共享 Prompt KV Prefix Caching 省显存。' });
  const r3 = evaluateCandidateMockTurn(q, turns, 'strict', 'P7');
  assert.equal(r3.done, true);
  assert.ok(r3.analysis.score > 60);
});

test('renderMarkdown escapes raw HTML tags safely and formats tables', () => {
  const html = renderMarkdown('### 标题\n<script>alert(1)</script>\n| A | B |\n|---|---|\n| 1 | `x` |');
  assert.ok(!html.includes('<script>'), 'Script tag must be escaped');
  assert.ok(html.includes('&lt;script&gt;'));
  assert.ok(html.includes('<table class="md-table">'));
  assert.ok(html.includes('<code>x</code>'));
});
