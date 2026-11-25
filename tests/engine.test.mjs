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
import { explainSelectionLocal } from '../public/js/engine/explainer.js';
import {
  CURATED_VIDEOS,
  getVideosForQuestion,
  getVideosForSelection,
  buildYoutubeSearchUrl,
} from '../public/js/data/videos.js';

test('Question bank has 32 balanced AI Infra questions with valid schema & 100% rubric weights', () => {
  assert.equal(ALL_QUESTIONS.length, 32);
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

  assert.equal(byTrack.training, 10);
  assert.equal(byTrack.inference, 9);
  assert.equal(byTrack.rl, 7);
  assert.equal(byTrack.kernel, 6);
});

test('Built-in reference answers cover their own local rubric vocabulary', () => {
  for (const q of ALL_QUESTIONS) {
    const { rubricDetails } = analyzeBoardAnswer(q, q.referenceAnswer);
    for (const item of rubricDetails) {
      assert.notEqual(item.state, 'miss', `${q.id} reference missed ${item.id}`);
    }
  }
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
    `先比较 FSDP2/HSDP 的参数 AllGather 与 TP 通信。TP 可能把 GEMM 切窄，PP 会引入气泡。
    GBS=4,194,304 tokens、seq=8192 时有 512 条序列。取 TP=2 × PP=1 × DP=512，DP 内 HSDP 64 路分片时，每卡静态显存约 1120/(2×64)=8.75 GB，还要预留激活和临时区。机内 NVLink 上开 TP+SP，每层前向两次 AllGather 和两次 ReduceScatter，每卡发送 8(T−1)/T × BSH 字节。
    等时 stage 的 1F1B 理想气泡率为 (p−1)/(m+p−1)，均匀 Interleaved VPP 还要核对层数整除及新增 P2P。
    128K 时只有 32 条序列，可取 TP=2、CP=16、DP=32。每个 TP rank 有 4 个 KV head，机内 Ulysses=4，再跨机 Ring Attention=4。`,
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
  const turns = [{ role: 'candidate', content: '经典 RLHF-PPO 若有独立等大 Actor、Critic、Ref 和 RM，示例静态状态总量是 36Ψ 字节。' }];
  const r1 = evaluateCandidateMockTurn(q, turns, 'strict', 'P7');
  assert.equal(r1.done, false);
  assert.ok(r1.reply.length > 20);

  turns.push({ role: 'interviewer', content: r1.reply });
  turns.push({ role: 'candidate', content: 'GRPO 去掉 Critic，用同 prompt 的 Group 采样算相对 Advantage。长 CoT 的自回归 Rollout 可能成为瓶颈，可分阶段计时。' });
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

test('explainSelectionLocal explains highlighted terms across modes and profiling topics', () => {
  const q1 = ALL_QUESTIONS.find((x) => x.id === 'tr-01');
  const exp1 = explainSelectionLocal({
    selectedText: '8(T−1)/T × bSH',
    surroundingContext: '典型 Megatron 稠密层的 Attention 和 MLP 各做一次 AG 和一次 RS',
    question: q1,
    mode: 'derivation',
  });
  assert.equal(exp1.source, 'local');
  assert.ok(exp1.content.includes('AllGather') && exp1.content.includes('ReduceScatter'));

  const qProf = ALL_QUESTIONS.find((x) => x.id === 'kn-06');
  const expProf = explainSelectionLocal({
    selectedText: 'Long Scoreboard',
    surroundingContext: 'Warp Stall Breakdown',
    question: qProf,
    mode: 'explain',
  });
  assert.ok(expProf.content.includes('L1TEX'));

  const expProbe = explainSelectionLocal({
    selectedText: 'expm1(d)−d',
    surroundingContext: 'k3 数值对照',
    question: ALL_QUESTIONS.find((x) => x.id === 'rl-07'),
    mode: 'probe',
  });
  assert.ok(expProbe.content.includes('追问'));
});

test('Curated YouTube videos use verified permanent watch URLs and match questions/selections accurately', () => {
  const validQIds = new Set(ALL_QUESTIONS.map((q) => q.id));
  const seenIds = new Set();
  assert.ok(CURATED_VIDEOS.length >= 20, `Expected >=20 curated visual videos, got ${CURATED_VIDEOS.length}`);

  for (const v of CURATED_VIDEOS) {
    assert.match(v.id, /^[A-Za-z0-9_-]{11}$/, `Invalid YouTube id: ${v.id}`);
    assert.ok(!seenIds.has(v.id), `Duplicate video id: ${v.id}`);
    seenIds.add(v.id);
    assert.equal(v.url, `https://www.youtube.com/watch?v=${v.id}`, `Non-canonical URL on ${v.id}`);
    assert.ok(v.title && v.note && v.channel && v.styleTag, `Missing metadata on ${v.id}`);
    assert.ok(Array.isArray(v.questionIds) && v.questionIds.length >= 1);
    for (const qId of v.questionIds) {
      assert.ok(validQIds.has(qId), `Video ${v.id} references unknown question ${qId}`);
    }
  }

  // Question lookup: core visual topics have 1-2 videos, unmapped niche topics cleanly empty
  const mlaVideos = getVideosForQuestion('inf-03');
  assert.equal(mlaVideos.length, 2);
  assert.ok(mlaVideos.some((v) => v.id === 'DBdeFbNrIZI'));
  assert.equal(getVideosForQuestion('rl-07').length, 0);

  // Selection lookup: direct term hit works across questions; unrelated niche terms stay empty
  const flashMatch = getVideosForSelection('Online Softmax', 'FlashAttention tiling', 'kn-01');
  assert.ok(flashMatch.length >= 1 && flashMatch.length <= 2);
  assert.equal(flashMatch[0].id, 'us1nzaQKq0Y');

  const ncuMatch = getVideosForSelection('Long Scoreboard', 'ncu warp stall', 'kn-06');
  assert.ok(ncuMatch.some((v) => v.id === 'Iuy_RAvguBM'));

  assert.equal(getVideosForSelection('完全无关的随机短语', '', 'rl-07').length, 0);

  // YouTube search URL fallback
  const ytUrl = buildYoutubeSearchUrl('CUDA VMM CuMem', ALL_QUESTIONS.find((q) => q.id === 'rl-05'));
  assert.ok(ytUrl.startsWith('https://www.youtube.com/results?search_query='));
  assert.ok(decodeURIComponent(ytUrl).includes('CUDA VMM CuMem'));
});
