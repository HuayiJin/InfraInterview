import {
  analyzeBoardAnswer,
  extractCandidateText,
  parseResumeAndSuggest,
  generateLocalFeedbackReport,
  evaluateCandidateMockTurn,
} from './evaluator.js';
import { explainSelectionLocal } from './explainer.js';
import { PERSONAS, TRACKS } from '../data/index.js';

async function callProxy(llmConfig, messages, maxTokens = 1400) {
  const res = await fetch('/api/llm-proxy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      baseUrl: llmConfig.baseUrl,
      apiKey: llmConfig.apiKey,
      model: llmConfig.model,
      temperature: Number(llmConfig.temperature ?? 0.35),
      max_tokens: maxTokens,
      messages,
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data.content || '';
}

function extractJsonBlock(text = '') {
  const trimmed = String(text).trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  const firstBrace = candidate.indexOf('{');
  const lastBrace = candidate.lastIndexOf('}');
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    return JSON.parse(candidate.slice(firstBrace, lastBrace + 1));
  }
  return JSON.parse(candidate);
}

export async function testLlmConnection(llmConfig) {
  const reply = await callProxy(
    llmConfig,
    [
      { role: 'system', content: 'You are a terse connectivity test assistant.' },
      { role: 'user', content: 'Reply with OK and your model readiness in 5 words.' },
    ],
    60
  );
  return { ok: true, reply: reply.trim() };
}

/**
 * Interviewer Mode: AI Analyze Candidate Answer (pasted / verbal notes / optional code board) & Generate Follow-ups
 */
export async function aiAnalyzeBoard(
  llmConfig,
  question,
  rawBoardText,
  targetLevel = 'P7',
  personaId = 'strict',
  existingRubricStates = null
) {
  const localResult = analyzeBoardAnswer(question, rawBoardText, targetLevel, existingRubricStates);
  if (!llmConfig?.enabled || !llmConfig?.baseUrl || !llmConfig?.model) {
    return localResult;
  }

  const candidateText = extractCandidateText(rawBoardText, question);
  const hasManual =
    existingRubricStates &&
    Object.values(existingRubricStates).some((s) => s === 'hit' || s === 'partial' || s === 'miss');
  if (candidateText.length < 10 && !hasManual) {
    return localResult;
  }

  const persona = PERSONAS[personaId] || PERSONAS.strict;
  const rubricListText = question.rubric
    .map((r) => {
      const manualSt = existingRubricStates?.[r.id];
      return `- id: "${r.id}" | 权重 ${r.weight}% | 当前标记: ${manualSt || 'unseen'} | 考点: ${r.label}`;
    })
    .join('\n');

  const systemPrompt = `你是顶级 AI Lab / 大厂的 Staff AI Infra 面试官副驾驶（风格：${persona.name} - ${persona.summaryTone}）。
你的任务是协助真人面试官评估候选人的口头/文字回答，并生成高杀伤力的下一轮口头追问。
严格只输出合法 JSON，不要输出任何额外说明文字。JSON Schema：
{
  "score": 0-100整数,
  "levelAssessment": "一句简洁的职级匹配结论（针对 ${targetLevel}）",
  "summary": "2-3句精准点评，指出回答核心质量、是否停留在八股还是具备一线定量与工程功底",
  "rubricStates": { "<rubricId>": "hit" | "partial" | "miss" },
  "strengths": ["亮点1", "亮点2"],
  "gaps": ["盲区或定量不足1", "盲区2"],
  "followUps": [
    { "type": "补漏追问", "intent": "考察意图", "question": "可直接口头追问候选人的精准问题" },
    { "type": "定量深挖", "intent": "考察意图", "question": "具体的数学估算/带宽/显存/复杂度追问" },
    { "type": "生产边界", "intent": "考察意图", "question": "极端规模/故障/Trade-off 追问" }
  ]
}`;

  const userPrompt = `【目标职级】${targetLevel}
【当前考题】[${TRACKS[question.track]?.name || question.track}] ${question.title}
【题干】${question.prompt}
【考点 Rubric】
${rubricListText}

【候选人回答 / 速记 / 代码记录】
${candidateText || '（面试官未贴文字，请严格依据上方已标记的 Rubric 状态针对未命中项给出追问）'}`;

  try {
    const raw = await callProxy(
      llmConfig,
      [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      1400
    );
    const parsed = extractJsonBlock(raw);
    const mergedStates = { ...localResult.rubricStates };
    if (parsed.rubricStates && typeof parsed.rubricStates === 'object') {
      for (const r of question.rubric) {
        const s = parsed.rubricStates[r.id];
        if (s === 'hit' || s === 'partial' || s === 'miss') {
          mergedStates[r.id] = s;
        }
      }
    }

    const badgeMap = { 补漏追问: 'badge-danger', 定量深挖: 'badge-training', 生产边界: 'badge-kernel' };
    const followUps = Array.isArray(parsed.followUps) && parsed.followUps.length
      ? parsed.followUps.slice(0, 3).map((f, i) => ({
          type: f.type || ['补漏追问', '定量深挖', '生产边界'][i] || '深度追问',
          badge: badgeMap[f.type] || ['badge-danger', 'badge-training', 'badge-kernel'][i] || 'badge-rl',
          intent: f.intent || '',
          question: f.question || '',
        }))
      : localResult.followUps;

    return {
      ...localResult,
      source: `llm:${llmConfig.model}`,
      score: typeof parsed.score === 'number' ? Math.max(0, Math.min(100, Math.round(parsed.score))) : localResult.score,
      levelAssessment: parsed.levelAssessment || localResult.levelAssessment,
      summary: parsed.summary || localResult.summary,
      rubricStates: mergedStates,
      strengths: Array.isArray(parsed.strengths) && parsed.strengths.length ? parsed.strengths : localResult.strengths,
      gaps: Array.isArray(parsed.gaps) && parsed.gaps.length ? parsed.gaps : localResult.gaps,
      followUps,
      analyzedAt: Date.now(),
    };
  } catch (err) {
    return {
      ...localResult,
      source: 'local-fallback',
      fallbackReason: err.message,
    };
  }
}

/**
 * Interviewer Mode: AI Parse Resume & Generate Bespoke 5-Why Interrogation Questions
 */
export async function aiParseResume(llmConfig, resumeText, targetLevel = 'P7', allQuestions = []) {
  const local = parseResumeAndSuggest(resumeText, targetLevel, allQuestions);
  if (!llmConfig?.enabled || !llmConfig?.baseUrl || !llmConfig?.model || !resumeText?.trim()) {
    return local;
  }

  const systemPrompt = `你是资深 AI Infra (LLM 训练/推理/RL/GPU 底层) 面试官。请根据候选人的简历或项目描述，生成极具针对性的 5-Why 深度拷打问题（直击简历里可能掺水的指标、架构选型权衡、规模瓶颈和排障细节）。
严格返回 JSON：
{
  "summary": "一句简历画像总结及面试摸底策略",
  "customProbes": [
    { "tag": "对应项目或技术模块", "track": "training|inference|rl|kernel", "question": "犀利具体的面试追问（可直接贴入 Online Txt 板）" }
  ]
}
生成 5~6 条高质量问题。`;

  try {
    const raw = await callProxy(
      llmConfig,
      [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `目标职级：${targetLevel}\n\n候选人简历/项目经历：\n${resumeText.slice(0, 6000)}` },
      ],
      1300
    );
    const parsed = extractJsonBlock(raw);
    return {
      ...local,
      source: `llm:${llmConfig.model}`,
      summary: parsed.summary || local.summary,
      customProbes:
        Array.isArray(parsed.customProbes) && parsed.customProbes.length ? parsed.customProbes.slice(0, 6) : local.customProbes,
      analyzedAt: Date.now(),
    };
  } catch (err) {
    return {
      ...local,
      source: 'local-fallback',
      fallbackReason: err.message,
    };
  }
}

/**
 * Interviewer Mode: AI Polish / Generate Hiring Committee Evaluation Report
 */
export async function aiGenerateFeedbackReport(llmConfig, session, allQuestions = []) {
  const localReport = generateLocalFeedbackReport(session, allQuestions);
  if (!llmConfig?.enabled || !llmConfig?.baseUrl || !llmConfig?.model) {
    return { report: localReport, source: 'local' };
  }

  const systemPrompt = `你是资深 AI Infra 面试官，请将下方原始面试记录、考点 Rubric 打分和 Online Txt 白板摘录整理润色为一份可直接粘贴进飞书/Moka/招聘系统的【正式中文技术面评报告】（Markdown 格式）。
要求：
1. 客观、有理有据，严禁编造候选人没答出的内容；
2. 明确给出综合结论、目标职级 (${session.targetLevel}) 匹配度、核心技术亮点 (Green Flags)、明确短板/风险 (Red Flags)；
3. 逐题提炼考察深度与定性/定量表现；
4. 最后给出给下一轮面试官的重点加面建议。`;

  try {
    const content = await callProxy(
      llmConfig,
      [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: localReport },
      ],
      1800
    );
    if (content && content.trim().length > 80) {
      return { report: content.trim(), source: `llm:${llmConfig.model}` };
    }
    return { report: localReport, source: 'local' };
  } catch (err) {
    return { report: localReport, source: 'local-fallback', fallbackReason: err.message };
  }
}

/**
 * Candidate Mock Mode (Secondary View): Dynamic Multi-Turn Interviewer Reply
 */
export async function aiCandidateNextTurn(llmConfig, question, turns = [], personaId = 'strict', targetLevel = 'P7') {
  const local = evaluateCandidateMockTurn(question, turns, personaId, targetLevel);
  if (!llmConfig?.enabled || !llmConfig?.baseUrl || !llmConfig?.model) {
    return { ...local, source: 'local' };
  }

  const persona = PERSONAS[personaId] || PERSONAS.strict;
  const candidateTurns = turns.filter((t) => t.role === 'candidate');
  const round = candidateTurns.length;
  const isFinal = round >= 3;

  const rubricText = question.rubric.map((r) => `- (${r.weight}%) ${r.label}`).join('\n');
  const systemPrompt = `你正在扮演一位顶级 AI Infra 面试官（风格：${persona.name}，${persona.summaryTone}），面试场景为纯文本 Online Txt 交互板，目标考察职级：${targetLevel}。
当前考题：${question.title}
原始题干：${question.prompt}
内部考察 Rubric（绝不直接把参考答案泄露给候选人）：
${rubricText}

${
  isFinal
    ? '这是本道题的最后一轮回复：请直接给出简洁专业的单题总结点评（指出亮点、遗漏盲区、定量能力评价与估分 0-100），不再抛出新追问。'
    : '请基于候选人刚才在 Online Txt 板的回答：用 1~2 句精准反馈指出对错或缺失点，然后直接抛出 1 个直击要害、可定量或贴近生产的追问（不要一次问 3 个问题）。'
}`;

  const messages = [
    { role: 'system', content: systemPrompt },
    ...turns.map((t) => ({
      role: t.role === 'candidate' ? 'user' : 'assistant',
      content: t.content,
    })),
  ];

  try {
    const reply = await callProxy(llmConfig, messages, 900);
    return {
      done: isFinal,
      analysis: local.analysis,
      reply: reply.trim() || local.reply,
      source: `llm:${llmConfig.model}`,
    };
  } catch (err) {
    return {
      ...local,
      source: 'local-fallback',
      fallbackReason: err.message,
    };
  }
}

/**
 * Page-wide Selection-to-Explain AI Assistant (Live LLM with offline glossary fallback)
 */
export async function aiExplainSelection(llmConfig, payload) {
  const {
    selectedText = '',
    surroundingContext = '',
    question = null,
    mode = 'explain',
    history = [],
    userFollowUp = '',
  } = payload || {};

  const local = explainSelectionLocal({ selectedText, surroundingContext, question, mode, userFollowUp });
  if (!llmConfig?.enabled || !llmConfig?.baseUrl || !llmConfig?.model) {
    return local;
  }

  const modeInstructions = {
    explain: '把选中片段拆开讲透：它在当前高密度参考答案里具体指什么、底层机制是什么、隐含了什么前提/边界条件。',
    derivation: '给出选中公式、字节数或复杂度结论的逐步推导过程，明确变量含义、单位（元素 vs 字节、GiB vs GB）与省略项。',
    example: '结合 H100 SXM 80GB / 70B Dense / DeepSeek-V3 等常见规格，给出一个可以直接手算复现的具体数字算例。',
    probe: '站在资深 AI Infra 面试官视角，围绕选中片段给出 2~3 句口头追问话术，并注明“答对的关键判据”和“常见误区”。',
  };

  const systemPrompt = `你是顶级 AI Lab / 大厂的 Principal AI Infra 技术顾问，正在辅助面试官或工程师精读高信息密度的 AI Infra 面试参考答案。
要求：
1. 严格精确、不编造数据、不夸大结论，凡涉及公式或硬件指标必注明前提与单位口径；
2. 回答精炼清晰（200~420字为宜），使用清晰 Markdown 列表或分段，方便阅读；
3. 紧扣用户选中的片段及其所在题目上下文作答。`;

  const baseContextMsg = `【所属题目】${question ? `[${TRACKS[question.track]?.name || question.track} · ${question.level}] ${question.title}` : '全局页面内容'}
【所在段落语境】${(surroundingContext || '').slice(0, 700)}
【用户划词选中的片段】
「${selectedText}」

【本次需求】${userFollowUp ? userFollowUp : modeInstructions[mode] || modeInstructions.explain}`;

  const messages = [{ role: 'system', content: systemPrompt }];
  if (Array.isArray(history) && history.length > 0 && userFollowUp) {
    messages.push({ role: 'user', content: `【所属题目】${question?.title || 'AI Infra'}\n【语境】${(surroundingContext || '').slice(0, 600)}\n【划词片段】「${selectedText}」` });
    for (const turn of history.slice(-6)) {
      messages.push({ role: turn.role === 'user' ? 'user' : 'assistant', content: turn.content });
    }
    messages.push({ role: 'user', content: userFollowUp });
  } else {
    messages.push({ role: 'user', content: baseContextMsg });
  }

  try {
    const content = await callProxy(llmConfig, messages, 1100);
    if (content && content.trim().length > 15) {
      return {
        source: `llm:${llmConfig.model}`,
        content: content.trim(),
      };
    }
    return local;
  } catch (err) {
    return {
      ...local,
      source: 'local-fallback',
      fallbackReason: err.message,
    };
  }
}
