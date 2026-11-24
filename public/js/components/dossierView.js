import { TRACKS } from '../data/index.js';
import { computeRubricScore, generateLocalFeedbackReport } from '../engine/evaluator.js';
import { aiGenerateFeedbackReport } from '../engine/llmClient.js';
import { escapeHtml } from '../engine/markdown.js';

const VERDICT_OPTIONS = [
  { id: 'strong_hire', label: 'Strong Hire', cls: 'v-strong-hire' },
  { id: 'hire', label: 'Hire', cls: 'v-hire' },
  { id: 'lean_hire', label: 'Lean Hire', cls: 'v-lean-hire' },
  { id: 'lean_no', label: 'Lean No Hire', cls: 'v-lean-no' },
  { id: 'no_hire', label: 'No Hire', cls: 'v-no-hire' },
];

export function renderDossierView(container, ctx) {
  if (ctx.role === 'candidate') {
    renderCandidateHistoryView(container, ctx);
    return;
  }
  renderInterviewerDossierView(container, ctx);
}

function renderInterviewerDossierView(container, ctx) {
  const {
    session,
    allQuestions,
    settings,
    archiveList,
    persistSession,
    archiveCurrentSession,
    loadArchivedIntoActive,
    removeArchivedSession,
    showToast,
    copyText,
    navigate,
  } = ctx;

  const qMap = new Map(allQuestions.map((q) => [q.id, q]));
  const questions = (session.questionIds || []).map((id) => qMap.get(id)).filter(Boolean);

  // Compute track breakdown & overall avg
  const trackMap = {};
  let totalScore = 0;
  let scoredCount = 0;

  const qRows = questions.map((q) => {
    const rec = session.records?.[q.id] || {};
    const stat = computeRubricScore(q, rec.rubricStates || {});
    const score = stat.coveredCount > 0 ? stat.score : rec.aiAnalysis?.score || 0;
    const hasText = (rec.candidateAnswer || '').trim().length > 15 || (rec.txtContent || '').trim().length > 15;
    const active = stat.coveredCount > 0 || hasText;
    if (active) {
      totalScore += score;
      scoredCount += 1;
      if (!trackMap[q.track]) trackMap[q.track] = [];
      trackMap[q.track].push(score);
    }
    return { q, rec, stat, score, active };
  });

  const avgScore = scoredCount > 0 ? Math.round(totalScore / scoredCount) : 0;
  const previewReport = session.generatedReport || generateLocalFeedbackReport(session, allQuestions);
  const isGenerating = Boolean(ctx.uiState.generatingReport);

  container.innerHTML = `
    <div style="max-width:1320px;margin:0 auto;display:flex;flex-direction:column;gap:18px;">
      <!-- Top Summary -->
      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title" style="font-size:16px;">面评生成工作台 · ${escapeHtml(session.candidateName || '未命名候选人')} (${escapeHtml(session.targetLevel)})</h2>
            <div class="card-sub">整合所有题目的 Rubric 打分、Online Txt 白板摘录与现场笔记，一键生成可贴入飞书 / Moka 的结构化面评</div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <button class="btn btn-ghost btn-sm" id="dos-back-iv">← 返回工作台继续面</button>
            <button class="btn btn-sm" id="dos-archive-btn">归档本次面试</button>
          </div>
        </div>

        <div class="grid-4" style="margin-bottom:16px;">
          <div class="metric-tile">
            <div class="metric-tile-lbl">综合评估均分</div>
            <div class="metric-tile-val" style="color:${avgScore >= 75 ? 'var(--track-inference)' : avgScore >= 60 ? 'var(--track-rl)' : 'var(--accent)'};">${avgScore} <span style="font-size:13px;color:var(--text-muted);">/ 100</span></div>
            <div class="metric-tile-sub">已评估 ${scoredCount} / ${questions.length} 题</div>
          </div>
          ${Object.values(TRACKS)
            .slice(0, 3)
            .map((t) => {
              const arr = trackMap[t.id] || [];
              const avg = arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : null;
              return `
                <div class="metric-tile">
                  <div class="metric-tile-lbl">${t.name}</div>
                  <div class="metric-tile-val" style="color:${t.color};">${avg !== null ? `${avg} 分` : '—'}</div>
                  <div class="metric-tile-sub">${arr.length ? `考察 ${arr.length} 题` : '本场未涉及'}</div>
                </div>
              `;
            })
            .join('')}
        </div>

        <div style="margin-bottom:14px;">
          <div style="font-size:12.5px;font-weight:700;margin-bottom:8px;color:var(--text-secondary);">录用建议结论 (Hiring Verdict)</div>
          <div class="verdict-row">
            ${VERDICT_OPTIONS.map(
              (v) => `
              <button class="verdict-btn ${v.cls} ${session.verdict === v.id ? 'selected' : ''}" data-verdict="${v.id}">
                ${v.label}
              </button>
            `
            ).join('')}
          </div>
        </div>

        <div class="grid-2">
          <div class="field">
            <label>核心亮点 (Green Flags)</label>
            <textarea id="dos-green" rows="3" placeholder="例如：对 3D 并行通信量和 Roofline 临界点心算极快，具备千卡稳定性实战经验…">${escapeHtml(session.greenFlags || '')}</textarea>
          </div>
          <div class="field">
            <label>风险短板 / 加面重点 (Red Flags)</label>
            <textarea id="dos-red" rows="3" placeholder="例如：RL 共置显存切换与 Train-Inference Precision Mismatch 偏概念，缺乏一手排障细节…">${escapeHtml(session.redFlags || '')}</textarea>
          </div>
        </div>

        <div class="field" style="margin-top:12px;">
          <label>面试官综合评语 &amp; 下一轮建议</label>
          <textarea id="dos-overall" rows="2" placeholder="总体评价及推荐定级理由…">${escapeHtml(session.overallNotes || '')}</textarea>
        </div>
      </div>

      <!-- Report Output -->
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">正式面评报告 (Markdown)</h3>
            <div class="card-sub">支持本地结构化直出或接入 Live LLM 深度润色</div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <button class="btn btn-primary btn-sm" id="dos-ai-gen-btn" ${isGenerating ? 'disabled' : ''}>
              ${isGenerating ? '<span class="spinner"></span> 生成面评中…' : settings.llm.enabled ? '⚡ AI 润色生成正式面评' : '⚡ 重新汇总面评'}
            </button>
            <button class="btn btn-sm" id="dos-copy-md-btn">复制 Markdown</button>
            <button class="btn btn-ghost btn-sm" id="dos-export-md-btn">导出 .md 文件</button>
          </div>
        </div>

        <div class="markdown-preview" id="dos-md-preview">${escapeHtml(previewReport)}</div>
      </div>

      <!-- Per-Question Breakdown -->
      <div class="card">
        <div class="card-header">
          <h3 class="card-title">本场各题得分明细</h3>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;">
          ${qRows
            .map(({ q, stat, score, active }, idx) => {
              const tr = TRACKS[q.track] || TRACKS.training;
              return `
                <div class="rubric-check-row">
                  <div>
                    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                      <span class="badge ${tr.badgeClass}">${tr.name}</span>
                      <span class="badge badge-mono">${q.level}</span>
                      <strong style="font-size:13px;">Q${idx + 1}. ${escapeHtml(q.title)}</strong>
                    </div>
                    <div style="font-size:12px;color:var(--text-muted);margin-top:3px;">
                      Rubric: 命中 ${stat.hitCount} / 一般 ${stat.partialCount} / 遗漏 ${stat.missCount} (共 ${stat.total} 项)
                    </div>
                  </div>
                  <div style="display:flex;align-items:center;gap:10px;">
                    <span class="mono" style="font-size:15px;font-weight:700;">${active ? `${score}分` : '未评'}</span>
                    <button class="btn btn-ghost btn-sm" data-jump-q="${idx}">回看白板</button>
                  </div>
                </div>
              `;
            })
            .join('')}
        </div>
      </div>

      <!-- Archived Interviews -->
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">历史面试归档 (${archiveList.length})</h3>
            <div class="card-sub">本地持久化保存，随时重新载入回看 Online Txt 记录或复盘</div>
          </div>
        </div>
        ${
          archiveList.length === 0
            ? `<div class="empty-state" style="padding:24px;">暂无已归档的面试记录，点击上方「归档本次面试」即可保存档案。</div>`
            : `<div style="display:flex;flex-direction:column;gap:8px;">
                ${archiveList
                  .map((item) => {
                    const dt = new Date(item.archivedAt || item.updatedAt || Date.now()).toLocaleString('zh-CN');
                    return `
                      <div class="rubric-check-row">
                        <div>
                          <strong style="font-size:13.5px;">${escapeHtml(item.candidateName || '未命名候选人')}</strong>
                          <span class="badge badge-mono" style="margin-left:6px;">${escapeHtml(item.targetLevel || 'P7')}</span>
                          ${item.verdict ? `<span class="badge badge-inference" style="margin-left:5px;">${escapeHtml(item.verdict)}</span>` : ''}
                          <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">归档时间：${dt} · 共 ${(item.questionIds || []).length} 题</div>
                        </div>
                        <div style="display:flex;gap:6px;">
                          <button class="btn btn-sm" data-load-arc="${item.id}">载入工作台</button>
                          <button class="btn btn-danger btn-sm" data-del-arc="${item.id}">删除</button>
                        </div>
                      </div>
                    `;
                  })
                  .join('')}
              </div>`
        }
      </div>
    </div>
  `;

  container.querySelector('#dos-back-iv')?.addEventListener('click', () => navigate('workspace'));
  container.querySelector('#dos-archive-btn')?.addEventListener('click', () => {
    archiveCurrentSession(true);
  });

  container.querySelectorAll('[data-verdict]').forEach((b) =>
    b.addEventListener('click', () => {
      session.verdict = b.dataset.verdict;
      session.generatedReport = generateLocalFeedbackReport(session, allQuestions);
      persistSession();
      ctx.rerender();
    })
  );

  const bindField = (sel, key) => {
    const el = container.querySelector(sel);
    el?.addEventListener('input', () => {
      session[key] = el.value;
      session.generatedReport = generateLocalFeedbackReport(session, allQuestions);
      persistSession();
      const previewEl = container.querySelector('#dos-md-preview');
      if (previewEl) previewEl.textContent = session.generatedReport;
    });
  };
  bindField('#dos-green', 'greenFlags');
  bindField('#dos-red', 'redFlags');
  bindField('#dos-overall', 'overallNotes');

  container.querySelector('#dos-ai-gen-btn')?.addEventListener('click', async () => {
    ctx.uiState.generatingReport = true;
    ctx.rerender();
    try {
      const res = await aiGenerateFeedbackReport(settings.llm, session, allQuestions);
      session.generatedReport = res.report;
      persistSession();
      showToast(res.source.startsWith('llm:') ? `面评已由 ${res.source.replace('llm:', '')} 润色生成` : '已重新汇总最新面评报告');
    } finally {
      ctx.uiState.generatingReport = false;
      ctx.rerender();
    }
  });

  container.querySelector('#dos-copy-md-btn')?.addEventListener('click', () => {
    copyText(session.generatedReport || previewReport, 'Markdown 面评报告已复制，可直接贴进飞书/Moka');
  });

  container.querySelector('#dos-export-md-btn')?.addEventListener('click', () => {
    const text = session.generatedReport || previewReport;
    const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI-Infra-面评-${(session.candidateName || 'candidate').replace(/\s+/g, '_')}-${session.targetLevel}.md`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('已导出 Markdown 面评文件');
  });

  container.querySelectorAll('[data-jump-q]').forEach((b) =>
    b.addEventListener('click', () => {
      session.activeIndex = Number(b.dataset.jumpQ);
      persistSession();
      navigate('workspace');
    })
  );

  container.querySelectorAll('[data-load-arc]').forEach((b) =>
    b.addEventListener('click', () => loadArchivedIntoActive(b.dataset.loadArc))
  );

  container.querySelectorAll('[data-del-arc]').forEach((b) =>
    b.addEventListener('click', () => {
      if (confirm('确定删除这条历史面试归档吗？')) {
        removeArchivedSession(b.dataset.delArc);
      }
    })
  );
}

function renderCandidateHistoryView(container, ctx) {
  const { candidateHistory, allQuestions, startSingleCandidateMock, navigate } = ctx;
  const trackTotals = { training: [], inference: [], rl: [], kernel: [] };
  for (const run of candidateHistory) {
    for (const r of run.results || []) {
      if (trackTotals[r.track]) trackTotals[r.track].push(r.score);
    }
  }

  container.innerHTML = `
    <div style="max-width:1200px;margin:0 auto;display:flex;flex-direction:column;gap:18px;">
      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">候选人模拟自测复盘</h2>
            <div class="card-sub">跟踪你在训练 / 推理 / RL / GPU 算子通信四大领域的实战得分</div>
          </div>
          <button class="btn btn-primary btn-sm" id="ch-go-mock">去开启新一轮模拟 →</button>
        </div>
        <div class="grid-4">
          ${Object.values(TRACKS)
            .map((t) => {
              const list = trackTotals[t.id] || [];
              const avg = list.length ? Math.round(list.reduce((a, b) => a + b, 0) / list.length) : null;
              return `
                <div class="metric-tile">
                  <div class="metric-tile-lbl">${t.name}</div>
                  <div class="metric-tile-val" style="color:${t.color};">${avg !== null ? `${avg} 分` : '未测'}</div>
                  <div class="metric-tile-sub">累计作答 ${list.length} 题</div>
                </div>
              `;
            })
            .join('')}
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3 class="card-title">自测历史记录 (${candidateHistory.length})</h3>
        </div>
        ${
          candidateHistory.length === 0
            ? `<div class="empty-state">还没有完成过候选人模拟自测，点击右上角去体验一轮 Online Txt 实战吧！</div>`
            : `<div style="display:flex;flex-direction:column;gap:10px;">
                ${candidateHistory
                  .map((run) => {
                    const dt = new Date(run.completedAt || Date.now()).toLocaleString('zh-CN');
                    return `
                      <div class="card" style="padding:14px;background:var(--bg-elevated);">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                          <div>
                            <strong>综合得分：${run.avgScore} / 100</strong>
                            <span class="badge badge-mono" style="margin-left:8px;">${escapeHtml(run.level || 'P7')}</span>
                          </div>
                          <span style="font-size:12px;color:var(--text-muted);">${dt}</span>
                        </div>
                        <div style="display:flex;flex-direction:column;gap:6px;">
                          ${(run.results || [])
                            .map((r) => `
                              <div class="rubric-check-row" style="background:var(--bg-surface);">
                                <div>
                                  <span class="badge ${TRACKS[r.track]?.badgeClass || ''}" style="margin-right:6px;">${TRACKS[r.track]?.short || r.track}</span>
                                  <span style="font-size:12.5px;">${escapeHtml(r.title)}</span>
                                </div>
                                <div style="display:flex;align-items:center;gap:8px;">
                                  <span class="mono" style="font-weight:700;">${r.score}分</span>
                                  <button class="btn btn-ghost btn-sm" data-retry-q="${r.questionId}">重练</button>
                                </div>
                              </div>
                            `)
                            .join('')}
                        </div>
                      </div>
                    `;
                  })
                  .join('')}
              </div>`
        }
      </div>
    </div>
  `;

  container.querySelector('#ch-go-mock')?.addEventListener('click', () => navigate('workspace'));
  container.querySelectorAll('[data-retry-q]').forEach((b) =>
    b.addEventListener('click', () => {
      const q = allQuestions.find((item) => item.id === b.dataset.retryQ);
      if (q) startSingleCandidateMock(q);
    })
  );
}
