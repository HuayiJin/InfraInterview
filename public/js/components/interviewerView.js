import { TRACKS, TARGET_LEVELS, PERSONAS } from '../data/index.js';
import { computeRubricScore, buildFollowUpsForStates } from '../engine/evaluator.js';
import { aiAnalyzeBoard, aiParseResume } from '../engine/llmClient.js';
import { escapeHtml, renderMarkdown } from '../engine/markdown.js';

function getRecord(session, qId) {
  if (!session.records) session.records = {};
  if (!session.records[qId]) {
    session.records[qId] = {
      candidateAnswer: '',
      txtContent: '',
      showCodeBoard: false,
      rubricStates: {},
      aiAnalysis: null,
      notes: '',
      updatedAt: Date.now(),
    };
  }
  const rec = session.records[qId];
  // Backward compatibility if an older localStorage record only has txtContent
  if (typeof rec.candidateAnswer !== 'string') rec.candidateAnswer = '';
  if (typeof rec.txtContent !== 'string') rec.txtContent = '';
  return rec;
}

function updateLineNumbers(gutterEl, text) {
  if (!gutterEl) return;
  const lineCount = Math.max(8, String(text || '').split('\n').length + 1);
  let out = '';
  for (let i = 1; i <= lineCount; i++) {
    out += i + (i === lineCount ? '' : '\n');
  }
  gutterEl.textContent = out;
}

export function renderInterviewerView(container, ctx) {
  const {
    session,
    allQuestions,
    settings,
    persistSession,
    showToast,
    copyText,
    navigate,
    openQuickAssembleModal,
    resetNewSession,
    archiveCurrentSession,
  } = ctx;

  const qMap = new Map(allQuestions.map((q) => [q.id, q]));
  let queue = (session.questionIds || []).map((id) => qMap.get(id)).filter(Boolean);
  if (queue.length === 0) {
    session.questionIds = [allQuestions[0].id];
    session.activeIndex = 0;
    queue = [allQuestions[0]];
    persistSession();
  }
  if (session.activeIndex >= queue.length || session.activeIndex < 0) {
    session.activeIndex = 0;
  }

  const activeQ = queue[session.activeIndex];
  const record = getRecord(session, activeQ.id);
  const trackMeta = TRACKS[activeQ.track] || TRACKS.training;
  const rubricStat = computeRubricScore(activeQ, record.rubricStates);
  const activeRightTab = ctx.uiState.copilotTab === 'resume' ? 'resume' : 'reference';
  const isAnalyzing = Boolean(ctx.uiState.analyzingQId === activeQ.id);
  const isParsingResume = Boolean(ctx.uiState.parsingResume);

  // Code whiteboard is only visible when explicitly toggled on OR when question is a coding scenario
  const codeBoardOpen = Boolean(record.showCodeBoard || activeQ.isCoding);

  const effectiveScore = rubricStat.coveredCount > 0 ? rubricStat.score : record.aiAnalysis?.score || 0;
  const followUpsToShow = record.aiAnalysis?.followUps?.length
    ? record.aiAnalysis.followUps
    : buildFollowUpsForStates(activeQ, record.rubricStates, session.targetLevel);

  container.innerHTML = `
    <div class="cockpit-shell">
      <!-- Top Candidate & Session Strip -->
      <div class="candidate-strip">
        <div class="candidate-strip-left">
          <span class="badge badge-training">面试官视角 · 口头主控</span>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:12.5px;color:var(--text-secondary);">
            <span>候选人</span>
            <input type="text" class="inline-input" id="iv-cand-name" value="${escapeHtml(session.candidateName || '')}" placeholder="输入姓名/编号" style="width:124px;" />
          </label>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:12.5px;color:var(--text-secondary);">
            <span>定级</span>
            <select class="inline-input" id="iv-target-level">
              ${TARGET_LEVELS.map(
                (lv) => `<option value="${lv.id}" ${session.targetLevel === lv.id ? 'selected' : ''}>${lv.label}</option>`
              ).join('')}
            </select>
          </label>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:12.5px;color:var(--text-secondary);">
            <span>追问风格</span>
            <select class="inline-input" id="iv-persona">
              ${Object.values(PERSONAS)
                .map((p) => `<option value="${p.id}" ${session.persona === p.id ? 'selected' : ''}>${p.name}</option>`)
                .join('')}
            </select>
          </label>
        </div>
        <div class="candidate-strip-right">
          <button class="btn btn-sm" id="iv-assemble-btn">快速组卷</button>
          <button class="btn btn-sm" id="iv-new-session-btn">新建面试</button>
          <button class="btn btn-primary btn-sm" id="iv-go-dossier-btn">生成面评报告 →</button>
        </div>
      </div>

      <!-- Question Queue Tabs -->
      <div class="question-queue-bar" id="iv-queue-bar">
        ${queue
          .map((q, idx) => {
            const r = getRecord(session, q.id);
            const st = computeRubricScore(q, r.rubricStates);
            const hasScore = st.coveredCount > 0 || (r.aiAnalysis && r.aiAnalysis.score > 0);
            const displayScore = st.coveredCount > 0 ? st.score : r.aiAnalysis?.score || 0;
            const tr = TRACKS[q.track] || TRACKS.training;
            return `
              <div class="q-tab ${idx === session.activeIndex ? 'active' : ''}" data-q-idx="${idx}">
                <span class="track-dot" style="background:${tr.color}"></span>
                <span>Q${idx + 1}. ${escapeHtml(q.title.slice(0, 12))}${q.title.length > 12 ? '…' : ''}</span>
                <span class="badge badge-mono" style="padding:0 5px;font-size:10px;">${q.level}</span>
                ${hasScore ? `<span class="q-tab-score">${displayScore}分</span>` : ''}
                ${queue.length > 1 ? `<span class="q-tab-close" data-remove-q="${q.id}" title="从本题单移除">✕</span>` : ''}
              </div>
            `;
          })
          .join('')}
        <button class="btn btn-ghost btn-sm" id="iv-add-from-bank">+ 题库选题</button>
      </div>

      <!-- Verbal-First Main Split: Left 62% Verbal Interview Flow | Right 38% Reference & Resume -->
      <div class="verbal-grid">
        <!-- Left Main Column: Question + Optional Code Pad + Candidate Answer Paste + AI Follow-ups + Rubric -->
        <section class="verbal-main-col" aria-label="口头面试主控区">
          <!-- 1. Current Question Hero Card -->
          <div class="prompt-hero-card">
            <div class="prompt-hero-head">
              <div style="flex:1;min-width:220px;">
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                  <span class="badge ${trackMeta.badgeClass}">${trackMeta.name}</span>
                  <span class="badge badge-mono">${activeQ.level}</span>
                  ${activeQ.incident ? '<span class="badge badge-danger">线上排障</span>' : ''}
                  ${activeQ.isCoding ? '<span class="badge badge-kernel">代码场景</span>' : ''}
                  ${(activeQ.contextSpecs || []).map((s) => `<span class="badge badge-mono">${escapeHtml(s)}</span>`).join('')}
                </div>
                <h2 class="prompt-hero-title">${escapeHtml(activeQ.title)}</h2>
              </div>
              <div style="display:flex;gap:6px;flex-wrap:wrap;flex-shrink:0;">
                <button class="btn btn-sm ${codeBoardOpen ? 'btn-primary' : 'btn-ghost'}" id="iv-toggle-code-btn" title="代码题或公式推导时按需展开纯文本白板">
                  ${codeBoardOpen ? '收起代码白板' : '📝 代码/公式白板'}
                </button>
                <button class="btn btn-sm" id="iv-copy-prompt-btn">复制题干</button>
              </div>
            </div>
            <p class="prompt-hero-text">${escapeHtml(activeQ.prompt)}</p>
          </div>

          <!-- 2. On-Demand Code / Formula Whiteboard (Visible ONLY in coding scenarios or when toggled) -->
          ${
            codeBoardOpen
              ? `
            <div class="code-board-collapsible" id="iv-code-board-wrap">
              <div class="txt-board-header">
                <div class="txt-board-title-group">
                  <span class="txt-board-label">CODE / FORMULA TXT</span>
                  <span style="font-size:12px;color:var(--text-secondary);">代码题 / 伪代码 / 公式推导白板（将自动合并参与 AI 分析）</span>
                </div>
                <div class="txt-board-actions">
                  <button class="btn btn-ghost btn-sm" id="iv-insert-prompt-btn">写入题干</button>
                  <button class="btn btn-ghost btn-sm" id="iv-copy-board-btn">复制白板</button>
                  <button class="btn btn-ghost btn-sm" id="iv-clear-board-btn">清空</button>
                </div>
              </div>
              <div class="txt-editor-wrap">
                <div class="txt-gutter" id="iv-txt-gutter" aria-hidden="true"></div>
                <textarea
                  class="txt-editor"
                  id="iv-txt-editor"
                  spellcheck="false"
                  placeholder="// 代码/公式白板：粘贴候选人写的代码、Triton/PyTorch 伪代码或手推公式…"
                >${escapeHtml(record.txtContent || '')}</textarea>
              </div>
            </div>
          `
              : ''
          }

          <!-- 3. Candidate Answer Paste / Verbal Notes Box -->
          <div class="answer-paste-card">
            <div class="answer-paste-head">
              <div>
                <strong style="font-size:13.5px;">候选人回答粘贴 / 口述速记区</strong>
                <span style="font-size:12px;color:var(--text-muted);margin-left:8px;">支持粘贴会议转写、候选人文字作答，或随手记口述关键词</span>
              </div>
              <span style="font-size:11.5px;color:var(--text-muted);" class="mono" id="iv-answer-stats">0 字</span>
            </div>

            <textarea
              class="answer-paste-textarea"
              id="iv-cand-answer"
              placeholder="在这里粘贴候选人的回答 / 会议实时字幕 / 口述要点速记…&#10;例如，4,194,304 tokens 按 8,192 分成 512 条序列。候选人给出 TP=2、DP=512，已算静态分片显存，还需补充 AllGather 和激活峰值。"
            >${escapeHtml(record.candidateAnswer || '')}</textarea>

            <div class="answer-paste-footer">
              <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                <button class="btn btn-ghost btn-sm" id="iv-clear-answer-btn">清空回答</button>
                <span style="font-size:11.5px;color:var(--text-muted);">
                  快捷键 <code class="mono">⌘/Ctrl + Enter</code> · 引擎：<strong style="color:var(--accent);">${settings.llm.enabled ? escapeHtml(settings.llm.model) : '本地专家引擎'}</strong>
                </span>
              </div>
              <button class="btn btn-primary btn-sm" id="iv-ai-analyze-btn" ${isAnalyzing ? 'disabled' : ''}>
                ${isAnalyzing ? '<span class="spinner"></span> AI 分析中…' : '⚡ AI 分析回答 &amp; 生成下一句追问'}
              </button>
            </div>
          </div>

          <!-- 4. AI Diagnosis + Ready-to-Ask Follow-up Cards -->
          <div class="card" style="padding:14px 16px;">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
              <div style="display:flex;align-items:center;gap:8px;">
                <strong style="font-size:13.5px;">${record.aiAnalysis ? 'AI 诊断 &amp; 下一句口头追问建议' : '下一轮口头追问建议（按当前考点动态排序）'}</strong>
                ${
                  record.aiAnalysis
                    ? `<span class="badge badge-mono">${record.aiAnalysis.source.startsWith('llm:') ? escapeHtml(record.aiAnalysis.source) : '本地专家引擎'}</span>`
                    : ''
                }
              </div>
              <span style="font-size:11.5px;color:var(--text-muted);">直接照着口头问，或点复制发给候选人</span>
            </div>

            ${
              isAnalyzing
                ? `<div class="callout" style="margin-bottom:10px;"><span class="loading-shimmer"><span class="spinner"></span> 正在评估候选人回答并结合 ${session.targetLevel} 标准组织追问…</span></div>`
                : record.aiAnalysis
                  ? `
                  <div class="callout ${record.aiAnalysis.score >= 75 ? 'callout-success' : record.aiAnalysis.score >= 50 ? 'callout-warn' : ''}" style="margin-bottom:12px;">
                    <div style="font-size:12.5px;color:var(--text-primary);"><strong>定级判断：</strong>${escapeHtml(record.aiAnalysis.levelAssessment)}</div>
                    <div style="font-size:12.5px;color:var(--text-secondary);margin-top:3px;">${escapeHtml(record.aiAnalysis.summary)}</div>
                    ${
                      record.aiAnalysis.gaps?.length
                        ? `<div style="font-size:12px;color:var(--danger);margin-top:5px;"><strong>待追问盲区：</strong>${escapeHtml(record.aiAnalysis.gaps.slice(0, 2).join('；'))}</div>`
                        : ''
                    }
                  </div>
                `
                  : ''
            }

            <div class="followup-grid">
              ${followUpsToShow
                .map(
                  (f) => `
                <div class="followup-card">
                  <div class="followup-card-head">
                    <span class="badge ${f.badge || 'badge-training'}">${escapeHtml(f.type)}</span>
                    <div class="followup-actions">
                      ${
                        codeBoardOpen
                          ? `<button class="btn btn-ghost btn-sm" data-insert-text="${encodeURIComponent(f.question)}">进白板</button>`
                          : ''
                      }
                      <button class="btn btn-sm" data-copy-text="${encodeURIComponent(f.question)}">复制</button>
                    </div>
                  </div>
                  ${f.intent ? `<div style="font-size:11.5px;color:var(--text-muted);">${escapeHtml(f.intent)}</div>` : ''}
                  <div class="followup-text">${escapeHtml(f.question)}</div>
                </div>
              `
                )
                .join('')}
            </div>
          </div>

          <!-- 5. Wide Interactive Rubric Scoring Matrix (with inline verbal probes!) -->
          <div class="card" style="padding:14px 16px;">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
              <div style="display:flex;align-items:center;gap:12px;">
                <span class="score-big" style="font-size:24px;color:${effectiveScore >= 75 ? 'var(--track-inference)' : effectiveScore >= 55 ? 'var(--track-rl)' : 'var(--accent)'};">${effectiveScore}</span>
                <div>
                  <strong style="font-size:13.5px;">考点 Rubric 实时打分（边听边点）</strong>
                  <div style="font-size:11.5px;color:var(--text-muted);">命中 ${rubricStat.hitCount} · 一般 ${rubricStat.partialCount} · 遗漏 ${rubricStat.missCount}（共 ${rubricStat.total} 项）</div>
                </div>
              </div>
              <button class="btn btn-ghost btn-sm" id="iv-reset-rubric-btn">重置打分</button>
            </div>

            <div class="rubric-list">
              ${activeQ.rubric
                .map((r) => {
                  const st = record.rubricStates?.[r.id] || 'unseen';
                  return `
                  <div class="rubric-wide-row state-${st}">
                    <div class="rubric-wide-top">
                      <div style="font-size:13px;line-height:1.5;color:var(--text-primary);flex:1;">
                        <span class="mono" style="color:var(--accent);font-weight:700;margin-right:6px;">[${r.weight}%]</span>${escapeHtml(r.label)}
                      </div>
                      <div class="rubric-state-group">
                        <button class="state-pill ${st === 'hit' ? 'active-hit' : ''}" data-rubric-set="${r.id}|hit">命中</button>
                        <button class="state-pill ${st === 'partial' ? 'active-partial' : ''}" data-rubric-set="${r.id}|partial">一般</button>
                        <button class="state-pill ${st === 'miss' ? 'active-miss' : ''}" data-rubric-set="${r.id}|miss">漏</button>
                      </div>
                    </div>
                    <div class="rubric-probe-line">
                      <span><strong>口头追问话术：</strong>${escapeHtml(r.probe)}</span>
                      <button class="btn btn-ghost btn-sm" style="padding:2px 7px;font-size:11px;flex-shrink:0;" data-copy-text="${encodeURIComponent(r.probe)}">复制</button>
                    </div>
                  </div>
                `;
                })
                .join('')}
            </div>
          </div>
        </section>

        <!-- Right Auxiliary Column: Reference Answer Cheat Sheet (Default) & Resume 5-Why / Notes -->
        <aside class="copilot-panel" aria-label="面试官速查与简历深挖">
          <div class="copilot-tabs">
            <button class="copilot-tab ${activeRightTab === 'reference' ? 'active' : ''}" data-cp-tab="reference">参考答案 &amp; 提示卡</button>
            <button class="copilot-tab ${activeRightTab === 'resume' ? 'active' : ''}" data-cp-tab="resume">简历深挖 &amp; 私密笔记</button>
          </div>

          <div class="copilot-body" id="iv-copilot-body">
            ${
              activeRightTab === 'reference'
                ? renderReferenceAndHintsTab(activeQ)
                : renderResumeNotesTab(session, record, allQuestions, isParsingResume)
            }
          </div>
        </aside>
      </div>
    </div>
  `;

  // Candidate answer paste textarea
  const answerEl = container.querySelector('#iv-cand-answer');
  const answerStatsEl = container.querySelector('#iv-answer-stats');
  const syncAnswerStats = () => {
    const chars = (answerEl?.value || '').replace(/\s+/g, '').length;
    if (answerStatsEl) answerStatsEl.textContent = `${chars} 字`;
  };
  syncAnswerStats();

  let saveTimer = null;
  answerEl?.addEventListener('input', () => {
    record.candidateAnswer = answerEl.value;
    record.updatedAt = Date.now();
    syncAnswerStats();
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => persistSession(), 250);
  });

  answerEl?.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      triggerAiAnalysis();
    }
  });

  // Optional Code Board wiring (only exists when codeBoardOpen)
  const editorEl = container.querySelector('#iv-txt-editor');
  const gutterEl = container.querySelector('#iv-txt-gutter');
  if (editorEl && gutterEl) {
    const syncCodeGutter = () => updateLineNumbers(gutterEl, editorEl.value || '');
    syncCodeGutter();
    editorEl.addEventListener('scroll', () => {
      gutterEl.scrollTop = editorEl.scrollTop;
    });
    editorEl.addEventListener('input', () => {
      record.txtContent = editorEl.value;
      record.updatedAt = Date.now();
      syncCodeGutter();
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => persistSession(), 250);
    });
    editorEl.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const s = editorEl.selectionStart;
        const en = editorEl.selectionEnd;
        editorEl.value = editorEl.value.slice(0, s) + '  ' + editorEl.value.slice(en);
        editorEl.selectionStart = editorEl.selectionEnd = s + 2;
        record.txtContent = editorEl.value;
        syncCodeGutter();
      } else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        triggerAiAnalysis();
      }
    });
  }

  async function triggerAiAnalysis() {
    if (ctx.uiState.analyzingQId === activeQ.id) return;
    if (answerEl) record.candidateAnswer = answerEl.value;
    if (editorEl) record.txtContent = editorEl.value;
    persistSession();

    const combinedInput = [record.candidateAnswer, codeBoardOpen ? record.txtContent : ''].filter(Boolean).join('\n\n');
    ctx.uiState.analyzingQId = activeQ.id;
    ctx.rerender();

    try {
      const result = await aiAnalyzeBoard(
        settings.llm,
        activeQ,
        combinedInput,
        session.targetLevel,
        session.persona,
        record.rubricStates
      );
      record.aiAnalysis = result;
      record.rubricStates = { ...(result.rubricStates || {}) };
      record.updatedAt = Date.now();
      persistSession();

      if (result.source === 'local-fallback') {
        showToast(`Live LLM 调用失败（${result.fallbackReason}），已自动降级为本地专家引擎`);
      } else if (result.source.startsWith('llm:')) {
        showToast(`AI 分析完成（${result.source.replace('llm:', '')}），已更新考点与追问`);
      } else {
        showToast('本地专家引擎分析完成，已更新考点打分与追问建议');
      }
    } finally {
      ctx.uiState.analyzingQId = null;
      ctx.rerender();
    }
  }

  // Top candidate strip controls
  container.querySelector('#iv-cand-name')?.addEventListener('change', (e) => {
    session.candidateName = e.target.value.trim() || '候选人';
    persistSession();
  });
  container.querySelector('#iv-target-level')?.addEventListener('change', (e) => {
    session.targetLevel = e.target.value;
    persistSession();
  });
  container.querySelector('#iv-persona')?.addEventListener('change', (e) => {
    session.persona = e.target.value;
    persistSession();
  });
  container.querySelector('#iv-assemble-btn')?.addEventListener('click', () => openQuickAssembleModal());
  container.querySelector('#iv-new-session-btn')?.addEventListener('click', () => {
    if (confirm('要归档当前面试并开启一场新面试吗？')) {
      archiveCurrentSession(false);
      resetNewSession();
    }
  });
  container.querySelector('#iv-go-dossier-btn')?.addEventListener('click', () => navigate('dossier'));
  container.querySelector('#iv-add-from-bank')?.addEventListener('click', () => navigate('bank'));

  // Queue tabs
  container.querySelectorAll('.q-tab').forEach((el) => {
    el.addEventListener('click', (e) => {
      const removeId = e.target?.dataset?.removeQ;
      if (removeId) {
        e.stopPropagation();
        session.questionIds = session.questionIds.filter((id) => id !== removeId);
        if (session.activeIndex >= session.questionIds.length) {
          session.activeIndex = Math.max(0, session.questionIds.length - 1);
        }
        persistSession();
        ctx.rerender();
        return;
      }
      session.activeIndex = Number(el.dataset.qIdx);
      persistSession();
      ctx.rerender();
    });
  });

  // Toggle Code Whiteboard (coding / formula scenario)
  container.querySelector('#iv-toggle-code-btn')?.addEventListener('click', () => {
    record.showCodeBoard = !codeBoardOpen;
    persistSession();
    ctx.rerender();
  });

  container.querySelector('#iv-copy-prompt-btn')?.addEventListener('click', () => {
    const promptTxt = `[${activeQ.title}]\n${activeQ.prompt}\n参考条件：${(activeQ.contextSpecs || []).join(' | ')}`;
    copyText(promptTxt, '题干已复制');
  });

  // Optional Code board toolbar buttons
  container.querySelector('#iv-insert-prompt-btn')?.addEventListener('click', () => {
    if (!editorEl) return;
    const block = `=== [题目] ${activeQ.title} (${activeQ.level}) ===\n${activeQ.prompt}\n====================================\n[候选人作答区]\n`;
    editorEl.value = (editorEl.value ? editorEl.value + '\n\n' : '') + block;
    record.txtContent = editorEl.value;
    updateLineNumbers(gutterEl, editorEl.value);
    persistSession();
  });
  container.querySelector('#iv-copy-board-btn')?.addEventListener('click', () => {
    copyText(editorEl?.value || '', '代码白板内容已复制');
  });
  container.querySelector('#iv-clear-board-btn')?.addEventListener('click', () => {
    if (!editorEl) return;
    editorEl.value = '';
    record.txtContent = '';
    updateLineNumbers(gutterEl, '');
    persistSession();
  });

  // Candidate answer paste actions
  container.querySelector('#iv-clear-answer-btn')?.addEventListener('click', () => {
    if (answerEl) {
      answerEl.value = '';
      record.candidateAnswer = '';
      syncAnswerStats();
      persistSession();
    }
  });
  container.querySelector('#iv-ai-analyze-btn')?.addEventListener('click', triggerAiAnalysis);

  // Rubric manual clicks
  container.querySelectorAll('[data-rubric-set]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const [rId, nextState] = btn.dataset.rubricSet.split('|');
      record.rubricStates[rId] = record.rubricStates[rId] === nextState ? 'unseen' : nextState;
      // Clear stale AI follow-ups if user manually toggles so dynamic ordering reflects latest gaps
      if (record.aiAnalysis?.source === 'local') {
        record.aiAnalysis.followUps = buildFollowUpsForStates(activeQ, record.rubricStates, session.targetLevel);
      }
      persistSession();
      ctx.rerender();
    });
  });

  container.querySelector('#iv-reset-rubric-btn')?.addEventListener('click', () => {
    record.rubricStates = {};
    record.aiAnalysis = null;
    persistSession();
    ctx.rerender();
  });

  // Insert follow-up into code board if open, or copy
  container.querySelectorAll('[data-insert-text]').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (!editorEl) return;
      const text = decodeURIComponent(btn.dataset.insertText);
      editorEl.value = (editorEl.value ? editorEl.value + '\n\n' : '') + `// [追问] ${text}\n`;
      record.txtContent = editorEl.value;
      updateLineNumbers(gutterEl, editorEl.value);
      persistSession();
      showToast('已写入代码白板');
    });
  });

  container.querySelectorAll('[data-copy-text]').forEach((btn) => {
    btn.addEventListener('click', () => {
      copyText(decodeURIComponent(btn.dataset.copyText), '已复制到剪贴板');
    });
  });

  // Right sidebar tabs
  container.querySelectorAll('.copilot-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      ctx.uiState.copilotTab = btn.dataset.cpTab;
      ctx.rerender();
    });
  });

  // Resume & Private Notes tab events
  const resumeInput = container.querySelector('#iv-resume-input');
  resumeInput?.addEventListener('input', () => {
    session.resumeText = resumeInput.value;
    persistSession();
  });

  container.querySelector('#iv-parse-resume-btn')?.addEventListener('click', async () => {
    const txt = (resumeInput?.value || session.resumeText || '').trim();
    if (!txt) {
      showToast('请先粘贴候选人简历或项目经历片段');
      return;
    }
    session.resumeText = txt;
    ctx.uiState.parsingResume = true;
    ctx.rerender();
    try {
      const res = await aiParseResume(settings.llm, txt, session.targetLevel, allQuestions);
      session.resumeAnalysis = res;
      persistSession();
      showToast(res.source.startsWith('llm:') ? 'AI 已生成简历定制 5-Why 追问清单' : '本地专家引擎已完成技术栈提取与追问生成');
    } finally {
      ctx.uiState.parsingResume = false;
      ctx.rerender();
    }
  });

  container.querySelectorAll('[data-add-rec-q]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const qId = btn.dataset.addRecQ;
      if (!session.questionIds.includes(qId)) {
        session.questionIds.push(qId);
        persistSession();
        showToast('已加入当前面试题单');
        ctx.rerender();
      } else {
        showToast('该题已在当前面试题单中');
      }
    });
  });

  const notesEl = container.querySelector('#iv-private-notes');
  notesEl?.addEventListener('input', () => {
    record.notes = notesEl.value;
    persistSession();
  });
}

function renderReferenceAndHintsTab(q) {
  return `
    <div class="ref-box" style="margin-top:0;">
      <div class="ref-box-title">
        <span>Staff 级参考答案与定量公式速查</span>
        <span class="badge badge-inference">仅面试官可见</span>
      </div>
      <div class="msg-content">${renderMarkdown(q.referenceAnswer || '暂无参考答案')}</div>
    </div>

    <details open style="border:1px solid var(--border-subtle);border-radius:var(--radius-sm);padding:10px 12px;background:var(--bg-elevated);">
      <summary style="cursor:pointer;font-size:12.5px;font-weight:700;color:var(--text-primary);">渐进提示卡（候选人卡壳时口头抛出 L1→L3）</summary>
      <div style="display:flex;flex-direction:column;gap:8px;margin-top:10px;">
        ${(q.hints || [])
          .map(
            (h) => `
          <div style="padding:8px 10px;border-radius:6px;background:var(--bg-surface);border:1px solid var(--border-subtle);font-size:12px;">
            <div style="margin-bottom:6px;color:var(--text-primary);">${escapeHtml(h)}</div>
            <button class="btn btn-ghost btn-sm" style="padding:2px 8px;font-size:11px;" data-copy-text="${encodeURIComponent(h)}">复制话术</button>
          </div>
        `
          )
          .join('')}
      </div>
    </details>
  `;
}

function renderResumeNotesTab(session, record, allQuestions, isParsingResume) {
  const analysis = session.resumeAnalysis;
  const qMap = new Map(allQuestions.map((q) => [q.id, q]));

  return `
    <div class="field">
      <label>
        <span>本题面试官私密备注（自动汇总入面评）</span>
      </label>
      <textarea
        id="iv-private-notes"
        rows="4"
        placeholder="记录候选人表达清晰度、是否真实做过项目、被追问时的反应速度、Green/Red Flags…"
      >${escapeHtml(record.notes || '')}</textarea>
    </div>

    <div class="field">
      <label>
        <span>候选人简历 / 项目经历片段</span>
        <span style="font-weight:400;color:var(--text-muted);">一键生成 5-Why 追问</span>
      </label>
      <textarea
        id="iv-resume-input"
        rows="5"
        placeholder="粘贴候选人项目描述，例如：负责千卡 H800 Megatron-LM 训 72B/MoE；落地 vLLM/SGLang PD分离；基于 veRL 搭建 GRPO 长 CoT 链路…"
      >${escapeHtml(session.resumeText || '')}</textarea>
      <div style="display:flex;justify-content:flex-end;margin-top:4px;">
        <button class="btn btn-primary btn-sm" id="iv-parse-resume-btn" ${isParsingResume ? 'disabled' : ''}>
          ${isParsingResume ? '<span class="spinner"></span> AI 解析简历中…' : '⚡ AI 提取项目 5-Why 追问'}
        </button>
      </div>
    </div>

    ${
      analysis
        ? `
      <div class="callout callout-success">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">
          <strong>简历画像 &amp; 拷打策略</strong>
          <span class="badge badge-mono">${analysis.source.startsWith('llm:') ? escapeHtml(analysis.source) : '本地专家提取'}</span>
        </div>
        <div style="font-size:12.5px;">${escapeHtml(analysis.summary)}</div>
        ${
          analysis.detectedStacks?.length
            ? `<div class="chip-row" style="margin-top:8px;">${analysis.detectedStacks
                .map((s) => `<span class="badge ${TRACKS[s.track]?.badgeClass || ''}">${escapeHtml(s.name)}</span>`)
                .join('')}</div>`
            : ''
        }
      </div>

      <div style="display:flex;flex-direction:column;gap:8px;">
        <strong style="font-size:13px;">简历定向 5-Why 追问建议</strong>
        ${(analysis.customProbes || [])
          .map(
            (p) => `
          <div class="followup-card">
            <div class="followup-card-head">
              <span class="badge ${TRACKS[p.track]?.badgeClass || 'badge-training'}">${escapeHtml(p.tag)}</span>
              <button class="btn btn-sm" data-copy-text="${encodeURIComponent(p.question)}">复制</button>
            </div>
            <div class="followup-text">${escapeHtml(p.question)}</div>
          </div>
        `
          )
          .join('')}
      </div>

      ${
        analysis.recommendedQuestionIds?.length
          ? `
        <div>
          <strong style="font-size:13px;display:block;margin-bottom:6px;">推荐加测的关联真题</strong>
          <div style="display:flex;flex-direction:column;gap:6px;">
            ${analysis.recommendedQuestionIds
              .map((id) => qMap.get(id))
              .filter(Boolean)
              .map(
                (rq) => `
              <div class="rubric-check-row">
                <div style="font-size:12px;">
                  <span class="badge ${TRACKS[rq.track]?.badgeClass || ''}" style="margin-right:5px;">${rq.level}</span>
                  ${escapeHtml(rq.title)}
                </div>
                <button class="btn btn-sm" data-add-rec-q="${rq.id}">+ 加入题单</button>
              </div>
            `
              )
              .join('')}
          </div>
        </div>
      `
          : ''
      }
    `
        : ''
    }
  `;
}
