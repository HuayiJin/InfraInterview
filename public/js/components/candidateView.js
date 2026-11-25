import { TRACKS, INTERVIEW_MODES, TARGET_LEVELS, PERSONAS, FORMULA_SNIPPETS } from '../data/index.js';
import { getVideosForQuestion } from '../data/videos.js';
import { aiCandidateNextTurn } from '../engine/llmClient.js';
import { escapeHtml, renderMarkdown } from '../engine/markdown.js';
import { renderVideoStrip } from './videoStrip.js';

let activeRecognition = null;

function stopVoiceRecognition(mockState) {
  if (activeRecognition) {
    try {
      activeRecognition.onresult = null;
      activeRecognition.onend = null;
      activeRecognition.onerror = null;
      activeRecognition.stop();
    } catch {
      // ignore
    }
    activeRecognition = null;
  }
  if (mockState) mockState.recording = false;
}

function selectMockQuestions(allQuestions, config) {
  const { mode, track, level } = config;
  const sortByLevelFit = (list) =>
    [...list].sort((a, b) => {
      const aFit = a.level === level ? 2 : 1;
      const bFit = b.level === level ? 2 : 1;
      if (bFit !== aFit) return bFit - aFit;
      return Math.random() - 0.5;
    });

  if (mode === 'incident_debug') {
    return allQuestions.filter((q) => q.incident).slice(0, 3);
  }
  if (mode === 'track_deep') {
    const pool = track && track !== 'all' ? allQuestions.filter((q) => q.track === track) : allQuestions;
    return sortByLevelFit(pool).slice(0, 3);
  }
  if (mode === 'stress_drill') {
    const hard = allQuestions.filter((q) => q.level === 'P8' || q.level === 'P7');
    return sortByLevelFit(hard).slice(0, 2);
  }
  const picks = [];
  for (const tid of ['training', 'inference', 'rl', 'kernel']) {
    const pool = sortByLevelFit(allQuestions.filter((q) => q.track === tid));
    if (pool.length) picks.push(pool[0]);
  }
  return picks;
}

export function startCandidateMockWithQuestions(mockState, questions, config) {
  stopVoiceRecognition(mockState);
  const persona = PERSONAS[config.persona] || PERSONAS.strict;
  const firstQ = questions[0];
  mockState.active = true;
  mockState.config = { ...config };
  mockState.questions = questions;
  mockState.currentIndex = 0;
  mockState.results = [];
  mockState.loading = false;
  mockState.hintIndex = 0;
  mockState.draftTxt = '';
  mockState.codeScratch = '';
  mockState.showCodePad = Boolean(firstQ?.isCoding);
  mockState.recording = false;
  mockState.finishedCurrent = false;
  mockState.currentAnalysis = null;
  mockState.turns = [
    {
      role: 'interviewer',
      content: `${persona.openingStyle}\n\n**第 1/${questions.length} 题 · [${TRACKS[firstQ.track]?.name} · ${firstQ.level}] ${firstQ.title}**\n\n${firstQ.prompt}`,
      specs: firstQ.contextSpecs || [],
    },
  ];
}

export function renderCandidateView(container, ctx) {
  const { mockState, settings, showToast, saveCandidateHistoryRun, switchToInterviewer } = ctx;

  if (!mockState.active) {
    stopVoiceRecognition(mockState);
    renderCandidateLobby(container, ctx);
    return;
  }

  const q = mockState.questions[mockState.currentIndex];
  const trackMeta = TRACKS[q.track] || TRACKS.training;
  const persona = PERSONAS[mockState.config.persona] || PERSONAS.strict;
  const showCodePad = Boolean(mockState.showCodePad || q.isCoding);

  container.innerHTML = `
    <div class="candidate-chat-shell">
      <!-- Header -->
      <div class="chat-header">
        <div class="chat-q-meta">
          <span class="badge badge-inference">候选人模拟</span>
          <span class="badge ${trackMeta.badgeClass}">${trackMeta.name}</span>
          <span class="badge badge-mono">${q.level}</span>
          <span class="badge badge-mono">Q${mockState.currentIndex + 1}/${mockState.questions.length}</span>
          <span style="font-size:12px;color:var(--text-muted);">${settings.llm.enabled ? escapeHtml(settings.llm.model) : '本地专家引擎'}</span>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button class="btn btn-sm ${showCodePad ? 'btn-primary' : 'btn-ghost'}" id="cm-toggle-code-btn">
            ${showCodePad ? '收起代码白板' : '📝 代码/公式白板'}
          </button>
          <button class="btn btn-ghost btn-sm" id="cm-exit-btn">退出模拟</button>
          <button class="btn btn-sm" id="cm-back-iv-btn">切回面试官</button>
        </div>
      </div>

      <!-- Main Dialogue Stream -->
      <div class="chat-messages" id="cm-messages">
        ${mockState.turns
          .map((t) => {
            const isIv = t.role === 'interviewer';
            return `
              <div class="msg ${isIv ? 'interviewer' : 'candidate'}">
                <div class="msg-avatar">${isIv ? persona.avatar : 'ME'}</div>
                <div class="msg-bubble">
                  <div class="msg-role-row">
                    <span class="msg-role-name">${isIv ? `AI 面试官 (${persona.name})` : '你的回答'}</span>
                    ${t.source ? `<span class="badge badge-mono" style="padding:0 5px;font-size:10px;">${escapeHtml(t.source)}</span>` : ''}
                  </div>
                  <div class="msg-content">${renderMarkdown(t.content)}</div>
                  ${
                    t.specs?.length
                      ? `<div class="context-specs">${t.specs.map((s) => `<span class="badge badge-mono">${escapeHtml(s)}</span>`).join('')}</div>`
                      : ''
                  }
                </div>
              </div>
            `;
          })
          .join('')}

        ${
          mockState.loading
            ? `<div class="msg interviewer"><div class="msg-avatar">${persona.avatar}</div><div class="msg-bubble"><span class="loading-shimmer"><span class="spinner"></span> 面试官正在思考下一句追问…</span></div></div>`
            : ''
        }

        ${
          mockState.finishedCurrent && mockState.currentAnalysis
            ? renderCandidateQuestionSummary(q, mockState.currentAnalysis, mockState.currentIndex < mockState.questions.length - 1)
            : ''
        }
      </div>

      <!-- Optional Code / Formula Whiteboard (Visible ONLY in coding scenarios or when toggled) -->
      ${
        showCodePad
          ? `
        <div style="border-top:1px solid var(--track-kernel-border);background:var(--bg-input);padding:10px 14px;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
            <span class="badge badge-kernel badge-mono">CODE / FORMULA 白板（提交回答时将自动附带）</span>
            <button class="btn btn-ghost btn-sm" id="cm-clear-code-btn">清空白板</button>
          </div>
          <textarea
            id="cm-code-scratch"
            class="composer-textarea mono"
            style="min-height:110px;font-size:12.5px;"
            spellcheck="false"
            ${mockState.finishedCurrent || mockState.loading ? 'disabled' : ''}
            placeholder="// 代码题或公式推导时在此编写伪代码 / CUDA / Triton / 定量推导…"
          >${escapeHtml(mockState.codeScratch || '')}</textarea>
        </div>
      `
          : ''
      }

      <!-- Bottom Conversational Composer (Input Box + Voice Button) -->
      <div class="composer">
        <div class="snippet-bar">
          <span class="snippet-label">常用公式速插：</span>
          ${FORMULA_SNIPPETS.map(
            (s, idx) => `<button class="snippet-btn" data-snip-idx="${idx}">${escapeHtml(s.label)}</button>`
          ).join('')}
        </div>

        <textarea
          class="composer-textarea"
          id="cm-txt-editor"
          ${mockState.finishedCurrent || mockState.loading ? 'disabled' : ''}
          placeholder="像真实口头面试一样输入你的回答思路、定量估算与架构权衡…（支持点击下方「🎤 语音口述」直接开口讲，按 Cmd/Ctrl + Enter 发送）"
        >${escapeHtml(mockState.draftTxt || '')}</textarea>

        <div class="composer-actions">
          <div class="composer-left-tools">
            <button
              class="btn btn-sm voice-btn ${mockState.recording ? 'recording' : ''}"
              id="cm-voice-btn"
              ${mockState.finishedCurrent || mockState.loading ? 'disabled' : ''}
              title="使用浏览器语音识别模拟口头作答（中文/中英混合）"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
              <span>${mockState.recording ? '停止语音口述…' : '语音口述'}</span>
            </button>
            <button class="btn btn-ghost btn-sm" id="cm-hint-btn" ${
              mockState.finishedCurrent || mockState.hintIndex >= (q.hints?.length || 0) ? 'disabled' : ''
            }>
              提示 (${mockState.hintIndex}/${q.hints?.length || 0})
            </button>
            <button class="btn btn-ghost btn-sm" id="cm-finish-q-btn" ${mockState.finishedCurrent ? 'disabled' : ''}>
              提前看解析
            </button>
          </div>
          <div style="display:flex;gap:8px;align-items:center;">
            ${
              mockState.finishedCurrent
                ? `<button class="btn btn-primary btn-sm" id="cm-next-q-btn">${
                    mockState.currentIndex < mockState.questions.length - 1 ? '进入下一题 →' : '完成模拟并查看复盘'
                  }</button>`
                : `<button class="btn btn-primary btn-sm" id="cm-submit-btn" ${mockState.loading ? 'disabled' : ''}>发送回答 (⌘↵)</button>`
            }
          </div>
        </div>
      </div>
    </div>
  `;

  // Scroll messages to bottom
  const msgBox = container.querySelector('#cm-messages');
  if (msgBox) msgBox.scrollTop = msgBox.scrollHeight;

  const editorEl = container.querySelector('#cm-txt-editor');
  const codeEl = container.querySelector('#cm-code-scratch');

  editorEl?.addEventListener('input', () => {
    mockState.draftTxt = editorEl.value;
  });
  editorEl?.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !mockState.finishedCurrent && !mockState.loading) {
      e.preventDefault();
      submitTurn();
    }
  });

  codeEl?.addEventListener('input', () => {
    mockState.codeScratch = codeEl.value;
  });
  codeEl?.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const s = codeEl.selectionStart;
      const en = codeEl.selectionEnd;
      codeEl.value = codeEl.value.slice(0, s) + '  ' + codeEl.value.slice(en);
      codeEl.selectionStart = codeEl.selectionEnd = s + 2;
      mockState.codeScratch = codeEl.value;
    }
  });

  container.querySelector('#cm-toggle-code-btn')?.addEventListener('click', () => {
    mockState.showCodePad = !showCodePad;
    ctx.rerender();
  });
  container.querySelector('#cm-clear-code-btn')?.addEventListener('click', () => {
    mockState.codeScratch = '';
    if (codeEl) codeEl.value = '';
  });

  // Voice Input (Web Speech API)
  container.querySelector('#cm-voice-btn')?.addEventListener('click', () => {
    if (mockState.recording) {
      stopVoiceRecognition(mockState);
      showToast('已停止语音输入');
      ctx.rerender();
      return;
    }
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      showToast('当前浏览器不支持 Web Speech 语音识别（建议使用 Chrome / Edge）');
      return;
    }
    try {
      const rec = new SpeechRec();
      rec.lang = 'zh-CN';
      rec.continuous = true;
      rec.interimResults = true;
      const baseText = (editorEl?.value || '').trim();
      let finalCommitted = '';

      rec.onresult = (event) => {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalCommitted += transcript;
          } else {
            interim += transcript;
          }
        }
        const combined = [baseText, (finalCommitted + interim).trim()].filter(Boolean).join('\n');
        mockState.draftTxt = combined;
        if (editorEl) editorEl.value = combined;
      };
      rec.onerror = (ev) => {
        stopVoiceRecognition(mockState);
        showToast(`语音识别中断：${ev.error || '请检查麦克风权限'}`);
        ctx.rerender();
      };
      rec.onend = () => {
        if (mockState.recording) {
          mockState.recording = false;
          activeRecognition = null;
          ctx.rerender();
        }
      };
      activeRecognition = rec;
      mockState.recording = true;
      rec.start();
      showToast('开始语音口述，直接开口讲即可实时转文字…');
      ctx.rerender();
    } catch (err) {
      stopVoiceRecognition(mockState);
      showToast(`无法启动语音输入：${err.message}`);
    }
  });

  // Snippet buttons
  container.querySelectorAll('[data-snip-idx]').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (mockState.finishedCurrent || !editorEl) return;
      const snip = FORMULA_SNIPPETS[Number(btn.dataset.snipIdx)];
      if (!snip) return;
      const cur = editorEl.value;
      editorEl.value = cur + (cur && !cur.endsWith('\n') ? '\n' : '') + snip.text + '\n';
      mockState.draftTxt = editorEl.value;
      editorEl.focus();
    });
  });

  async function submitTurn() {
    stopVoiceRecognition(mockState);
    const spoken = (editorEl?.value || '').trim();
    const code = showCodePad ? (codeEl?.value || '').trim() : '';
    const answer = [spoken, code ? `\`\`\`\n${code}\n\`\`\`` : ''].filter(Boolean).join('\n\n');
    if (answer.replace(/`/g, '').trim().length < 8) {
      showToast('请先输入或口述你的核心思路再发送');
      return;
    }
    mockState.turns.push({ role: 'candidate', content: answer });
    mockState.draftTxt = '';
    mockState.codeScratch = '';
    mockState.loading = true;
    ctx.rerender();

    try {
      const res = await aiCandidateNextTurn(
        settings.llm,
        q,
        mockState.turns,
        mockState.config.persona,
        mockState.config.level
      );
      mockState.turns.push({
        role: 'interviewer',
        content: res.reply,
        source: res.source === 'local' ? '本地专家引擎' : res.source,
      });
      mockState.currentAnalysis = res.analysis;
      if (res.done) {
        mockState.finishedCurrent = true;
        recordMockResult();
      }
    } finally {
      mockState.loading = false;
      ctx.rerender();
    }
  }

  function recordMockResult() {
    const exists = mockState.results.find((r) => r.questionId === q.id);
    const penalty = mockState.hintIndex * 4;
    const finalScore = Math.max(0, (mockState.currentAnalysis?.score || 0) - penalty);
    const entry = {
      questionId: q.id,
      title: q.title,
      track: q.track,
      level: q.level,
      score: finalScore,
      hintsUsed: mockState.hintIndex,
      analysis: mockState.currentAnalysis,
    };
    if (exists) Object.assign(exists, entry);
    else mockState.results.push(entry);
  }

  container.querySelector('#cm-submit-btn')?.addEventListener('click', submitTurn);

  container.querySelector('#cm-hint-btn')?.addEventListener('click', () => {
    const hint = q.hints?.[mockState.hintIndex];
    if (!hint) return;
    mockState.hintIndex += 1;
    mockState.turns.push({ role: 'interviewer', content: `**[提示 L${mockState.hintIndex}]** ${hint}` });
    ctx.rerender();
  });

  container.querySelector('#cm-finish-q-btn')?.addEventListener('click', async () => {
    stopVoiceRecognition(mockState);
    if (!mockState.currentAnalysis) {
      const res = await aiCandidateNextTurn(
        { enabled: false },
        q,
        mockState.turns,
        mockState.config.persona,
        mockState.config.level
      );
      mockState.currentAnalysis = res.analysis;
    }
    mockState.finishedCurrent = true;
    recordMockResult();
    ctx.rerender();
  });

  const handleNext = () => {
    stopVoiceRecognition(mockState);
    if (mockState.currentIndex < mockState.questions.length - 1) {
      mockState.currentIndex += 1;
      const nextQ = mockState.questions[mockState.currentIndex];
      mockState.hintIndex = 0;
      mockState.draftTxt = '';
      mockState.codeScratch = '';
      mockState.showCodePad = Boolean(nextQ?.isCoding);
      mockState.finishedCurrent = false;
      mockState.currentAnalysis = null;
      mockState.turns = [
        {
          role: 'interviewer',
          content: `我们进入下一道题。\n\n**第 ${mockState.currentIndex + 1}/${mockState.questions.length} 题 · [${TRACKS[nextQ.track]?.name} · ${nextQ.level}] ${nextQ.title}**\n\n${nextQ.prompt}`,
          specs: nextQ.contextSpecs || [],
        },
      ];
      ctx.rerender();
    } else {
      const avg =
        mockState.results.length > 0
          ? Math.round(mockState.results.reduce((s, r) => s + r.score, 0) / mockState.results.length)
          : 0;
      saveCandidateHistoryRun({
        id: `cm-${Date.now()}`,
        mode: mockState.config.mode,
        level: mockState.config.level,
        avgScore: avg,
        results: mockState.results,
      });
      mockState.active = false;
      showToast(`模拟面试已完成！综合得分 ${avg} 分，已存入复盘记录`);
      ctx.navigate('dossier');
    }
  };

  container.querySelector('#cm-next-q-btn')?.addEventListener('click', handleNext);
  container.querySelector('#cm-inline-next-btn')?.addEventListener('click', handleNext);

  container.querySelector('#cm-exit-btn')?.addEventListener('click', () => {
    if (confirm('确定退出当前模拟自测吗？')) {
      stopVoiceRecognition(mockState);
      mockState.active = false;
      ctx.rerender();
    }
  });
  container.querySelector('#cm-back-iv-btn')?.addEventListener('click', () => {
    stopVoiceRecognition(mockState);
    switchToInterviewer();
  });
}

function renderCandidateQuestionSummary(q, analysis, hasNext) {
  return `
    <div class="ref-box">
      <div class="ref-box-title">
        <span>本题揭晓 · 评估得分 ${analysis.score} / 100</span>
        <button class="btn btn-primary btn-sm" id="cm-inline-next-btn">${hasNext ? '下一题 →' : '查看总结复盘'}</button>
      </div>
      <div class="rubric-list" style="margin-bottom:12px;">
        ${(analysis.rubricDetails || [])
          .map(
            (r) => `
          <div class="rubric-item ${r.state === 'hit' ? 'hit' : r.state === 'miss' ? 'miss' : ''}">
            <span class="rubric-icon">${r.state === 'hit' ? '✓' : r.state === 'partial' ? '△' : '✕'}</span>
            <span>[${r.weight}%] ${escapeHtml(r.label)}</span>
          </div>
        `
          )
          .join('')}
      </div>
      <details open>
        <summary style="cursor:pointer;font-weight:700;font-size:13px;margin-bottom:8px;color:var(--track-inference);">展开 Staff 级参考架构解析</summary>
        <div class="msg-content">${renderMarkdown(q.referenceAnswer || '')}</div>
      </details>
      ${renderVideoStrip(getVideosForQuestion(q.id))}
    </div>
  `;
}

function renderCandidateLobby(container, ctx) {
  const { mockState, allQuestions, showToast, switchToInterviewer } = ctx;
  const cfg = mockState.config;

  container.innerHTML = `
    <div class="lobby-wrap">
      <div class="hero-banner">
        <div>
          <div class="hero-kicker">CANDIDATE MOCK · SECONDARY VIEW</div>
          <h2 class="hero-title">候选人模拟自测（对话 + 语音口述模式）</h2>
          <p class="hero-desc">严格隐藏参考答案与 Rubric，以自然一问一答方式接受 AI 面试官多轮追问拷打；支持打字与语音口述，遇到代码/公式题可随时展开代码白板。</p>
        </div>
        <div>
          <button class="btn btn-ghost btn-sm" id="cm-lobby-back-iv">← 返回面试官工作台</button>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">1. 选择模拟模式</h3>
            <div class="card-sub">全栈覆盖训练 / 推理 / RL / GPU 算子与通信</div>
          </div>
        </div>
        <div class="option-grid">
          ${INTERVIEW_MODES.map(
            (m) => `
            <button class="option-card ${cfg.mode === m.id ? 'selected' : ''}" data-cm-mode="${m.id}">
              <div class="option-card-head">
                <span class="option-card-title">${m.title}</span>
                <span class="badge">${m.badge}</span>
              </div>
              <div class="option-card-desc">${m.desc}</div>
            </button>
          `
          ).join('')}
        </div>

        ${
          cfg.mode === 'track_deep'
            ? `
          <div style="margin-top:14px;">
            <div style="font-size:12.5px;font-weight:600;margin-bottom:8px;color:var(--text-secondary);">选择专项方向：</div>
            <div class="chip-row">
              ${Object.values(TRACKS)
                .map(
                  (t) => `<button class="chip ${cfg.track === t.id ? 'selected' : ''}" data-cm-track="${t.id}">${t.name}</button>`
                )
                .join('')}
            </div>
          </div>
        `
            : ''
        }
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-header"><h3 class="card-title">2. 目标职级</h3></div>
          <div style="display:flex;flex-direction:column;gap:8px;">
            ${TARGET_LEVELS.map(
              (lv) => `
              <button class="option-card ${cfg.level === lv.id ? 'selected' : ''}" data-cm-level="${lv.id}">
                <div class="option-card-title">${lv.label}</div>
                <div class="option-card-desc">${lv.desc}</div>
              </button>
            `
            ).join('')}
          </div>
        </div>

        <div class="card">
          <div class="card-header"><h3 class="card-title">3. AI 面试官风格</h3></div>
          <div style="display:flex;flex-direction:column;gap:8px;">
            ${Object.values(PERSONAS)
              .map(
                (p) => `
              <button class="option-card ${cfg.persona === p.id ? 'selected' : ''}" data-cm-persona="${p.id}">
                <div class="option-card-head">
                  <span class="option-card-title">${p.name}</span>
                  <span class="badge">${p.tag}</span>
                </div>
                <div class="option-card-desc">${p.summaryTone}</div>
              </button>
            `
              )
              .join('')}
          </div>
        </div>
      </div>

      <div style="display:flex;justify-content:flex-end;">
        <button class="btn btn-primary btn-lg" id="cm-start-btn">开始模拟面试 →</button>
      </div>
    </div>
  `;

  container.querySelector('#cm-lobby-back-iv')?.addEventListener('click', () => switchToInterviewer());
  container.querySelectorAll('[data-cm-mode]').forEach((b) =>
    b.addEventListener('click', () => {
      cfg.mode = b.dataset.cmMode;
      ctx.rerender();
    })
  );
  container.querySelectorAll('[data-cm-track]').forEach((b) =>
    b.addEventListener('click', () => {
      cfg.track = b.dataset.cmTrack;
      ctx.rerender();
    })
  );
  container.querySelectorAll('[data-cm-level]').forEach((b) =>
    b.addEventListener('click', () => {
      cfg.level = b.dataset.cmLevel;
      ctx.rerender();
    })
  );
  container.querySelectorAll('[data-cm-persona]').forEach((b) =>
    b.addEventListener('click', () => {
      cfg.persona = b.dataset.cmPersona;
      ctx.rerender();
    })
  );
  container.querySelector('#cm-start-btn')?.addEventListener('click', () => {
    const picks = selectMockQuestions(allQuestions, cfg);
    if (!picks.length) {
      showToast('未匹配到题目');
      return;
    }
    startCandidateMockWithQuestions(mockState, picks, cfg);
    ctx.rerender();
  });
}
