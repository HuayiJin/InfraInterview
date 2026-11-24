import { TRACKS } from '../data/index.js';
import { escapeHtml, renderMarkdown } from '../engine/markdown.js';

export function renderBankView(container, ctx) {
  const {
    allQuestions,
    role,
    session,
    persistSession,
    showToast,
    copyText,
    navigate,
    openModal,
    closeModal,
    startSingleCandidateMock,
    saveNewCustomQuestion,
    removeCustomQuestion,
  } = ctx;

  const filter = ctx.uiState.bankFilter;
  const qTerm = (filter.search || '').trim().toLowerCase();

  const filtered = allQuestions.filter((q) => {
    if (filter.track !== 'all' && q.track !== filter.track) return false;
    if (filter.level !== 'all' && q.level !== filter.level) return false;
    if (filter.special === 'incident' && !q.incident) return false;
    if (filter.special === 'must' && q.frequency !== '高频必考') return false;
    if (filter.special === 'custom' && !q.custom) return false;
    if (qTerm) {
      const hay = `${q.title} ${q.summary} ${(q.tags || []).join(' ')} ${q.prompt}`.toLowerCase();
      if (!hay.includes(qTerm)) return false;
    }
    return true;
  });

  container.innerHTML = `
    <div style="max-width:1360px;margin:0 auto;">
      <div class="bank-toolbar">
        <input type="text" class="search-input" id="bank-search" value="${escapeHtml(filter.search || '')}" placeholder="搜索题目、框架或考点（如 GRPO、MLA、DualPipe、Roofline、CuMem、NCCL…）" />
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <button class="btn btn-primary btn-sm" id="bank-create-q-btn">+ 新建自定义题</button>
          ${
            role === 'interviewer'
              ? `<button class="btn btn-sm" id="bank-back-cockpit-btn">返回工作台（已选 ${session.questionIds.length} 题）</button>`
              : ''
          }
        </div>
      </div>

      <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:16px;">
        <div class="chip-row">
          <button class="chip ${filter.track === 'all' ? 'selected' : ''}" data-bf-track="all">全部领域 (${allQuestions.length})</button>
          ${Object.values(TRACKS)
            .map((t) => {
              const cnt = allQuestions.filter((q) => q.track === t.id).length;
              return `<button class="chip ${filter.track === t.id ? 'selected' : ''}" data-bf-track="${t.id}">${t.name} (${cnt})</button>`;
            })
            .join('')}
        </div>
        <div class="chip-row">
          ${['all', 'P6', 'P7', 'P8']
            .map(
              (lv) =>
                `<button class="chip ${filter.level === lv ? 'selected' : ''}" data-bf-level="${lv}">${lv === 'all' ? '全部职级' : lv}</button>`
            )
            .join('')}
        </div>
        <div class="chip-row">
          <button class="chip ${filter.special === 'all' ? 'selected' : ''}" data-bf-spec="all">全部类型</button>
          <button class="chip ${filter.special === 'must' ? 'selected' : ''}" data-bf-spec="must">高频必考</button>
          <button class="chip ${filter.special === 'incident' ? 'selected' : ''}" data-bf-spec="incident">线上排障</button>
          <button class="chip ${filter.special === 'custom' ? 'selected' : ''}" data-bf-spec="custom">我的自定义题</button>
        </div>
      </div>

      ${
        filtered.length === 0
          ? `<div class="card empty-state">没有匹配的面试题，可以调整筛选条件或点击右上角「新建自定义题」。</div>`
          : `<div class="grid-3">
            ${filtered
              .map((q) => {
                const tr = TRACKS[q.track] || TRACKS.training;
                const inQueue = session.questionIds.includes(q.id);
                return `
                  <article class="q-card">
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap;">
                      <div style="display:flex;gap:5px;flex-wrap:wrap;">
                        <span class="badge ${tr.badgeClass}">${tr.name}</span>
                        <span class="badge badge-mono">${q.level}</span>
                        ${q.incident ? '<span class="badge badge-danger">线上排障</span>' : ''}
                        ${q.custom ? '<span class="badge">自定义</span>' : ''}
                      </div>
                      <span style="font-size:11px;color:var(--text-muted);">${escapeHtml(q.frequency || '')}</span>
                    </div>
                    <h3 class="q-card-title">${escapeHtml(q.title)}</h3>
                    <p class="q-card-summary">${escapeHtml(q.summary)}</p>
                    <div class="chip-row">
                      ${(q.tags || []).slice(0, 4).map((tag) => `<span class="badge badge-mono" style="font-weight:500;">${escapeHtml(tag)}</span>`).join('')}
                    </div>
                    <div class="q-card-footer">
                      <div style="display:flex;gap:6px;">
                        <button class="btn btn-ghost btn-sm" data-q-preview="${q.id}">解析与Rubric</button>
                        <button class="btn btn-ghost btn-sm" data-q-copy="${q.id}">复制题干</button>
                        ${q.custom ? `<button class="btn btn-danger btn-sm" data-q-del="${q.id}">删</button>` : ''}
                      </div>
                      ${
                        role === 'interviewer'
                          ? inQueue
                            ? `<button class="btn btn-sm" data-q-open-iv="${q.id}">已在题单 · 去提问</button>`
                            : `<button class="btn btn-primary btn-sm" data-q-add-iv="${q.id}">+ 加入题单</button>`
                          : `<button class="btn btn-primary btn-sm" data-q-mock="${q.id}">单题自测</button>`
                      }
                    </div>
                  </article>
                `;
              })
              .join('')}
          </div>`
      }
    </div>
  `;

  const searchEl = container.querySelector('#bank-search');
  searchEl?.addEventListener('input', () => {
    filter.search = searchEl.value;
    ctx.rerender();
    // restore focus
    const next = container.querySelector('#bank-search');
    if (next) {
      next.focus();
      next.setSelectionRange(next.value.length, next.value.length);
    }
  });

  container.querySelector('#bank-back-cockpit-btn')?.addEventListener('click', () => navigate('workspace'));

  container.querySelectorAll('[data-bf-track]').forEach((b) =>
    b.addEventListener('click', () => {
      filter.track = b.dataset.bfTrack;
      ctx.rerender();
    })
  );
  container.querySelectorAll('[data-bf-level]').forEach((b) =>
    b.addEventListener('click', () => {
      filter.level = b.dataset.bfLevel;
      ctx.rerender();
    })
  );
  container.querySelectorAll('[data-bf-spec]').forEach((b) =>
    b.addEventListener('click', () => {
      filter.special = b.dataset.bfSpec;
      ctx.rerender();
    })
  );

  // Add to interviewer queue
  container.querySelectorAll('[data-q-add-iv]').forEach((b) =>
    b.addEventListener('click', () => {
      const id = b.dataset.qAddIv;
      if (!session.questionIds.includes(id)) {
        session.questionIds.push(id);
        persistSession();
        showToast('已加入当前面试题单');
        ctx.rerender();
      }
    })
  );

  container.querySelectorAll('[data-q-open-iv]').forEach((b) =>
    b.addEventListener('click', () => {
      const idx = session.questionIds.indexOf(b.dataset.qOpenIv);
      if (idx >= 0) {
        session.activeIndex = idx;
        persistSession();
      }
      navigate('workspace');
    })
  );

  container.querySelectorAll('[data-q-mock]').forEach((b) =>
    b.addEventListener('click', () => {
      const q = allQuestions.find((item) => item.id === b.dataset.qMock);
      if (q) startSingleCandidateMock(q);
    })
  );

  container.querySelectorAll('[data-q-copy]').forEach((b) =>
    b.addEventListener('click', () => {
      const q = allQuestions.find((item) => item.id === b.dataset.qCopy);
      if (q) copyText(`[${q.title}]\n${q.prompt}`, '题干已复制');
    })
  );

  container.querySelectorAll('[data-q-del]').forEach((b) =>
    b.addEventListener('click', () => {
      if (confirm('确定删除这道自定义题吗？')) {
        removeCustomQuestion(b.dataset.qDel);
      }
    })
  );

  container.querySelectorAll('[data-q-preview]').forEach((b) =>
    b.addEventListener('click', () => {
      const q = allQuestions.find((item) => item.id === b.dataset.qPreview);
      if (!q) return;
      const tr = TRACKS[q.track] || TRACKS.training;
      openModal(`
        <div class="card-header">
          <div>
            <div style="display:flex;gap:6px;margin-bottom:6px;">
              <span class="badge ${tr.badgeClass}">${tr.name}</span>
              <span class="badge badge-mono">${q.level}</span>
            </div>
            <h3 class="card-title">${escapeHtml(q.title)}</h3>
          </div>
          <button class="icon-btn" id="modal-close-x">✕</button>
        </div>
        <div class="callout" style="margin-bottom:12px;">${escapeHtml(q.prompt)}</div>
        <div style="margin-bottom:12px;">
          <strong style="font-size:13px;display:block;margin-bottom:6px;">考点 Rubric 权重</strong>
          <div class="rubric-list">
            ${q.rubric.map((r) => `<div class="rubric-item"><span class="mono" style="color:var(--accent);">[${r.weight}%]</span><span>${escapeHtml(r.label)}</span></div>`).join('')}
          </div>
        </div>
        <div class="ref-box">
          <div class="ref-box-title"><span>Staff 级参考答案</span></div>
          <div class="msg-content">${renderMarkdown(q.referenceAnswer || '')}</div>
        </div>
      `);
      document.querySelector('#modal-close-x')?.addEventListener('click', closeModal);
    })
  );

  // Create custom question modal
  container.querySelector('#bank-create-q-btn')?.addEventListener('click', () => {
    openModal(`
      <div class="card-header">
        <h3 class="card-title">新建自定义 AI Infra 面试题</h3>
        <button class="icon-btn" id="modal-close-x">✕</button>
      </div>
      <div style="display:flex;flex-direction:column;gap:12px;">
        <div class="field">
          <label>题目标题 *</label>
          <input type="text" id="cq-title" placeholder="例如：团队内部异构集群混合部署 32B 推理与 RL Rollout 调度设计" />
        </div>
        <div class="form-grid">
          <div class="field">
            <label>所属领域</label>
            <select id="cq-track">
              ${Object.values(TRACKS).map((t) => `<option value="${t.id}">${t.name}</option>`).join('')}
            </select>
          </div>
          <div class="field">
            <label>难度职级</label>
            <select id="cq-level">
              <option value="P6">P6 / Senior</option>
              <option value="P7" selected>P7 / Staff</option>
              <option value="P8">P8 / Principal</option>
            </select>
          </div>
        </div>
        <label style="display:inline-flex;align-items:center;gap:8px;font-size:12.5px;color:var(--text-secondary);cursor:pointer;">
          <input type="checkbox" id="cq-is-coding" />
          <span>标记为代码/手撕题场景（切到该题时默认展开代码白板）</span>
        </label>
        <div class="field">
          <label>完整题干 *（面试官直接发给候选人的内容）</label>
          <textarea id="cq-prompt" rows="3" placeholder="描述硬件背景、业务规模与核心问题…"></textarea>
        </div>
        <div class="field">
          <label>核心考点 Rubric（每行一条，AI 分析与勾选打分将基于此生成）</label>
          <textarea id="cq-rubric" rows="4" placeholder="显存与 KV Cache 容量定量估算&#10;并行度切分与通信 Overlap 策略&#10;极端长尾与故障隔离设计"></textarea>
        </div>
        <div class="field">
          <label>参考答案要点（Markdown，仅面试官可见）</label>
          <textarea id="cq-ref" rows="4" placeholder="核心公式、理想答案要点与常见坑…"></textarea>
        </div>
        <div style="display:flex;justify-content:flex-end;gap:8px;">
          <button class="btn btn-ghost" id="cq-cancel">取消</button>
          <button class="btn btn-primary" id="cq-save">保存并加入题库</button>
        </div>
      </div>
    `);
    document.querySelector('#modal-close-x')?.addEventListener('click', closeModal);
    document.querySelector('#cq-cancel')?.addEventListener('click', closeModal);
    document.querySelector('#cq-save')?.addEventListener('click', () => {
      const title = document.querySelector('#cq-title').value.trim();
      const prompt = document.querySelector('#cq-prompt').value.trim();
      if (!title || !prompt) {
        showToast('请填写题目标题和完整题干');
        return;
      }
      const rubricLines = document
        .querySelector('#cq-rubric')
        .value.split('\n')
        .map((s) => s.trim())
        .filter(Boolean);
      const effectiveLines = rubricLines.length ? rubricLines : ['核心机制与瓶颈定位', '定量估算与架构权衡', '生产落地与容错考量'];
      const weightEach = Math.round(100 / effectiveLines.length);
      const newQ = {
        id: `custom-${Date.now()}`,
        custom: true,
        isCoding: Boolean(document.querySelector('#cq-is-coding')?.checked),
        track: document.querySelector('#cq-track').value,
        level: document.querySelector('#cq-level').value,
        frequency: '自定义',
        incident: false,
        title,
        summary: prompt.slice(0, 72) + (prompt.length > 72 ? '…' : ''),
        tags: ['自定义题', '团队题库'],
        contextSpecs: ['自定义场景'],
        prompt,
        rubric: effectiveLines.map((line, i) => ({
          id: `cr-${i}`,
          label: line,
          weight: weightEach,
          keywords: line.split(/[\s、,，/：:()（）]+/).filter((w) => w.length >= 2),
          probe: `关于「${line}」，能结合定量估算或线上案例再展开讲一下吗？`,
        })),
        hints: ['L1：先明确硬件约束与核心瓶颈是 Compute-bound、Memory-bound 还是通信瓶颈。'],
        followUps: [
          { stage: 'deepen', question: '如果把集群规模或并发流量再放大 10 倍，你刚才的设计哪里会最先出现瓶颈？' },
        ],
        referenceAnswer: document.querySelector('#cq-ref').value.trim() || '暂无预设参考答案。',
      };
      saveNewCustomQuestion(newQ);
      closeModal();
      showToast('自定义面试题已保存');
    });
  });
}
