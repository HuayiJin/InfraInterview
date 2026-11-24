import { LLM_PRESETS } from '../engine/storage.js';
import { testLlmConnection } from '../engine/llmClient.js';
import { escapeHtml } from '../engine/markdown.js';

export function openSettingsModal(ctx) {
  const { settings, updateSettings, openModal, closeModal, showToast } = ctx;
  const draft = structuredClone(settings.llm);

  const render = () => {
    openModal(`
      <div class="card-header">
        <div>
          <h3 class="card-title">AI 面试官引擎配置（双引擎）</h3>
          <div class="card-sub">默认使用零配置「本地专家引擎」；开启 Live LLM API 后可解锁深度动态追问、定制简历拷打与面评润色</div>
        </div>
        <button class="icon-btn" id="st-close">✕</button>
      </div>

      <div style="display:flex;flex-direction:column;gap:14px;">
        <div class="option-grid">
          <button class="option-card ${!draft.enabled ? 'selected' : ''}" id="st-mode-local">
            <div class="option-card-head">
              <span class="option-card-title">本地专家引擎 (离线零配置)</span>
              <span class="badge badge-inference">默认</span>
            </div>
            <div class="option-card-desc">内置 AI Infra 领域 Rubric 评估器、技术栈提取器与结构化追问树，零延迟、完全隐私离线。</div>
          </button>

          <button class="option-card ${draft.enabled ? 'selected' : ''}" id="st-mode-llm">
            <div class="option-card-head">
              <span class="option-card-title">Live LLM API 增强引擎</span>
              <span class="badge badge-training">AI 动态驱动</span>
            </div>
            <div class="option-card-desc">通过本地 Node 代理无 CORS 接入 DeepSeek / OpenAI / Qwen / 本地 Ollama，接口报错自动降级本地引擎。</div>
          </button>
        </div>

        <div id="st-llm-form" style="display:${draft.enabled ? 'flex' : 'none'};flex-direction:column;gap:12px;padding:14px;border-radius:var(--radius-sm);background:var(--bg-elevated);border:1px solid var(--border-default);">
          <div>
            <div style="font-size:12px;font-weight:600;color:var(--text-secondary);margin-bottom:6px;">快捷服务商预设</div>
            <div class="chip-row">
              ${LLM_PRESETS.map(
                (p) =>
                  `<button class="chip ${draft.preset === p.id ? 'selected' : ''}" data-preset="${p.id}" title="${escapeHtml(p.hint)}">${escapeHtml(p.name)}</button>`
              ).join('')}
            </div>
          </div>

          <div class="field">
            <label>Base URL (OpenAI Chat Completions 兼容端点)</label>
            <input type="text" id="st-base-url" value="${escapeHtml(draft.baseUrl || '')}" placeholder="https://api.deepseek.com/v1" />
          </div>

          <div class="form-grid">
            <div class="field">
              <label>Model 名称</label>
              <input type="text" id="st-model" value="${escapeHtml(draft.model || '')}" placeholder="deepseek-chat / gpt-4o / qwen-max" />
            </div>
            <div class="field">
              <label>Temperature (${draft.temperature})</label>
              <input type="range" id="st-temp" min="0" max="1" step="0.05" value="${draft.temperature ?? 0.35}" />
            </div>
          </div>

          <div class="field">
            <label>
              <span>API Key（本地 Ollama/vLLM 可留空）</span>
              <span style="font-weight:400;color:var(--text-muted);">仅存于本机浏览器 localStorage</span>
            </label>
            <input type="password" id="st-api-key" value="${escapeHtml(draft.apiKey || '')}" placeholder="sk-..." autocomplete="off" />
          </div>

          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;">
            <button class="btn btn-sm" id="st-test-btn">测试 API 连通性</button>
            <span id="st-test-status" style="font-size:12px;color:var(--text-secondary);"></span>
          </div>
        </div>

        <div style="display:flex;justify-content:flex-end;gap:8px;padding-top:4px;">
          <button class="btn btn-ghost" id="st-cancel">取消</button>
          <button class="btn btn-primary" id="st-save">保存生效</button>
        </div>
      </div>
    `);

    document.querySelector('#st-close')?.addEventListener('click', closeModal);
    document.querySelector('#st-cancel')?.addEventListener('click', closeModal);

    document.querySelector('#st-mode-local')?.addEventListener('click', () => {
      draft.enabled = false;
      render();
    });
    document.querySelector('#st-mode-llm')?.addEventListener('click', () => {
      draft.enabled = true;
      render();
    });

    document.querySelectorAll('[data-preset]').forEach((b) =>
      b.addEventListener('click', () => {
        const preset = LLM_PRESETS.find((p) => p.id === b.dataset.preset);
        if (!preset) return;
        draft.preset = preset.id;
        if (preset.baseUrl) draft.baseUrl = preset.baseUrl;
        if (preset.model) draft.model = preset.model;
        render();
      })
    );

    const syncInputs = () => {
      const u = document.querySelector('#st-base-url');
      const m = document.querySelector('#st-model');
      const k = document.querySelector('#st-api-key');
      const t = document.querySelector('#st-temp');
      if (u) draft.baseUrl = u.value.trim();
      if (m) draft.model = m.value.trim();
      if (k) draft.apiKey = k.value.trim();
      if (t) draft.temperature = Number(t.value);
    };

    document.querySelector('#st-temp')?.addEventListener('input', (e) => {
      draft.temperature = Number(e.target.value);
      const lbl = e.target.previousElementSibling;
      if (lbl) lbl.textContent = `Temperature (${draft.temperature})`;
    });

    document.querySelector('#st-test-btn')?.addEventListener('click', async () => {
      syncInputs();
      const statusEl = document.querySelector('#st-test-status');
      if (!draft.baseUrl || !draft.model) {
        if (statusEl) statusEl.textContent = '请先填写 Base URL 和 Model';
        return;
      }
      if (statusEl) statusEl.innerHTML = '<span class="loading-shimmer"><span class="spinner"></span> 正在连接测试…</span>';
      try {
        const res = await testLlmConnection(draft);
        if (statusEl) statusEl.innerHTML = `<span style="color:var(--track-inference);">✓ 连接成功：${escapeHtml(res.reply)}</span>`;
      } catch (err) {
        if (statusEl) statusEl.innerHTML = `<span style="color:var(--danger);">✕ 连接失败：${escapeHtml(err.message)}</span>`;
      }
    });

    document.querySelector('#st-save')?.addEventListener('click', () => {
      syncInputs();
      if (draft.enabled && (!draft.baseUrl || !draft.model)) {
        showToast('启用 Live LLM 需要填写 Base URL 和 Model');
        return;
      }
      updateSettings({ ...settings, llm: draft });
      closeModal();
      showToast(draft.enabled ? `已启用 Live LLM (${draft.model})` : '已切换为本地专家引擎');
    });
  };

  render();
}
