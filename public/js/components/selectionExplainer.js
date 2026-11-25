import { getVideosForSelection, buildYoutubeSearchUrl } from '../data/videos.js';
import { aiExplainSelection } from '../engine/llmClient.js';
import { escapeHtml, renderMarkdown } from '../engine/markdown.js';
import { renderVideoStrip } from './videoStrip.js';

const MODES = [
  { id: 'explain', label: '原理解释' },
  { id: 'derivation', label: '公式推导' },
  { id: 'example', label: '数字算例' },
  { id: 'probe', label: '转为追问' },
];

function isEditableOrIgnored(node) {
  const el = node?.nodeType === 1 ? node : node?.parentElement;
  if (!el) return true;
  if (el.closest('input, textarea, select, [contenteditable="true"]')) return true;
  if (el.closest('#selection-explainer-pill, #selection-explainer-popover')) return true;
  return false;
}

function getSurroundingText(range) {
  let node = range.commonAncestorContainer;
  if (node.nodeType !== 1) node = node.parentElement;
  const block = node?.closest('p, li, td, th, blockquote, .ref-box, .prompt-hero-card, .rubric-wide-row, .msg-bubble, .q-card, .callout') || node;
  return String(block?.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 700);
}

export function initSelectionExplainer(getRuntimeCtx) {
  // Inject singleton floating pill & popover onto body
  const pill = document.createElement('button');
  pill.id = 'selection-explainer-pill';
  pill.className = 'selection-pill';
  pill.type = 'button';
  pill.hidden = true;
  pill.innerHTML = `<span class="selection-pill-spark">✨</span><span>AI 解释</span>`;
  document.body.appendChild(pill);

  const popover = document.createElement('aside');
  popover.id = 'selection-explainer-popover';
  popover.className = 'explainer-popover';
  popover.hidden = true;
  popover.setAttribute('aria-label', 'AI 划词解释');
  document.body.appendChild(popover);

  const st = {
    open: false,
    pinned: false,
    loading: false,
    selectedText: '',
    surroundingContext: '',
    question: null,
    mode: 'explain',
    source: 'local',
    fallbackReason: '',
    messages: [], // { role: 'user' | 'assistant', content, modeLabel? }
    anchorRect: null,
  };

  let pendingSelection = null;

  function hidePill() {
    pill.hidden = true;
    pendingSelection = null;
  }

  function closePopover(force = false) {
    if (st.pinned && !force) return;
    st.open = false;
    popover.hidden = true;
  }

  function positionPill(rect) {
    const vw = window.innerWidth;
    const pillW = 104;
    const left = Math.max(12, Math.min(vw - pillW - 12, rect.left + rect.width / 2 - pillW / 2));
    const top = rect.top > 52 ? rect.top - 40 : rect.bottom + 8;
    pill.style.left = `${Math.round(left)}px`;
    pill.style.top = `${Math.round(Math.max(8, Math.min(window.innerHeight - 44, top)))}px`;
    pill.hidden = false;
  }

  function positionPopover(rect) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const minTop = 66; // stay below 58px topbar
    const popW = Math.min(460, vw - 24);
    const popH = Math.min(520, vh - minTop - 16);

    if (vw < 780 || !rect) {
      popover.style.left = `${Math.max(12, Math.round((vw - popW) / 2))}px`;
      popover.style.top = `${Math.max(minTop, vh - popH - 16)}px`;
      return;
    }

    let left = rect.left;
    if (left + popW > vw - 16) left = vw - popW - 16;
    if (left < 16) left = 16;

    let top = rect.bottom + 12;
    if (top + popH > vh - 16) {
      top = Math.max(minTop, rect.top - popH - 12);
    }
    if (top + popH > vh - 16) {
      top = Math.max(minTop, vh - popH - 16);
    }

    popover.style.left = `${Math.round(left)}px`;
    popover.style.top = `${Math.round(top)}px`;
  }

  function renderPopover() {
    if (!st.open) return;
    const ctx = getRuntimeCtx();
    const llmEnabled = Boolean(ctx.settings?.llm?.enabled);
    const shortQuote =
      st.selectedText.length > 76 ? st.selectedText.slice(0, 76) + '…' : st.selectedText;
    const matchedVideos = getVideosForSelection(st.selectedText, st.surroundingContext, st.question?.id);
    const ytSearchUrl = buildYoutubeSearchUrl(st.selectedText, st.question);

    popover.innerHTML = `
      <div class="explainer-head">
        <div style="display:flex;align-items:center;gap:7px;min-width:0;">
          <span class="explainer-spark-badge">✨ AI 划词解释</span>
          <span class="badge badge-mono" style="font-size:10.5px;">
            ${
              st.source.startsWith('llm:')
                ? escapeHtml(st.source.replace('llm:', ''))
                : llmEnabled
                  ? escapeHtml(ctx.settings.llm.model)
                  : '本地知识库'
            }
          </span>
        </div>
        <div style="display:flex;align-items:center;gap:4px;">
          <button class="icon-btn" id="exp-pin-btn" title="${st.pinned ? '取消固定' : '固定窗口便于对照阅读'}" style="width:28px;height:28px;font-size:12px;${st.pinned ? 'border-color:var(--accent);color:var(--accent);background:var(--accent-bg);' : ''}">📌</button>
          <button class="icon-btn" id="exp-close-btn" title="关闭 (Esc)" style="width:28px;height:28px;font-size:12px;">✕</button>
        </div>
      </div>

      <div class="explainer-quote" title="${escapeHtml(st.selectedText)}">
        「${escapeHtml(shortQuote)}」
      </div>

      <div class="explainer-modes">
        ${MODES.map(
          (m) =>
            `<button class="explainer-mode-btn ${st.mode === m.id ? 'active' : ''}" data-exp-mode="${m.id}" ${st.loading ? 'disabled' : ''}>${m.label}</button>`
        ).join('')}
      </div>

      <div class="explainer-body" id="exp-body">
        ${
          st.fallbackReason
            ? `<div class="callout callout-warn" style="margin-bottom:8px;padding:7px 10px;font-size:11.5px;">Live LLM 请求异常（${escapeHtml(st.fallbackReason)}），已自动回退本地词库解释。</div>`
            : ''
        }
        ${st.messages
          .map((m) =>
            m.role === 'user'
              ? `<div class="explainer-user-q"><strong>追问：</strong>${escapeHtml(m.content)}</div>`
              : `<div class="msg-content explainer-ans">${renderMarkdown(m.content)}</div>`
          )
          .join('')}
        ${
          st.loading
            ? `<div class="loading-shimmer" style="padding:8px 0;"><span class="spinner"></span> 正在结合上下文生成深度解释…</div>`
            : matchedVideos.length
              ? renderVideoStrip(matchedVideos, { heading: '关联 YouTube 图解视频', compact: true })
              : ''
        }
      </div>

      <form class="explainer-ask-bar" id="exp-ask-form">
        <input
          type="text"
          id="exp-ask-input"
          class="inline-input"
          style="flex:1;"
          placeholder="对这段解释继续追问 AI（如：为什么系数是 2 而不是 4？）"
          ${st.loading ? 'disabled' : ''}
        />
        <button class="btn btn-primary btn-sm" type="submit" ${st.loading ? 'disabled' : ''}>提问</button>
      </form>

      <div class="explainer-footer">
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
          <button class="btn btn-ghost btn-sm" id="exp-copy-btn">复制解释</button>
          <a
            class="btn btn-ghost btn-sm explainer-yt-link"
            href="${escapeHtml(ytSearchUrl)}"
            target="_blank"
            rel="noopener noreferrer"
            title="在 YouTube 按此知识点搜索垂类图解视频"
          >
            <span class="yt-play-dot" aria-hidden="true">▶</span> YouTube 搜图解 ↗
          </a>
        </div>
        ${
          !llmEnabled
            ? `<button class="btn btn-ghost btn-sm" id="exp-open-llm-btn" style="color:var(--accent);">接入 Live LLM →</button>`
            : `<span style="font-size:11px;color:var(--text-muted);">支持连续追问</span>`
        }
      </div>
    `;

    const bodyEl = popover.querySelector('#exp-body');
    if (bodyEl) {
      bodyEl.scrollTop = st.messages.length > 1 ? bodyEl.scrollHeight : 0;
    }

    popover.querySelector('#exp-close-btn')?.addEventListener('click', () => closePopover(true));
    popover.querySelector('#exp-pin-btn')?.addEventListener('click', () => {
      st.pinned = !st.pinned;
      renderPopover();
    });
    popover.querySelector('#exp-copy-btn')?.addEventListener('click', () => {
      const lastAssistant = [...st.messages].reverse().find((m) => m.role === 'assistant');
      if (lastAssistant) ctx.copyText(lastAssistant.content, 'AI 解释已复制');
    });
    popover.querySelector('#exp-open-llm-btn')?.addEventListener('click', () => {
      closePopover(true);
      ctx.openSettings();
    });

    popover.querySelectorAll('[data-exp-mode]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const nextMode = btn.dataset.expMode;
        st.mode = nextMode;
        runExplain(nextMode, '');
      });
    });

    popover.querySelector('#exp-ask-form')?.addEventListener('submit', (e) => {
      e.preventDefault();
      const input = popover.querySelector('#exp-ask-input');
      const qTxt = (input?.value || '').trim();
      if (!qTxt || st.loading) return;
      input.value = '';
      runExplain(st.mode, qTxt);
    });
  }

  async function runExplain(mode, userFollowUp = '') {
    const ctx = getRuntimeCtx();
    st.loading = true;
    st.fallbackReason = '';
    if (userFollowUp) {
      st.messages.push({ role: 'user', content: userFollowUp });
    } else {
      st.messages = [];
    }
    renderPopover();

    try {
      const res = await aiExplainSelection(ctx.settings.llm, {
        selectedText: st.selectedText,
        surroundingContext: st.surroundingContext,
        question: st.question,
        mode,
        history: userFollowUp ? st.messages.slice(0, -1) : [],
        userFollowUp,
      });
      st.source = res.source || 'local';
      st.fallbackReason = res.fallbackReason || '';
      st.messages.push({ role: 'assistant', content: res.content });
    } finally {
      st.loading = false;
      renderPopover();
    }
  }

  function handleSelectionCheck() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
      hidePill();
      return;
    }
    const range = sel.getRangeAt(0);
    if (isEditableOrIgnored(range.commonAncestorContainer)) {
      hidePill();
      return;
    }
    const text = sel.toString().replace(/\s+/g, ' ').trim();
    if (text.length < 2 || text.length > 500) {
      hidePill();
      return;
    }
    const rect = range.getBoundingClientRect();
    if (!rect || (rect.width === 0 && rect.height === 0)) {
      hidePill();
      return;
    }
    pendingSelection = {
      text,
      surroundingContext: getSurroundingText(range),
      rect: { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: rect.width, height: rect.height },
    };
    positionPill(pendingSelection.rect);
  }

  let debounceTimer = null;
  document.addEventListener('mouseup', (e) => {
    if (e.target.closest('#selection-explainer-pill, #selection-explainer-popover')) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(handleSelectionCheck, 20);
  });

  document.addEventListener('keyup', (e) => {
    if (e.key === 'Escape') {
      hidePill();
      closePopover(true);
      return;
    }
    if (e.shiftKey || e.key.startsWith('Arrow')) {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(handleSelectionCheck, 40);
    }
  });

  document.addEventListener('mousedown', (e) => {
    if (e.target.closest('#selection-explainer-pill, #selection-explainer-popover')) return;
    hidePill();
    if (st.open && !st.pinned) {
      closePopover(false);
    }
  });

  // Trigger popover from floating pill (prevent mousedown from clearing selection prematurely)
  pill.addEventListener('mousedown', (e) => {
    e.preventDefault();
    e.stopPropagation();
  });

  pill.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!pendingSelection) return;
    const ctx = getRuntimeCtx();
    st.selectedText = pendingSelection.text;
    st.surroundingContext = pendingSelection.surroundingContext;
    st.question = ctx.getActiveQuestion();
    st.mode = 'explain';
    st.anchorRect = pendingSelection.rect;
    st.open = true;
    hidePill();
    popover.hidden = false;
    positionPopover(st.anchorRect);
    runExplain('explain', '');
  });

  return {
    close: () => {
      hidePill();
      closePopover(true);
    },
  };
}
