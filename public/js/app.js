import { ALL_QUESTIONS, TRACKS, TARGET_LEVELS } from './data/index.js';
import {
  loadSettings,
  saveSettings,
  loadActiveInterviewerSession,
  saveActiveInterviewerSession,
  createDefaultInterviewerSession,
  loadInterviewerArchive,
  archiveInterviewerSession,
  deleteArchivedSession,
  loadCandidateHistory,
  saveCandidateRun,
  loadCustomQuestions,
  saveCustomQuestion,
  deleteCustomQuestion,
} from './engine/storage.js';
import { renderInterviewerView } from './components/interviewerView.js';
import { renderCandidateView, startCandidateMockWithQuestions } from './components/candidateView.js';
import { renderBankView } from './components/bankView.js';
import { renderDossierView } from './components/dossierView.js';
import { openSettingsModal } from './components/settingsModal.js';
import { escapeHtml } from './engine/markdown.js';

const state = {
  settings: loadSettings(),
  role: 'interviewer', // Default: Interviewer Cockpit! Secondary: 'candidate'
  route: 'workspace', // 'workspace' | 'bank' | 'dossier'
  customQuestions: loadCustomQuestions(),
  session: loadActiveInterviewerSession(),
  archiveList: loadInterviewerArchive(),
  candidateHistory: loadCandidateHistory(),
  clock: {
    running: false,
    seconds: 0,
    timerId: null,
  },
  mockState: {
    active: false,
    config: { mode: 'full_loop', track: 'rl', level: 'P7', persona: 'strict' },
    questions: [],
    currentIndex: 0,
    turns: [],
    results: [],
    loading: false,
    hintIndex: 0,
    draftTxt: '',
    finishedCurrent: false,
    currentAnalysis: null,
  },
  uiState: {
    copilotTab: 'reference',
    analyzingQId: null,
    parsingResume: false,
    generatingReport: false,
    bankFilter: { search: '', track: 'all', level: 'all', special: 'all' },
  },
};

function getAllQuestions() {
  return [...state.customQuestions, ...ALL_QUESTIONS];
}

function showToast(msg) {
  const wrap = document.querySelector('#toast-container');
  if (!wrap) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  wrap.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transition = 'opacity 200ms';
    setTimeout(() => el.remove(), 220);
  }, 2600);
}

async function copyText(text, successMsg = '已复制到剪贴板') {
  try {
    await navigator.clipboard.writeText(text);
    showToast(successMsg);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    showToast(successMsg);
  }
}

function openModal(html) {
  const backdrop = document.querySelector('#modal-backdrop');
  const dialog = document.querySelector('#modal-dialog');
  if (!backdrop || !dialog) return;
  dialog.innerHTML = html;
  backdrop.hidden = false;
}

function closeModal() {
  const backdrop = document.querySelector('#modal-backdrop');
  if (backdrop) backdrop.hidden = true;
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
}

function formatClock(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(sec % 60).padStart(2, '0');
  return `${m}:${s}`;
}

function toggleClock() {
  const dot = document.querySelector('#topbar-clock-dot');
  const timeEl = document.querySelector('#topbar-clock-time');
  if (state.clock.running) {
    clearInterval(state.clock.timerId);
    state.clock.running = false;
    dot?.classList.add('paused');
    showToast(`面试计时已暂停 (${formatClock(state.clock.seconds)})`);
  } else {
    state.clock.running = true;
    dot?.classList.remove('paused');
    state.clock.timerId = setInterval(() => {
      state.clock.seconds += 1;
      if (timeEl) timeEl.textContent = formatClock(state.clock.seconds);
    }, 1000);
    showToast('面试计时已启动');
  }
}

function updateShellChrome() {
  const allQs = getAllQuestions();

  // Role buttons
  document.querySelectorAll('.role-btn').forEach((btn) => {
    const active = btn.dataset.role === state.role;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  });

  // Nav active state & dynamic labels
  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.route === state.route);
  });

  const wsLabel = document.querySelector('#nav-workspace-label');
  const dosLabel = document.querySelector('#nav-dossier-label');
  const bankCount = document.querySelector('#nav-bank-count');
  const sessionPill = document.querySelector('#nav-session-pill');

  if (wsLabel) wsLabel.textContent = state.role === 'interviewer' ? '面试官工作台' : '候选人模拟舱';
  if (dosLabel) dosLabel.textContent = state.role === 'interviewer' ? '面评 & 归档' : '自测复盘记录';
  if (bankCount) bankCount.textContent = String(allQs.length);
  if (sessionPill) sessionPill.hidden = !(state.role === 'candidate' && state.mockState.active);

  // Topbar title
  const titleEl = document.querySelector('#page-title');
  const subEl = document.querySelector('#page-subtitle');
  const titles = {
    workspace:
      state.role === 'interviewer'
        ? ['面试官工作台', '口头面试主控 · 考点 Rubric 实时打分、候选人回答分析与代码白板按需呼出']
        : ['候选人模拟自测', '副视图 · 对话式 AI 面试官多轮拷打（支持文字输入与语音口述）'],
    bank: ['AI Infra 全栈题库 & 组卷', '训练 + 推理 + RL + GPU 算子通信 · P6/P7/P8 难度分层'],
    dossier:
      state.role === 'interviewer'
        ? ['面评报告 & 面试归档', '一键汇总 Rubric 得分与候选人回答摘录，输出结构化 Markdown 面评']
        : ['模拟自测复盘', '查看历史模拟得分与四大方向掌握度'],
  };
  const [t, s] = titles[state.route] || titles.workspace;
  if (titleEl) titleEl.textContent = t;
  if (subEl) subEl.textContent = s;

  // Engine status badges
  const isApi = Boolean(state.settings.llm.enabled);
  const dot = document.querySelector('#engine-status-dot');
  const topDot = document.querySelector('#topbar-engine-dot');
  const modeLbl = document.querySelector('#engine-mode-label');
  const subLbl = document.querySelector('#engine-sub-label');
  const topTxt = document.querySelector('#topbar-engine-text');

  dot?.classList.toggle('api-mode', isApi);
  topDot?.classList.toggle('api-mode', isApi);
  if (modeLbl) modeLbl.textContent = isApi ? 'Live LLM 增强模式' : '本地专家引擎';
  if (subLbl) subLbl.textContent = isApi ? `${state.settings.llm.model}` : '零配置 · 点击接入 LLM API';
  if (topTxt) topTxt.textContent = isApi ? state.settings.llm.model : '本地引擎 / 配置 API';

  // Sidebar track stats
  const trackWrap = document.querySelector('#sidebar-track-stats');
  if (trackWrap) {
    const queueSet = new Set(state.session.questionIds || []);
    trackWrap.innerHTML = Object.values(TRACKS)
      .map((tr) => {
        const qs = allQs.filter((q) => q.track === tr.id);
        const selectedCnt = qs.filter((q) => queueSet.has(q.id)).length;
        const pct = Math.min(100, Math.round((qs.length / allQs.length) * 100 * 2.4));
        return `
          <div class="track-mini-item">
            <div class="track-mini-row">
              <span class="track-mini-name"><span class="track-dot" style="background:${tr.color}"></span>${tr.name}</span>
              <span class="mono" style="font-size:11px;color:var(--text-muted);">${qs.length}题${selectedCnt ? ` · 已选${selectedCnt}` : ''}</span>
            </div>
            <div class="track-mini-bar">
              <div class="track-mini-fill" style="width:${pct}%;background:${tr.color};"></div>
            </div>
          </div>
        `;
      })
      .join('');
  }
}

function openQuickAssembleModal() {
  const allQs = getAllQuestions();
  openModal(`
    <div class="card-header">
      <div>
        <h3 class="card-title">智能快速组卷（替换当前面试题单）</h3>
        <div class="card-sub">按目标定级与重点方向一键抽取平衡题组</div>
      </div>
      <button class="icon-btn" id="qa-close">✕</button>
    </div>
    <div style="display:flex;flex-direction:column;gap:12px;">
      <div class="field">
        <label>目标定级倾向</label>
        <select id="qa-level">
          ${TARGET_LEVELS.map((l) => `<option value="${l.id}" ${state.session.targetLevel === l.id ? 'selected' : ''}>${l.label}</option>`).join('')}
        </select>
      </div>
      <div class="option-grid">
        <button class="option-card" data-qa-preset="balanced">
          <div class="option-card-title">全栈均衡 4 题</div>
          <div class="option-card-desc">训练 + 推理 + RL + GPU/通信各抽 1 题（标准全栈面）</div>
        </button>
        <button class="option-card" data-qa-preset="rl_inf">
          <div class="option-card-title">RL + 推理引擎专项 (4 题)</div>
          <div class="option-card-desc">2 道 RL/GRPO 系统 + 2 道推理引擎/PD分离/Wide-EP</div>
        </button>
        <button class="option-card" data-qa-preset="train_kern">
          <div class="option-card-title">预训练 + 底层集群专项 (4 题)</div>
          <div class="option-card-desc">2 道分布式训练/MoE + 2 道 FlashAttention/NCCL/硬件</div>
        </button>
        <button class="option-card" data-qa-preset="incident">
          <div class="option-card-title">真实线上排障专场 (3 题)</div>
          <div class="option-card-desc">万卡 Hang/SDC + 推理 SLO 雪崩 + RL 精度偏移</div>
        </button>
      </div>
    </div>
  `);

  document.querySelector('#qa-close')?.addEventListener('click', closeModal);
  document.querySelectorAll('[data-qa-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const preset = btn.dataset.qaPreset;
      const lvl = document.querySelector('#qa-level').value;
      state.session.targetLevel = lvl;

      const pickFrom = (track, count) => {
        const pool = allQs
          .filter((q) => q.track === track)
          .sort((a, b) => (b.level === lvl ? 1 : 0) - (a.level === lvl ? 1 : 0) + (Math.random() - 0.5) * 0.6);
        return pool.slice(0, count).map((q) => q.id);
      };

      let ids = [];
      if (preset === 'balanced') {
        ids = [...pickFrom('training', 1), ...pickFrom('inference', 1), ...pickFrom('rl', 1), ...pickFrom('kernel', 1)];
      } else if (preset === 'rl_inf') {
        ids = [...pickFrom('rl', 2), ...pickFrom('inference', 2)];
      } else if (preset === 'train_kern') {
        ids = [...pickFrom('training', 2), ...pickFrom('kernel', 2)];
      } else if (preset === 'incident') {
        ids = allQs.filter((q) => q.incident).map((q) => q.id);
      }

      if (ids.length) {
        state.session.questionIds = ids;
        state.session.activeIndex = 0;
        saveActiveInterviewerSession(state.session);
        closeModal();
        showToast(`已生成 ${ids.length} 道 ${lvl} 面试题单`);
        render();
      }
    });
  });
}

function buildCtx() {
  return {
    role: state.role,
    route: state.route,
    settings: state.settings,
    session: state.session,
    archiveList: state.archiveList,
    candidateHistory: state.candidateHistory,
    mockState: state.mockState,
    uiState: state.uiState,
    allQuestions: getAllQuestions(),
    rerender: render,
    showToast,
    copyText,
    openModal,
    closeModal,
    openQuickAssembleModal,
    navigate: (route) => {
      state.route = route;
      document.querySelector('#sidebar')?.classList.remove('mobile-open');
      render();
    },
    switchToInterviewer: () => {
      state.role = 'interviewer';
      state.route = 'workspace';
      render();
    },
    persistSession: () => {
      saveActiveInterviewerSession(state.session);
      updateShellChrome();
    },
    resetNewSession: () => {
      state.session = createDefaultInterviewerSession();
      saveActiveInterviewerSession(state.session);
      state.clock.seconds = 0;
      const tEl = document.querySelector('#topbar-clock-time');
      if (tEl) tEl.textContent = '00:00';
      showToast('已开启新的面试会话');
      render();
    },
    archiveCurrentSession: (notify = true) => {
      state.archiveList = archiveInterviewerSession(state.session);
      if (notify) showToast(`已归档「${state.session.candidateName || '候选人'}」的面试记录`);
      render();
    },
    loadArchivedIntoActive: (id) => {
      const found = state.archiveList.find((x) => x.id === id);
      if (!found) return;
      state.session = structuredClone(found);
      saveActiveInterviewerSession(state.session);
      state.route = 'workspace';
      state.role = 'interviewer';
      showToast(`已载入「${found.candidateName}」的面试档案`);
      render();
    },
    removeArchivedSession: (id) => {
      state.archiveList = deleteArchivedSession(id);
      showToast('已删除归档记录');
      render();
    },
    saveCandidateHistoryRun: (run) => {
      state.candidateHistory = saveCandidateRun(run);
    },
    startSingleCandidateMock: (question) => {
      state.role = 'candidate';
      state.route = 'workspace';
      startCandidateMockWithQuestions(state.mockState, [question], {
        ...state.mockState.config,
        level: question.level,
      });
      render();
    },
    saveNewCustomQuestion: (q) => {
      state.customQuestions = saveCustomQuestion(q);
      render();
    },
    removeCustomQuestion: (id) => {
      state.customQuestions = deleteCustomQuestion(id);
      state.session.questionIds = state.session.questionIds.filter((x) => x !== id);
      saveActiveInterviewerSession(state.session);
      render();
    },
    updateSettings: (next) => {
      state.settings = next;
      saveSettings(next);
      applyTheme(next.theme);
      render();
    },
  };
}

function render() {
  updateShellChrome();
  const main = document.querySelector('#main-content');
  if (!main) return;
  const ctx = buildCtx();

  if (state.route === 'workspace') {
    if (state.role === 'interviewer') {
      renderInterviewerView(main, ctx);
    } else {
      renderCandidateView(main, ctx);
    }
  } else if (state.route === 'bank') {
    renderBankView(main, ctx);
  } else if (state.route === 'dossier') {
    renderDossierView(main, ctx);
  }
}

function init() {
  applyTheme(state.settings.theme || 'dark');

  // Navigation buttons
  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.route = btn.dataset.route;
      document.querySelector('#sidebar')?.classList.remove('mobile-open');
      render();
    });
  });

  // Role switcher (Default Interviewer <-> Secondary Candidate Mock)
  document.querySelectorAll('.role-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const nextRole = btn.dataset.role;
      if (state.role === nextRole) return;
      state.role = nextRole;
      if (state.route !== 'bank') {
        state.route = 'workspace';
      }
      showToast(nextRole === 'interviewer' ? '已切换至：面试官工作台（默认视角）' : '已切换至：候选人模拟自测（副视图）');
      render();
    });
  });

  // Clock
  document.querySelector('#topbar-clock')?.addEventListener('click', toggleClock);

  // Settings buttons
  const openSettings = () => openSettingsModal(buildCtx());
  document.querySelector('#open-settings-btn')?.addEventListener('click', openSettings);
  document.querySelector('#topbar-ai-btn')?.addEventListener('click', openSettings);

  // Theme toggle
  document.querySelector('#theme-toggle-btn')?.addEventListener('click', () => {
    state.settings.theme = state.settings.theme === 'dark' ? 'light' : 'dark';
    saveSettings(state.settings);
    applyTheme(state.settings.theme);
  });

  // Mobile menu
  document.querySelector('#mobile-menu-btn')?.addEventListener('click', () => {
    document.querySelector('#sidebar')?.classList.toggle('mobile-open');
  });

  // Modal backdrop click close
  document.querySelector('#modal-backdrop')?.addEventListener('click', (e) => {
    if (e.target.id === 'modal-backdrop') closeModal();
  });

  // Escape closes modal
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });

  render();
}

init();
