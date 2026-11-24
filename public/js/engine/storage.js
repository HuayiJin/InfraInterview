const STORAGE_KEYS = {
  SETTINGS: 'tb_settings_v2',
  INTERVIEWER_ACTIVE: 'tb_interviewer_active_v2',
  INTERVIEWER_ARCHIVE: 'tb_interviewer_archive_v2',
  CANDIDATE_HISTORY: 'tb_candidate_history_v2',
  CUSTOM_QUESTIONS: 'tb_custom_questions_v2',
};

export const LLM_PRESETS = [
  {
    id: 'deepseek',
    name: 'DeepSeek API',
    baseUrl: 'https://api.deepseek.com/v1',
    model: 'deepseek-chat',
    hint: '性价比极高，中文 AI Infra 术语精准',
  },
  {
    id: 'openai',
    name: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    model: 'gpt-4o',
    hint: '官方 OpenAI 兼容接口',
  },
  {
    id: 'qwen',
    name: '通义千问 DashScope',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'qwen-max',
    hint: '阿里云百炼 OpenAI 兼容模式',
  },
  {
    id: 'ollama',
    name: '本地 Ollama / vLLM',
    baseUrl: 'http://127.0.0.1:11434/v1',
    model: 'qwen2.5:32b',
    hint: '无需 Key，纯内网本地推理服务',
  },
  {
    id: 'custom',
    name: '自定义 OpenAI 兼容',
    baseUrl: '',
    model: '',
    hint: '支持硅基流动、OpenRouter、内部网关等',
  },
];

export const DEFAULT_SETTINGS = {
  theme: 'dark',
  defaultRole: 'interviewer',
  persona: 'strict',
  llm: {
    enabled: false,
    preset: 'deepseek',
    baseUrl: 'https://api.deepseek.com/v1',
    apiKey: '',
    model: 'deepseek-chat',
    temperature: 0.35,
  },
};

function safeRead(key, fallback) {
  try {
    if (typeof localStorage === 'undefined') return structuredClone(fallback);
    const raw = localStorage.getItem(key);
    if (!raw) return structuredClone(fallback);
    return JSON.parse(raw);
  } catch {
    return structuredClone(fallback);
  }
}

function safeWrite(key, value) {
  try {
    if (typeof localStorage === 'undefined') return;
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore storage quota errors
  }
}

export function loadSettings() {
  const saved = safeRead(STORAGE_KEYS.SETTINGS, DEFAULT_SETTINGS);
  return {
    ...DEFAULT_SETTINGS,
    ...saved,
    llm: {
      ...DEFAULT_SETTINGS.llm,
      ...(saved?.llm || {}),
    },
  };
}

export function saveSettings(settings) {
  safeWrite(STORAGE_KEYS.SETTINGS, settings);
}

export function createDefaultInterviewerSession(defaultQuestionIds = ['tr-01', 'inf-02', 'rl-02', 'kn-01']) {
  const now = Date.now();
  return {
    id: `iv-${now}`,
    createdAt: now,
    updatedAt: now,
    candidateName: '候选人 A',
    targetLevel: 'P7',
    persona: 'strict',
    resumeText: '',
    resumeAnalysis: null,
    questionIds: [...defaultQuestionIds],
    activeIndex: 0,
    records: {}, // questionId -> { txtContent, rubricStates: { [rubricId]: 'unseen'|'hit'|'partial'|'miss' }, aiAnalysis, notes, updatedAt }
    verdict: '', // 'strong_hire' | 'hire' | 'lean_hire' | 'lean_no' | 'no_hire'
    greenFlags: '',
    redFlags: '',
    overallNotes: '',
    generatedReport: '',
  };
}

export function loadActiveInterviewerSession() {
  const saved = safeRead(STORAGE_KEYS.INTERVIEWER_ACTIVE, null);
  if (!saved || !Array.isArray(saved.questionIds) || saved.questionIds.length === 0) {
    return createDefaultInterviewerSession();
  }
  return saved;
}

export function saveActiveInterviewerSession(session) {
  session.updatedAt = Date.now();
  safeWrite(STORAGE_KEYS.INTERVIEWER_ACTIVE, session);
}

export function loadInterviewerArchive() {
  return safeRead(STORAGE_KEYS.INTERVIEWER_ARCHIVE, []);
}

export function archiveInterviewerSession(session) {
  const list = loadInterviewerArchive();
  const idx = list.findIndex((item) => item.id === session.id);
  const snapshot = structuredClone({ ...session, archivedAt: Date.now() });
  if (idx >= 0) {
    list[idx] = snapshot;
  } else {
    list.unshift(snapshot);
  }
  safeWrite(STORAGE_KEYS.INTERVIEWER_ARCHIVE, list.slice(0, 50));
  return list;
}

export function deleteArchivedSession(id) {
  const list = loadInterviewerArchive().filter((item) => item.id !== id);
  safeWrite(STORAGE_KEYS.INTERVIEWER_ARCHIVE, list);
  return list;
}

export function loadCandidateHistory() {
  return safeRead(STORAGE_KEYS.CANDIDATE_HISTORY, []);
}

export function saveCandidateRun(run) {
  const list = loadCandidateHistory();
  list.unshift({ ...run, completedAt: Date.now() });
  safeWrite(STORAGE_KEYS.CANDIDATE_HISTORY, list.slice(0, 50));
  return list;
}

export function loadCustomQuestions() {
  return safeRead(STORAGE_KEYS.CUSTOM_QUESTIONS, []);
}

export function saveCustomQuestion(q) {
  const list = loadCustomQuestions();
  const idx = list.findIndex((item) => item.id === q.id);
  if (idx >= 0) list[idx] = q;
  else list.unshift(q);
  safeWrite(STORAGE_KEYS.CUSTOM_QUESTIONS, list);
  return list;
}

export function deleteCustomQuestion(id) {
  const list = loadCustomQuestions().filter((item) => item.id !== id);
  safeWrite(STORAGE_KEYS.CUSTOM_QUESTIONS, list);
  return list;
}
