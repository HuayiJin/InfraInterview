# TensorBench · AI Infra 面试官工作台 (LLM 训练 / 推理 / RL / GPU 底层)

专为 LLM AI Infra 面试打造的 Web 应用：**默认以面试官工作台为主视角，支持一键切换候选人模拟副视图**，紧贴真实面试平台的 **Online Txt 纯文本交互板** 体验，内置双引擎 AI 能力（零配置离线专家引擎 + 可选 Live LLM API）。

## 核心特性

1. **面试官工作台（默认主视角）**
   - **Online Txt 交互板**：带行号、等宽字体、Tab 缩进、一键写入题干/追问分隔线/复制外发；面试官可粘贴候选人在外部面试平台（飞书/牛客/CoderPad）上的实时作答或现场速记。
   - **⚡ AI 实时分析 & 分层追问**：一键评估白板作答的考点命中率与定量深度，自动预填 Rubric 勾选状态，生成「补漏追问」「定量深挖」「生产边界」3 条梯度追问，点一下即可复制发回或写入白板。
   - **考点 Rubric 实时打分 & 提示卡**：支持手动校准「命中 / 一般 / 遗漏」，实时换算得分；内置 L1→L3 渐进提示卡和仅面试官可见的 Staff 级参考答案与定量公式。
   - **简历 5-Why 深挖**：粘贴候选人项目经历，自动提取 AI Infra 技术栈并生成定制化拷打问题与关联真题推荐。
   - **面评生成 & 归档中心**：汇总整场 Rubric 得分、录用结论（Strong Hire ~ No Hire）、Green/Red Flags 与白板摘录，一键复制 Markdown 面评贴入招聘系统或导出 `.md`。

2. **候选人模拟自测（副视图）**
   - 顶部一键切换，严格隐藏参考答案与 Rubric，通过同款 Online Txt 白板接受 AI 面试官多轮追问拷打，单题结束后揭晓解析与复盘。

3. **28 道全栈均衡 AI Infra 题库（P6 / P7 / P8 分层）**
   - **分布式训练 (8 题)**：3D/5D 并行、ZeRO/FSDP2、MoE EP & DualPipe/DeepEP、FP8 细粒度训练、128K+ Context Parallelism、万卡容错与异步 Checkpoint、Zero-Bubble 流水线、千卡 MFU 定量估算。
   - **推理加速与引擎 (8 题)**：vLLM PagedAttention & Chunked Prefill、PD 分离与 KV 传输、MHA/GQA/MLA 矩阵吸收 & RadixAttention、Roofline 定量推导、W4A16/FP8/KV 量化、投机采样 EAGLE/MTP、DeepSeek-R1 Wide-EP & TBO、线上 SLO 排障。
   - **RL / 后训练系统 (7 题)**：PPO vs GRPO vs DPO、Rollout 与训练的共置/分池及同步/异步更新（veRL、OpenRLHF）、Weight Sync & 3D Resharding、长 CoT Rollout 长尾 Straggler、CUDA VMM/CuMem Sleep Mode、Agentic RL 沙箱集群、Train-Inference Precision Mismatch 排障。
   - **GPU 算子与通信 (5 题)**：FlashAttention-1/2/3、NCCL Ring/Tree & Rail-Optimized/RoCEv2、CUDA/Nsight/Triton 调优、A100→GB200 NVL72 代际演进、Chunked Linear Cross-Entropy (Liger-Kernel)。
   - 支持自定义新建团队独家面试题。

4. **双引擎 AI 接入**
   - 零配置即用本地专家引擎；
   - 点击左下角/右上角「AI 配置」可接入 **DeepSeek / OpenAI / 通义千问 DashScope / 本地 Ollama & vLLM**（由 `server.mjs` 的 `/api/llm-proxy` 无 CORS 转发，API 异常时自动无缝降级为本地引擎）。

## 启动与测试

```bash
# 运行自动化测试
npm test

# 启动本地服务 (默认 http://127.0.0.1:3456)
npm start
```
