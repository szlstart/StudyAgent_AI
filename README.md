# StudyAgent AI — 多模态个性化学习智能体

面向一年级至九年级学生的本机 AI 学习平台。姓名 + 年级 + 学期登录 → 拍照上传作业 → GPT-5.5 视觉批改 → 对应教材 RAG → 自动归档错题 → 个性化复习。

> 学生操作、项目启动、教材维护、备份和故障排查请查看：[StudyAgent AI 详细使用手册](./使用手册.md)
>
> 项目当前优点、已修复风险、剩余限制与验证证据请查看：[项目审计与改进记录](./项目审计与改进记录.md)
>
> 最新的安全、用户体验、Agent 合理性与真实运行审计请查看：[严格安全与 Agent 质量审计报告](./严格安全与Agent质量审计报告.md)

接入本地部署的 **EverOS（EverMemOS）** 长期记忆服务，让 AI 家教跨会话“记住”每个学生——批改历史、薄弱知识点和学习偏好都保存在本机 Docker 数据卷中。

---

## 演示视频

<video src="https://github.com/user-attachments/assets/eb99171f-246f-4a90-99e0-91e803813c4b" controls width="100%"></video>

---

## 功能概览

- **作业批改**：支持 JPG/PNG/WEBP/GIF/HEIC/HEIF，AI 自动识别题目、判断对错、给出解析和错误类型
- **知识点标注**：每道题自动标注涉及的知识点和难度等级，命中历史薄弱点即高亮提示
- **错题本**：只归档错题/部分正确题，完整原题截图、错误原因、教材页码、正确答案和解析直接展开
- **AI 问答**：多轮对话问难题，AI 会结合你的历史学情给出个性化讲解
- **学情追踪**：自动统计正确率、错误类型分布、各科薄弱知识点趋势
- **分用户长期记忆**：姓名唯一、首次登录自动建档，每位学生的本地资料和 EverMemOS 用户空间相互隔离
- **年级教材知识库**：70 本教材按年级/学期/科目匹配，Qwen 向量化后持久化到 Docker Qdrant

---

## 当前约束

- 不提供注册页：第一次用唯一姓名登录时自动注册；没有密码保护，姓名创建后不可修改。
- 只支持一年级至九年级，不支持高中课程。
- 作业识别、批改和错题解析固定使用 `.env` 配置的 GPT-5.5，不允许前端切换模型。
- 服务只监听 `127.0.0.1`，EverOS 和 Qdrant 也只映射本机回环地址。

---

## 部署指南

### 环境要求

- Miniconda（本机环境固定为 `/opt/miniconda3/envs/studybuddy`）
- Docker Desktop（用于 EverOS 长期记忆和 Qdrant 教材向量库）
- 支持图片输入的 GPT-5.5 OpenAI 兼容接口

### 第一步：克隆项目

```bash
git clone https://github.com/szlstart/StudyAgent-AI.git
cd StudyAgent-AI
```

### 第二步：安装依赖

```bash
/opt/miniconda3/bin/conda create -y -p /opt/miniconda3/envs/studybuddy python=3.13
/opt/miniconda3/bin/conda activate /opt/miniconda3/envs/studybuddy
python -m pip install -r requirements.txt
```

本项目不要创建 `.venv`。已存在环境时只需执行第二行激活命令。

### 第三步：配置环境变量

直接编辑项目根目录的 `.env`，至少填写以下必填项：

```bash
# 主 LLM（必须支持图片输入，用于 OCR 识别和批改）
LLM_BINDING=openai
LLM_MODEL=gpt-5.5
LLM_API_KEY=sk-xxx
LLM_HOST=https://api.openai.com/v1

# 千问教材 embedding（兼容 OpenAI embeddings 接口）
EMBEDDING_MODEL=qwen3.7-text-embedding
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_HOST=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_DIMENSION=1024

# 本地数据库服务
EVERMEMOS_BASE_URL=http://127.0.0.1:1995
QDRANT_URL=http://127.0.0.1:6333
```

`.env` 含真实密钥，已被 `.gitignore` 忽略；不要提交或分享。

### 第四步：启动本地数据库

```bash
docker compose up -d qdrant everos
docker compose ps
```

EverOS 仅映射到 `127.0.0.1:1995`，Qdrant 仅映射到 `127.0.0.1:6333`。数据分别持久化在 `studybuddy_everos_data` 与 `studybuddy_qdrant_data` Docker 数据卷中。它们是后端内部地址，不作为浏览器入口。

### 第五步：OCR 并索引教材

教材 PDF 因体积和版权原因不随公开代码仓库发布。请按照 `TextBook/README.md` 的目录与命名规则自行放入教材后再执行：

```bash
# 仅扫描版 PDF 走 macOS Apple Vision OCR，支持断点续跑
python scripts/ocr_textbooks.py

# 全部 70 本教材写入 Qdrant；已完成的教材自动跳过
python scripts/index_textbooks.py
```

### 第六步：启动 StudyAgent AI

```bash
./run.sh
```

也可以在激活 Conda 环境后执行 `python src/api/run_server.py`。默认只监听 `127.0.0.1:8001`，不接受局域网访问；浏览器统一使用 `localhost`，访问 `127.0.0.1` 会自动跳转。端口可在 `.env` 中通过 `BACKEND_PORT` 修改。

### 第七步：打开应用

浏览器访问：

```
http://localhost:8001/ui/
```

---

## EverMemOS：让 AI 真正记住每一个学生

### 为什么需要长期记忆？

普通 AI 没有记忆，每次对话都从零开始。使用 StudyAgent AI 一个月后，AI 依然不知道这个学生数学上的"因式分解"一直是薄弱点，也不记得上次讲解时学生在哪里卡住了。

[EverOS](https://github.com/EverMind-AI/EverOS) 是 EverMemOS 的本地开源实现。StudyAgent AI 每次批改后把学情写入本机 EverOS；下次 AI 开口前先做关键词和向量混合检索，再给出个性化讲解。

### 写入了哪些记忆？

| 触发时机 | 写入的内容 |
|---------|-----------|
| 每次作业批改完成 | 科目、题数、正确率和薄弱知识点 |
| 每道错题 | 完整题目、学生作答、正确答案、AI 针对本次作答自由判断的具体错误原因 |
| 检测到薄弱知识点 | 薄弱点列表，写为"复习提醒"记忆 |
| AI 对话达到轮次上限 | 本次会话摘要（学生问了什么 + AI 分析结论） |
| 登录或修改个人设置 | 姓名、年级/学期、讲解风格偏好；性别和 MBTI 只保存在本机，不发送到 EverOS |

### 记忆在哪里发挥作用？

**批改时（KnowPointAgent）**：注入学生历史薄弱知识点，自动标记哪些题命中了已知盲区（`is_weak_area=true`）。

**问答时（ExplainAgent）**：开口前先检索该学生的学情记忆，了解讲解风格偏好、近期薄弱点、常犯错误类型，再用个性化方式解答。

### 记忆类型说明

StudyAgent AI 写入对话型学情消息，EverOS 1.2.3 会提取并检索：

- **Episode（情节记忆）**：一次作业、错题或会话的上下文摘要
- **Atomic Fact（原子事实）**：成绩、薄弱点、学习偏好等可独立检索的事实
- **Profile（用户画像）**：随着记忆累积而聚合出的长期学习画像

EverOS 还保留 Foresight 能力，但 1.2.3 默认关闭该提取策略，因此本项目不把它作为当前可用能力宣传。

### 数据库与运维

`.env` 中保留以下配置，然后执行 `docker compose up -d everos`：

```bash
EVERMEMOS_BASE_URL=http://127.0.0.1:1995
EVERMEMOS_API_KEY=
EVERMEMOS_RETRIEVE_METHOD=hybrid
```

EverMemOS 的用户 ID 由登录用户 UUID 自动生成，不能在 `.env` 中配置为一个全局固定值，否则不同学生会串记忆。

本地 EverOS 没有内置鉴权，因此 Compose 端口必须保持为 `127.0.0.1:1995:8000`。查看健康状态：

```bash
curl http://127.0.0.1:1995/health
docker compose logs -f everos
```

备份和恢复整个长期记忆数据卷：

```bash
# 备份到当前目录 studybuddy-everos-backup.tar.gz
docker run --rm -v studybuddy_everos_data:/data:ro -v "$PWD":/backup \
  alpine tar czf /backup/studybuddy-everos-backup.tar.gz -C /data .

# 恢复前先停止 EverOS，再把备份解压回数据卷
docker compose stop everos
docker run --rm -v studybuddy_everos_data:/data -v "$PWD":/backup \
  alpine sh -c 'cd /data && tar xzf /backup/studybuddy-everos-backup.tar.gz'
docker compose up -d everos
```

---

## 技术架构

### 数据流

```
手机拍照
   │
   ▼
[前端 index.html]        Alpine.js 单文件 SPA，无需 npm / 编译
   │  POST /api/v1/homework/grade
   ▼
[FastAPI 后端]
   │
   ▼ 视觉 + RAG 流水线
   ├─ OcrAgent         GPT-5.5 看原图 → 识别文字、公式、图形和学生圈画
   ├─ Qdrant           按登录年级/学期/科目检索对应教材页
   ├─ GradeAgent       GPT-5.5 同时核对原图、识别结果与教材后逐题批改
   ├─ KnowPointAgent   标注知识点 + 难度等级
   └─ ExamTagAgent     分析试卷类型/来源
   │
   ├─→ WrongBookService     错题写入当前用户独立目录
   └─→ EverMemOSService     学情写入当前用户独立 EverOS 空间
   │
   ▼
返回 JSON → 前端渲染批改结果
```

### 项目结构

```
StudyAgent-AI/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app、中间件、路由注册
│   │   ├── run_server.py        # 启动入口（直接运行这个）
│   │   └── routers/             # HTTP 路由
│   │       ├── auth.py          # 姓名 + 年级 + 学期登录
│   │       ├── homework.py      # POST /api/v1/homework/grade
│   │       ├── explain.py       # POST /api/v1/explain
│   │       ├── wrong_book.py    # GET/POST /api/v1/wrong-book
│   │       ├── memory.py        # GET /api/v1/profile
│   │       └── textbook.py      # 当前年级/学期教材与索引状态
│   │
│   ├── agents/
│   │   ├── base_agent.py        # 所有 Agent 的父类（统一 LLM 调用、日志）
│   │   ├── homework/            # 批改流水线（OCR → Grade → KnowPoint → ExamTag）
│   │   ├── explain/             # 多轮问答 Agent（支持历史记忆注入）
│   │   └── memory/              # 用户画像更新 Agent
│   │
│   └── services/
│       ├── llm/                 # GPT-5.5 的 OpenAI 兼容调用封装
│       ├── evermemos/           # EverMemOS 客户端 + 业务逻辑层
│       ├── textbook_vector_store.py # Qdrant 教材向量读写
│       ├── rag/                 # RAG 组件
│       └── embedding/           # 向量化服务
│
├── web/
│   └── index.html               # 整个前端（Alpine.js 单文件 SPA）
│
├── config/
│   ├── main.yaml                # 全局配置
│   └── agents.yaml              # 各 Agent 的 LLM 参数
│
├── TextBook/                    # 一年级～九年级/学科_年级_上册或下册.pdf
├── data/                        # SQLite、分用户资料、OCR 缓存和索引清单
├── requirements.txt
└── .env                         # 你的真实配置（已 gitignore，勿提交）
```

### 模型链路

主模型固定读取 `.env` 中的 `LLM_MODEL=gpt-5.5`；教材向量由 `qwen3.7-text-embedding` 生成 1024 维向量并写入 Qdrant。前端没有模型选择器，避免视觉批改被切换到不兼容模型。

---

## API 文档

启动后访问 Swagger UI：`http://localhost:8001/docs`

主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/auth/login` | 姓名 + 年级 + 学期登录，首次自动建档 |
| `PATCH` | `/api/v1/auth/profile` | 修改年级与学期（姓名不可修改） |
| `POST` | `/api/v1/homework/grade` | 上传作业图片，返回批改结果 |
| `POST` | `/api/v1/explain` | 文字 AI 问答 |
| `POST` | `/api/v1/explain/with-image` | 图片题目视觉解析 |
| `POST` | `/api/v1/explain/extract-from-file` | 提取图片或 PDF 内容（PDF 问答预处理） |
| `GET` | `/api/v1/wrong-book` | 获取当前用户错题列表 |
| `PUT` | `/api/v1/wrong-book/{entry_id}/mastered` | 标记或取消掌握错题 |
| `GET` | `/api/v1/profile` | 获取用户学情画像 |
| `PATCH` | `/api/v1/profile/preferences` | 修改讲解风格等学习偏好 |
| `GET` | `/api/v1/textbook/status` | 获取当前学期教材及索引状态 |
| `POST` | `/api/v1/textbook/reindex/{subject}` | 重建当前科目教材索引 |

---

## PWA

应用仍保留 PWA 配置，但当前服务仅允许本机访问，不提供手机局域网安装入口。

---
