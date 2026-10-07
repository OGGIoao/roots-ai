# 🤖 LLM 客户端模板工具箱

> **一句话**：一套零配置、可切换、拿来即用的 LLM API 客户端。改 `.env` 一行切换模型，业务代码零改动。

---

## 一览

```
templates/llm-client/
├── README.md           ← 使用说明（你在这里）
├── .env.example        ← 环境变量模板（复制为 .env 后填入密钥）
└── llm_client.py       ← LLM 客户端核心代码（开箱即用，可单独复制）
```

> 依赖（openai + python-dotenv）在本仓库内统一由根目录 `requirements-mlc.txt` 声明，
> 本目录不单独维护 requirements 文件，避免版本漂移。
> 若把模板**复制到别的项目**（见下方快速开始），则按第 2 步自行安装这两个包。

| 关键特性 | 说明 |
|---------|------|
| **供应商** | DeepSeek · Qwen · OpenAI · Ollama |
| **切换成本** | 改 `.env` 一行 `LLM_PROVIDER` |
| **结构化输出** | `chat_json()` 自动解析 + 容错 |
| **多轮对话** | `chat_with_messages()` 直接传历史 |
| **标准化错误** | 限流 / 超时 / API 异常分别捕获 |

---

## 为什么要这个模板

你每次写 AI 应用项目，都要重复做这几件事：

| 问题 | 模板解决 |
|------|---------|
| **对接 API 的样板代码** | 封装了 OpenAI 兼容接口，开箱即用 |
| **每次切换模型都要改代码** | 配置驱动：改 `LLM_PROVIDER` 即可，代码不动 |
| **API Key 管理混乱** | `.env` + `python-dotenv`，环境变量隔离 |
| **不知道什么场景该用哪个模型** | 内置 DeepSeek / Qwen / OpenAI / Ollama 四家配置，附选型建议 |
| **JSON mode 容易踩坑** | `chat_json()` 自动解析 + 容错处理 |
| **多轮 / Agent 场景没有现成入口** | `chat_with_messages()` 直接传历史 |

---

## 快速上手

```bash
# 1. 把模板复制到你的项目
cp -r templates/llm-client/ your-project/src/llm/

# 2. 安装依赖
pip install openai python-dotenv

# 3. 配置 API Key（将 .env.example 复制为 .env 并填入密钥）
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 4. 测试连接
python llm_client.py
# ✅ LLM 连接成功！
#    模型: deepseek-chat
#    回复: 我是一个由深度求索公司开发的AI助手...
```

---

## 在项目中使用

```python
from llm_client import LLMClient

# 零配置初始化（自动从 .env 读取配置）
client = LLMClient()

# ─── 简单问答 ───
reply = client.chat("用一句话解释量子计算")

# ─── 带系统提示词的分类任务 ───
category = client.chat(
    "这个用户评论是正面还是负面：「产品质量太差了」",
    system_prompt="你是一个情感分析师。只回复「正面」或「负面」。",
    temperature=0.0,       # 确定性任务：temperature=0
)

# ─── 结构化输出 ───
info = client.chat_json(
    "从以下邮件中提取：日期、金额、客户姓名：\n\n...",
    system_prompt="你是一个信息提取助手。输出 JSON 格式。",
)

# ─── 多轮 / Agent 对话 ───
messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "我的名字是张三。"},
    {"role": "assistant", "content": "你好张三！有什么可以帮你的？"},
    {"role": "user", "content": "我刚才自我介绍了吗？"},
]
reply = client.chat_with_messages(messages)
```

---

## 供应商选型指南

| 供应商 | 推荐场景 | 强度 | 成本 |
|--------|---------|------|------|
| **DeepSeek** | 日常开发、Agent 推理 | 推理强，中文好 | 💰 极低 |
| **Qwen** | 中文场景、信息提取 | 中文 SOTA，函数调用好 | 💰 低 |
| **OpenAI (GPT-4o)** | 复杂推理、代码生成 | 综合最强 | 💰💰💰 |
| **Ollama** | 本地部署、隐私敏感 | 取决于本地模型 | 免费 |

**切换方法**：改 `.env` 中一行 `LLM_PROVIDER=deepseek` → `LLM_PROVIDER=qwen`，重新运行，零代码改动。

---

## 扩展新供应商

在 `PROVIDER_CONFIGS` 中增加一条即可：

```python
PROVIDER_CONFIGS["siliconflow"] = {
    "api_key_env": "SILICONFLOW_API_KEY",
    "base_url_env": "SILICONFLOW_BASE_URL",
    "default_model": "Qwen/Qwen2.5-7B-Instruct",
}
```

然后在 `.env` 中添加对应的 API Key 和 Base URL。无需修改任何业务代码。

---

## 项目迁移清单

将模板应用到已有项目时，建议同时做以下工作：

- [ ] 复制 `llm_client.py` 到项目 `src/` 或 `app/` 下
- [ ] 复制 `.env.example` → `.env`，填入真实 API Key
- [ ] 将 `.env` 添加到 `.gitignore`
- [ ] 在入口文件（`main.py`/`app.py`）初始化 `LLMClient` 单例
- [ ] 所有 LLM 调用改为 `client.chat()` / `client.chat_json()` / `client.chat_with_messages()`
- [ ] 在 CI 中增加 `python llm_client.py` 作为健康检查步骤
