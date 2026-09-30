"""
LLM API 客户端 — 统一调用 DeepSeek / Qwen / OpenAI / Ollama
=============================================================

本文件是 AI 应用的 LLM 基础设施层，
职责是封装不同 LLM 供应商的 API 差异，提供统一的调用接口。

支持：
- 多 provider 自动切换（环境变量 LLM_PROVIDER 控制，改 .env 即可切换）
- Chat Completion（单轮 / 多轮对话）
- 结构化输出（JSON mode，用于意图分类、信息提取等场景）
- 错误分类处理（限流 / 超时 / API 异常各自抛出）
- 本地模型接入（Ollama / vLLM 等兼容 OpenAI 接口的方案）

用法示例：
    from llm_client import LLMClient
    client = LLMClient()
    reply = client.chat("你好，请介绍一下自己")

设计理念：
    LLMClient 是整个 AI 应用的"通信基座"，
    所有需要 LLM 的地方 —— 问答、信息提取、意图分类、Agent 推理 ——
    都通过这一个客户端完成，而非每个模块各自对接 API。

    这样做的收益：
      1. 切换模型不碰业务代码（改 .env 里一行的成本）
      2. 统一的错误处理和日志策略
      3. 新项目拿来就用的"零配置"体验
"""

# ===================================================================
# 标准库导入
# ===================================================================
import os          # 读取环境变量（API Key、模型选择等）
import json        # 解析 LLM 返回的 JSON 结构化输出
import logging     # 记录运行日志，方便排查问题
from typing import Optional    # 类型提示：表示参数可传 None
from dataclasses import dataclass, field   # 数据类，用来简洁地定义配置结构体

# ===================================================================
# 第三方库导入
# ===================================================================
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
#   ^^^^^^  OpenAI Python SDK ✦
#   虽然是 OpenAI 的库，但由于所有主流 LLM API 都兼容 OpenAI 的接口格式，
#   所以用它来统一调用 DeepSeek / Qwen / OpenAI，只需换 base_url 和 api_key。
#
#   OpenAI       — 核心客户端，发送 chat completion 请求
#   APIError     — 通用的 API 错误（网络问题、服务端异常等）
#   RateLimitError — 请求太频繁被限流
#   APITimeoutError — 请求超时

from dotenv import load_dotenv
import sys

# Windows 控制台默认 GBK, 无法编码脚本结尾处的 ✅/❌ 等符号,
# print() 会抛 UnicodeEncodeError 让脚本在跑通后异常退出。
for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r:
        try:
            _r(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

#   ^^^^^^^^ 从 .env 文件加载环境变量 ✦
#   开发时把 API Key 写在 .env 里，不上传到 Git

# ===================================================================
# 日志配置
# ===================================================================
# 格式说明：时间 | 级别(左对齐5字符) | 模块名 | 消息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
)
log = logging.getLogger("llm_client")   # 获取当前模块的日志器

# ===================================================================
# 供应商配置表
# ===================================================================
# 加载 .env 文件（把文件中的 KEY=VALUE 读入环境变量）
load_dotenv()

# 这张表定义了各 LLM 供应商的接入信息：
#   api_key_env  — .env 中哪个变量存着 API Key
#   base_url_env — .env 中哪个变量存着 API 地址
#   default_model — 默认使用的模型名
#
# ⚡ 切换供应商只需改 .env 里的 LLM_PROVIDER=deepseek|qwen|openai|ollama
#
# ✨ 模板使用提示：在 PROVIDER_CONFIGS 中新增条目即可扩展支持新的供应商，
#    无需修改任何业务代码。典型的扩展对象包括：
#    - 硅基流动 / Groq 等第三方代理服务
#    - 自部署 vLLM / Triton 推理端点
#    - 企业内部专有模型网关
PROVIDER_CONFIGS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_model": "deepseek-chat",          # DeepSeek-V3
    },
    "qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "default_model": "qwen-plus",              # 通义千问-Plus
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_model": "gpt-4o-mini",            # 性价比最高的 GPT-4o 变体
    },
    "ollama": {
        "api_key_env": "OLLAMA_API_KEY",
        "base_url_env": "OLLAMA_BASE_URL",
        "default_model": "llama3",                 # 或 qwen2.5 / mistral 等
    },
}


# ===================================================================
# LLMConfig — 配置数据类
# ===================================================================
# @dataclass 是 Python 的"数据类"装饰器：
# 自动生成 __init__、__repr__、__eq__ 等方法，省去手写模板代码。
# field(default_factory=...) 表示字段的默认值通过调用一个函数来生成。
@dataclass
class LLMConfig:
    """
    LLM 客户端配置，从环境变量自动加载。

    工作流程：
    1. 从 .env 读取 LLM_PROVIDER 确定用哪个供应商
    2. 根据供应商名称查找 PROVIDER_CONFIGS 表
    3. 从 .env 读取对应的 API Key 和 Base URL
    4. 校验必填项是否齐全
    """

    # — 供应商选择（默认 deepseek）———————————————
    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek")
    )

    # — API 凭证（可手动传入，否则从 .env 自动读）—————
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # — 模型名（可手动传入，否则读 .env → 回退到供应商默认）—
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat")
    )

    # — 网络与重试 ———————————————————————————————
    timeout: int = 60          # 单次请求超时（秒）
    max_retries: int = 2       # 失败后最多重试次数

    # ------------------------------------------------------------------
    # __post_init__ ：@dataclass 的初始化后钩子
    # 普通 __init__ 执行完后自动调用，适合做字段的补充验证和自动填充
    # ------------------------------------------------------------------
    def __post_init__(self):
        # 1) 检查供应商是否在支持列表中
        provider = self.provider.lower()
        cfg = PROVIDER_CONFIGS.get(provider)
        if cfg is None:
            raise ValueError(
                f"不支持的 LLM 提供商: {provider}，可选: {list(PROVIDER_CONFIGS.keys())}"
            )

        # 2) 自动补全 API Key 和 Base URL（优先用外部传进来的值）
        if not self.api_key:
            self.api_key = os.getenv(cfg["api_key_env"])
        if not self.base_url:
            self.base_url = os.getenv(cfg["base_url_env"])

        # 3) 模型选择优先级：
        #    手动传入 > 环境变量 LLM_MODEL > 该供应商的 default_model
        self.model = os.getenv("LLM_MODEL") or cfg["default_model"]

        # 4) 最终校验：API Key 是必填项
        if not self.api_key:
            raise ValueError(
                f"缺少 {provider} 的 API Key。请设置环境变量 {cfg['api_key_env']} "
                f"或在 .env 文件中配置。"
            )


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

# ===================================================================
# LLMClient — 统一 LLM API 客户端
# ===================================================================
# 核心职责：
#   1. 接收上层传来的消息（字符串或消息列表）
#   2. 组装请求参数，调用 OpenAI 兼容接口
#   3. 返回文本或解析后的 JSON
#
# 典型使用链路：
#   用户输入 → 业务模块（如 Agent / RAG / 分类器）→ LLMClient → LLM API → 响应
#   LLMClient 不关心上层业务逻辑，只做一件事：把消息送进去，把回复取出来。
class LLMClient:
    """统一 LLM API 客户端。"""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化客户端。

        参数：
            config — 可传入自定义 LLMConfig，不传则自动从 .env 读取。

        内部创建了一个 OpenAI 客户端实例（self._client），
        后续所有请求都由它发出。
        """
        self.config = config or LLMConfig()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
        log.info(
            "LLMClient 初始化: provider=%s model=%s base_url=%s",
            self.config.provider,
            self.config.model,
            self.config.base_url,
        )

    # ================================================================
    # 对外公开方法
    # ================================================================

    # ------------------------------------------------------------------
    # chat() — 单轮对话（最常用的入口）
    # ------------------------------------------------------------------
    # 参数说明：
    #   message      — 用户问题（字符串）
    #   system_prompt — 系统提示词，设定 AI 的角色和行为规则
    #   temperature  — 生成随机性，0=确定性最高，1=创造力最高
    #                 对于分类/提取等任务通常用 0.0~0.3（比较稳定）
    #                 对于创意/写作等任务可以用 0.7~1.0（发挥创造力）
    #   max_tokens   — 最大输出 token 数（防止 AI 无限生成）
    #   json_mode    — 是否要求 LLM 输出 JSON 格式
    #
    # 返回值：LLM 回复的文本（字符串）
    # ------------------------------------------------------------------
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """单轮对话：一条 user 消息 + 可选的 system 消息 → 回复文本。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return self._chat_completion(messages, temperature, max_tokens, json_mode)

    # ------------------------------------------------------------------
    # chat_with_messages() — 多轮对话
    # ------------------------------------------------------------------
    # 适用于 Agent 场景：需要把对话历史传给 LLM 做上下文理解。
    # messages 格式示例：
    #   [
    #       {"role": "system", "content": "..."},
    #       {"role": "user", "content": "..."},
    #       {"role": "assistant", "content": "..."},  ← 之前的回复
    #       {"role": "user", "content": "最新的问题"},
    #   ]
    # ------------------------------------------------------------------
    def chat_with_messages(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """多轮对话：直接传入完整的消息列表。"""
        return self._chat_completion(messages, temperature, max_tokens, json_mode)

    # ================================================================
    # 内部核心方法
    # ================================================================

    def _chat_completion(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """
        所有对话最终都汇聚到这里，完成一次 LLM API 调用。

        步骤：
          1. 组装请求参数字典（model / messages / temperature / max_tokens）
          2. 如果启用 json_mode，添加 response_format 参数
          3. 发送请求，捕获三类常见异常
          4. 提取回复文本，去首尾空白后返回
        """
        # — 请求参数准备 ———————————————————————————————
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # JSON mode 下，要求 LLM 的输出必须是合法的 JSON 对象
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # — 发送请求 & 异常处理 —————————————————————————
        # 使用了 OpenAI SDK 内置的重试机制（max_retries=2），
        # 对偶发的网络抖动有自动恢复能力
        try:
            resp = self._client.chat.completions.create(**kwargs)
            # 响应结构：choices[0].message.content 就是模型回复
            content = resp.choices[0].message.content
            if content is None:
                log.warning("LLM 返回了空内容")
                return ""
            return content.strip()
        except RateLimitError as e:
            # API 限流：通常是因为请求频率过高
            # 应对策略：加入指数退避等待（后续版本可增加此逻辑）
            log.error("API 限流: %s", e)
            raise
        except APITimeoutError as e:
            # 请求超时：网络问题或模型生成过慢
            log.error("API 超时: %s", e)
            raise
        except APIError as e:
            # 其他 API 错误：如认证失败、参数非法等
            log.error("API 错误: %s", e)
            raise

    # ================================================================
    # 结构化输出（JSON mode）
    # ================================================================

    # ------------------------------------------------------------------
    # chat_json() — 让 LLM 返回结构化的 JSON 数据
    # ------------------------------------------------------------------
    # 典型使用场景：
    #   - 文本分类："判断以下评论的情感倾向（positive / negative / neutral）"
    #   - 信息提取："从以下文档中提取：日期、金额、对方账户"
    #   - 意图识别："判断用户的请求属于哪一个意图类别"
    #
    # 系统会自动将 LLM 的文本回复解析为 Python dict，
    # 解析失败时返回包含错误信息的 dict，不会直接崩溃。
    #
    # 注意：不是所有模型都支持 strict JSON mode。
    # DeepSeek / GPT-4o 支持良好，部分小模型可能不稳定。
    # ------------------------------------------------------------------
    def chat_json(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict:
        """返回 JSON 格式的响应。"""
        content = self.chat(
            message=message,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            log.error("JSON 解析失败，原始内容: %s", content[:200])
            # 返回一个容错结构，调用方可以检查是否有 "error" 键
            return {"error": f"JSON parse failed: {e}", "raw": content[:500]}


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

# ===================================================================
# 快速测试入口
# ===================================================================
# 直接运行本文件即可测试 LLM API 是否连通。
#
# 命令行：
#   python llm_client.py
#
# 预期输出：
#   ✅ LLM 连接成功！
#      模型: deepseek-chat
#      回复: 我是一个由深度求索公司开发的AI助手...
#
# 如果失败，检查 .env 中的 API Key 是否正确。
# ===================================================================

def test_connection():
    """
    测试 LLM API 连接是否正常。

    用途：
      1. 开发期验证 API Key 和网络连通性
      2. CI/CD 中作为健康检查
      3. 快速排查配置问题（Key 过期、余额不足、网络不通等）

    返回 True / False，同时打印详细信息到控制台。
    """
    try:
        client = LLMClient()
        # 用最轻量的请求测试连接：temperature 设低确保稳定，max_tokens 设小节省额度
        reply = client.chat(
            "请用一句话介绍你自己。",
            system_prompt="你是一个有帮助的 AI 助手。请简洁回答。",
            temperature=0.3,
            max_tokens=100,
        )
        print(f"✅ LLM 连接成功！")
        print(f"   模型: {client.config.model}")
        print(f"   回复: {reply}")
        return True
    except Exception as e:
        print(f"❌ LLM 连接失败: {e}")
        print(f"   请检查 .env 文件中的 API Key 配置是否正确。")
        return False


if __name__ == "__main__":
    """
    直接运行本文件时执行连接测试。
    快速验证 API Key 和网络是否正常。
    """
    test_connection()
