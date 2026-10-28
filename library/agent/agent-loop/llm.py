"""
llm.py — 三种大模型后端, 对 Agent 循环暴露同一个接口:

    reply = llm.chat(messages, tools)      # tools 为 None 时不提供工具 (讲义案例 ①)
    reply = llm.chat(messages, tools, require_tool=True)   # 强制这一轮必须调用工具 (讲义第 6 节)
    reply.content      模型说的话 (str)
    reply.tool_calls   [{"id", "name", "arguments"}]; arguments 是模型给出的原始 JSON 字符串, 由安全层去解析
    reply.usage        {"prompt_tokens", "completion_tokens"}
    reply.latency_s    这一次调用花了多少秒

  LocalLLM   本地 transformers 跑 Qwen2.5-Instruct (不需要 API Key, 数字可复现; 讲义的主数据)
  APILLM     云端 OpenAI 兼容接口 (DeepSeek / 通义千问 / OpenAI), 复用 templates/llm-client 的配置
  ScriptedLLM 按剧本依次返回预先写好的回复 (测试与练习用, 不需要真模型)

消息格式统一用 OpenAI 的格式 (assistant 消息带 tool_calls, 工具结果是 role="tool" 的消息),
本地模型在 LocalLLM 内部转成 Qwen 聊天模板认的格式。
"""

import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # library/agent/agent-loop → 仓库根


class Reply:
    def __init__(self, content="", tool_calls=None, usage=None, latency_s=0.0, raw=""):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0}
        self.latency_s = latency_s
        self.raw = raw                       # 模型的原始输出 (调试用)


# ─────────────────────────────── 剧本 ───────────────────────────────


class ScriptedLLM:
    """
    按顺序返回预先写好的回复。每条回复是 dict: {"content": str, "tool_calls": [(name, arguments), ...]}。
    arguments 可以是 dict 或 JSON 字符串。用来在没有真模型的情况下测试 Agent 循环。
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []                      # 每次被调用时收到的 messages (测试可检查)
        self.required = []                   # 每次调用时是否要求强制调用工具

    def chat(self, messages, tools=None, require_tool=False):
        self.calls.append([dict(m) for m in messages])
        self.required.append(require_tool)
        if not self.script:
            return Reply(content="(剧本用完了)")
        step = self.script.pop(0)
        calls = []
        for i, (name, args) in enumerate(step.get("tool_calls", [])):
            calls.append({"id": f"call_{len(self.calls)}_{i}", "name": name,
                          "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)})
        return Reply(content=step.get("content", ""), tool_calls=calls)


# ─────────────────────────────── 本地 ───────────────────────────────

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_qwen_output(text):
    """
    从 Qwen2.5 的输出里取出工具调用。它的聊天模板约定的格式是:
        <tool_call>
        {"name": "walk", "arguments": {"distance_m": 0.5}}
        </tool_call>
    返回 (content, tool_calls)。JSON 写坏的工具调用不丢掉, 记成 name="__invalid__", 让安全层报错给模型。
    """
    calls = []
    for i, block in enumerate(TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(block)
            calls.append({"id": f"local_{i}", "name": str(obj.get("name")),
                          "arguments": json.dumps(obj.get("arguments", {}), ensure_ascii=False)})
        except (json.JSONDecodeError, AttributeError):
            calls.append({"id": f"local_{i}", "name": "__invalid__", "arguments": block})
    content = TOOL_CALL_RE.sub("", text).strip()
    # 小模型有时把工具调用直接写成一行 JSON, 忘了包 <tool_call> 标签 (本仓库实测 Qwen2.5-1.5B 出现过)。
    # 如果不处理, 这次调用会被当成普通文字悄悄丢掉: 模型以为自己 look 过了, 其实没有。
    # 这里把它也认出来, 但标记 bare=True, 评测时单独统计 (讲义第 4 节)。
    for m in BARE_CALL_RE.finditer(content):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and "arguments" in obj:
            calls.append({"id": f"local_bare_{len(calls)}", "name": obj["name"],
                          "arguments": json.dumps(obj.get("arguments") or {}, ensure_ascii=False), "bare": True})
            content = content.replace(m.group(0), "").strip()
    return content, calls


# 一行内的 {"name": ..., "arguments": {...}} (arguments 里最多一层花括号, 工具参数都是扁平的)
BARE_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}')


class LocalLLM:
    """
    本地 transformers 推理。默认贪心解码 (do_sample=False), 同样的输入得到同样的输出, 评测可复现。
    注意模型目录里 generation_config.json 的设置会悄悄生效: Qwen2.5 自带 repetition_penalty=1.1, 贪心时也照样执行
    (知识库 llm/decoding 讲义第 6.3 节)。gen_kwargs 可以覆盖, 例如 {"repetition_penalty": 1.0} 或 {"do_sample": True};
    采样时用 seed 固定随机数, 结果可复现。
    model_id 例: "Qwen/Qwen2.5-1.5B-Instruct"; 权重从 Hugging Face 缓存加载
    (国内网络可设置 HF_ENDPOINT=https://hf-mirror.com 先下载)。
    """

    def __init__(self, model_id, device=None, dtype="auto", max_new_tokens=256, gen_kwargs=None, seed=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto" and self.device == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = torch.float16            # 老显卡 (例如 Colab 免费的 T4) 不支持 bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.gen_kwargs = {"do_sample": False, **(gen_kwargs or {})}
        if seed is not None:
            torch.manual_seed(seed)

    @staticmethod
    def _to_template(messages):
        """OpenAI 格式 → Qwen 模板格式: assistant 的 tool_calls 里 arguments 要是 dict。"""
        out = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    f = c["function"]
                    try:
                        args = json.loads(f["arguments"]) if isinstance(f["arguments"], str) else f["arguments"]
                    except json.JSONDecodeError:
                        args = {"_raw": f["arguments"]}
                    calls.append({"type": "function", "function": {"name": f["name"], "arguments": args}})
                m["tool_calls"] = calls
            out.append(m)
        return out

    def chat(self, messages, tools=None, require_tool=False):
        prompt = self.tok.apply_chat_template(self._to_template(messages), tools=tools,
                                              add_generation_prompt=True, tokenize=False)
        # 强制调用工具: 替模型把回复的开头写好 "<tool_call>" + 换行, 它只能接着写工具调用的 JSON。
        # 云端 API 里对应的是 tool_choice="required"。
        prefix = "<tool_call>\n" if (require_tool and tools) else ""
        prompt += prefix
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tok.eos_token_id, **self.gen_kwargs)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        new = out[0, ids["input_ids"].shape[1]:]
        text = prefix + self.tok.decode(new, skip_special_tokens=True)
        content, calls = parse_qwen_output(text)
        usage = {"prompt_tokens": int(ids["input_ids"].shape[1]), "completion_tokens": int(new.shape[0])}
        return Reply(content=content, tool_calls=calls, usage=usage, latency_s=latency, raw=text)


# ─────────────────────────────── 云端 ───────────────────────────────


class APILLM:
    """
    OpenAI 兼容接口 (DeepSeek / 通义千问 / OpenAI)。
    配置复用 templates/llm-client: 在仓库根目录放一个 .env (照 templates/llm-client/.env.example 填 Key),
    LLM_PROVIDER 选供应商。temperature=0, 但云端模型不保证完全确定, 评测时多跑几次取平均。
    """

    def __init__(self, provider=None, model=None):
        import sys
        sys.path.insert(0, os.path.join(ROOT, "templates", "llm-client"))
        from llm_client import LLMClient, LLMConfig
        cfg = LLMConfig(provider=provider) if provider else LLMConfig()
        if model:
            cfg.model = model
        self.client = LLMClient(cfg)
        self.model_id = f"{cfg.provider}:{cfg.model}"

    def chat(self, messages, tools=None, require_tool=False):
        kwargs = {"model": self.client.config.model, "messages": messages, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
            if require_tool:
                kwargs["tool_choice"] = "required"
        t0 = time.perf_counter()
        resp = self.client._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        msg = resp.choices[0].message
        calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"}
                 for c in (msg.tool_calls or [])]
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return Reply(content=msg.content or "", tool_calls=calls, usage=usage, latency_s=latency)
