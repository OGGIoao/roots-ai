"""M1 Agent原子循环：手写最小Agent，不用LangChain。
底层解析：Agent = Prompt + Context + Tool + Loop
对比Transformer attention：Agent的"上下文窗口"不是attention矩阵，而是Memory/History管理。
"""
import json
from typing import List, Dict, Callable, Optional

class AgentLoop:
    """最小Agent循环：手写，不依赖任何框架。"""
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
        self.history: List[Dict] = []

    def run(self, prompt: str, max_steps: int = 3) -> str:
        # 1. 初始化上下文（类似Transformer的input_ids，但这里是文本历史）
        self.history = [{"role": "user", "content": prompt}]
        for step in range(max_steps):
            # 2. 生成（这里简化为直接处理，真实场景接LLM API）
            # 3. 工具调用判断：解析JSON，决定是否调用Tool
            tool_call = self._parse_tool_call(prompt)
            if tool_call:
                result = self.tools[tool_call["name"]](**tool_call["args"])
                self.history.append({"role": "tool", "content": str(result)})
                prompt = f"Tool result: {result}"
            else:
                # 直接输出
                return f"Agent response (step {step}): {prompt}"
        return "Max steps reached"

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        # 底层：手写JSON解析，不用框架
        try:
            if "{" in text and "}" in text:
                return json.loads(text[text.find("{"):text.rfind("}")+1])
        except Exception:
            pass
        return None

# 工具示例：计算 + 查询（模拟业务链路）
def calc(x: int, y: int) -> int:
    return x + y

def query_db(key: str) -> str:
    return f"DB result for {key}"

if __name__ == "__main__":
    agent = AgentLoop({"calc": calc, "query_db": query_db})

    # 测试：工具调用链路（解析 JSON → 执行工具 → 写回 History）
    # 注意 prompt 必须是 AgentLoop 能解析的 JSON 形式，
    # 自然语言（"Calculate 3+5"）不会被 _parse_tool_call 识别，会直接原样回显。
    result = agent.run('{"name": "calc", "args": {"x": 3, "y": 5}}')
    print(result)
    print("History:", agent.history)
