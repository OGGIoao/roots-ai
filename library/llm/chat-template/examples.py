"""
examples.py — 本单元贯穿始终的例子: 一个通用的小助手和它的三个工具 (不依赖任何项目)

讲义、实验、练习都用这里的对话和工具, 换成任何应用 (问答、客服、机器人) 道理都一样。
项目里的用法 (例如小车的工具) 只在讲义最后的 "实践" 一节出现。
"""

# 三个工具的说明书 (JSON Schema, OpenAI 格式): 查天气、算数、查资料
GET_WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "description": "查询某个城市现在的天气",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名, 例如 台北"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}},
        "required": ["city"]}}}

CALCULATOR = {"type": "function", "function": {
    "name": "calculator",
    "description": "计算一个算术表达式",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "算术表达式, 例如 3*(4+5)"}},
        "required": ["expression"]}}}

SEARCH_DOCS = {"type": "function", "function": {
    "name": "search_docs",
    "description": "在资料库里搜索和问题相关的段落",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "搜索词"},
        "top_k": {"type": "integer", "description": "返回几段, 默认 3"}},
        "required": ["query"]}}}

TOOLS = [GET_WEATHER, CALCULATOR, SEARCH_DOCS]

SYSTEM = "你是一个乐于助人的助手。需要实时信息或精确计算时, 调用工具。"

# 一段完整的对话: 用户提问 → 模型调用工具 → 工具返回 → 模型回答
CONVERSATION = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "台北现在几度?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "台北"}}}]},
    {"role": "tool", "content": '{"city": "台北", "temperature": 26, "condition": "多云"}'},
    {"role": "assistant", "content": "台北现在 26 度, 多云。"},
]

# 只有两条消息的最小例子 (讲义第 3 节手算用)
TINY = [
    {"role": "system", "content": "你是助手。"},
    {"role": "user", "content": "你好"},
]
