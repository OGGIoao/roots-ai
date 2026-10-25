"""
tools.py — 小车的工具定义 (给大模型看的说明书) 与安全层 (大模型说什么都要先过这一关)

工具接口和 PRODUCT.md 第 4 节一致: walk / turn / stop / look / get_pose。

两部分:
  1. TOOL_SCHEMAS: OpenAI 风格的 JSON Schema。大模型靠它知道有哪些工具、参数叫什么、单位是什么。
     本地模型 (Qwen2.5 的聊天模板) 和云端 API (DeepSeek / 通义千问) 都认这个格式。
  2. validate_call(): 安全层。大模型会出错: 工具名拼错、少参数、把 "1米" 写成 "1m"、要求一口气走 10 米……
     这些都在这里拦下, **不执行**, 把错误原因作为观测返回给大模型, 让它改了再试 (讲义第 4 节)。
     注意分工: 速度上限是小车"固件"里的 (sim/car2d.py 的 max_speed_mps, 上层越不过去);
     这里管的是 Agent 层的策略, 比如"一次最多走 1 米"——走太远时误差和碰撞风险都变大, 要求它走一段看一眼。
"""

import json

MAX_WALK_M = 1.0        # 一次 walk 的最大距离 (米)
MAX_TURN_DEG = 180.0    # 一次 turn 的最大角度 (度)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "walk",
        "description": f"沿车头方向直走一段距离, 负数表示后退。一次最多走 {MAX_WALK_M} 米。撞到东西会自动停下并报告。",
        "parameters": {"type": "object", "properties": {
            "distance_m": {"type": "number", "description": f"距离, 单位米, 范围 -{MAX_WALK_M} 到 {MAX_WALK_M}"},
            "speed": {"type": "number", "description": "速度, 单位米/秒, 可选, 默认 0.3, 最大 0.5"}},
            "required": ["distance_m"]}}},
    {"type": "function", "function": {
        "name": "turn",
        "description": "原地转向。正数左转 (逆时针), 负数右转 (顺时针)。",
        "parameters": {"type": "object", "properties": {
            "angle_deg": {"type": "number", "description": f"角度, 单位度, 范围 -{MAX_TURN_DEG:g} 到 {MAX_TURN_DEG:g}"}},
            "required": ["angle_deg"]}}},
    {"type": "function", "function": {
        "name": "stop",
        "description": "立即停车。任务完成或遇到危险时调用。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "look",
        "description": "用摄像头看一眼正前方 (视场左右各 30 度, 最远 2 米)。返回看到的物体: 名称、距离 (米, 车身到物体表面)、方位角 (度, 左正右负)。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_pose",
        "description": "读取小车当前位置和朝向的估计值 (由轮子转了多少推算, 会有累积误差)。",
        "parameters": {"type": "object", "properties": {}}}},
]

TOOL_NAMES = [s["function"]["name"] for s in TOOL_SCHEMAS]


def _number(v):
    """把参数转成 float; 大模型偶尔把数字写成字符串 ("0.5"), 能无歧义地转就转, 带单位的 ("0.5m") 不猜。"""
    if isinstance(v, bool):
        raise ValueError("不能是布尔值")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return float(v.strip())          # "0.5m" 会在这里抛 ValueError
    raise ValueError(f"应为数字, 收到 {type(v).__name__}")


def validate_call(name, args):
    """
    检查一次工具调用。返回 (ok, 结果):
      ok=True  → 结果是清洗后的参数 dict, 可以交给小车执行;
      ok=False → 结果是一句给大模型看的错误说明, 不执行。
    args 可以是 dict, 也可以是 JSON 字符串 (云端 API 返回的 arguments 就是字符串)。
    """
    if name not in TOOL_NAMES:
        return False, f"没有叫 {name!r} 的工具, 可用的工具: {', '.join(TOOL_NAMES)}"
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            return False, f"参数不是合法的 JSON: {e.msg}"
    if not isinstance(args, dict):
        return False, "参数必须是一个 JSON 对象"
    schema = next(s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == name)
    allowed = set(schema["properties"])
    extra = set(args) - allowed
    if extra:
        return False, f"{name} 不接受参数 {sorted(extra)}, 只接受 {sorted(allowed) or '无参数'}"
    for req in schema.get("required", []):
        if req not in args:
            return False, f"{name} 缺少必需的参数 {req!r}"
    clean = {}
    try:
        for k, v in args.items():
            clean[k] = _number(v)
    except ValueError as e:
        return False, f"{name} 的参数有误: {e}。请只给数字, 不要带单位"
    if name == "walk":
        d = clean["distance_m"]
        if abs(d) > MAX_WALK_M:
            return False, f"一次最多走 {MAX_WALK_M} 米, 你要求走 {d:g} 米。请分几段走, 每走一段先 look 确认"
        if "speed" in clean and clean["speed"] <= 0:
            return False, "speed 必须为正数"
    if name == "turn" and abs(clean["angle_deg"]) > MAX_TURN_DEG:
        return False, f"一次最多转 {MAX_TURN_DEG:g} 度, 你要求转 {clean['angle_deg']:g} 度"
    return True, clean


def execute(car, name, args, safety=True):
    """
    执行一次工具调用, 返回给大模型看的观测 (dict)。
    safety=False 时跳过安全层 (只用于讲义第 4 节的对照实验: 看看没有安全层会发生什么)。
    即使跳过安全层, 也只能调用已知工具, 参数也要能转成数字, 否则小车本身就执行不了。
    """
    if safety:
        ok, res = validate_call(name, args)
        if not ok:
            return {"ok": False, "error": res}
        args = res
    else:
        if name not in TOOL_NAMES:
            return {"ok": False, "error": f"没有叫 {name!r} 的工具"}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as e:
                return {"ok": False, "error": f"参数不是合法的 JSON: {e.msg}"}
        try:
            args = {k: _number(v) for k, v in args.items()}
        except (ValueError, AttributeError) as e:
            return {"ok": False, "error": str(e)}
    try:
        return getattr(car, name)(**args)
    except TypeError as e:                       # 参数名不对 (只在 safety=False 时可能走到这里)
        return {"ok": False, "error": f"调用失败: {e}"}
