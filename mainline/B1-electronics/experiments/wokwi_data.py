"""
读取 Wokwi 逻辑分析仪的录波, 以及串口记录。

两种录波格式都支持:
  1. 标准 VCD (.vcd): 在 Wokwi 网页版停止仿真时自动下载的 "b1-logic.vcd", 自己跑实验时用它;
  2. 仓库存档用的紧凑格式 (.cvcd): 本仓库 results/wokwi/ 下的文件。和 VCD 记录的内容完全相同, 只是写法更短:
       每个通道一行: 通道名, 然后是一串记录;
       每条记录 = 电平符号 (+ 表示 1, - 表示 0) + 距离本通道上一条记录的时间差 (纳秒);
       连续重复的一对记录写成 [+a -b]*n。
     之所以要这个格式: 1 kHz 的 PWM 录 1 秒就有 2000 次跳变, VCD 文本很长, 紧凑格式只有它的 1/4 左右。

读出来的结果统一是 {通道名: [(时间_ns, 电平), ...]}, 按时间排序。
"""

import re


def parse_vcd(text):
    ids = {}
    for m in re.finditer(r"\$var\s+wire\s+1\s+(\S+)\s+(\S+)\s+\$end", text):
        ids[m.group(1)] = m.group(2)
    chans = {name: [] for name in ids.values()}
    t = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            t = int(line[1:])
            continue
        m = re.match(r"^([01])(\S+)$", line)
        if m and m.group(2) in ids:
            chans[ids[m.group(2)]].append((t, int(m.group(1))))
    return chans


def _expand_tokens(tokens):
    """把 "[+a", "-b]*n" 这样的重复写法展开成普通记录。"""
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("["):
            a = tok[1:]
            b, n = tokens[i + 1].split("]*")
            out.extend([a, b] * int(n))
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


def expand_cvcd(text):
    """紧凑格式 → 没有重复写法的紧凑格式 (仅用于校验; 分析用 parse_cvcd)。"""
    lines = []
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        parts = line.split(" ")
        lines.append(" ".join([parts[0]] + _expand_tokens(parts[1:])))
    return "\n".join(lines)


def parse_cvcd(text):
    chans = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ")
        t = 0
        recs = []
        for tok in _expand_tokens(parts[1:]):
            t += int(tok[1:])
            recs.append((t, 1 if tok[0] == "+" else 0))
        chans[parts[0]] = recs
    return chans


def load_waves(path):
    text = open(path, encoding="utf-8").read()
    return parse_cvcd(text) if path.endswith(".cvcd") else parse_vcd(text)


def load_records(path, tag):
    """串口记录里形如 "@TAG,k=v,k=v;" 的行 → [dict, ...] (能转成数字的值转成数字)。"""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("@" + tag + ","):
            continue
        body = line[len(tag) + 2:].rstrip(";")
        rec = {}
        for kv in body.split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                rec[k] = int(v)
            except ValueError:
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
        out.append(rec)
    return out


def checksum(s):
    """与采集时浏览器里用的校验和相同, 用来确认存档和采到的数据逐字一致。"""
    x = 0
    for ch in s:
        x = (x * 31 + ord(ch)) & 0xFFFFFFFF
    return len(s), x
