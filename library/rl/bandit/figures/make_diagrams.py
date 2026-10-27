"""
make_diagrams.py — 老虎机讲义里的结构图 (SVG)

  fig_loop.svg   策略与环境的一步交互, 以及每个数组的形状 (讲义第 2 节)
  fig_ucb.svg    UCB 在第 30 步的真实状态: 估计值 + 奖励, 为什么选的不是估计值最高的臂 (讲义第 6 节)
                 图里的数字是现场跑出来的 (env_seed=0 的第 0 局), 不是示意值

运行: python figures/make_diagrams.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from agents import UCB  # noqa: E402
from env import BanditTestbed  # noqa: E402

from common.svgfig import (BG, CACHE, GRID, INK, INK2, KV_FILL, KV_LINE, MUTED, OP_LINE, Q_FILL,  # noqa: E402
                           Q_LINE, Svg)


def fig_loop():
    s = Svg(820, 300)
    s.text(20, 22, "一步交互: 策略选臂 → 环境给收益 → 策略更新估计", 15, INK, "start", "bold")
    s.text(20, 42, "R 局同时进行, 每个数组的第一维都是局; 策略和环境各有自己的随机数发生器", 12, INK2, "start")
    # 左: 策略; 右: 环境
    s.rect(40, 80, 260, 170, Q_FILL, Q_LINE)
    s.text(170, 100, "策略 (agents.py)", 14, INK, weight="bold")
    s.text(170, 130, "Q   (R, k)  每个臂收益的估计", 12, INK2)
    s.text(170, 152, "N   (R, k)  每个臂被拉过几次", 12, INK2)
    s.text(170, 186, "select(): 按 Q (和 N) 选臂", 12, INK2)
    s.text(170, 208, "update(): Q ← Q + 步长 × (r − Q)", 12, INK2)
    s.rect(520, 80, 260, 170, KV_FILL, KV_LINE)
    s.text(650, 100, "环境 (env.py)", 14, INK, weight="bold")
    s.text(650, 130, "q*  (R, k)  真实均值, 策略看不到", 12, INK2)
    s.text(650, 164, "pull(a): r = q*[a] + N(0, 1) 噪声", 12, INK2)
    s.text(650, 198, "optimal_arm() / regret_of(a)", 12, MUTED)
    s.text(650, 216, "只用来记指标, 不告诉策略", 12, MUTED)
    # 上箭头: 动作; 下箭头: 收益
    s.line(300, 125, 518, 125)
    s.text(410, 112, "actions  (R,)  每局拉哪个臂", 12, INK)
    s.line(520, 205, 302, 205)
    s.text(410, 222, "rewards  (R,)  拿到的收益", 12, INK)
    s.text(410, 275, "runner.py 把这一步重复 steps 次, 每步记下 R 局的平均收益、最优臂比例、遗憾", 12, MUTED)
    s.save(os.path.join(HERE, "fig_loop.svg"))


def ucb_state(t_stop=30, run=0):
    """跑 UCB 到第 t_stop 次 select, 返回这一刻第 run 局的 Q、N、奖励、选中的臂和真实均值。"""
    env = BanditTestbed(2000, 10, seed=0)
    agent = UCB(2000, 10, c=2.0)
    for _ in range(t_stop - 1):
        a = agent.select()
        agent.update(a, env.pull(a))
    chosen = int(agent.select()[run])                  # 第 t_stop 次 select (agent.t == t_stop)
    Q, N = agent.Q[run].copy(), agent.N[run].copy()
    bonus = agent.c * np.sqrt(np.log(agent.t) / N)
    return Q, N, bonus, chosen, env.q_star[run].copy()


def fig_ucb():
    Q, N, bonus, chosen, q_star = ucb_state()
    k = len(Q)
    s = Svg(820, 440)
    s.text(20, 22, f"UCB 第 30 步: 估计值最高的是臂 {int(Q.argmax())}, 却选了臂 {chosen}", 15, INK, "start", "bold")
    s.text(20, 42, "实心柱 = 估计值 Q; 虚线柱 = 不确定性奖励 c·√(ln t / N); 两者之和 (柱顶数字) 最高的臂被选中 "
                   "(env_seed=0 的第 0 局, c=2)", 12, INK2, "start")
    x0, x1, top, bottom = 80, 790, 90, 360
    vmin, vmax = -3.0, 3.0
    Y = lambda v: bottom - (v - vmin) / (vmax - vmin) * (bottom - top)  # noqa: E731
    for v in range(-3, 4):                                             # 网格与刻度
        s.items.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        s.text(x0 - 10, Y(v), f"{v:+d}" if v else "0", 11, MUTED, "end")
    s.items.append(f'<text x="30" y="{(top + bottom) / 2}" font-size="12" fill="{INK2}" text-anchor="middle" '
                   f'transform="rotate(-90 30 {(top + bottom) / 2})">收益</text>')
    slot = (x1 - x0) / k
    for a in range(k):
        cx = x0 + slot * (a + 0.5)
        w = slot * 0.5
        lo, hi = sorted([0.0, Q[a]])
        line = CACHE if a == chosen else Q_LINE
        # 先画奖励部分 (从 Q 到 Q + 奖励), 再画 Q 的柱子 (从 0 到 Q):
        # Q 为负时两者会重叠, 后画的实心柱必须在上面才看得见
        s.rect(cx - w / 2, Y(Q[a] + bonus[a]), w, Y(Q[a]) - Y(Q[a] + bonus[a]),
               BG, line, dash=True, rx=2, sw=2.2 if a == chosen else 1.5)
        s.rect(cx - w / 2, Y(hi), w, Y(lo) - Y(hi), Q_FILL, line, rx=2)
        s.text(cx, Y(Q[a] + bonus[a]) - 10, f"{Q[a] + bonus[a]:.2f}", 11, INK if a == chosen else INK2,
               weight="bold" if a == chosen else "normal")
        # 真实均值: 一条短横线 (策略看不到, 画出来只为对照)
        s.items.append(f'<line x1="{cx - w / 2 - 6:.1f}" y1="{Y(q_star[a]):.1f}" x2="{cx + w / 2 + 6:.1f}" '
                       f'y2="{Y(q_star[a]):.1f}" stroke="{KV_LINE}" stroke-width="2.5"/>')
        s.text(cx, bottom + 18, f"臂 {a}", 12, INK if a == chosen else INK2,
               weight="bold" if a == chosen else "normal")
        s.text(cx, bottom + 36, f"N={N[a]}", 11, MUTED)
        s.text(cx, bottom + 52, f"Q={Q[a]:+.2f}", 11, MUTED)
    s.items.append(f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x1}" y2="{Y(0):.1f}" stroke="{OP_LINE}"/>')
    # 图例
    lx, ly = 20, 64
    s.rect(lx, ly - 5, 14, 10, Q_FILL, Q_LINE, rx=1)
    s.text(lx + 20, ly, "估计值 Q", 11, INK2, "start")
    s.rect(lx + 90, ly - 5, 14, 10, BG, Q_LINE, dash=True, rx=1)
    s.text(lx + 110, ly, "奖励", 11, INK2, "start")
    s.items.append(f'<line x1="{lx + 155}" y1="{ly}" x2="{lx + 175}" y2="{ly}" stroke="{KV_LINE}" stroke-width="2.5"/>')
    s.text(lx + 180, ly, "真实均值 q* (策略看不到, 只为对照)", 11, INK2, "start")
    s.rect(lx + 400, ly - 5, 14, 10, BG, CACHE, dash=True, rx=1)
    s.text(lx + 420, ly, "绿框 = 本步选中的臂", 11, INK2, "start")
    s.save(os.path.join(HERE, "fig_ucb.svg"))


if __name__ == "__main__":
    fig_loop()
    fig_ucb()
