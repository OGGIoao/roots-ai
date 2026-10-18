"""
runner.py — 让一个策略在测试台上玩 steps 步, 记录三条曲线

所有实验 (experiments/*.py) 都通过这里跑, 保证同样的机器 (env_seed)、同样的指标。

    run(make_agent, steps=1000, n_runs=2000, k=10, env_seed=0, **env_kwargs) -> dict

    make_agent(n_runs, k) -> 策略对象 (有 select / update 两个方法, 见 agents.py)

返回的每条曲线长度都是 steps, 第 t 个值是 R 局在第 t+1 步的平均:
    avg_reward      平均收益
    optimal_frac    拉到 (当前) 最优臂的比例
    cum_regret      累计遗憾 (期望遗憾按步累加)
另外 locked_frac: 最后 100 步里一次都没拉过最优臂的局所占的比例 (讲义第 4 节用它衡量"锁死")。
"""

import json
import os

import numpy as np

from env import BanditTestbed

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run(make_agent, steps=1000, n_runs=2000, k=10, env_seed=0, **env_kwargs):
    env = BanditTestbed(n_runs, k, seed=env_seed, **env_kwargs)
    agent = make_agent(n_runs, k)
    avg_reward = np.zeros(steps)
    optimal_frac = np.zeros(steps)
    regret = np.zeros(steps)
    hit_tail = np.zeros(n_runs, dtype=bool)             # 每局在最后 100 步里是否拉到过最优臂
    for t in range(steps):
        actions = agent.select()                         # (R,)
        best = env.optimal_arm()                         # (R,) 要在 pull 之前取 (非平稳时 pull 后会变)
        regret_t = env.regret_of(actions)                # 同理, 用拉杆这一刻的真实均值
        rewards = env.pull(actions)                      # (R,)
        agent.update(actions, rewards)
        avg_reward[t] = rewards.mean()
        optimal_frac[t] = (actions == best).mean()
        regret[t] = regret_t.mean()
        if t >= steps - 100:
            hit_tail |= actions == best
    return {
        "avg_reward": avg_reward.round(4).tolist(),
        "optimal_frac": optimal_frac.round(4).tolist(),
        "cum_regret": np.cumsum(regret).round(3).tolist(),
        "locked_frac": round(float((~hit_tail).mean()), 4),
    }


def save(name, data):
    """把实验结果写入 results/<name>.json (讲义和画图脚本都从这里读)。"""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print("结果写入", os.path.relpath(path))


def summary(result, last=100):
    """一行摘要: 最后 last 步的平均收益与最优臂比例、总遗憾、锁死比例。"""
    return {
        "reward_last": round(float(np.mean(result["avg_reward"][-last:])), 3),
        "optimal_last": round(float(np.mean(result["optimal_frac"][-last:])), 3),
        "regret_total": round(result["cum_regret"][-1], 1),
        "locked_frac": result["locked_frac"],
    }
