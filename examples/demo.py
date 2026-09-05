"""示例：不接 LLM，直接用工具层完成一个迷你研究流程。

模拟场景：有一段合成地震道，先做频谱分析，再和"含气砂岩"标签一起
喂给基线模型，看看能以什么精度区分 —— 这是 agent 自动化的典型一环。
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from geoagent import GeoAgent

agent = GeoAgent()

# 1. 合成 3 口"井"的属性：频率衰减 + 振幅异常 + 噪声
rng = np.random.default_rng(7)
t = np.arange(1000) / 1000.0
features, target = [], []
for i in range(60):
    peak = rng.uniform(5, 40)  # 主频 Hz
    amp = rng.uniform(0.5, 2.0)
    x = amp * np.sin(2 * math.pi * peak * t) * np.exp(-3 * t)
    x += rng.normal(0, 0.1, t.size)
    # 含气砂岩（标签1）特征上主频偏低、振幅偏高 —— 人为加上可分的趋势
    gas = int(i % 2 == 0)
    f0, f1, f2 = peak / (1.3 if gas else 1.0), amp * (1.5 if gas else 1.0), len(x) / 1000
    features.append([f0, f1, f2])
    target.append(gas)

# 2. agent 调用工具：数据画像 -> 快速建模 -> 交叉验证
print(agent.run_tool("dataset_profile", {"features": features, "target": target}))
print()
print(agent.run_tool("quick_train", {"features": features, "target": target}))
print()
print(agent.run_tool("cross_validate", {"features": features, "target": target, "k": 5}))
print()
# 3. 频谱分析一个合成地震道
trace = (np.sin(2 * math.pi * 15 * t)).tolist()
print(agent.run_tool("signal_spectrum", {"data": trace, "dt_ms": 1.0}))
