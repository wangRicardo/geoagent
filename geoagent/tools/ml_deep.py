"""PyTorch 深度学习工作流：MLP 分类器训练（早停 + 学习曲线）与预测。

面向地球物理表格数据（地震属性/测井曲线 → 岩性/相分类）的轻量训练封装：
- train_mlp_classifier: 自动标准化、早停、保存最优权重到工作区、输出损失曲线
- mlp_predict: 加载模型预测并返回概率

依赖 torch（CPU 即可）：pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import registry

MODEL_DIR = "ricardo_models"


def _torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except ImportError as exc:
        raise RuntimeError(
            "需要 PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc


def _model_path(name: str) -> str:
    root = os.getcwd()
    full = os.path.abspath(os.path.join(root, MODEL_DIR, f"{name}.json"))
    if os.path.commonpath([root, full]) != root:
        raise ValueError(f"路径越出工作目录: {name}")
    return full


def _build_nn(nn, n_in: int, hidden: list[int], n_out: int):
    layers = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


@registry.register(category="ml")
def train_mlp_classifier(
    features: list[list[float]],
    labels: list[float],
    hidden: list[int] = None,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    early_stopping: int = 8,
    model_name: str = "mlp_model",
) -> str:
    """训练 MLP 分类器：自动标准化、验证集早停、保存最优模型与损失曲线到工作区。

    hidden 为隐藏层尺寸列表（默认 [64, 32]）。返回测试/验证指标，模型存到
    ricardo_models/<model_name>.json，可用 mlp_predict 加载预测。
    """
    torch, nn = _torch()
    torch.manual_seed(42)
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        return "ERROR: 至少需要 2 个类别"
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_i = np.array([cls_idx[v] for v in y], dtype=np.int64)

    n = len(X)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(2, int(n * val_split))
    va, tr = perm[:n_val], perm[n_val:]
    mu, sigma = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
    Xn = (X - mu) / sigma

    net = _build_nn(nn, X.shape[1], list(hidden or [64, 32]), len(classes))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xn[tr])
    yt = torch.tensor(y_i[tr])
    Xv = torch.tensor(Xn[va])
    yv = torch.tensor(y_i[va])

    best, best_state, bad = float("inf"), None, 0
    curve = {"train": [], "val": []}
    for _ in range(max(5, epochs)):
        net.train()
        perm_t = torch.randperm(len(Xt))
        ep_loss = 0.0
        for i in range(0, len(Xt), batch_size):
            b = perm_t[i : i + batch_size]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(b)
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv), yv))
        curve["train"].append(ep_loss / len(Xt))
        curve["val"].append(vl)
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= early_stopping:
                break
    if best_state:
        net.load_state_dict(best_state)
    with torch.no_grad():
        pred = net(Xv).argmax(1).numpy()
        acc = float((pred == yv.numpy()).mean())

    os.makedirs(os.path.dirname(_model_path(model_name)), exist_ok=True)
    payload = {
        "state": {k: v.tolist() for k, v in net.state_dict().items()},
        "hidden": list(hidden or [64, 32]),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "classes": [float(c) for c in classes],
        "val_acc": acc,
        "epochs_run": None,
    }
    with open(_model_path(model_name), "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # 损失曲线
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(curve["train"], label="train")
        ax.plot(curve["val"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        ax.set_title(f"{model_name} (best val loss {best:.3f})")
        curve_path = os.path.join(MODEL_DIR, f"{model_name}_loss.png")
        fig.savefig(curve_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:  # noqa: BLE001 - 画图失败不影响训练结果
        curve_path = None

    return (
        f"训练完成: {model_name} (轮数 {len(curve['train'])}, 早停 patience={early_stopping})\n"
        f"  验证集准确率: {acc:.3f} | 类别: {classes} | 最优验证损失: {best:.4f}\n"
        f"  模型: {MODEL_DIR}/{model_name}.json | 损失曲线: {curve_path}\n"
        f"  用 mlp_predict(model_name='{model_name}', features=...) 进行预测"
    )


@registry.register(category="ml")
def mlp_predict(model_name: str, features: list[list[float]]) -> str:
    """加载已训练的 MLP 模型对 新样本预测，返回类别与概率。"""
    torch, nn = _torch()
    path = _model_path(model_name)
    if not os.path.exists(path):
        return f"ERROR: 模型 {model_name} 不存在（先 train_mlp_classifier 训练）"
    payload = json.load(open(path, encoding="utf-8"))
    X = np.asarray(features, dtype=np.float32)
    Xn = (X - np.asarray(payload["mu"], dtype=np.float32)) / np.asarray(payload["sigma"], dtype=np.float32)
    net = _build_nn(nn, X.shape[1], payload["hidden"], len(payload["classes"]))
    net.load_state_dict({k: torch.tensor(v) for k, v in payload["state"].items()})
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(torch.tensor(Xn)), dim=1).numpy()
    classes = payload["classes"]
    lines = [f"模型 {model_name} (验证准确率 {payload['val_acc']:.3f}):"]
    for i, p in enumerate(prob):
        top = int(np.argmax(p))
        lines.append(f"  样本{i}: → {classes[top]} (p={p[top]:.3f}) | 全概率 {np.round(p, 3).tolist()}")
    return "\n".join(lines)
