"""机器学习领域工具。

依赖 scikit-learn（未安装时优雅降级）；numpy/pandas 开箱即用。
工具粒度刻意保持"研究者常用操作"级别：数据画像、快速建模、交叉验证、
特征重要性 —— 让 agent 能自主完成一轮「看数据 -> 建模型 -> 报告指标」。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import registry


@registry.register(category="ml")
def dataset_profile(features: List[List[float]], target: List[float]) -> str:
    """对特征矩阵 X 与标签 y 做快速画像：形状、统计量、缺失值、类别分布。"""
    X = np.asarray(features, dtype=float)
    y = np.asarray(target)
    if X.ndim != 2:
        return f"ERROR: features 需要是二维列表，当前形状 {X.shape}"
    lines = [f"X: {X.shape[0]} 样本 x {X.shape[1]} 特征, y: {y.shape[0]} 条"]
    nan_count = int(np.isnan(X).sum())
    lines.append(f"X 缺失值: {nan_count}")
    lines.append("特征统计 (min/mean/max):")
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            lines.append(f"  f{j}: {col.min():.4g} / {col.mean():.4g} / {col.max():.4g}")
    uniq = np.unique(y)
    if uniq.size <= 20:
        dist = {str(u): int((y == u).sum()) for u in uniq}
        lines.append(f"y 取值分布: {dist}")
    else:
        lines.append(f"y 为回归目标: min={y.min():.4g}, max={y.max():.4g}, mean={y.mean():.4g}")
    return "\n".join(lines)


def _classify_or_regress(y: np.ndarray) -> str:
    """粗略判断任务类型：整型/少取值 -> 分类，否则回归。"""
    uniq = np.unique(y[~np.isnan(y)]) if y.dtype.kind == "f" else np.unique(y)
    if y.dtype.kind in "iu" or uniq.size <= 20:
        return "classification"
    return "regression"


@registry.register(category="ml")
def quick_train(
    features: List[List[float]],
    target: List[float],
    test_size: float = 0.25,
    random_state: int = 42,
) -> str:
    """一键基线建模：自动判断分类/回归，训练随机森林并报告测试集指标。"""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, f1_score, r2_score, mean_absolute_error,
        )
    except ImportError:
        return "ERROR: 需要 scikit-learn。请先安装: pip install scikit-learn"

    X = np.asarray(features, dtype=float)
    y = np.asarray(target)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if X.shape[0] < 4:
        return f"ERROR: 有效样本过少 ({X.shape[0]})，无法划分训练/测试集"

    task = _classify_or_regress(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state).fit(Xtr, ytr)
        pred = model.predict(Xte)
        metrics = {
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "f1_macro": round(float(f1_score(yte, pred, average="macro", zero_division=0)), 4),
        }
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state).fit(Xtr, ytr)
        pred = model.predict(Xte)
        metrics = {
            "r2": round(float(r2_score(yte, pred)), 4),
            "mae": round(float(mean_absolute_error(yte, pred)), 4),
        }
    importances = model.feature_importances_
    top = sorted(enumerate(importances), key=lambda kv: -kv[1])[:5]
    return (
        f"任务类型: {task} (随机森林, {len(Xtr)} 训练 / {len(Xte)} 测试)\n"
        f"指标: {metrics}\n"
        f"特征重要性Top5: {[(f'f{i}', round(v, 4)) for i, v in top]}"
    )


@registry.register(category="ml")
def cross_validate(
    features: List[List[float]],
    target: List[float],
    k: int = 5,
    random_state: int = 42,
) -> str:
    """对随机森林基线做 k 折交叉验证，返回各折分数与均值±标准差。"""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return "ERROR: 需要 scikit-learn。请先安装: pip install scikit-learn"
    X = np.asarray(features, dtype=float)
    y = np.asarray(target)
    task = _classify_or_regress(y)
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        scoring = "r2"
    scores = cross_val_score(model, X, y, cv=k, scoring=scoring)
    return (
        f"{k} 折 {scoring}: {np.round(scores, 4).tolist()}\n"
        f"均值 {scores.mean():.4f} ± {scores.std():.4f}"
    )


@registry.register(category="ml")
def feature_correlation(features: List[List[float]], threshold: float = 0.8) -> str:
    """计算特征间 Pearson 相关矩阵，列出相关系数超过阈值的特征对（共线性预警）。"""
    X = np.asarray(features, dtype=float)
    if X.shape[1] < 2:
        return "ERROR: 至少需要 2 个特征"
    corr = np.corrcoef(X, rowvar=False)
    pairs = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            if abs(corr[i, j]) >= threshold:
                pairs.append(f"f{i}~f{j}: {corr[i, j]:.3f}")
    return (
        f"相关矩阵:\n{np.round(corr, 3)}\n"
        + (f"高相关特征对(|r|>={threshold}): {'; '.join(pairs)}" if pairs else "没有超过阈值的特征对")
    )


@registry.register(category="ml")
def confusion_report(features: List[List[float]], target: List[float]) -> str:
    """训练随机森林分类器并输出测试集混淆矩阵与分类报告（precision/recall/F1）。"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
    except ImportError:
        return "ERROR: 需要 scikit-learn。请先安装: pip install scikit-learn"
    X = np.asarray(features, dtype=float)
    y = np.asarray(target)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    pred = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr).predict(Xte)
    cm = confusion_matrix(yte, pred)
    report = classification_report(yte, pred, zero_division=0)
    return f"混淆矩阵 (行=真实, 列=预测):\n{cm}\n\n{report}"


@registry.register(category="ml")
def pca_reduce(features: List[List[float]], n_components: int = 2) -> str:
    """PCA 降维，返回各主成分解释方差比与前几个样本的投影坐标。"""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return "ERROR: 需要 scikit-learn。请先安装: pip install scikit-learn"
    X = np.asarray(features, dtype=float)
    X = np.nan_to_num(X, nan=np.nanmean(X))
    n_components = min(n_components, X.shape[1], X.shape[0])
    p = PCA(n_components=n_components).fit(X)
    proj = p.transform(X)
    return (
        f"解释方差比: {np.round(p.explained_variance_ratio_, 4).tolist()}\n"
        f"前5样本投影: {np.round(proj[:5], 4).tolist()}"
    )
