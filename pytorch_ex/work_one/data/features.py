# -*- coding: utf-8 -*-
"""特征工程：生成二次（多项式）特征，以学习非线性决策边界。"""
import numpy as np


def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """为二维输入 X 生成多项式特征，包含偏置列。

    输入 X 形状 (n_samples, 2)，列依次为 x1, x2。
    当 degree == 2 时输出列顺序为：
        [1, x1, x2, x1^2, x1*x2, x2^2]
    仅要求输入为二维（本数据集正好两列特征），可通过 padding 处理高维输入。

    参数
    ----
    X : (n_samples, n_features) 原始特征，仅取前两列做二次展开
    degree : 多项式最高次数（当前仅支持 2）

    返回
    ----
    X_poly : (n_samples, n_features_poly) 含偏置的多项式特征
    """
    X = np.atleast_2d(X)
    n_samples, n_in = X.shape
    if n_in != 2:
        raise ValueError(
            f"polynomial_features 当前仅支持二维输入 (n_features==2)，收到 {n_in}。"
        )

    if degree == 1:
        return np.hstack([np.ones((n_samples, 1)), X])

    if degree != 2:
        raise NotImplementedError("polynomial_features 仅实现了 degree=2 的特征展开。")

    x1 = X[:, 0]
    x2 = X[:, 1]
    cols = [
        np.ones(n_samples),  # 偏置
        x1,
        x2,
        x1 ** 2,
        x1 * x2,
        x2 ** 2,
    ]
    return np.column_stack(cols)
