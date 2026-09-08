# -*- coding: utf-8 -*-
"""损失函数：BaseLoss 基类 + MSELoss + CrossEntropyLoss（含 L2 正则化项）。"""
import numpy as np


class BaseLoss:
    """损失函数基类。

    子类需实现 forward(y_pred, y_true)，返回标量损失；
    gradient(y_pred, y_true) 返回对模型线性输出 z 的梯度（用于手工优化器）。
    若包含可学习参数的可导正则项，请实现 reg_term / reg_grad。
    """

    def __init__(self, l2_reg: float = 0.0):
        # l2_reg 为 0 表示不使用 L2 正则
        self.l2_reg = l2_reg

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """返回 d(loss)/d(y_pred)。"""
        raise NotImplementedError

    def reg_term(self, weights: np.ndarray) -> float:
        """(1/2) * l2_reg * sum(w^2) —— 偏置不在 weights 内时适用。"""
        if self.l2_reg == 0:
            return 0.0
        return 0.5 * self.l2_reg * float(np.sum(weights ** 2))

    def reg_grad(self, weights: np.ndarray) -> np.ndarray:
        if self.l2_reg == 0:
            return np.zeros_like(weights)
        return self.l2_reg * weights

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class MSELoss(BaseLoss):
    """均方误差损失，用于线性回归。"""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pred = np.ravel(y_pred)
        true = np.ravel(y_true)
        return float(np.mean((pred - true) ** 2))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pred = np.ravel(y_pred)
        true = np.ravel(y_true)
        n = len(true)
        return (2.0 / n) * (pred - true).reshape(-1, 1)


class CrossEntropyLoss(BaseLoss):
    """二分类交叉熵损失（对数损失），带数值稳定性处理。

    输入 y_pred 为 sigmoid 概率（取值 0~1），y_true ∈ {0,1}。
    公式：loss = -mean[ y*log(p) + (1-y)*log(1-p) ]
    梯度：dL/dz = p - y （z 为线性输出，即 sigmoid 反函数）。
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        p = np.clip(np.ravel(y_pred), 1e-12, 1.0 - 1e-12)
        y = np.ravel(y_true)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """返回 d(loss)/dz，其中 z = logit = sigmoid^{-1}(p)。

        对 LogisticRegression，模型前向得到的是概率 p，
        反向用 dL/dz = p - y 可让优化器直接对线性权重求导。
        """
        p = np.clip(np.ravel(y_pred), 1e-12, 1.0 - 1e-12)
        y = np.ravel(y_true)
        return (p - y).reshape(-1, 1)
