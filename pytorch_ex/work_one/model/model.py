# -*- coding: utf-8 -*-
"""模型定义：BaseModel 基类 + LinearRegression + LogisticRegression。

模型使用 numpy 手写实现。两类模型都支持带偏置的线性参数。
"""
import pickle
import numpy as np

# 避免循环导入：在函数内部导入所需模块
def _sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


class BaseModel:
    """所有模型共有的接口：拟合 / 保存 / 加载 / 重置。

    子类需实现：
      _init_params(n_features)
      predict_proba(X)  -> (n,) 概率或实数预测
      predict(X)
      grad(X, y, pred)  -> (n_features+1,) 参数梯度（含偏置，见子类约定）
    """

    def __init__(self, **kwargs):
        self.weights = None      # 形状 (n_features+1, 1)，最后一行为偏置
        self.n_features = None
        self.metric_fn = None    # 由具体任务决定

    # ------------------------------------------------------------ 参数初始化
    def _init_params(self, n_features: int):
        bound = 1.0 / np.sqrt(n_features)
        w = np.random.uniform(-bound, bound, size=(n_features + 1, 1))
        self.weights = w
        self.n_features = n_features

    # ---------------------------------------------------------------- 工具
    def num_params(self) -> int:
        return 0 if self.weights is None else int(self.weights.size)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_features={self.n_features}, params={self.num_params()})"
        )

    # ---------------------------------------------------------------- 保存
    def save(self, path: str):
        """将完整模型（含标准化参数）序列化到磁盘。"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseModel":
        with open(path, "rb") as f:
            return pickle.load(f)


class LinearRegression(BaseModel):
    """线性回归：y = X·w。损失 MSE（含可选 L2）。"""

    def __init__(self, l2_reg: float = 0.0):
        super().__init__()
        self.l2_reg = l2_reg
        from loss.loss import MSELoss
        self._loss = MSELoss(l2_reg=l2_reg)

    def _init_params(self, n_features: int):
        super()._init_params(n_features)
        # 线性回归对初始化不敏感，但保持较小扰动更稳
        self.weights = np.zeros((n_features + 1, 1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """返回 (n,1) 预测值 = [X, 1] @ w。"""
        Xb = np.hstack([X, np.ones((len(X), 1))])
        return Xb @ self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ravel(self.forward(X))

    # ------------------------------------------------------------ 梯度计算
    def grad(self, X, y, pred):
        """返回 (n_features+1, 1) 梯度（含偏置项，偏置不参与 L2）。"""
        Xb = np.hstack([X, np.ones((len(X), 1))])
        n = len(y)
        pred_col = np.ravel(pred).reshape(-1, 1)      # 统一成 (n,1)
        y_col = np.ravel(y).reshape(-1, 1)            # 统一成 (n,1)
        base = (2.0 / n) * (Xb.T @ (pred_col - y_col))
        w_no_bias = self.weights[:-1]
        base[:-1] += (self.l2_reg * w_no_bias).reshape(-1, 1)
        return base

    def score(self, X, y):
        """R^2 决定系数。"""
        pred = np.ravel(self.forward(X))
        y = np.ravel(y)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


class LogisticRegression(BaseModel):
    """逻辑回归（二分类）：概率 p = sigmoid(z)。损失为交叉熵（含可选 L2）。"""

    def __init__(self, l2_reg: float = 0.0):
        super().__init__()
        self.l2_reg = l2_reg
        from loss.loss import CrossEntropyLoss
        self._loss = CrossEntropyLoss(l2_reg=l2_reg)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """返回 (n,1) 概率值 p = sigmoid([X,1]@w)。"""
        Xb = np.hstack([X, np.ones((len(X), 1))])
        z = Xb @ self.weights
        return _sigmoid(z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ravel(self.forward(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = np.ravel(self.forward(X))
        return (p >= threshold).astype(np.int64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """返回线性得分 z（用于决策边界绘制）。"""
        Xb = np.hstack([X, np.ones((len(X), 1))])
        return np.ravel(Xb @ self.weights)

    # ------------------------------------------------------------ 梯度计算
    def grad(self, X, y, pred):
        """返回 (n_features+1, 1) 梯度：dL/dz = p - y。

        注意这里 pred 传入的是 sigmoid 概率。
        """
        Xb = np.hstack([X, np.ones((len(X), 1))])
        pred_col = np.ravel(pred).reshape(-1, 1)      # (n,1)
        y_col = np.ravel(y).reshape(-1, 1)            # (n,1)
        g = Xb.T @ (pred_col - y_col)
        base = g / len(y)
        w_no_bias = self.weights[:-1]
        base[:-1] += (self.l2_reg * w_no_bias).reshape(-1, 1)
        return base

    def loss_value(self, X, y):
        p = np.ravel(self.forward(X))
        y = np.ravel(y)
        loss = self._loss.forward(p, y)
        return loss + self._loss.reg_term(self.weights[:-1])
