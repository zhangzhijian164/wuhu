# -*- coding: utf-8 -*-
"""7 种经典优化器：SGD / Momentum / NAG / AdaGrad / RMSProp / AdaDelta / Adam。

统一接口：step(model, grad) 依据传入的梯度更新 model.weights。
设计为「无自动求导」场景：由上层计算好损失对参数的梯度 grad，
此处只负责用不同动量 / 自适应学习率策略更新参数。
"""

import numpy as np


class BaseOptimizer:
    """优化器基类。l2 正则已在 loss / model 层合并进 grad，这里不重复。"""

    def __init__(self, params_shape, lr=0.01, **kwargs):
        self.lr = lr
        self.params_shape = params_shape
        self.step_count = 0

    def step(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """输入当前参数 w 与梯度 grad（同形状），返回更新后的参数。"""
        raise NotImplementedError

    def zero_grad(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"


class SGD(BaseOptimizer):
    """随机梯度下降（vanilla）。"""

    def step(self, w, grad):
        self.step_count += 1
        return w - self.lr * grad


class Momentum(BaseOptimizer):
    """带动量的 SGD。v = mu*v - lr*grad; w += v"""

    def __init__(self, params_shape, lr=0.01, momentum=0.9, **kwargs):
        super().__init__(params_shape, lr)
        self.mu = momentum
        self.v = np.zeros(params_shape)

    def step(self, w, grad):
        self.step_count += 1
        self.v = self.mu * self.v - self.lr * grad
        return w + self.v


class NAG(BaseOptimizer):
    """Nesterov 加速梯度：先按动量外推一步，再在该处计算梯度。

    近似实现：w_ahead = w + mu*v，然后用同样的 grad 更新 v。
    （精确 NAG 需在外推点重新算梯度，这里用标准近似形式。）
    """

    def __init__(self, params_shape, lr=0.01, momentum=0.9, **kwargs):
        super().__init__(params_shape, lr)
        self.mu = momentum
        self.v = np.zeros(params_shape)

    def step(self, w, grad):
        self.step_count += 1
        w_ahead = w + self.mu * self.v
        self.v = self.mu * self.v - self.lr * grad
        return w_ahead + self.v


class AdaGrad(BaseOptimizer):
    """自适应梯度：对每个参数累计梯度平方，用其平方根缩放学习率。"""

    def __init__(self, params_shape, lr=0.01, eps=1e-8, **kwargs):
        super().__init__(params_shape, lr)
        self.eps = eps
        self.G = np.zeros(params_shape)

    def step(self, w, grad):
        self.step_count += 1
        self.G += grad ** 2
        lr_eff = self.lr / (np.sqrt(self.G) + self.eps)
        return w - lr_eff * grad


class RMSProp(BaseOptimizer):
    """RMSProp：指数滑动平均的梯度平方归一化。"""

    def __init__(self, params_shape, lr=0.01, beta=0.9, eps=1e-8, **kwargs):
        super().__init__(params_shape, lr)
        self.beta = beta
        self.eps = eps
        self.s = np.zeros(params_shape)

    def step(self, w, grad):
        self.step_count += 1
        self.s = self.beta * self.s + (1 - self.beta) * grad ** 2
        lr_eff = self.lr / (np.sqrt(self.s) + self.eps)
        return w - lr_eff * grad


class AdaDelta(BaseOptimizer):
    """AdaDelta：无需全局学习率，用参数更新量的 RMS 做二阶归一化。

    更新规则：
        E[g^2] = rho*E[g^2] + (1-rho)*g^2
        dx = - sqrt(E[dx^2] + eps) / sqrt(E[g^2] + eps) * g
        E[dx^2] = rho*E[dx^2] + (1-rho)*dx^2
        w = w + dx
    """

    def __init__(self, params_shape, rho=0.95, eps=1e-6, **kwargs):
        super().__init__(params_shape, lr=None)
        self.rho = rho
        self.eps = eps
        self.Eg2 = np.zeros(params_shape)
        self.Ed2 = np.zeros(params_shape)

    def step(self, w, grad):
        self.step_count += 1
        self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * grad ** 2
        dx = -np.sqrt(self.Ed2 + self.eps) / np.sqrt(self.Eg2 + self.eps) * grad
        self.Ed2 = self.rho * self.Ed2 + (1 - self.rho) * dx ** 2
        return w + dx


class Adam(BaseOptimizer):
    """Adam：一阶动量（带偏差校正）+ 二阶矩 + 自适应学习率。"""

    def __init__(self, params_shape, lr=0.01, beta1=0.9, beta2=0.999,
                 eps=1e-8, **kwargs):
        super().__init__(params_shape, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)
        self.t = 0

    def step(self, w, grad):
        self.t += 1
        self.step_count = self.t
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# 名称 -> 类 的映射，便于按字符串创建
OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adadelta": AdaDelta,
    "adam": Adam,
}
