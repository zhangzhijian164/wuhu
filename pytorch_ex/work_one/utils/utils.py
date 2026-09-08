# -*- coding: utf-8 -*-
"""通用工具：随机种子、计时器、指标均值记录器。"""
import random
import time
import numpy as np


def set_seed(seed: int = 42) -> None:
    """固定 Python / NumPy 随机种子，保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)


class Timer:
    """简单的上下文计时器，用于统计训练耗时。"""

    def __init__(self):
        self.start = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start

    def stop(self) -> float:
        """结束计时并返回耗时秒数。"""
        if self.start is not None:
            self.elapsed = time.perf_counter() - self.start
            self.start = None
        return self.elapsed

    def __str__(self):
        return f"{self.elapsed:.3f}s"


class AverageMeter:
    """记录并平滑一个标量的平均值 / 当前值 / 累计计数。

    用法：meter = AverageMeter(); meter.update(loss, n=batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return f"{self.avg:.6f}"
