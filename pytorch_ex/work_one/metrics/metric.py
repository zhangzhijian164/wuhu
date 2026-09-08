# -*- coding: utf-8 -*-
"""分类评价指标：Accuracy / Precision / Recall / F1 / 混淆矩阵。"""
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """返回 2x2 混淆矩阵 [[TN, FP], [FN, TP]]。"""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    assert len(yt) == len(yp), "标签与预测长度不一致"
    tp = int(np.sum((yp == 1) & (yt == 1)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return float(np.mean(yt == yp))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """查准率 = TP / (TP + FP)。无正预测时返回 0。"""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """召回率 = TP / (TP + FN)。无真实正样本时返回 0。"""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """一次性返回所有指标与混淆矩阵。"""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    cm = confusion_matrix(yt, yp)
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy(yt, yp),
        "precision": precision(yt, yp),
        "recall": recall(yt, yp),
        "f1": f1_score(yt, yp),
        "n": int(len(yt)),
    }
