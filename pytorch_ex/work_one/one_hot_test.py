# -*- coding: utf-8 -*-
"""推理脚本：加载 model/model.pkl，在测试集上做预测与评估。

同时演示对任意新样本 (Age, EstimatedSalary) 的单条预测。
"""
import os
import pickle
import sys

import numpy as np

# 保证能 import 到同目录下的 data / metrics 等包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import LocalDataset            # noqa: E402
from metrics.metric import evaluate, accuracy    # noqa: E402
from main import PackagedClassifier              # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "Social_Network_Ads.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.pkl")


def main():
    if not os.path.exists(MODEL_PATH):
        print("未找到 model/model.pkl，请先运行 main.py 训练。")
        return 1

    # 1. 加载完整模型（含标准化参数 + 特征变换器）
    pkg = PackagedClassifier.load(MODEL_PATH)
    model = pkg.base
    print(f"加载模型: {model!r}")
    print(f"特征均值 mean = {pkg.feat_mean}")
    print(f"特征标准差 std = {pkg.feat_std}")

    # 2. 重新读取原始数据并按相同 7:2:1 切分测试集
    #    注意：LocalDataset 固定 seed=42，切分与训练完全一致
    ds = LocalDataset(CSV_PATH, seed=42)
    Xte_raw, yte = ds.get_split("test")   # 标准化后的两列特征（原始量纲需还原）

    # 还原成原始 Age / Salary 以便演示新样本
    Xte_orig = Xte_raw * ds.std + ds.mean

    # 3. 预测
    prob = pkg.predict_proba(Xte_orig)
    pred = pkg.predict(Xte_orig)
    true = yte.astype(np.int64)

    # 4. 评估
    report = evaluate(true, pred)
    print("\n========== 测试集评估（加载 model.pkl） ==========")
    print(f"样本数      = {report['n']}")
    print("混淆矩阵  [[TN FP]")
    print(f"           {report['confusion_matrix'][0]}")
    print(f"           {report['confusion_matrix'][1]}]  [FN TP]")
    print(f"Accuracy  = {report['accuracy']:.4f}")
    print(f"Precision = {report['precision']:.4f}")
    print(f"Recall    = {report['recall']:.4f}")
    print(f"F1        = {report['f1']:.4f}")

    # 5. 抽样展示前 10 条预测
    print("\n--- 前 10 条测试样本预测 ---")
    print(f"{'Age':>4} {'Salary':>8} {'真值':>4} {'预测':>4} {'P(购)':>7}")
    for i in range(min(10, len(true))):
        print(f"{Xte_orig[i, 0]:>4.0f} {Xte_orig[i, 1]:>8.0f} "
              f"{true[i]:>4} {pred[i]:>4} {prob[i]:>7.4f}")

    # 6. 单条新样本演示
    print("\n--- 新样本演示 ---")
    for age, sal in [(35, 70000), (50, 100000), (25, 30000)]:
        p_ = pkg.predict_proba(np.array([[age, sal]]))[0]
        lab = pkg.predict(np.array([[age, sal]]))[0]
        print(f"Age={age}, Salary={sal} -> P(Purchased)={p_:.4f} "
              f"=> {'购买(1)' if lab == 1 else '不购买(0)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
