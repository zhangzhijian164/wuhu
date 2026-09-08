# -*- coding: utf-8 -*-
"""训练入口：配置超参数、切分数据、训练逻辑回归（多项式特征）、绘图、保存模型。

用法：
    python main.py --optimizer adam --lr 0.1 --epochs 500

流程：
  1. 读取 Social_Network_Ads.csv，按 7:2:1 切分；
  2. 生成二次多项式特征（在标准化之后做，保持稳定）；
  3. 用所选优化器在 train 上做 mini-batch 训练，val 上早停/选优；
  4. 保存最优模型到 model/model.pkl（含标准化参数 + 特征变换器）；
  5. 绘制训练损失曲线与验证精度曲线；
  6. 用测试集评估并打印指标，输出决策边界图 shannon_result.png。
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 项目内模块
from data.dataset import LocalDataset
from data.features import polynomial_features
from model.model import LogisticRegression
from optim.optimizer import OPTIMIZERS
from metrics.metric import evaluate, accuracy
from utils.utils import set_seed, Timer

# 统一以本文件所在目录为项目根，保证任意 cwd 都能 import
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "Social_Network_Ads.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "out_dir")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


# ---------------------------------------------------------------------- 包装模型
class PackagedClassifier:
    """把「特征变换 + 标准化 + 逻辑回归」打包成一个可直接保存的完整模型。

    保存后 one_hot_test.py 只需 load 即可对新数据 (Age, Salary) 做预测。
    """

    def __init__(self, base_model, feat_mean, feat_std, degree=2, threshold=0.5):
        self.base = base_model          # LogisticRegression
        self.feat_mean = feat_mean      # 原始两列特征均值
        self.feat_std = feat_std
        self.degree = degree
        self.threshold = threshold

    # ---- 变换链：原始两列 -> 标准化 -> 二次多项式
    def _transform(self, X_raw):
        X = np.atleast_2d(X_raw).astype(np.float64)
        Xs = (X - self.feat_mean) / (self.feat_std + 1e-8)
        return polynomial_features(Xs, degree=self.degree)

    def predict_proba(self, X_raw):
        Xp = self._transform(X_raw)
        return self.base.predict_proba(Xp)

    def predict(self, X_raw):
        Xp = self._transform(X_raw)
        return self.base.predict(Xp, threshold=self.threshold)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------- 训练逻辑
def train_step(model, optimizer, Xb, y, n_train):
    """单步 batch 更新，返回该 batch 的损失。"""
    prob = model.forward(Xb)                      # (batch,1) sigmoid
    loss_obj = model._loss
    # 交叉熵主损失
    p = np.clip(np.ravel(prob), 1e-12, 1 - 1e-12)
    y = np.ravel(y)
    loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    loss += loss_obj.reg_term(model.weights[:-1])
    # 梯度 dL/dw
    grad = model.grad(Xb, y, np.ravel(prob))      # (n_feat+1,1)
    model.weights = optimizer.step(model.weights, grad)
    return loss


def make_batches(X, y, batch_size, rng):
    n = len(y)
    order = rng.permutation(n)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        yield X[idx], y[idx]


def train(train_cfg):
    set_seed(train_cfg["seed"])
    opt_name = train_cfg["optimizer"]
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    l2 = train_cfg["l2_reg"]

    # 1. 读取 + 切分
    ds = LocalDataset(CSV_PATH, OUT_DIR, seed=train_cfg["seed"])
    Xtr_raw, ytr = ds.get_split("train")
    Xva_raw, yva = ds.get_split("val")
    Xte_raw, yte = ds.get_split("test")

    # 2. 二次多项式特征（此时 Xtr_raw 已是标准化后的两列）
    deg = 2
    Xtr = polynomial_features(Xtr_raw, degree=deg)
    Xva = polynomial_features(Xva_raw, degree=deg)
    Xte = polynomial_features(Xte_raw, degree=deg)

    n_feat = Xtr.shape[1]
    print(f"[train] 特征维度（含偏置与二次项）= {n_feat}")

    # 3. 模型与优化器
    model = LogisticRegression(l2_reg=l2)
    model._init_params(n_feat)
    opt_cls = OPTIMIZERS[opt_name.lower()]
    optimizer = opt_cls(model.weights.shape, lr=lr)

    # 4. 训练
    best_val_acc = -1.0
    best_w = None
    train_loss_hist = []
    val_acc_hist = []
    rng = np.random.default_rng(train_cfg["seed"])

    timer = Timer()
    with timer:
        for ep in range(epochs):
            ep_losses = []
            for Xb, yb in make_batches(Xtr, ytr, batch_size, rng):
                loss = train_step(model, optimizer, Xb, yb, len(ytr))
                ep_losses.append(loss)
            avg_loss = float(np.mean(ep_losses))
            train_loss_hist.append(avg_loss)

            # 验证精度（全量）
            p_va = model.forward(Xva)
            pred_va = (np.ravel(p_va) >= 0.5).astype(np.int64)
            acc_va = accuracy(yva, pred_va)
            val_acc_hist.append(acc_va)

            if acc_va > best_val_acc:
                best_val_acc = acc_va
                best_w = model.weights.copy()

            if (ep + 1) % 100 == 0:
                print(f"[epoch {ep+1:>4}/{epochs}] loss={avg_loss:.5f} "
                      f"val_acc={acc_va:.4f}")

    dur = timer.stop()
    print(f"[train] 训练完成，耗时 {dur:.2f}s")

    # 恢复验证集最优权重
    model.weights = best_w

    # 5. 保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    pkg = PackagedClassifier(model, ds.mean, ds.std, degree=deg, threshold=0.5)
    pkg_path = os.path.join(MODEL_DIR, "model.pkl")
    pkg.save(pkg_path)
    print(f"[train] 模型已保存 -> {pkg_path}")

    # 同时备份纯权重（供 runs/weights.pkl）
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, "weights.pkl"), "wb") as f:
        pickle.dump({"weights": best_w, "mean": ds.mean, "std": ds.std,
                     "degree": deg, "feature_names": ds.feature_names}, f)
    print(f"[train] 参数副本已保存 -> runs/weights.pkl")

    # 6. 测试评估
    p_te = model.forward(Xte)
    pred_te = (np.ravel(p_te) >= 0.5).astype(np.int64)
    report = evaluate(yte, pred_te)
    print("\n========== Test 集评估 ==========")
    print(f"confusion matrix:\n{report['confusion_matrix']}")
    print(f"accuracy = {report['accuracy']:.4f}")
    print(f"precision= {report['precision']:.4f}")
    print(f"recall   = {report['recall']:.4f}")
    print(f"f1       = {report['f1']:.4f}")

    # 7. 绘图
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    _plot(train_cfg, model, ds, Xva_raw, yva, Xtr_raw, ytr,
          train_loss_hist, val_acc_hist, report)
    return report


# ---------------------------------------------------------------------- 绘图
def _plot(cfg, model, ds, Xva_raw, yva, Xtr_raw, ytr,
          loss_hist, val_acc_hist, report):
    """生成两张图：损失/精度曲线 + 决策边界。合并输出为一张大图。"""
    fig = plt.figure(figsize=(15, 6))

    # (1) 损失曲线
    ax1 = fig.add_subplot(1, 2, 1)
    eps = list(range(1, len(loss_hist) + 1))
    ax1.plot(eps, loss_hist, color="#d62728", lw=1.8,
             label="train loss (CE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#d62728")
    ax1.set_title(f"Training Loss  [{cfg['optimizer'].upper()}]")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # 右侧纵轴：验证精度
    ax1b = ax1.twinx()
    ax1b.plot(eps, val_acc_hist, color="#1f77b4", lw=1.6, ls="--",
              label="val accuracy")
    ax1b.set_ylabel("Accuracy", color="#1f77b4")
    ax1b.set_ylim(0, 1.02)
    ax1b.legend(loc="lower right")

    # (2) 决策边界
    ax2 = fig.add_subplot(1, 2, 2)
    # 生成标准空间网格（基于 val 的统计量一致的坐标 → 直接用 val 的 min/max 反标准化）
    Xva_u = Xva_raw * ds.std + ds.mean      # 还原到原始量纲便于绘图
    margin = 2.0
    x_min, x_max = Xva_u[:, 0].min() - margin, Xva_u[:, 0].max() + margin
    y_min, y_max = Xva_u[:, 1].min() - margin, Xva_u[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_raw = np.c_[xx.ravel(), yy.ravel()]          # 原始 Age, Salary
    # 标准化 -> 多项式 -> 概率
    Xs = (grid_raw - ds.mean) / (ds.std + 1e-8)
    Xp = polynomial_features(Xs, degree=2)
    prob = model.predict_proba(Xp).reshape(xx.shape)
    levels = np.array([0.05, 0.5, 0.95])
    cf = ax2.contourf(xx, yy, prob, levels=50, cmap="coolwarm", alpha=0.35)
    # 决策边界（p=0.5）用白色实线加粗
    cs = ax2.contour(xx, yy, prob, levels=[0.5], colors="white", linewidths=2.5)
    ax2.clabel(cs, inline=True, fontsize=9, fmt="%.1f")

    # 训练点（原始量纲）
    Xtr_u = Xtr_raw * ds.std + ds.mean
    ax2.scatter(Xtr_u[ytr == 0, 0], Xtr_u[ytr == 0, 1],
                c="#1f77b4", marker="o", s=26, alpha=0.6, label="Not Purchased (0)")
    ax2.scatter(Xtr_u[ytr == 1, 0], Xtr_u[ytr == 1, 1],
                c="#d62728", marker="x", s=32, alpha=0.8, label="Purchased (1)")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Estimated Salary")
    ax2.set_title("Decision Boundary (quadratic features)\n"
                  f"Test Accuracy = {report['accuracy']:.4f}")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.2)

    fig.colorbar(cf, ax=ax2, label="P(Purchased=1)")
    fig.suptitle("Logistic Regression on Social_Network_Ads — "
                 f"optimizer={cfg['optimizer'].upper()}, "
                 f"lr={cfg['lr']}, epochs={cfg['epochs']}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_png = os.path.join(PROJECT_ROOT, "shannon_result.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] 图已保存 -> {out_png}")


# ---------------------------------------------------------------------- 命令行
def parse_args():
    p = argparse.ArgumentParser(description="手写逻辑回归训练 (Social Network Ads)")
    p.add_argument("--optimizer", default="adam", choices=list(OPTIMIZERS),
                   help="优化器")
    p.add_argument("--lr", type=float, default=0.1, help="学习率")
    p.add_argument("--epochs", type=int, default=500, help="迭代轮数")
    p.add_argument("--batch_size", type=int, default=32, help="batch 大小")
    p.add_argument("--l2_reg", type=float, default=0.0, help="L2 正则系数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "l2_reg": args.l2_reg,
        "seed": args.seed,
    }
    print("=" * 60)
    print("手写逻辑回归训练 (Social_Network_Ads)")
    print(f"配置: {cfg}")
    print("=" * 60)
    train(cfg)
