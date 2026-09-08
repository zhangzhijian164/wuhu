# -*- coding: utf-8 -*-
"""LocalDataset：读取 CSV、按 7:2:1 切分、z-score 标准化、按 batch 迭代。"""
import os
import numpy as np


class LocalDataset:
    """封装 "Social_Network_Ads.csv" 数据集的读取与切分。

    数据集三列：Age, EstimatedSalary, Purchased（二分类标签 0/1）。

    功能：
      1. read_csv 读取原始数据；
      2. 固定随机切分为 train / val / test = 7 : 2 : 1；
      3. 使用 训练集 的均值 / 标准差对特征做标准化（防止信息泄漏）；
      4. 以 batch 为单位迭代训练数据；
      5. 将切分后的数据写回 out_dir，便于人工核对。
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __init__(self, csv_path: str, out_dir: str = None, seed: int = 42):
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.seed = seed

        self.raw_features = None          # (N, 2) 原始 Age, EstimatedSalary
        self.raw_labels = None            # (N,)   Purchased
        self.feature_names = None

        self.split = {self.TRAIN: None, self.VAL: None, self.TEST: None}
        self.mean = None                  # 训练集特征均值
        self.std = None                   # 训练集特征标准差

        self._read()

    # ------------------------------------------------------------------ 读取
    def _read(self):
        """从 CSV 读取数据。"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"找不到数据集文件: {self.csv_path}")

        rows = []
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            header = f.readline().strip().split(",")
            self.feature_names = header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                age = float(parts[0])
                salary = float(parts[1])
                purchase = int(parts[2])
                rows.append([age, salary, purchase])

        arr = np.asarray(rows, dtype=np.float64)
        self.raw_features = arr[:, :2]
        self.raw_labels = arr[:, 2].astype(np.int64)
        print(f"[LocalDataset] 读取 {len(rows)} 条样本, 特征数={self.raw_features.shape[1]}")

    # -------------------------------------------------------------- 划分/标准化
    def make_split(self):
        """按 7:2:1 划分 train/val/test 并基于训练集做标准化。"""
        rng = np.random.default_rng(self.seed)
        n = len(self.raw_labels)
        idx = rng.permutation(n)

        n_train = int(round(n * 0.7))
        n_val = int(round(n * 0.2))
        n_test = n - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        # 特征标准化：只用训练集统计量
        train_feat = self.raw_features[train_idx]
        self.mean = train_feat.mean(axis=0)
        self.std = train_feat.std(axis=0) + 1e-8

        def std(x):
            return (x - self.mean) / self.std

        self.split[self.TRAIN] = (
            std(self.raw_features[train_idx]), self.raw_labels[train_idx])
        self.split[self.VAL] = (
            std(self.raw_features[val_idx]), self.raw_labels[val_idx])
        self.split[self.TEST] = (
            std(self.raw_features[test_idx]), self.raw_labels[test_idx])

        print(
            f"[LocalDataset] 切分: train={n_train}, val={n_val}, test={n_test} "
            f"(比例 7:2:1)"
        )
        print(f"[LocalDataset] 标准化参数 mean={self.mean}, std={self.std}")

        if self.out_dir:
            self._save_split()

        return self.split

    # ------------------------------------------------------------------ 保存
    def _save_split(self):
        """将切分结果写回 out_dir/*.csv（原始量纲）。"""
        os.makedirs(self.out_dir, exist_ok=True)
        name2idx = {self.TRAIN: 0, self.VAL: 1, self.TEST: 2}
        idx_by_split = {}
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(self.raw_labels))
        n_train = int(round(len(self.raw_labels) * 0.7))
        n_val = int(round(len(self.raw_labels) * 0.2))
        idx_by_split[self.TRAIN] = idx[:n_train]
        idx_by_split[self.VAL] = idx[n_train:n_train + n_val]
        idx_by_split[self.TEST] = idx[n_train + n_val:]

        header = "Age,EstimatedSalary,Purchased"
        for name in (self.TRAIN, self.VAL, self.TEST):
            sub_idx = idx_by_split[name]
            feats = self.raw_features[sub_idx]
            labs = self.raw_labels[sub_idx]
            path = os.path.join(self.out_dir, f"{name}.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write(header + "\n")
                for i in range(len(sub_idx)):
                    f.write(
                        f"{feats[i, 0]:.0f},{feats[i, 1]:.0f},{int(labs[i])}\n"
                    )
            print(f"[LocalDataset] 已保存 {os.path.basename(path)}")

    # ---------------------------------------------------------------- 访问器
    def get_split(self, name: str):
        if name not in self.split:
            raise KeyError(f"未知切分 '{name}'，可用: train/val/test")
        if self.split[name] is None:
            self.make_split()
        return self.split[name]

    # ----------------------------------------------------------------- batch
    def iter_batches(self, name: str, batch_size: int, shuffle: bool = True):
        """按 batch 迭代某个切分。

        返回迭代器，每次 yield (X_batch, y_batch)。
        """
        X, y = self.get_split(name)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            yield X[idx], y[idx]

    @property
    def n_features(self) -> int:
        return self.raw_features.shape[1]

    @property
    def n_samples(self) -> int:
        return len(self.raw_labels)
