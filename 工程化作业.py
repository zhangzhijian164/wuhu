import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体与负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. zgtorch.data (数据集基类与三分法数据集)
# ==============================================================================
class Dataset:
    """数据集父类：规范迭代器接口"""

    def __init__(self):
        self.batches = []

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx < len(self):
            item = self[self._iter_idx]
            self._iter_idx += 1
            return item
        else:
            raise StopIteration


class SocialAdsDataset(Dataset):
    """具体数据集类：前处理（Min-Max 归一化、拼接偏置项、封装 Mini-Batch）"""

    def __init__(self, X_data, y_data, batch_size=32, min_val=None, max_val=None):
        super().__init__()

        # 1. 前处理：归一化基准对齐（验证集/测试集复用训练集的 Min/Max）
        if min_val is None or max_val is None:
            self.min_val = X_data.min(axis=0)
            self.max_val = X_data.max(axis=0)
        else:
            self.min_val = min_val
            self.max_val = max_val

        # 特征 Min-Max 归一化到 [0, 1]
        X_norm = (X_data - self.min_val) / (self.max_val - self.min_val + 1e-8)

        # 2. 拼接偏置列 (Bias = 1)
        self.X = np.column_stack([np.ones((len(X_norm), 1)), X_norm])
        self.y = y_data.reshape(-1, 1)

        # 3. 切分为 Mini-Batches
        num_samples = len(self.X)
        for i in range(0, num_samples, batch_size):
            bx = self.X[i:i + batch_size]
            by = self.y[i:i + batch_size]
            self.batches.append((bx, by))


# ==============================================================================
# 2. zgtorch.model (模型父类与逻辑回归模型)
# ==============================================================================
class Module:
    """模型父类：定规范 + 字典数据结构权重存取"""

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def save(self, filepath):
        """保存权重字典至 .pkl 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state_dict = {'params': self.params, 'grads': self.grads}
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
        print(f"模型权重字典已成功保存至: {filepath}")

    def load(self, filepath):
        """从 .pkl 文件加载权重字典"""
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        self.params = state_dict['params']
        self.grads = state_dict['grads']


class LogicRegression(Module):
    """逻辑回归模型：数据与计算分离，权重与梯度保存在模型内部"""

    def __init__(self, input_dim=3):
        super().__init__()
        # input_dim=3: [Bias, Age_norm, Salary_norm]
        self.params['w'] = np.zeros((input_dim, 1), dtype=np.float32)
        self.grads['w'] = np.zeros((input_dim, 1), dtype=np.float32)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x):
        self.x_cache = x  # 缓存输入以备反向传播使用
        z = x @ self.params['w']
        self.output_cache = self.sigmoid(z)
        return self.output_cache


# ==============================================================================
# 3. zgtorch.loss (损失函数父类与二分类交叉熵损失)
# ==============================================================================
class Loss:
    """损失函数父类"""

    def __init__(self, model):
        self.model = model

    def forward(self, output, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, output, target):
        return self.forward(output, target)


class CrossEntropyLoss(Loss):
    """二分类交叉熵损失：backward() 负责计算梯度并回写给 model.grads"""

    def __init__(self, model):
        super().__init__(model)

    def forward(self, output, target):
        self.output = output
        self.target = target
        eps = 1e-15
        p = np.clip(output, eps, 1.0 - eps)
        loss = -np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
        return loss

    def backward(self):
        m = len(self.target)
        x = self.model.x_cache
        grad_w = (1.0 / m) * (x.T @ (self.output - self.target))
        self.model.grads['w'] = grad_w


# ==============================================================================
# 4. zgtorch.optim (7种梯度下降算法与优化器实现)
# ==============================================================================
class Optimizer:
    """优化器父类"""

    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        pass

    def step(self, grads):
        raise NotImplementedError


# 算法 1: SGD (随机梯度下降)
class SGD(Optimizer):
    """1. Vanilla SGD (标准随机梯度下降)
    公式: w = w - lr * grad
    """

    def __init__(self, params, lr=0.1):
        super().__init__(params)
        self.lr = lr

    def step(self, grads):
        for k in self.params.keys():
            self.params[k] -= self.lr * grads[k]


# 算法 2: Momentum (动量梯度下降)
class Momentum(Optimizer):
    """2. SGD with Momentum (动量梯度下降)
    公式: v = gamma * v + lr * grad
         w = w - v
    """

    def __init__(self, params, lr=0.05, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for k in self.params.keys():
            self.v[k] = self.momentum * self.v[k] + self.lr * grads[k]
            self.params[k] -= self.v[k]


# 算法 3: Nesterov Accelerated Gradient (NAG)
class NAG(Optimizer):
    """3. Nesterov Accelerated Gradient (牛顿加速动量)
    公式: v_prev = v
         v = gamma * v + lr * grad
         w = w - (gamma * v + lr * grad)
    """

    def __init__(self, params, lr=0.05, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for k in self.params.keys():
            v_prev = self.v[k].copy()
            self.v[k] = self.momentum * self.v[k] + self.lr * grads[k]
            self.params[k] -= (self.momentum * self.v[k] + self.lr * grads[k])


# 算法 4: AdaGrad
class AdaGrad(Optimizer):
    """4. AdaGrad (自适应梯度算法)
    公式: G = G + grad^2
         w = w - (lr / sqrt(G + eps)) * grad
    """

    def __init__(self, params, lr=0.1, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.G = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for k in self.params.keys():
            self.G[k] += grads[k] ** 2
            self.params[k] -= (self.lr / (np.sqrt(self.G[k]) + self.eps)) * grads[k]


# 算法 5: RMSprop
class RMSprop(Optimizer):
    """5. RMSprop (均方根传播算法)
    公式: v = beta * v + (1 - beta) * grad^2
         w = w - (lr / sqrt(v + eps)) * grad
    """

    def __init__(self, params, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for k in self.params.keys():
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * (grads[k] ** 2)
            self.params[k] -= (self.lr / (np.sqrt(self.v[k]) + self.eps)) * grads[k]


# 算法 6: AdaDelta
class AdaDelta(Optimizer):
    """6. AdaDelta (自适应 Delt 算法，无需手动设定学习率)
    公式: E_g2 = rho * E_g2 + (1 - rho) * grad^2
         delta_w = - (sqrt(E_p2 + eps) / sqrt(E_g2 + eps)) * grad
         E_p2 = rho * E_p2 + (1 - rho) * delta_w^2
         w = w + delta_w
    """

    def __init__(self, params, rho=0.95, eps=1e-6):
        super().__init__(params)
        self.rho = rho
        self.eps = eps
        self.E_g2 = {k: np.zeros_like(v) for k, v in params.items()}
        self.E_p2 = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for k in self.params.keys():
            g = grads[k]
            self.E_g2[k] = self.rho * self.E_g2[k] + (1 - self.rho) * (g ** 2)
            delta_w = - (np.sqrt(self.E_p2[k] + self.eps) / (np.sqrt(self.E_g2[k]) + self.eps)) * g
            self.E_p2[k] = self.rho * self.E_p2[k] + (1 - self.rho) * (delta_w ** 2)
            self.params[k] += delta_w


# 算法 7: Adam
class Adam(Optimizer):
    """7. Adam (自适应矩估计)
    公式: m = beta1 * m + (1 - beta1) * g
         v = beta2 * v + (1 - beta2) * g^2
         m_hat = m / (1 - beta1^t)
         v_hat = v / (1 - beta2^t)
         w = w - (lr / sqrt(v_hat + eps)) * m_hat
    """

    def __init__(self, params, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.params.keys():
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==============================================================================
# 5. 可视化绘图与评估辅助函数
# ==============================================================================
def plot_eda_charts(train_df, save_dir):
    """前处理可视：生成纯柱状图的特征直方图和二维散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist([train_df[train_df['Purchased'] == 0]['Age'],
                  train_df[train_df['Purchased'] == 1]['Age']],
                 bins=15, color=['#FF7F7F', '#2CA02C'], label=['未购买 (0)', '已购买 (1)'],
                 edgecolor='black', alpha=0.8, stacked=False)
    axes[0].set_title('训练集 - 年龄(Age)特征分布直方图', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Age (年龄)')
    axes[0].set_ylabel('频数 (Count)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].hist([train_df[train_df['Purchased'] == 0]['EstimatedSalary'],
                  train_df[train_df['Purchased'] == 1]['EstimatedSalary']],
                 bins=15, color=['#FF7F7F', '#2CA02C'], label=['未购买 (0)', '已购买 (1)'],
                 edgecolor='black', alpha=0.8, stacked=False)
    axes[1].set_title('训练集 - 预估薪资(EstimatedSalary)特征分布直方图', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('EstimatedSalary (预估薪资)')
    axes[1].set_ylabel('频数 (Count)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_histograms.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(train_df[train_df['Purchased'] == 0]['Age'], train_df[train_df['Purchased'] == 0]['EstimatedSalary'],
                c='red', label='未购买 (0)', alpha=0.7, edgecolors='k')
    plt.scatter(train_df[train_df['Purchased'] == 1]['Age'], train_df[train_df['Purchased'] == 1]['EstimatedSalary'],
                c='green', label='已购买 (1)', alpha=0.7, edgecolors='k')
    plt.title('训练集 - Age vs EstimatedSalary 特征分布散点图', fontsize=12, fontweight='bold')
    plt.xlabel('Age (年龄)')
    plt.ylabel('Estimated Salary (预估薪资)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_scatterplot.png'), dpi=300)
    plt.show()


def evaluate(model, dataset):
    """计算模型在评估集上的指标"""
    y_true = dataset.y.flatten()
    y_prob = model(dataset.X).flatten()
    y_pred = (y_prob >= 0.5).astype(np.int32)

    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

    return loss, acc, prec, rec, f1, (tp, fp, fn, tn)


# ==============================================================================
# 6. 主程序与 7 种优化器切换测试
# ==============================================================================
def train():
    runs_dir = 'zgtorch/runs'
    os.makedirs(runs_dir, exist_ok=True)

    df = pd.read_csv('Social_Network_Ads.csv')

    np.random.seed(42)
    indices = np.random.permutation(len(df))
    df_shuffled = df.iloc[indices]

    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_df = df_shuffled.iloc[:n_train]
    val_df = df_shuffled.iloc[n_train:n_train + n_val]
    test_df = df_shuffled.iloc[n_train + n_val:]

    print(f"数据三分法切分完成: 训练集={len(train_df)}条, 验证集={len(val_df)}条, 测试集={len(test_df)}条\n")

    plot_eda_charts(train_df, runs_dir)

    train_dataset = SocialAdsDataset(train_df[['Age', 'EstimatedSalary']].values, train_df['Purchased'].values, batch_size=32)
    val_dataset = SocialAdsDataset(val_df[['Age', 'EstimatedSalary']].values, val_df['Purchased'].values, batch_size=32,
                                   min_val=train_dataset.min_val, max_val=train_dataset.max_val)
    test_dataset = SocialAdsDataset(test_df[['Age', 'EstimatedSalary']].values, test_df['Purchased'].values, batch_size=32,
                                    min_val=train_dataset.min_val, max_val=train_dataset.max_val)

    # --- 模型与损失函数实例化 ---
    model = LogicRegression(input_dim=3)
    loss_fn = CrossEntropyLoss(model)

    # 💡【在这里自由选择你想用的梯度下降优化器】
    # 可选列表: "sgd", "momentum", "nag", "adagrad", "rmsprop", "adadelta", "adam"
    OPTIMIZER_TYPE = "adam"

    if OPTIMIZER_TYPE == "sgd":
        optimizer = SGD(model.params, lr=0.1)
    elif OPTIMIZER_TYPE == "momentum":
        optimizer = Momentum(model.params, lr=0.05, momentum=0.9)
    elif OPTIMIZER_TYPE == "nag":
        optimizer = NAG(model.params, lr=0.05, momentum=0.9)
    elif OPTIMIZER_TYPE == "adagrad":
        optimizer = AdaGrad(model.params, lr=0.2)
    elif OPTIMIZER_TYPE == "rmsprop":
        optimizer = RMSprop(model.params, lr=0.01)
    elif OPTIMIZER_TYPE == "adadelta":
        optimizer = AdaDelta(model.params)
    elif OPTIMIZER_TYPE == "adam":
        optimizer = Adam(model.params, lr=0.05)
    else:
        raise ValueError(f"未知的优化器类型: {OPTIMIZER_TYPE}")

    print(f"当前使用的是优化器: {optimizer.__class__.__name__}\n")

    n_epochs = 200
    train_loss_history, val_loss_history = [], []

    # 训练循环（依然保持标准 SOP 步骤，无需改动）
    for epoch in range(n_epochs):
        total_train_loss = 0.0
        for input, target in train_dataset:
            optimizer.zero_grad()
            output = model(input)
            loss_val = loss_fn(output, target)
            loss_fn.backward()
            optimizer.step(model.grads)
            total_train_loss += loss_val * len(input)

        avg_train_loss = total_train_loss / len(train_df)
        val_loss, val_acc, _, _, _, _ = evaluate(model, val_dataset)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:3d}/{n_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

    model.save(os.path.join(runs_dir, 'logic_regression.pkl'))

    test_loss, test_acc, test_prec, test_rec, test_f1, (tp, fp, fn, tn) = evaluate(model, test_dataset)

    print("\n" + "=" * 55)
    print(f"       独立测试集最终评估结果 ({optimizer.__class__.__name__})        ")
    print("=" * 55)
    print(f"测试集准确率 (Accuracy) : {test_acc * 100:.2f}%")
    print(f"测试集精确率 (Precision): {test_prec * 100:.2f}%")
    print(f"测试集召回率 (Recall)   : {test_rec * 100:.2f}%")
    print(f"测试集 F1-Score         : {test_f1:.4f}")
    print("-" * 55)
    print(f"混淆矩阵: [ TP: {tp:2d} | FP: {fp:2d} ]")
    print(f"          [ FN: {fn:2d} | TN: {tn:2d} ]")
    print("=" * 55 + "\n")

    # 绘制结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, n_epochs + 1), train_loss_history, label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(range(1, n_epochs + 1), val_loss_history, label='Val Loss', color='orange', linestyle='--', linewidth=2)
    axes[0].set_title(f'[{optimizer.__class__.__name__}] 训练与验证 Loss 下降曲线', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    X_all = df[['Age', 'EstimatedSalary']].values
    x1_min, x1_max = X_all[:, 0].min() - 5, X_all[:, 0].max() + 5
    x2_min, x2_max = X_all[:, 1].min() - 5000, X_all[:, 1].max() + 5000
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))

    grid_pts = np.c_[xx1.ravel(), xx2.ravel()]
    grid_pts_norm = (grid_pts - train_dataset.min_val) / (train_dataset.max_val - train_dataset.min_val + 1e-8)
    grid_pts_bias = np.column_stack([np.ones((len(grid_pts_norm), 1)), grid_pts_norm])

    probs = model(grid_pts_bias).reshape(xx1.shape)

    axes[1].contourf(xx1, xx2, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['red', 'green'])
    axes[1].contour(xx1, xx2, probs, levels=[0.5], colors='black', linewidths=2)
    axes[1].scatter(test_df['Age'], test_df['EstimatedSalary'], c=test_df['Purchased'], cmap='bwr', edgecolors='k', alpha=0.8)
    axes[1].set_title(f'测试集分类决策边界 (Acc: {test_acc * 100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Age (年龄)')
    axes[1].set_ylabel('Estimated Salary (预估薪资)')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, 'training_result.png'), dpi=300)
    plt.show()
if __name__ == '__main__':
    train()









