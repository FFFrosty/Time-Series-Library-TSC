import argparse
import os
import numpy as np
from tqdm import tqdm
import time

# ================= 替换为分类器 =================
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder  # 新增：用于处理标签编码

# 新增：导入第三方 xgboost 库
from xgboost import XGBClassifier

# 导入 TSLib 原生的组件
from data_provider.data_factory import data_provider

import random
import torch
def fix_random_seed(seed):
    """固定所有相关的随机种子，确保实验 100% 可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 针对 PyTorch 底层 CuDNN 的确定性设置 (虽然 ML 基线不用 GPU 计算，但严谨起见一并加上)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RocketWrapper(BaseEstimator, ClassifierMixin):
    """
    为 Rocket 分类模型编写的适配器。
    将 2D 展平的数组转换为 sktime 要求的 3D 数组。
    """

    def __init__(self, seq_len, enc_in, num_kernels=1000, random_state=None):
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        try:
            # 注意：这里改为了 classification
            from sktime.classification.kernel_based import RocketClassifier
        except ImportError:
            raise ImportError("请先运行 `pip install sktime` 安装依赖，才能使用 Rocket 模型")

        self.model = RocketClassifier(num_kernels=self.num_kernels, random_state=self.random_state,
                                      rocket_transform='minirocket')

        # 将 2D 展平的数据还原为 3D: [Batch, seq_len, features]
        # 并调整为 sktime 要求的维度顺序: [Batch, features, seq_len]
        X_3d = X.reshape(-1, self.seq_len, self.enc_in).transpose(0, 2, 1)
        self.model.fit(X_3d, y)
        return self

    def predict(self, X):
        X_3d = X.reshape(-1, self.seq_len, self.enc_in).transpose(0, 2, 1)
        return self.model.predict(X_3d)


def get_flattened_data(args, flag):
    """
    复用 TSLib 的 DataLoader，提取并展平数据
    """
    dataset, dataloader = data_provider(args, flag)

    x_list = []
    y_list = []

    print(f"正在加载 {flag} 数据集...")
    for i, batch_data in enumerate(tqdm(dataloader)):
        # 动态获取前两个值 (特征和标签)
        batch_x = batch_data[0]
        batch_y = batch_data[1]

        # 展平输入 X: [Batch, seq_len * features]
        batch_x_flat = batch_x.numpy().reshape(batch_x.shape[0], -1)

        # 展平目标 Y: 对于分类任务，标签通常就是单个整数 [Batch]
        true_y_flat = batch_y.numpy().reshape(-1)

        x_list.append(batch_x_flat)
        y_list.append(true_y_flat)

    # 将所有的 batch 拼接成一个 numpy 数组
    X_all = np.vstack(x_list)
    Y_all = np.concatenate(y_list)  # 分类标签使用 concatenate 拼接成 1D 数组

    return X_all, Y_all, None


def main():
    parser = argparse.ArgumentParser(description='TSLib ML Baselines for Classification')

    # 核心参数 (默认设为分类任务)
    parser.add_argument('--task_name', type=str, default='classification')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    # 新增模型选项：XGBoost
    parser.add_argument('--model', type=str, default='XGBoost', help='HistGB, RandomForest, Rocket, or XGBoost')

    # TSLib需要的参数
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # 机器学习基础参数
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--random_state', type=int, default=2026)
    # Rocket 的卷积核数量
    parser.add_argument('--num_kernels', type=int, default=10000)

    # 数据相关参数
    parser.add_argument('--data', type=str, default='UEA')  # 分类常用 UEA 数据集
    parser.add_argument('--root_path', type=str, default='./data/EthanolConcentration/')
    parser.add_argument('--data_path', type=str, default='EthanolConcentration_TEST.ts')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')

    # 长度参数 (分类任务通常用整个 seq_len 作为输入，不需要 pred_len)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=0)

    # DataLoader 需要的参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--drop_last', action='store_true', default=False)

    args = parser.parse_args()

    # =========== 新增：在解析参数后立即固定全局种子 ===========
    fix_random_seed(args.random_state)

    # 1. 提取数据
    X_train, Y_train_raw, _ = get_flattened_data(args, flag='train')
    X_test, Y_test_raw, _ = get_flattened_data(args, flag='test')

    enc_in = X_train.shape[1] // args.seq_len  # 自动推断特征维度

    # ================= 新增：标签编码 =================
    # XGBoost 强制要求分类标签必须是从 0 开始的整数 (0, 1, 2...)
    # 使用 LabelEncoder 可以自动把字符串标签或非 0 起点的标签转换成标准格式
    print("正在进行标签编码对齐...")
    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(Y_train_raw)
    Y_test = label_encoder.transform(Y_test_raw)

    # 2. 初始化分类模型
    print(f"\n初始化 {args.model} 分类器...")

    if args.model == 'XGBoost':
        # 调用第三方 xgboost 库
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=0.1,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            eval_metric='mlogloss'  # 避免输出警告信息
        )
    elif args.model == 'HistGB':
        # sklearn 中的高效直方图梯度提升分类树
        model = HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            learning_rate=0.1,
            random_state=args.random_state
        )
    elif args.model == 'Rocket':
        # 包装后的 Rocket 分类模型
        model = RocketWrapper(
            seq_len=args.seq_len,
            enc_in=enc_in,
            num_kernels=args.num_kernels,
            random_state=args.random_state,
        )
    else:
        # 随机森林分类器
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.random_state
        )

    # 3. 训练
    print(f"开始训练 {args.model}...")
    model.fit(X_train, Y_train)
    print("训练完成！")

    # 4. 预测
    print("开始在测试集上进行预测...")
    Y_pred = model.predict(X_test)

    # 5. 计算分类指标
    print("计算指标...")
    # 计算指标时使用编码后的标签
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='macro')

    print(f"\n================ 最终测试结果 ({args.model}) ================")
    print(f"Dataset: {args.data} | seq_len: {args.seq_len}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("==============================================================")

    # 6. 保存结果
    run_id = time.strftime("%m%d_%H%M%S")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_rs{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.random_state,
        run_id)
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = 'result_classification.txt'
    f = open(os.path.join(folder_path, file_name), 'a')
    f.write(setting + "  \n")
    f.write('accuracy:{}'.format(acc))
    f.write('f1:{}'.format(f1))
    f.write('\n')
    f.write('\n')
    f.close()

    # 保存预测结果和真实标签 (保存的是经过编码后的 0,1,2 格式，方便后续计算混淆矩阵)
    np.save(folder_path + 'pred.npy', Y_pred)
    np.save(folder_path + 'true.npy', Y_test)

    # 如果你需要保存原始标签，可以将上面的 Y_test 改回 Y_test_raw，
    # 并使用 label_encoder.inverse_transform(Y_pred) 还原预测标签后再保存。


if __name__ == '__main__':
    main()