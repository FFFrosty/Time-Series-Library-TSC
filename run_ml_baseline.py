import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import warnings

# ================= 导入分类器与回归器 =================
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# 导入 TSLib 原生的组件
from data_provider.data_factory import data_provider

import random
import torch


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ARIMAWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, seq_len, pred_len, enc_in, order=(1, 1, 1)):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.order = order

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_3d = X.reshape(-1, self.seq_len, self.enc_in)
        preds = []

        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', UserWarning)

        print(f"ARIMA 正在逐样本拟合与预测 (共 {X_3d.shape[0]} 个样本)...")
        for i in tqdm(range(X_3d.shape[0]), desc="ARIMA Predicting"):
            sample_preds = []
            for f in range(self.enc_in):
                history = X_3d[i, :, f]
                try:
                    model = ARIMA(history, order=self.order)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=self.pred_len)
                except Exception:
                    forecast = np.full(self.pred_len, history[-1])
                sample_preds.append(forecast)

            sample_preds = np.column_stack(sample_preds)
            preds.append(sample_preds)

        preds = np.array(preds).reshape(X_3d.shape[0], -1)
        return preds


class RocketWrapper(BaseEstimator):
    def __init__(self, task_name, seq_len, enc_in, num_kernels=10000, random_state=None):
        self.task_name = task_name
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        try:
            if self.task_name == 'classification':
                from sktime.classification.kernel_based import RocketClassifier
                self.model = RocketClassifier(num_kernels=self.num_kernels, random_state=self.random_state,
                                              rocket_transform='minirocket')
            else:
                from sktime.regression.kernel_based import RocketRegressor
                self.model = RocketRegressor(num_kernels=self.num_kernels, random_state=self.random_state,
                                             rocket_transform='minirocket')
        except ImportError:
            raise ImportError("请先运行 `pip install sktime` 安装依赖")

        X_3d = X.reshape(-1, self.seq_len, self.enc_in).transpose(0, 2, 1)
        self.model.fit(X_3d, y)
        return self

    def predict(self, X):
        X_3d = X.reshape(-1, self.seq_len, self.enc_in).transpose(0, 2, 1)
        return self.model.predict(X_3d)


def get_flattened_data(args, flag):
    # ✅ 修改点 1：返回 dataset 对象，以便后续提取 inverse_transform
    dataset, dataloader = data_provider(args, flag)
    x_list, y_list = [], []

    print(f"正在加载 {flag} 数据集...")
    for i, batch_data in enumerate(tqdm(dataloader)):
        batch_x = batch_data[0]
        batch_y = batch_data[1]

        batch_x_flat = batch_x.numpy().reshape(batch_x.shape[0], -1)

        if args.task_name == 'classification':
            true_y_flat = batch_y.numpy().reshape(-1)
        else:
            true_y = batch_y.numpy()[:, -args.pred_len:, :]
            true_y_flat = true_y.reshape(true_y.shape[0], -1)

        x_list.append(batch_x_flat)
        y_list.append(true_y_flat)

    X_all = np.vstack(x_list)

    if args.task_name == 'classification':
        Y_all = np.concatenate(y_list)
    else:
        Y_all = np.vstack(y_list)

    # 返回 X_all, Y_all, dataset
    return X_all, Y_all, dataset


def main():
    parser = argparse.ArgumentParser(description='TSLib ML Baselines')

    parser.add_argument('--task_name', type=str, default='classification', help='classification or long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--model', type=str, default='ARIMA',
                        help='HistGB, RandomForest, Rocket, XGBoost, SVM, or ARIMA')

    # 新增：是否在测试集上进行反归一化 (默认开启)
    parser.add_argument('--inverse', action='store_true', default=True, help='inverse output data')

    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--random_state', type=int, default=2026)
    parser.add_argument('--num_kernels', type=int, default=10000)
    parser.add_argument('--arima_order', type=str, default='1,1,1', help='ARIMA order (p,d,q)')

    parser.add_argument('--data', type=str, default='UEA')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='data.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--drop_last', action='store_true', default=False)

    # 维持运行的基本参数
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")

    args = parser.parse_args()
    fix_random_seed(args.random_state)

    # 1. 提取数据
    X_train, Y_train_raw, dataset_train = get_flattened_data(args, flag='train')
    # ✅ 接收 dataset_test
    X_test, Y_test_raw, dataset_test = get_flattened_data(args, flag='test')

    enc_in = X_train.shape[1] // args.seq_len

    # 2. 标签处理
    if args.task_name == 'classification':
        print("分类任务：进行标签编码...")
        label_encoder = LabelEncoder()
        Y_train = label_encoder.fit_transform(Y_train_raw)
        Y_test = label_encoder.transform(Y_test_raw)
    else:
        Y_train = Y_train_raw
        Y_test = Y_test_raw

    if args.model == 'ARIMA':
        try:
            order = tuple(map(int, args.arima_order.split(',')))
        except ValueError:
            order = (1, 1, 1)

    # 3. 初始化模型
    print(f"\n初始化 {args.model} 模型 (任务: {args.task_name})...")

    if args.task_name == 'classification':
        if args.model == 'XGBoost':
            model = XGBClassifier(n_estimators=args.n_estimators, learning_rate=0.1, n_jobs=args.n_jobs,
                                  random_state=args.random_state, eval_metric='mlogloss')
        elif args.model == 'HistGB':
            model = HistGradientBoostingClassifier(max_iter=args.n_estimators, learning_rate=0.1,
                                                   random_state=args.random_state)
        elif args.model == 'SVM':
            model = SVC(kernel='rbf', probability=False, random_state=args.random_state)
        elif args.model == 'Rocket':
            model = RocketWrapper(task_name=args.task_name, seq_len=args.seq_len, enc_in=enc_in,
                                  num_kernels=args.num_kernels, random_state=args.random_state)
        else:
            model = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=args.n_jobs,
                                           random_state=args.random_state)
    else:
        if args.model == 'XGBoost':
            model = MultiOutputRegressor(
                XGBRegressor(n_estimators=args.n_estimators, learning_rate=0.1, n_jobs=args.n_jobs,
                             random_state=args.random_state))
        elif args.model == 'HistGB':
            model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=args.n_estimators, learning_rate=0.1,
                                                                       random_state=args.random_state))
        elif args.model == 'SVM':
            model = MultiOutputRegressor(SVR(kernel='rbf'))
        elif args.model == 'Rocket':
            model = RocketWrapper(task_name=args.task_name, seq_len=args.seq_len, enc_in=enc_in,
                                  num_kernels=args.num_kernels, random_state=args.random_state)
        elif args.model == 'ARIMA':
            model = ARIMAWrapper(seq_len=args.seq_len, pred_len=args.pred_len, enc_in=enc_in, order=order)
        else:
            model = RandomForestRegressor(n_estimators=args.n_estimators, n_jobs=args.n_jobs,
                                          random_state=args.random_state)

    # 4. 训练与预测
    print(f"开始训练 {args.model}...")
    start_time = time.time()
    model.fit(X_train, Y_train)
    print(f"训练完成！耗时: {time.time() - start_time:.2f} 秒")

    print("开始在测试集上进行预测...")
    pred_start_time = time.time()
    Y_pred = model.predict(X_test)
    print(f"预测完成！耗时: {time.time() - pred_start_time:.2f} 秒")

    # ================= ✅ 修改点 2：反归一化处理 =================
    if args.task_name != 'classification' and args.inverse:
        if hasattr(dataset_test, 'inverse_transform'):
            print("\n正在对预测结果和真实标签进行反归一化...")
            # 计算目标维度 (比如 M 任务就是所有的特征数，S 任务就是 1)
            target_dim = Y_test.shape[1] // args.pred_len

            # TSLib 的 inverse_transform 通常依赖底层的 sklearn StandardScaler
            # scaler 强制要求输入是 2D 的 [samples, features]
            Y_pred_2d = Y_pred.reshape(-1, target_dim)
            Y_test_2d = Y_test.reshape(-1, target_dim)

            try:
                # 调用数据集自带的反归一化方法
                Y_pred_inv = dataset_test.inverse_transform(Y_pred_2d)
                Y_test_inv = dataset_test.inverse_transform(Y_test_2d)

                # 重新展平回模型输出的形状，用于后续指标计算
                Y_pred = Y_pred_inv.reshape(Y_pred.shape)
                Y_test = Y_test_inv.reshape(Y_test.shape)
                print("✅ 反归一化成功！计算的 MSE/MAE 将是原始数据尺度。")
            except Exception as e:
                print(
                    f"⚠️ 反归一化失败。这通常是因为特征维度不匹配（例如使用了 --features MS，但预测输出维度为1，而 scaler 是在多维上 fit 的）。")
                print(f"报错信息: {e}")
                print("将继续使用【归一化后】的数据计算指标。")
        else:
            print("当前数据集不支持 inverse_transform，跳过反归一化。")

    # 5. 计算指标与保存
    print("\n================ 测试结果 ================")
    print(f"Task: {args.task_name} | Dataset: {args.data} | Model: {args.model}")

    run_id = time.strftime("%m%d_%H%M%S")
    setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_rs{args.random_state}_{run_id}'
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = f'result_{args.task_name}.txt'
    with open(os.path.join(folder_path, file_name), 'a') as f:
        f.write(setting + "  \n")

        if args.task_name == 'classification':
            acc = accuracy_score(Y_test, Y_pred)
            f1 = f1_score(Y_test, Y_pred, average='macro')
            print(f"Accuracy: {acc:.4f} \nMacro F1: {f1:.4f}")
            f.write(f'accuracy:{acc}  \nf1:{f1}\n\n')
        else:
            mse = mean_squared_error(Y_test, Y_pred)
            mae = mean_absolute_error(Y_test, Y_pred)
            print(f"MSE: {mse:.4f} \nMAE: {mae:.4f}")
            f.write(f'mse:{mse}  \nmae:{mae}\n\n')

    print("==========================================")

    np.save(folder_path + 'pred.npy', Y_pred)
    np.save(folder_path + 'true.npy', Y_test)


if __name__ == '__main__':
    main()