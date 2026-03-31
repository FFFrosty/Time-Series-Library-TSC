import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def save_args_to_txt(args, save_dir):
    """将超参数以友好的格式保存到 txt 文件中"""
    file_path = os.path.join(save_dir, 'args_config.txt')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(" " * 15 + "EXPERIMENT ARGUMENTS\n")
        f.write("=" * 50 + "\n\n")

        # 将参数按字母顺序排序，看起来更清爽
        for key, value in sorted(vars(args).items()):
            f.write(f"{key:<25}: {value}\n")

    # print(f"🚀 超参数已安全保存至: {file_path}")


class NumpyEncoder(json.JSONEncoder):
    """
    一个自定义的 JSON 编码器，用于将 NumPy 数据类型转换为标准 Python 类型。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_args_to_json(args, save_dir):
    """将超参数保存为标准 JSON，自动过滤掉 device 等无法序列化的对象"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'args_config.json')

    args_dict = vars(args)
    safe_dict = {}
    # 核心逻辑：逐个试探，只保留安全的参数
    for key, value in args_dict.items():
        try:
            # 试探性地对这个 value 进行 JSON 转换
            json.dumps(value, cls=NumpyEncoder)
            safe_dict[key] = value
        except TypeError:
            # 如果抛出 TypeError，说明是 device, function, logger 等奇怪的对象，直接丢弃
            # print(f"   [自动过滤] 参数 '{key}' (类型: {type(value).__name__}) 无法序列化，已跳过。")
            continue
    # 将清洗后的干净字典写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(safe_dict, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    # print(f"✅ 超参数已安全清洗并保存至: {file_path}")
