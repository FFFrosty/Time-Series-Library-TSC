import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    裁剪模块：为了实现因果卷积（Causal Convolution），
    我们需要在序列左侧进行 padding，然后通过 Chomp1d 裁剪掉右侧多余的输出，
    确保时刻 t 的输出只依赖于 t 及之前的输入，不会发生“未来信息泄露”。
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 的核心基础块：包含两层膨胀因果卷积和残差连接
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一层卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 残差连接：如果输入输出通道数不一致，用 1x1 卷积调整维度
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    TCN 主干网络：堆叠多个 TemporalBlock，感受野随着层数指数级扩大
    """

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # 膨胀系数以 2 的指数倍递增：1, 2, 4, 8...
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 严格计算 padding 大小以满足因果卷积的要求
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Model(nn.Module):
    """
    包装成 TSLib 标准接口的 TCN 模型
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in  # 输入特征维度

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # ==================== 构建 TCN ====================
        # 从 configs 获取通道数（通常 TSLib 里叫 d_model），如果没有就默认 64
        d_model = getattr(configs, 'd_model', 64)
        # 我们使用 3 层 TCN 结构，通道数保持为 d_model
        num_channels = [d_model, d_model, d_model]
        kernel_size = configs.d_conv
        dropout = getattr(configs, 'dropout', 0.1)

        self.tcn = TemporalConvNet(self.enc_in, num_channels, kernel_size=kernel_size, dropout=dropout)

        # ==================== 任务输出映射头 ====================
        out_channels = num_channels[-1]

        if self.task_name == 'classification':
            self.projection = nn.Linear(out_channels, configs.c_out)
        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(out_channels, self.pred_len * self.enc_in)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(out_channels, self.seq_len * self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # TSLib 输入 shape: [Batch, seq_len, features]
        # PyTorch Conv1d 期望 shape: [Batch, Channels(features), Length(seq_len)]
        x = x_enc.permute(0, 2, 1)

        # [Batch, out_channels, seq_len]
        y = self.tcn(x)

        if self.task_name == 'classification':
            # 分类任务：使用全局平均池化 (GAP) 提取整个序列的特征，鲁棒性更强
            # [Batch, out_channels, seq_len] -> [Batch, out_channels, 1] -> [Batch, out_channels]
            pooled_y = F.adaptive_avg_pool1d(y, 1).squeeze(-1)
            # [Batch, num_class]
            return self.projection(pooled_y)

        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 取最后一个时间步的输出作为整个序列的表征向量
            # [Batch, out_channels]
            last_y = y[:, :, -1]
            output = self.projection(last_y)
            return output.view(output.shape[0], self.pred_len, self.enc_in)

        elif self.task_name in ['imputation', 'anomaly_detection']:
            # 保持序列维度不变进行映射
            # 将 [Batch, out_channels, seq_len] 转换回 [Batch, seq_len, out_channels]
            y = y.permute(0, 2, 1)
            output = self.projection(y)
            return output.view(output.shape[0], self.seq_len, self.enc_in)

        return None