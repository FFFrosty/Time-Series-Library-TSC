import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

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
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Model(nn.Module):
    """
    双分支 TCN 模型 (Dual-Stream TCN)
    自动识别时域和频域通道，分别提取特征后在特征层进行特征拼接融合。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in  # 传入的总特征通道数 (例如: 3时域 + 3频域 = 6)

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # 获取 TCN 的隐藏层维度和卷积核大小
        d_model = getattr(configs, 'd_model', 64)
        num_channels = [d_model, d_model, d_model]
        kernel_size = getattr(configs, 'd_conv', 3)  # 添加了 fallback 防止 configs 里没有 d_conv
        dropout = getattr(configs, 'dropout', 0.1)

        # ==================== 构建双分支 TCN ====================
        # 1. 强制前 3 个通道为时域
        self.time_channels = 3
        # 2. 剩余通道分配给频域 (如果 enc_in 只有 3，那么 freq_channels 为 0)
        self.freq_channels = self.enc_in - self.time_channels

        # 时域分支 (输入通道固定为 3)
        self.tcn_time = TemporalConvNet(self.time_channels, num_channels, kernel_size=kernel_size, dropout=dropout)

        # 频域分支 (动态构建)
        if self.freq_channels > 0:
            self.tcn_freq = TemporalConvNet(self.freq_channels, num_channels, kernel_size=kernel_size, dropout=dropout)
            # 如果双分支都激活，融合后的通道数是单分支的两倍
            out_channels_total = num_channels[-1] * 2
        else:
            self.tcn_freq = None
            out_channels_total = num_channels[-1]

        # ==================== 任务输出映射头 ====================
        if self.task_name == 'classification':
            self.projection = nn.Linear(out_channels_total, configs.c_out)

        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(out_channels_total, self.pred_len * self.enc_in)

        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(out_channels_total, self.seq_len * self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # TSLib 输入 shape: [Batch, seq_len, features]
        # PyTorch Conv1d 期望 shape: [Batch, Channels(features), Length(seq_len)]
        x = x_enc.permute(0, 2, 1)

        # 1. 切分出时域数据送入时域分支
        # 取出前 3 个通道, Shape: [Batch, 3, seq_len]
        x_time = x[:, :self.time_channels, :]
        y_time = self.tcn_time(x_time)  # 输出 Shape: [Batch, d_model, seq_len]

        # 2. 切分出频域数据送入频域分支 (如果存在的话)
        if self.freq_channels > 0:
            x_freq = x[:, self.time_channels:, :]
            y_freq = self.tcn_freq(x_freq)

            # 特征融合：在通道维度(dim=1)拼接时域和频域提取的高维特征
            # [Batch, d_model, seq_len] + [Batch, d_model, seq_len] -> [Batch, d_model*2, seq_len]
            y_fused = torch.cat([y_time, y_freq], dim=1)
        else:
            y_fused = y_time

        # 3. 后端映射输出
        if self.task_name == 'classification':
            # 分类任务：使用全局平均池化 (GAP)
            # [Batch, out_channels_total, seq_len] -> [Batch, out_channels_total]
            pooled_y = F.adaptive_avg_pool1d(y_fused, 1).squeeze(-1)
            # [Batch, num_class]
            return self.projection(pooled_y)

        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 取最后一个时间步
            last_y = y_fused[:, :, -1]
            output = self.projection(last_y)
            return output.view(output.shape[0], self.pred_len, self.enc_in)

        elif self.task_name in ['imputation', 'anomaly_detection']:
            # 序列映射
            y_fused = y_fused.permute(0, 2, 1)
            output = self.projection(y_fused)
            return output.view(output.shape[0], self.seq_len, self.enc_in)

        return None