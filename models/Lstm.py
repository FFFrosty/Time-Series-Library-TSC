import torch
import torch.nn as nn


class Model(nn.Module):
    """
    基于 LSTM 的时间序列多任务模型，完全兼容 DLinear 的外部接口
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        # 判断是需要输出预测长度，还是原始序列长度
        if self.task_name in ['classification', 'anomaly_detection', 'imputation']:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.hidden_size = configs.d_ff  # 可以根据需要暴露到 configs 中
        self.num_layers = configs.e_layers

        # 核心 LSTM 层
        self.lstm = nn.LSTM(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.1
        )

        # 针对不同任务的输出映射层
        if self.task_name == 'classification':
            # 分类任务：将最后一步的隐藏状态映射到类别数
            self.projection = nn.Linear(self.hidden_size, configs.num_class)
        else:
            # 预测/异常检测/插补：将隐藏状态映射回特征维度
            self.feature_projection = nn.Linear(self.hidden_size, self.channels)

            # 针对预测任务，还需要调整序列长度 (从 seq_len 映射到 pred_len)
            if self.task_name in ['long_term_forecast', 'short_term_forecast']:
                self.seq_projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc 的形状通常为: [Batch, seq_len, Features]

        # 1. 过 LSTM 提取时序特征
        lstm_out, _ = self.lstm(x_enc)  # lstm_out: [B, seq_len, hidden_size]

        # 2. 根据任务类型处理输出
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 映射特征维度: [B, seq_len, hidden_size] -> [B, seq_len, Features]
            out = self.feature_projection(lstm_out)
            # 调整时间步维度: [B, seq_len, Features] -> [B, Features, seq_len] -> [B, Features, pred_len] -> [B, pred_len, Features]
            out = self.seq_projection(out.permute(0, 2, 1)).permute(0, 2, 1)
            return out

        elif self.task_name in ['imputation', 'anomaly_detection']:
            # 保持输入长度，直接映射回特征维度: [B, seq_len, Features]
            return self.feature_projection(lstm_out)

        elif self.task_name == 'classification':
            # 取 LSTM 序列的最后一步的输出: [B, hidden_size]
            last_hidden_state = lstm_out[:, -1, :]
            # 映射到类别概率: [B, num_classes]
            return self.projection(last_hidden_state)

        return None