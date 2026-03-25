import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AMS import AMS_intralinear
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    TSLib-compatible Pathformer Model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # TSLib 通常使用 enc_in 表示变量数，兼容 Pathformer 的 num_nodes
        self.num_nodes = getattr(configs, 'num_nodes', configs.enc_in)

        # Pathformer 超参数
        self.layer_nums = configs.e_layers
        self.k = configs.k
        self.num_experts_list = [int(x) for x in configs.num_experts_list.split(',')]
        self.patch_size_list = [int(x) for x in configs.patch_size_list.split(',')]
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.batch_norm = configs.batch_norm
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        # 1. 数据预处理
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=self.num_nodes, affine=False, subtract_last=False)

        # 2. 初始特征嵌入
        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)

        # 3. 核心特征提取 (AMS 多尺度路由块)
        self.AMS_lists = nn.ModuleList()
        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS_intralinear(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm)
            )

        # ==================== 针对 TSLib 的多任务 Head ====================
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projections = nn.Sequential(
                nn.Linear(self.seq_len * self.d_model, self.pred_len)
            )

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            # 分类任务：将 (seq_len * num_nodes * d_model) 展平后映射到类别数
            self.projections = nn.Linear(self.seq_len * self.num_nodes * self.d_model, configs.num_class)

    def forecast(self, x_enc):
        balance_loss = 0
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')

        out = self.start_fc(x_enc.unsqueeze(-1))

        # AMS 特征提取
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        # 预测头映射
        batch_size = x_enc.shape[0]
        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, balance_loss

    def classification(self, x_enc, x_mark_enc):
        balance_loss = 0
        # 注意：在分类任务中，绝对幅值通常是重要特征，一般不建议使用 RevIN。
        # 如果你明确需要抗漂移，可以解除下面的注释。
        # if self.revin:
        #     x_enc = self.revin_layer(x_enc, 'norm')

        out = self.start_fc(x_enc.unsqueeze(-1))

        # AMS 特征提取
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        # 分类头映射
        out = self.act(out)
        out = self.dropout(out)

        # 展平特征矩阵 [B, seq_len, num_nodes, d_model] -> [B, -1]
        out = out.reshape(out.shape[0], -1)
        out = self.projections(out)  # 输出 [B, num_class]

        return out, balance_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        TSLib 标准 Forward 接口
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, balance_loss = self.forecast(x_enc)
            # TSLib 通常截取最后 pred_len 长度
            return dec_out[:, -self.pred_len:, :], balance_loss

        if self.task_name == 'classification':
            dec_out, balance_loss = self.classification(x_enc, x_mark_enc)
            return dec_out, balance_loss

        return None