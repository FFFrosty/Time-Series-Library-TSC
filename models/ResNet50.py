import torch
import torch.nn as nn


class Bottleneck1D(nn.Module):
    """
    1D 版本的 ResNet Bottleneck 模块
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Model(nn.Module):
    """
    基于 1D ResNet50 的时间序列模型，完全兼容 TSLib 接口
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in  # 输入特征维度 (即 Conv1d 的输入通道数)

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # ==================== 构建 1D ResNet50 ====================
        self.inplanes = 64
        # 初始卷积层：将时间序列的特征维度作为通道数输入
        self.conv1 = nn.Conv1d(self.enc_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet50 结构: [3, 4, 6, 3] 个 Bottleneck
        self.layer1 = self._make_layer(Bottleneck1D, 64, 3)
        self.layer2 = self._make_layer(Bottleneck1D, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck1D, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck1D, 512, 3, stride=2)

        # 全局自适应平均池化，将任意长度的序列池化到长度为 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 提取特征后的最终维度: 512 * 4 = 2048
        resnet_out_dim = 512 * Bottleneck1D.expansion

        # ==================== 任务输出映射头 ====================
        if self.task_name == 'classification':
            self.projection = nn.Linear(resnet_out_dim, configs.c_out)
        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 预测任务：将 2048 维映射到 [pred_len * features]，以便之后 Reshape
            self.projection = nn.Linear(resnet_out_dim, self.pred_len * self.enc_in)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(resnet_out_dim, self.seq_len * self.enc_in)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 输入 shape: [Batch, seq_len, features]
        # Conv1d 期望的 shape: [Batch, Channels, Length]
        # 因此我们需要置换维度：[Batch, features, seq_len]
        x = x_enc.permute(0, 2, 1)

        # 过 ResNet 特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # shape: [Batch, 2048, 1]
        x = torch.flatten(x, 1)  # shape: [Batch, 2048]

        # 根据任务类型进行映射
        if self.task_name == 'classification':
            # shape: [Batch, num_classes]
            output = self.projection(x)
            return output

        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # shape: [Batch, pred_len * features]
            output = self.projection(x)
            # 恢复为标准的 TSLib 输出 shape: [Batch, pred_len, features]
            output = output.view(output.shape[0], self.pred_len, self.enc_in)
            return output

        elif self.task_name in ['imputation', 'anomaly_detection']:
            # shape: [Batch, seq_len * features]
            output = self.projection(x)
            # 恢复 shape: [Batch, seq_len, features]
            output = output.view(output.shape[0], self.seq_len, self.enc_in)
            return output

        return None