import sys
import os

# 将项目根目录（profiling 的上一级）加入 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import torch
import torch.nn as nn
from thop import profile, clever_format
import json
import importlib


# 假设这里导入你 TSLib 中的模型类，例如：
# from models import Pathformer, Informer, TimesNet, ResNet

class DummyArgs:
    """
    通过读取 JSON 文件动态生成的参数类，完美复原 TSLib 训练时的配置。
    """

    def __init__(self, json_path, **override_kwargs):
        # 1. 从 JSON 文件中加载所有原始超参数
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[!] 找不到参数文件: {json_path}。请检查路径！")

        with open(json_path, 'r', encoding='utf-8') as f:
            args_dict = json.load(f)

        # 动态将字典键值对转换为类的属性 (等价于 argparse.Namespace)
        for key, value in args_dict.items():
            setattr(self, key, value)

        print(f"✅ 已成功从 {json_path} 加载 {len(args_dict)} 个模型超参数。")

        # 2. 覆盖特定参数 (用于 Profiling)
        # 训练时的 batch_size 可能是 32，但评测时我们往往想强制测算 batch_size=1 的单样本开销
        for key, value in override_kwargs.items():
            setattr(self, key, value)
            print(f"   -> [覆盖] {key} = {value}")

        # # 3. 容错处理：确保一些基准属性存在 (以防 JSON 里漏存了)
        # if not hasattr(self, 'num_nodes') and hasattr(self, 'enc_in'):
        #     self.num_nodes = self.enc_in  # Pathformer 等模型可能需要的别名兼容

class DLProfiler:
    def __init__(self, args, device='cuda:0'):
        self.args = args
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 按照 TSLib 统一维度: [Batch, Seq_Len, Channels]
        # 注意：这里我们生成全 1 矩阵或正态分布矩阵作为 Dummy Data
        self.x_enc = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(self.device)

        # 大部分分类任务在 TSLib 中不需要 decoder 输入，这里设为 None 或 dummy
        self.x_mark_enc = None
        self.x_dec = None
        self.x_mark_dec = None

    def _get_file_size_mb(self, file_path):
        if not os.path.exists(file_path):
            return 0.0
        return os.path.getsize(file_path) / (1024 ** 2)

    def profile_model(self, model_class, checkpoint_path=None, model_name="DL_Model"):
        print(f"\n" + "=" * 55)
        print(f"🚀 Profiling DL Model: {model_name.upper()}")
        print(f"📊 Input Shape: [Batch={self.args.batch_size}, SeqLen={self.args.seq_len}, Channels={self.args.enc_in}]")
        print("=" * 55)

        # 1. 实例化模型并加载权重
        model = model_class(self.args).float().to(self.device)

        # 测算磁盘大小
        if checkpoint_path and os.path.exists(checkpoint_path):
            disk_size = self._get_file_size_mb(checkpoint_path)
            print(f"[1] 磁盘占用 (Checkpoint Size): {disk_size:.2f} MB")

            # 加载权重 (如果是 strictly load，建议加上 strict=False 容错)
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            except Exception as e:
                print(f"    [警告] 权重加载失败，使用随机初始化权重继续测算: {e}")
        else:
            print(f"[1] 磁盘占用: 找不到 {checkpoint_path}，当前使用随机初始化权重测算。")

        model.eval()  # 切换到推理模式

        # 2. 测算静态显存 (仅加载模型参数所占的显存)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        static_vram = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        print(f"[2] 静态显存 (Model Weights VRAM): {static_vram:.2f} MB")

        # 3. 测算计算量(FLOPs)和参数量(Params)
        try:
            # thop 的 inputs 需要是一个 tuple，对齐 forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
            macs, params = profile(model, inputs=(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec),
                                   verbose=False)
            macs_str, params_str = clever_format([macs, params], "%.3f")
            # 通常 MACs * 2 约等于 FLOPs
            print(f"[3] 理论计算量 (MACs): {macs_str}")
            print(f"[4] 参数总量 (Params): {params_str}")
        except Exception as e:
            print(f"[3/4] 理论计算量与参数量: 【动态图测算受限】(thop 无法追踪该计算流)\n    -> 报错原因: {e}")

        # 4. 预热 (Warm-up) - 极其重要，为了 CUDA 的懒加载机制
        print(f"[*] 正在 GPU 上进行 Warm-up 预热 (50 次)...")
        with torch.no_grad():
            for _ in range(50):
                _ = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        torch.cuda.synchronize()  # 确保预热任务在 GPU 上完全执行完毕

        # 5. 测算单样本/Batch前向耗时与峰值显存
        n_runs = 100
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before_infer = torch.cuda.memory_allocated(self.device)

        start_time = time.perf_counter()

        with torch.no_grad():  # 推理必须关闭梯度，否则显存会爆炸
            for _ in range(n_runs):
                _ = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)

        torch.cuda.synchronize()  # 必须加上同步原语，否则计时不准！
        end_time = time.perf_counter()

        # 获取峰值显存
        peak_mem_during_infer = torch.cuda.max_memory_allocated(self.device)
        infer_extra_vram = (peak_mem_during_infer - mem_before_infer) / (1024 ** 2)

        # 耗时计算
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_batch_ms = total_time_ms / n_runs
        avg_time_per_sample_ms = avg_time_per_batch_ms / self.args.batch_size

        print(f"[5] 推理额外峰值显存 (Peak Inference VRAM): {infer_extra_vram:.2f} MB (批次={self.args.batch_size})")

        if self.args.batch_size == 1:
            print(f"[6] 单样本前向耗时 (Latency): {avg_time_per_sample_ms:.4f} ms")
        else:
            print(f"[6] Batch前向耗时: {avg_time_per_batch_ms:.4f} ms (折合单样本: {avg_time_per_sample_ms:.4f} ms)")


class LazyModelDict(dict):
    """
    Smart Lazy-Loading Dictionary
    """

    def __init__(self):
        self.model_map = self._scan_models_directory()
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)

        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")

        module_path = self.model_map[key]
        try:
            print(f"🚀 Lazy Loading: {key} ...")
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # Try to find the model class
        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        self[key] = model_class
        return model_class

    def _scan_models_directory(self):
        """
        Automatically scan all .py files in the models folder
        """
        model_map = {}
        models_dir = 'models'

        # Iterate through all files in 'models' directory
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                # Ignore __init__.py and non-.py files
                if filename.endswith('.py') and filename != '__init__.py':
                    # Remove .py extension to get module name
                    module_name = filename[:-3]

                    # Build full import path
                    full_path = f"{models_dir}.{module_name}"

                    # loading dict: {'Transformer': 'models.Transformer'}
                    model_map[module_name] = full_path

        return model_map


# ==========================================
# 执行测试的样例逻辑
# ==========================================
base_path = 'checkpoints/'
check_id = 'classification_LandingGearOrigin_DLinear_UEA_ftM_sl7990_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_115354'
model_name = 'DLinear'
if __name__ == "__main__":
    args_path = base_path + check_id + '/args_config.json'
    # 1. 构造实验参数
    # 如果要测单样本，将 batch_size 设为 1
    args = DummyArgs(args_path, batch_size=1, seq_len=7990, enc_in=3)

    # 2. 初始化 Profiler
    profiler = DLProfiler(args, device='cuda:0')


    # 3. 假设我们要测试之前写的那个 Model
    # 引入你本地的代码： from models.Pathformer import Model as PathformerModel
    model_dict = LazyModelDict()
    Model = model_dict[args.model]

    # # 【示例占位】：这里用一个假的 nn.Module 演示接口调用
    # class MockTSLibModel(nn.Module):
    #     def __init__(self, configs):
    #         super().__init__()
    #         self.linear = nn.Linear(configs.seq_len * configs.enc_in, configs.num_class)
    #
    #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    #         x = x_enc.reshape(x_enc.shape[0], -1)
    #         return self.linear(x)


    # 4. 执行 Profile
    path = base_path + check_id + '/checkpoint.pth'
    profiler.profile_model(
        model_class=Model,
        checkpoint_path=path,
        model_name=model_name
    )