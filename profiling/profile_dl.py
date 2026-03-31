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

        # 唤醒 CUDA 上下文 (极度重要：避免首次测算显存时把 CUDA 初始化的几百兆算进去)
        if self.device.type == 'cuda':
            _ = torch.zeros(1).to(self.device)
            torch.cuda.synchronize()

        # 延迟初始化数据，初始设为 None
        self.x_enc = None
        self.x_mark_enc = None
        self.x_dec = None
        self.x_mark_dec = None

    def _generate_data(self):
        """延迟生成 Dummy 数据到指定设备"""
        if self.x_enc is None:
            self.x_enc = torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in).to(self.device)
            # 大部分分类任务在 TSLib 中不需要 decoder 输入
            self.x_mark_enc = torch.ones(self.args.batch_size, self.args.seq_len).to(self.device)
            self.x_dec = None
            self.x_mark_dec = None

    def _get_file_size_mb(self, file_path):
        if not os.path.exists(file_path):
            return 0.0
        return os.path.getsize(file_path) / (1024 ** 2)

    def profile_model(self, model_class, checkpoint_path=None, cal_method="fvcore"):
        print(f"\n" + "=" * 55)
        print(f"🚀 Profiling Method of MAC and Params: {cal_method.upper()}")
        print(f"📊 Input Shape: [Batch={self.args.batch_size}, SeqLen={self.args.seq_len}, Channels={self.args.enc_in}]")
        print("=" * 55)

        # 1. 测算磁盘大小
        if checkpoint_path and os.path.exists(checkpoint_path):
            disk_size = self._get_file_size_mb(checkpoint_path)
            print(f"[1] 磁盘占用 (Checkpoint Size): {disk_size:.2f} MB")
        else:
            print(f"[1] 磁盘占用: 找不到权重，当前使用随机初始化测算。")

        # ================= 显存精细剥离核心逻辑 =================
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            mem_baseline = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        else:
            mem_baseline = 0

        # 2a. 测算【输入数据静态显存】
        self._generate_data()
        if self.device.type == 'cuda':
            mem_after_data = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            data_static_vram = mem_after_data - mem_baseline
        else:
            data_static_vram = 0
        print(f"[2a] 输入数据占用 (Data VRAM): {data_static_vram:.2f} MB")

        # 2b. 实例化模型并测算【模型纯净显存】
        model = model_class(self.args).float().to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            try: # 多卡训练参数名会出现module.前缀，这里做了一个鲁棒加载
                # 1. 先把带有 module. 前缀的原始权重加载进内存
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                # 2. 创建一个干净的新字典
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                # 3. 遍历原权重，把 'module.' 前缀统统切掉
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # 截取掉前面7个字符 'module.'
                    new_state_dict[name] = v
                # 4. 把洗干净的权重装进单卡模型里
                model.load_state_dict(new_state_dict)
                # print(f"    ✅ 权重加载成功！(已自动剥离 DataParallel 壳)")
            except Exception as e:
                print(f"    [警告] 权重加载失败: {e}")
        del state_dict
        del new_state_dict
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        if self.device.type == 'cuda':
            mem_after_model = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            model_static_vram = mem_after_model - mem_after_data
        else:
            model_static_vram = 0
        print(f"[2b] 模型纯净占用 (Model Weights VRAM): {model_static_vram:.2f} MB")
        # ========================================================

        model.eval()  # 切换到推理模式

        # 3. 测算计算量(FLOPs)和参数量(Params)
        if cal_method == 'fvcore':
            try:
                from fvcore.nn import FlopCountAnalysis, parameter_count

                # 构造输入元组 (过滤掉 None，因为 fvcore 对 None 的解析比较严格)
                # inputs = tuple([x for x in (self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec) if x is not None])
                inputs = (self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)

                # 使用 fvcore 进行底层追踪
                flops_analyzer = FlopCountAnalysis(model, inputs)

                # 屏蔽掉 fvcore 烦人的 warning 输出
                flops_analyzer.unsupported_ops_warnings(False)
                flops_analyzer.uncalled_modules_warnings(False)

                macs = flops_analyzer.total()
                params = sum(parameter_count(model).values())

                # 格式化输出 (转换为 M 或 G)
                def format_number(num):
                    if num > 1e9:
                        return f"{num / 1e9:.3f} G"
                    elif num > 1e6:
                        return f"{num / 1e6:.3f} M"
                    else:
                        return f"{num / 1000:.3f} K"

                print(f"[3] 理论计算量 (MACs): {format_number(macs)}")
                print(f"[4] 参数总量 (Params): {format_number(params)}")

            except ImportError:
                print("[3/4] 警告: 未安装 fvcore，请运行 `pip install fvcore`")
            except Exception as e:
                # 连 fvcore 都崩了的话，提供一个纯原生的参数量保底方案
                fallback_params = sum(p.numel() for p in model.parameters())
                print(f"[3] 理论计算量 (MACs): 【动态图测算极其受限，无法追踪】")
                print(f"[4] 参数总量 (Params): {fallback_params / 1e6:.3f} M (原生 API 保底统计)")
                print(f"    -> 报错原因: {e}")
        else:
            try:
                macs, params = profile(model, inputs=(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec),
                                       verbose=False)
                macs_str, params_str = clever_format([macs, params], "%.3f")
                print(f"[3] 理论计算量 (MACs): {macs_str}")
                print(f"[4] 参数总量 (Params): {params_str}")
            except Exception as e:
                print(f"[3/4] 理论计算量与参数量: 【动态图测算受限】\n    -> 原因: {e}")

        # 4. 预热 (Warm-up)
        print(f"[*] 正在 GPU 上进行 Warm-up 预热 (50 次)...")
        with torch.no_grad():
            for _ in range(50):
                _ = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # 5. 测算单样本/Batch前向耗时与峰值显存
        n_runs = 100
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            mem_before_infer = torch.cuda.memory_allocated(self.device)

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # 获取峰值显存
        if self.device.type == 'cuda':
            peak_mem_during_infer = torch.cuda.max_memory_allocated(self.device)
            infer_extra_vram = (peak_mem_during_infer - mem_before_infer) / (1024 ** 2)
        else:
            infer_extra_vram = 0

        # 耗时计算
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_batch_ms = total_time_ms / n_runs
        avg_time_per_sample_ms = avg_time_per_batch_ms / self.args.batch_size

        print(f"[5] 推理额外峰值显存 (Peak Inference VRAM): {infer_extra_vram:.2f} MB (批次={self.args.batch_size})")

        if self.args.batch_size == 1:
            print(f"[6] 单样本前向耗时 (Latency): {avg_time_per_sample_ms:.4f} ms")
        else:
            print(f"[6] Batch前向耗时: {avg_time_per_batch_ms:.4f} ms (单样本分摊: {avg_time_per_sample_ms:.4f} ms)")


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
check_id = 'classification_LandingGearOrigin_Lstm_UEA_ftM_sl7990_ll48_pl0_dm512_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_121630'
check_id = 'classification_LandingGearOrigin_DLinear_UEA_ftM_sl7990_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_121226'
check_id = 'classification_LandingGearOrigin_LightTS_UEA_ftM_sl7990_ll48_pl0_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_121452'
check_id = 'classification_LandingGearOrigin_ResNet50_UEA_ftM_sl7990_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_122021'
check_id = 'classification_LandingGearOrigin_TCN_UEA_ftM_sl7990_ll48_pl0_dm128_nh8_el2_dl1_df2048_expand2_dc100_fc1_ebtimeF_dtTrue_Exp_0_tm0331_122244'
check_id = 'classification_LandingGearOrigin_Transformer_UEA_ftM_sl7990_ll48_pl0_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_123744'
check_id = 'classification_LandingGearOrigin_Informer_UEA_ftM_sl7990_ll48_pl0_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_131008'
check_id = 'classification_LandingGearOrigin_Crossformer_UEA_ftM_sl7990_ll48_pl0_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_131933'
check_id = 'classification_LandingGearOrigin_TimesNet_UEA_ftM_sl7990_ll48_pl0_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_123259'
check_id = 'classification_LandingGearOrigin_iTransformer_UEA_ftM_sl7990_ll48_pl0_dm64_nh8_el2_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_tm0331_132200'
MAC_cal_method = 'fvcore' # fvcore or thop
if __name__ == "__main__":
    args_path = base_path + check_id + '/args_config.json'
    # 1. 构造实验参数
    # 如果要测单样本，将 batch_size 设为 1
    args = DummyArgs(args_path, batch_size=1, seq_len=7990, enc_in=3)

    # 2. 初始化 Profiler
    profiler = DLProfiler(args, device='cuda:0')

    # 3. Model导入
    model_dict = LazyModelDict()
    Model = model_dict[args.model]

    # 4. 执行 Profile
    path = base_path + check_id + '/checkpoint.pth'
    profiler.profile_model(
        model_class=Model,
        checkpoint_path=path,
        cal_method=MAC_cal_method
    )