import os
import time
import psutil
import joblib
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin
# run_ml_baseline.py使用了__main__，因此RocketWrapper的索引被标记为__main__.RocketWrapper，这里需要补一个模型定义
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


class MLProfiler:
    def __init__(self, batch_size=1, seq_len=7990, n_channels=3):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.process = psutil.Process(os.getpid())

        # # 统一生成 Dummy 数据
        # # Rocket 需要 3D: (Batch, Channels, Seq_Len)
        # self.X_3d = np.random.randn(self.batch_size, self.n_channels, self.seq_len).astype(np.float32)
        # # RF/XGB 需要 2D 展平: (Batch, Channels * Seq_Len)
        # self.X_2d = self.X_3d.reshape(self.batch_size, -1)
        self.X_2d = None
        self.X_3d = None

    def _generate_data(self):
        """延迟生成Dummy数据，便于被监控"""
        # Rocket 需要 3D: (Batch, Channels, Seq_Len)
        self.X_3d = np.random.randn(self.batch_size, self.n_channels, self.seq_len).astype(np.float32)
        # RF/XGB 需要 2D 展平: (Batch, Channels * Seq_Len)
        self.X_2d = self.X_3d.reshape(self.batch_size, -1)

    def _get_memory_mb(self):
        """获取当前进程的物理内存占用 (MB)"""
        return self.process.memory_info().rss / (1024 ** 2)

    def _get_file_size_mb(self, file_path):
        """获取模型文件在磁盘上的大小 (MB)"""
        if not os.path.exists(file_path):
            return 0.0
        return os.path.getsize(file_path) / (1024 ** 2)

    def _extract_complexity(self, model, model_type):
        """提取传统模型的复杂度指标 (替代 FLOPs)"""
        if model_type == 'rf':
            # 统计随机森林的总节点数和树的数量
            n_trees = len(model.estimators_)
            n_nodes = sum([tree.tree_.node_count for tree in model.estimators_])
            return f"{n_trees} Trees, {n_nodes} Total Nodes"

        elif model_type == 'xgb':
            # XGBoost 的树结构信息
            booster = model.get_booster()
            df_trees = booster.trees_to_dataframe()
            n_trees = df_trees['Tree'].nunique()
            n_nodes = len(df_trees)
            return f"{n_trees} Trees, {n_nodes} Total Nodes"

        elif model_type == 'rocket':
            # Rocket 主要是固定数量的随机卷积核
            # sktime 的 MiniRocket 默认或自定义的 kernel 数量
            n_kernels = getattr(model, 'num_kernels', 'Unknown')
            return f"{n_kernels} Random Kernels"
        return "N/A"

    def profile_model(self, model_path, model_type):
        print(f"\n" + "=" * 50)
        print(f"🚀 Profiling Model: {model_type.upper()}")
        print(f"📦 File: {model_path}")
        print(f"📊 Input Shape: Batch={self.batch_size}, Channels={self.n_channels}, SeqLen={self.seq_len}")
        print("=" * 50)

        # 1. 测算磁盘大小
        disk_size = self._get_file_size_mb(model_path)
        print(f"[1] 磁盘占用 (Disk Size): {disk_size:.2f} MB")

        # 2. 测算模型加载后的静态内存
        # 记录最原始的进程底噪内存
        mem_baseline = self._get_memory_mb()
        # 2a. 测算【输入数据静态内存】
        self._generate_data()
        mem_after_data = self._get_memory_mb()
        data_static_mem = mem_after_data - mem_baseline
        print(f"[2a] 输入数据占用 (Data RAM): {data_static_mem:.2f} MB")

        # 2b. 测算模型静态内存
        if model_type == 'xgb':
            model = xgb.XGBClassifier()
            model.load_model(model_path)
        else:
            model = joblib.load(model_path)

        mem_after_model = self._get_memory_mb()
        # 注意这里是减去数据加载后的内存，从而单独剥离出模型的纯体积
        model_static_mem = mem_after_model - mem_after_data
        print(f"[2b] 模型纯净占用 (Model RAM): {model_static_mem:.2f} MB")

        # 3. 提取算法复杂度
        complexity = self._extract_complexity(model, model_type)
        print(f"[3] 算法复杂度 (Complexity): {complexity}")

        # 4. 准备数据并进行预热 (Warm-up)
        # 预热是为了让底层 C 库完成初始化，避免首次前向传播时间异常长
        X_input = self.X_3d if model_type == 'rocket' else self.X_2d

        print(f"[*] 正在预热 (Warm-up) 10 次...")
        for _ in range(10):
            _ = model.predict(X_input)

        # 5. 测算前向传播耗时与峰值内存
        n_runs = 100
        mem_before_infer = self._get_memory_mb()
        max_mem_during_infer = mem_before_infer

        start_time = time.perf_counter()
        for _ in range(n_runs):
            _ = model.predict(X_input)
            # 在循环内部极高频采样内存，捕捉瞬间峰值
            current_mem = self._get_memory_mb()
            if current_mem > max_mem_during_infer:
                max_mem_during_infer = current_mem

        end_time = time.perf_counter()

        # 计算平均耗时
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_batch_ms = total_time_ms / n_runs
        avg_time_per_sample_ms = avg_time_per_batch_ms / self.batch_size

        # 计算推理瞬间增加的额外内存
        infer_peak_mem = max_mem_during_infer - mem_before_infer

        print(f"[4] 推理峰值内存 (Peak RAM during predict): {infer_peak_mem:.2f} MB")
        if self.batch_size == 1:
            print(f"[5] 单样本前向传播耗时 (Latency): {avg_time_per_sample_ms:.4f} ms")
        else:
            print(f"[5] 整个Batch耗时: {avg_time_per_batch_ms:.4f} ms (单样本分摊: {avg_time_per_sample_ms:.4f} ms)")


# ==========================================
# 执行测试
# ==========================================
base_path = 'checkpoints/'
check_id = 'classification_LandingGearOrigin_RandomForest_UEA_ftM_sl7990_ll48_pl0_rs2024_0331_132410' # 20260331 RF开销测试
check_id = 'classification_LandingGearOrigin_XGBoost_UEA_ftM_sl7990_ll48_pl0_rs2024_0331_132907' # 20260331 XGBoost开销测试
check_id = 'classification_LandingGearOrigin_Rocket_UEA_ftM_sl7990_ll48_pl0_rs2026_0331_132615' # 20260331 Rocket开销测试
path_head = base_path + check_id + '/'
if __name__ == "__main__":
    # 你可以通过修改这里的 batch_size 来自由切换单样本测试 (batch=1) 还是批量测试 (batch=128 等)
    profiler = MLProfiler(batch_size=1, seq_len=7990, n_channels=3)

    # 请确保对应目录下存在这三个模型文件
    model_configs = [
        ("rf_model.joblib", "rf"),
        ("xgb_model.json", "xgb"),
        ("rocket_model.joblib", "rocket")
    ]

    for param_file, m_type in model_configs:
        path = path_head + param_file
        if os.path.exists(path):
            profiler.profile_model(path, m_type)
        else:
            print(f"\n[!] 找不到文件: {path}，跳过 {m_type.upper()} 测试。")