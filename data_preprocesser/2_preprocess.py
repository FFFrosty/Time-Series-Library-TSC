import numpy as np
from sklearn.model_selection import train_test_split

# ================= 1. 配置参数 =================
INPUT_FILE = "data_preprocesser/SensorData.ts"
TRAIN_FILE = "data_preprocesser/LandingGear_TRAIN.ts"
TEST_FILE = "data_preprocesser/LandingGear_TEST.ts"
PROBLEM_NAME = "LandingGear"

START_IDX = 1000
END_IDX = 4000
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 传感器通道索引 (假设顺序为 FirstA:0, LaLi:1, SecA:2)
LALI_INDEX = 1


# ================= 2. 核心功能函数 =================

def load_ts_to_numpy(filepath):
    """高效读取 .ts 文件并转为 NumPy 数组"""
    print(f"正在读取 {filepath} ...")
    X_list, y_list = [], []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@'):
                continue

            # 【修复 1】：将逗号改为冒号，正确剥离末尾的类别标签
            data_str, label = line.rsplit(':', 1)
            # 提取 3 个通道的数据
            dimensions = [np.fromstring(dim, sep=',') for dim in data_str.split(':')]

            X_list.append(dimensions)
            y_list.append(label)

    X = np.array(X_list)  # 形状: (N, M, T)
    y = np.array(y_list)
    print(f"读取完毕！数据形状: {X.shape}")
    return X, y


def save_numpy_to_ts(X, y, filepath, problem_name):
    """将 NumPy 数组写回为标准 .ts 格式"""
    print(f"正在写入 {filepath} ...")
    N, M, T = X.shape
    unique_classes = np.unique(y)

    with open(filepath, 'w') as f:
        # 写入元数据 Header
        f.write(f"@problemName {problem_name}\n")
        f.write("@timeStamps false\n")
        f.write("@missing false\n")
        f.write(f"@univariate {'true' if M == 1 else 'false'}\n")
        f.write("@equalLength true\n")
        class_str = " ".join([str(c) for c in unique_classes])
        f.write(f"@classLabel true {class_str}\n")
        f.write("@data\n")

        # 写入数据 Data
        for i in range(N):
            dimensions = []
            for j in range(M):
                # 转换为字符串并用逗号连接同一通道的时间步
                series_str = ",".join(X[i, j, :].astype(str))
                dimensions.append(series_str)

            # 【修复 2】：将类别标签加入列表，统一用冒号连接
            dimensions.append(str(y[i]))
            instance_str = ":".join(dimensions) + "\n"

            f.write(instance_str)

    print(f"写入成功！数据形状: {X.shape}")


# ================= 3. 执行主流程 =================

if __name__ == "__main__":
    # 步骤 A: 加载原始数据
    X, y = load_ts_to_numpy(INPUT_FILE)

    # 步骤 B：【修复 3】使用 np.clip 解决负极值被翻转为正极值的 Bug
    print("正在处理极端值...")
    threshold_FA = 50
    threshold_LL = 3000
    threshold_SA = 50

    # clip(a, a_min, a_max) 会把超出范围的值限制在上下界之内，保留符号方向
    X[:, 0, :] = np.clip(X[:, 0, :], -threshold_FA, threshold_FA)
    X[:, 1, :] = np.clip(X[:, 1, :], -threshold_LL, threshold_LL)
    X[:, 2, :] = np.clip(X[:, 2, :], -threshold_SA, threshold_SA)

    # 步骤 C: 对 LaLi 通道进行一阶差分 (保持原数组长度不变)
    print("正在对 LaLi 传感器进行一阶差分...")
    lali_data = X[:, LALI_INDEX, :]
    # 这里你的写法很棒，prepend=0 是 Numpy 更现代的补齐方式
    X[:, LALI_INDEX, :] = np.diff(lali_data, axis=1, prepend=0)

    # 步骤 D: 时间窗口裁剪 (提取 1000 到 4000)
    print(f"正在裁剪时间窗口 [{START_IDX}:{END_IDX}]...")
    X_cropped = X[:, :, START_IDX:END_IDX]

    # 步骤 E: 分层划分训练集和测试集
    print("正在进行训练集/测试集划分...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_cropped, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # 步骤 F: 通道独立归一化 (Z-Score) -> 严防数据泄露！
    print("正在执行通道独立的 Z-Score 归一化...")
    # 仅使用训练集计算均值和标准差
    train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    train_std = np.std(X_train, axis=(0, 2), keepdims=True)

    # 将训练集的统计参数同时应用于训练集和测试集
    X_train_norm = (X_train - train_mean) / (train_std + 1e-8)
    X_test_norm = (X_test - train_mean) / (train_std + 1e-8)

    # 步骤 G: 导出最终文件
    save_numpy_to_ts(X_train_norm, y_train, TRAIN_FILE, PROBLEM_NAME)
    save_numpy_to_ts(X_test_norm, y_test, TEST_FILE, PROBLEM_NAME)

    print("\n所有预处理和划分任务已圆满完成！")