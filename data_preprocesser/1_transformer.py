import os
import pandas as pd
import numpy as np

# ================= 1. 配置参数 =================
data_dir = r'D:\ExperimentData' # 替换为存放那 21 个 xlsx 文件的文件夹路径
output_ts_file = "SensorData.ts"  # 输出的 .ts 文件名
problem_name = "SensorClassification"

sensors = ["FirstA", "LaLi", "SecA"]
classes = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

X_all = []
y_all = []

# ================= 2. 读取并重组数据 =================
print("开始读取和重组数据...")

for c, label_ in enumerate(classes):
    class_data = []  # 临时存放当前类别的 3 个传感器数据

    for sensor in sensors:
        filename = f"{sensor}{c+1}.xlsx"
        filepath = os.path.join(data_dir, filename)

        print(f"正在读取: {filename} ...")
        # 假设你的 Excel 没有表头纯数据。如果有表头，请将 header=None 改为 header=0
        df = pd.read_excel(filepath, header=None)

        # 当前 df 形状是 (7000, 200)，即 (时间步, 样本数)
        # 我们需要转置它，变成 (200, 7000)，即 (样本数, 时间步)
        sensor_data_transposed = df.values.T
        class_data.append(sensor_data_transposed)

    # 将当前类别的 3 个传感器数据堆叠起来
    # class_data 包含 3 个 (200, 7000) 的数组
    # 堆叠后 class_X 形状变为 (200, 3, 7000)，完全符合 (N, M, T)
    class_X = np.stack(class_data, axis=1)
    X_all.append(class_X)

    # 生成对应的标签，当前类别有 200 个样本
    num_samples = class_X.shape[0]
    class_y = np.full(num_samples, label_)
    y_all.append(class_y)

# 沿样本维度(axis=0)合并所有 7 个类别的数据
X = np.concatenate(X_all, axis=0)  # 最终形状: (1400, 3, 7000)
y = np.concatenate(y_all, axis=0)  # 最终形状: (1400,)

print(f"\n数据组装完成！")
print(f"X 的形状: {X.shape} (样本数, 变量数, 序列长度)")
print(f"y 的形状: {y.shape}")

# ================= 3. 写入 .ts 格式文件 =================
print(f"\n开始写入 {output_ts_file} (数据量较大，请耐心等待)...")


def write_to_ts(X, y, output_file, dataset_name):
    N, M, T = X.shape
    unique_classes = np.unique(y)

    with open(output_file, 'w') as f:
        # 写入元数据 (Header)
        f.write(f"@problemName {dataset_name}\n")
        f.write("@timeStamps false\n")
        f.write("@missing false\n")
        f.write(f"@univariate {'true' if M == 1 else 'false'}\n")
        f.write("@equalLength true\n")
        class_str = " ".join([str(c) for c in unique_classes])
        f.write(f"@classLabel true {class_str}\n")
        f.write("@data\n")

        # 写入实际数据 (Data)
        for i in range(N):
            dimensions = []
            for j in range(M):
                # 用逗号连接同一传感器的时间序列
                # 因为数据量大，这里先将其转为字符串列表再 join，速度更快
                series_str = ",".join(X[i, j, :].astype(str))
                dimensions.append(series_str)

            # 将类别标签也转为字符串并加入列表
            dimensions.append(str(y[i]))
            # 用冒号统一连接所有变量以及最后的类别标签
            instance_str = ":".join(dimensions) + "\n"
            f.write(instance_str)


write_to_ts(X, y, output_ts_file, problem_name)
print("转换成功！文件已保存。")