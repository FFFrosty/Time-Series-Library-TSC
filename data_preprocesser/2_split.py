import os
from sklearn.model_selection import train_test_split

# ================= 1. 配置参数 =================
input_ts_file = "SensorData.ts"
train_ts_file = "SensorData_TRAIN.ts"
test_ts_file = "SensorData_TEST.ts"

test_ratio = 0.2  # 测试集占比，例如 0.2 表示 20% 测试集，80% 训练集
random_seed = 42  # 随机种子，确保每次运行划分结果一致

# ================= 2. 读取文本行与拆分 =================
print(f"正在读取 {input_ts_file} ...")

header_lines = []
data_lines = []
labels = []

with open(input_ts_file, 'r') as f:
    is_data_section = False

    for line in f:
        # 如果还没遇到 @data，说明是 Header 部分
        if not is_data_section:
            header_lines.append(line)
            if line.strip().lower() == "@data":
                is_data_section = True
        else:
            # 遇到空行跳过
            if not line.strip():
                continue

            data_lines.append(line)
            # 提取标签：.ts 数据行格式是 "变量1:变量2:变量3,标签"
            # 我们只需要通过最后一个逗号切分，拿到最后的字符即可
            label = line.strip().split(',')[-1]
            labels.append(label)

print(f"读取完毕！共找到 {len(data_lines)} 个样本，准备进行分层划分...")

# ================= 3. 分层划分 (Stratified Split) =================
# 使用 stratify=labels 确保 7 个类别在训练集和测试集中的比例一致
train_lines, test_lines = train_test_split(
    data_lines,
    test_size=test_ratio,
    random_state=random_seed,
    stratify=labels
)

print(f"划分结果: 训练集 {len(train_lines)} 个样本，测试集 {len(test_lines)} 个样本。")


# ================= 4. 写入新文件 =================
def write_ts_subset(output_file, header, lines):
    print(f"正在写入 {output_file} ...")
    with open(output_file, 'w') as f:
        # 写入元数据头
        f.writelines(header)
        # 写入数据行
        f.writelines(lines)


write_ts_subset(train_ts_file, header_lines, train_lines)
write_ts_subset(test_ts_file, header_lines, test_lines)

print("划分完成！")