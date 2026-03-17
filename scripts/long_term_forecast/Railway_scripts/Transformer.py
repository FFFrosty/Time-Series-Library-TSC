import subprocess
import os
import sys
import torch

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（上三级）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# run.py 的完整路径
run_py_path = os.path.join(project_root, 'run.py')

def run_experiment(dataset_name, **kwargs):
    """运行单个实验"""

    # 基础命令
    cmd = [
        'python', '-u', run_py_path,
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model', 'Transformer',
        '--data', 'custom',
        '--feature', 'M',
        '--target', '40_C',
        '--seq_len', '96',
        '--label_len', '48',
        '--pred_len', '96',
        '--des', 'Exp',
        '--itr', '1',
        '--learning_rate', '0.001',
        '--patience', '10',
        '--batch_size', '8',
        '--inverse',
    ]

    # 添加数据集特定的参数
    cmd.extend(['--root_path', f'./dataset/{dataset_name}/'])
    cmd.extend(['--data_path', f'railway_passenger_flow.csv'])
    cmd.extend(['--model_id', dataset_name])

    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])

    print(' '.join(cmd))
    subprocess.run(cmd)


# 使用示例
if __name__ == '__main__':
    # ✅ 默认使用原数据集，方便你单独测试这个脚本
    dataset = 'Railway'

    # ✅ 如果从 run_all.py (或其他命令行方式) 传来了参数，则覆盖默认数据集
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    print(f"🛠️  模型 Transformer 准备就绪，即将处理数据集: {dataset}")

    # 运行实验（直接传入替换好的 dataset 变量）
    run_experiment(dataset, e_layers=3, d_layers=1, factor=3, enc_in=55, dec_in=55, c_out=55,
                   d_model=128, d_ff=256, top_k=3, train_epochs=100)