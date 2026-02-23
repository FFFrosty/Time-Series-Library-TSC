import subprocess
import os
import sys
import torch

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（上两级）
project_root = os.path.dirname(os.path.dirname(current_dir))
# run.py 的完整路径
run_py_path = os.path.join(project_root, 'run.py')

def run_experiment(dataset_name, **kwargs):
    """运行单个实验"""

    # 基础命令
    cmd = [
        'python', '-u', run_py_path,
        '--task_name', 'classification',
        '--is_training', '1',
        '--model', 'Transformer',
        '--data', 'UEA',
        '--des', 'Exp',
        '--itr', '1',
        '--learning_rate', '0.001',
        '--patience', '10',
        '--batch_size', '8'
    ]

    # 添加数据集特定的参数
    cmd.extend(['--root_path', f'./dataset/{dataset_name}/'])
    cmd.extend(['--model_id', dataset_name])

    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])

    print(' '.join(cmd))
    subprocess.run(cmd)


# 使用示例
if __name__ == '__main__':
    # 如果想运行所有实验，取消下面的注释
    # experiments = [...上面列表...]

    # 或者只运行你想运行的实验
    dataset = 'LandingGear'
    run_experiment(dataset, e_layers=2, d_model=64, d_ff=128, top_k=3, train_epochs=100)

    # if len(sys.argv) > 1:
    #     # 从命令行参数指定数据集
    #     dataset = sys.argv[1]
    #     if dataset == 'EthanolConcentration':
    #         run_experiment('EthanolConcentration', e_layers=2, d_model=16, d_ff=32, top_k=3, train_epochs=30)
    #     elif dataset == 'FaceDetection':
    #         run_experiment('FaceDetection', e_layers=2, d_model=64, d_ff=256, top_k=3, num_kernels=4, train_epochs=30)
    #     # ... 添加其他数据集
    # else:
    #     print("请指定要运行的数据集，例如：")
    #     print("python TimesNet.py EthanolConcentration")