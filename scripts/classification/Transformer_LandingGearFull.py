import subprocess
import os
import sys
import torch

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 解决显存问题，该问题目前只在服务器上发现
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
        '--batch_size', '8',
        '--use_multi_gpu',
        '--devices', '0,1,2,3',
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
    # ✅ 默认使用原数据集，方便你单独测试这个脚本
    dataset = 'LandingGearFull'

    # ✅ 如果从 run_all.py (或其他命令行方式) 传来了参数，则覆盖默认数据集
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    print(f"🛠️  模型 Transformer 准备就绪，即将处理数据集: {dataset}")

    # 运行实验（直接传入替换好的 dataset 变量）
    run_experiment(dataset, e_layers=2, d_model=64, d_ff=128, top_k=3, train_epochs=100)