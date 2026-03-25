import subprocess
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（上三级）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# run.py 的完整路径
run_py_path = os.path.join(project_root, 'run_ml_baseline.py')

def run_experiment(dataset_name, **kwargs):
    cmd = [
        'python', '-u', run_py_path,
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model', 'RandomForest',           # ✅ 调用 RandomForest
        '--data', 'custom',
        '--feature', 'M',
        '--target', '74_A',
        '--batch_size', '16',                # 树模型可以适当调大 batch_size 以加快数据加载
        '--seq_len', '96',
        '--pred_len', '96',
        '--label_len', '48'
    ]

    # 添加数据集特定的参数
    cmd.extend(['--root_path', f'./dataset/{dataset_name}/'])
    cmd.extend(['--data_path', f'railway_passenger_flow_small.csv'])
    cmd.extend(['--model_id', dataset_name])

    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])

    print(' '.join(cmd))
    subprocess.run(cmd)

if __name__ == '__main__':
    dataset = 'Railway'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    print(f"🛠️  模型 RandomForest (回归) 准备就绪，即将处理数据集: {dataset}")
    run_experiment(dataset, n_estimators=5, random_state=2026)