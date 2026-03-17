import subprocess
import os
import sys

# 设置环境变量 (虽然 ARIMA 只用 CPU，但保持你的原有格式)
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
        '--task_name', 'long_term_forecast', # ✅ 改为预测任务
        '--is_training', '1',
        '--model', 'ARIMA',                  # ✅ 调用 ARIMA 模型
        '--data', 'custom',                  # ✅ 预测通常为 custom 或 ETTh1 等
        '--feature', 'M',
        '--target', '74_A',
        '--batch_size', '8',
        '--seq_len', '96',                   # ✅ 历史步长 96
        '--pred_len', '96',                  # ✅ 预测步长 96
        '--label_len', '48'                  # TSLib dataloader 的常规搭配参数
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
    dataset = 'Railway' # 替换为你的预测数据集文件夹名
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    print(f"🛠️  模型 ARIMA 准备就绪，即将处理数据集: {dataset}")
    # ARIMA 的 order 默认是 1,1,1，你可以通过 kwargs 传入 arima_order 覆盖
    run_experiment(dataset, random_state=2026, arima_order='1,1,1')