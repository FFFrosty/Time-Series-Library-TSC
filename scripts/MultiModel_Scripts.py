import os
import subprocess
import time
import sys

BASE_DIR = os.path.join('', 'scripts/classification')

# ✅ 在这里统一设置你想跑的数据集名称
TARGET_DATASET = 'LandingGearOrigin'

scripts_to_run = [
    'DLinear_LandingGearFull.py',
    'LightTS_LandingGearFull.py',
    'Lstm_LandingGearFull.py',
    'ResNet50_LandingGearFull.py',
    'TCN_LandingGearFull.py',
    'TimesNet_LandingGearFull.py',
    'Transformer_LandingGearFull.py',
    'Informer_LandingGearFull.py',
    'Crossformer_LandingGearFull.py',
    'iTransformer_LandingGearFull.py',
]

def run_experiments(scripts, target_dataset):
    print(f"总计需要运行 {len(scripts)} 个实验脚本...")
    print(f"🎯 统一目标数据集设定为: {target_dataset}\n") # 打印当前使用的数据集

    for idx, script_name in enumerate(scripts, 1):
        script_path = os.path.join(BASE_DIR, script_name)

        print("=" * 60)
        print(f"🚀 正在启动第 {idx}/{len(scripts)} 个脚本: {script_path}")
        start_time = time.time()

        try:
            # ✅ 修改点：将 target_dataset 作为额外参数传入
            # 相当于在终端执行: python script_path TARGET_DATASET
            result = subprocess.run([sys.executable, script_path, target_dataset], check=True)

            end_time = time.time()
            elapsed_mins = (end_time - start_time) / 60
            print(f"\n✅ {script_name} 顺利执行完毕！耗时: {elapsed_mins:.2f} 分钟")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ {script_path} 运行崩溃，退出状态码: {e.returncode}")
            print("⚠️ 终止后续实验队伍...")
            break
        except FileNotFoundError:
            print(f"\n❌ 找不到文件: {script_path}，请检查该文件是否确实存在于 {BASE_DIR} 目录下。")
            break

    print("=" * 60)
    print("🎉 批处理队列执行结束！")

if __name__ == "__main__":
    # 传入数据集参数
    run_experiments(scripts_to_run, TARGET_DATASET)