import os
import subprocess
import time
import sys

# 设置脚本所在的相对目录
BASE_DIR = os.path.join('scripts', 'classification')

# 在这里填入你要顺次运行的脚本列表（只需要写文件名）
scripts_to_run = [
    # 'DLinear_LandingGearFull.py',
    # 'LightTS_LandingGearFull.py',
    # 'Lstm_LandingGearFull.py',
    'ResNet50_LandingGearFull.py',
    'TCN_LandingGearFull.py',
    'TimesNet_LandingGearFull.py',
    'Transformer_LandingGearFull.py',
    # 'exp3.py',
]


def run_experiments(scripts):
    print(f"总计需要运行 {len(scripts)} 个实验脚本...\n")

    for idx, script_name in enumerate(scripts, 1):
        # 自动拼接出完整路径：例如 scripts/classification/exp1.py
        script_path = os.path.join(BASE_DIR, script_name)

        print("=" * 60)
        print(f"🚀 正在启动第 {idx}/{len(scripts)} 个脚本: {script_path}")
        start_time = time.time()

        try:
            # 执行拼接好路径的脚本
            result = subprocess.run([sys.executable, script_path], check=True)

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
    run_experiments(scripts_to_run)