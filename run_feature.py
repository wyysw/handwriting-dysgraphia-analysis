#!/usr/bin/env python3
"""
运行特征提取脚本的统一入口
用法:
    python run_feature.py              # 运行所有三个脚本
    python run_feature.py --game circle   # 只运行 circle 脚本
    python run_feature.py --game maze     # 只运行 maze 脚本
    python run_feature.py --game sym      # 只运行 sym 脚本
"""

import argparse
import subprocess
import sys
import os

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'features')

# 脚本映射
SCRIPTS = {
    'circle': 'test_circle_extract.py',
    'maze': 'test_maze_extract.py',
    'sym': 'test_sym_extract.py'
}

def run_script(script_name):
    """运行指定的Python脚本"""
    script_path = os.path.join(FEATURES_DIR, script_name)
    
    if not os.path.exists(script_path):
        print(f"错误: 脚本不存在 - {script_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"正在运行: {script_name}")
    print(f"{'='*60}")
    
    try:
        # 使用与当前Python解释器相同的环境运行脚本
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,  # 在项目根目录下运行
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"错误: 运行 {script_name} 失败，返回码: {e.returncode}")
        return False
    except Exception as e:
        print(f"错误: 运行 {script_name} 时发生异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='运行特征提取脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_feature.py                # 运行所有脚本
  python run_feature.py --game circle  # 只运行 circle 脚本
  python run_feature.py --game maze    # 只运行 maze 脚本
  python run_feature.py --game sym     # 只运行 sym 脚本
        """
    )
    
    parser.add_argument(
        '--game',
        choices=['circle', 'maze', 'sym'],
        help='指定要运行的游戏类型（不指定则运行所有）'
    )
    
    args = parser.parse_args()
    
    # 检查 features 目录是否存在
    if not os.path.exists(FEATURES_DIR):
        print(f"错误: features 目录不存在 - {FEATURES_DIR}")
        sys.exit(1)
    
    # 确定要运行的脚本
    if args.game:
        scripts_to_run = [(args.game, SCRIPTS[args.game])]
    else:
        scripts_to_run = list(SCRIPTS.items())
        print("未指定游戏类型，将运行所有脚本\n")
    
    # 运行脚本
    success_count = 0
    total_count = len(scripts_to_run)
    
    for game_name, script_file in scripts_to_run:
        if run_script(script_file):
            success_count += 1
    
    # 输出总结
    print(f"\n{'='*60}")
    print(f"完成! 成功运行 {success_count}/{total_count} 个脚本")
    if success_count < total_count:
        print("部分脚本运行失败，请检查上述错误信息")
        sys.exit(1)
    else:
        print("所有脚本运行成功!")
        sys.exit(0)

if __name__ == '__main__':
    main()