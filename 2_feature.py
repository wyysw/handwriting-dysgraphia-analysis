#!/usr/bin/env python3
import argparse, glob, os, subprocess, sys
from config import GAME_CONFIGS

def run_game(game, cfg):
    ids = [os.path.splitext(os.path.basename(f))[0]
           for f in glob.glob(f"{cfg['raw_dir']}/*.txt")]
    print(f"\n{'='*50}\n{game}: 共 {len(ids)} 个样本\n{'='*50}")

    for sid in ids:
        cmd = [
            sys.executable, cfg["extractor"],
            "--txt", f"{cfg['raw_dir']}/{sid}.txt",
            "--png", f"{cfg['raw_dir']}/{sid}.png",
            "--out", f"{cfg['feat_dir']}/{sid}.json",
            "--sample_id", sid,
            *cfg["extra_args"],
        ]
        
        # maze 添加可视化（目录）
        if "maze_feature_extractor" in cfg["extractor"]:
            vis_dir = f"{cfg['feat_dir']}/vis_{sid}"
            cmd += ["--vis_dir", vis_dir, "--game", game]
        # sym 添加可视化（文件）
        elif "sym_feature_extractor" in cfg["extractor"]:
            cmd += ["--vis", f"{cfg['feat_dir']}/{sid}_vis.png"]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ✗ {sid} 失败")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=list(GAME_CONFIGS), help="不指定则运行全部")
    args = parser.parse_args()

    targets = {args.game: GAME_CONFIGS[args.game]} if args.game else GAME_CONFIGS
    for game, cfg in targets.items():
        run_game(game, cfg)

if __name__ == "__main__":
    main()