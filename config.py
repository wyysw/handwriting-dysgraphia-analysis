GAME_CONFIGS = {
    "maze": {
        "extractor":  "features/maze_feature_extractor.py",
        "raw_image":  "data/raw/34migong.png",
        "raw_dir":    "data/raw/maze",
        "feat_dir":   "data/feature/maze",
        "out_dir":    "data/shape_out/maze",
        "extra_args": ["--mask", "data/shape_out/maze_mask.png"],
    },
    "sym": {
        "extractor":  "features/sym_feature_extractor.py",
        "raw_image":  "data/raw/35duichen.png",
        "raw_dir":    "data/raw/sym",
        "feat_dir":   "data/feature/sym",
        "out_dir":    "data/shape_out/sym",
        "extra_args": [
            "--blue",   "data/shape_out/sym_blue_mask.png",
            "--helper", "data/shape_out/sym_helper_mask.png",
        ],
    },
    "circle": {
        "extractor":  "features/maze_feature_extractor.py",
        "raw_image":  "data/raw/36circle.png",
        "raw_dir":    "data/raw/circle",
        "feat_dir":   "data/feature/circle",
        "out_dir":    "data/shape_out/circle",
        "extra_args": ["--mask", "data/shape_out/circle_mask.png"],
    },
}

# 中间产物汇总目录（shape.py 的 shutil.copy 目标）
SHAPE_OUT_DIR = "data/shape_out"