import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

from shape import final_shape_migong as migong
from shape import final_shape_sym as sym
from shape import final_shape_circle as circle
from config import GAME_CONFIGS, SHAPE_OUT_DIR

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def read_image(path: str, flags=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def to_display_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def show_group(images, titles, fig_title=None, figsize=(14, 6)):
    n = len(images)
    plt.figure(figsize=figsize)
    if fig_title:
        plt.suptitle(fig_title, fontsize=14)
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, n, i)
        disp = to_display_image(img)
        if disp.ndim == 2:
            plt.imshow(disp, cmap="gray")
        else:
            plt.imshow(disp)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_maze():
    cfg = GAME_CONFIGS["maze"]
    migong.extract_maze(cfg["raw_image"], cfg["out_dir"])
    return cfg["out_dir"]


def run_sym():
    cfg = GAME_CONFIGS["sym"]
    sym.extract_symmetry(
        image_path=cfg["raw_image"],
        out_dir=cfg["out_dir"],
        ignore_side_width=190,
        expected_width=1201,
        expected_height=1601,
        ignore_mode="white",
    )
    return cfg["out_dir"]


def run_circle():
    cfg = GAME_CONFIGS["circle"]
    circle.extract_maze(cfg["raw_image"], cfg["out_dir"])
    return cfg["out_dir"]


def main():
    # 1. 运行三个提取程序
    maze_out_dir   = run_maze()
    sym_out_dir    = run_sym()
    circle_out_dir = run_circle()

    # 2. 读取路径（从 config 取 raw_image，mask 路径由 out_dir 拼接）
    maze_cfg   = GAME_CONFIGS["maze"]
    sym_cfg    = GAME_CONFIGS["sym"]
    circle_cfg = GAME_CONFIGS["circle"]

    maze_roi_path  = maze_cfg["raw_image"]
    maze_mask_path = os.path.join(maze_out_dir,   "maze_mask.png")

    sym_raw_path         = sym_cfg["raw_image"]
    sym_blue_mask_path   = os.path.join(sym_out_dir, "sym_blue_mask.png")
    sym_helper_mask_path = os.path.join(sym_out_dir, "sym_helper_mask.png")

    circle_roi_path  = circle_cfg["raw_image"]
    circle_mask_path = os.path.join(circle_out_dir, "circle_mask.png")

    # 3. 读取图片
    maze_roi   = read_image(maze_roi_path)
    maze_mask  = read_image(maze_mask_path)

    sym_raw         = read_image(sym_raw_path)
    sym_blue_mask   = read_image(sym_blue_mask_path)
    sym_helper_mask = read_image(sym_helper_mask_path)

    circle_roi  = read_image(circle_roi_path)
    circle_mask = read_image(circle_mask_path)

    # 4. 展示
    show_group(
        images=[maze_roi, maze_mask],
        titles=["原始迷宫", "迷宫线条"],
        fig_title="方形迷宫图形",
        figsize=(12, 6),
    )
    show_group(
        images=[sym_raw, sym_blue_mask, sym_helper_mask],
        titles=["原始对称游戏", "对称图形", "辅助线框"],
        fig_title="对称图形",
        figsize=(18, 6),
    )
    show_group(
        images=[circle_roi, circle_mask],
        titles=["原始圆形迷宫", "圆形迷宫线条"],
        fig_title="圆形迷宫图形",
        figsize=(12, 6),
    )

    # 5. 复制 mask 文件到汇总目录
    os.makedirs(SHAPE_OUT_DIR, exist_ok=True)
    mask_files = [
        sym_helper_mask_path,
        sym_blue_mask_path,
        maze_mask_path,
        circle_mask_path,
    ]
    for mask_file in mask_files:
        shutil.copy(mask_file, SHAPE_OUT_DIR)
        print(f"已复制 {mask_file} 到 {SHAPE_OUT_DIR}")


if __name__ == "__main__":
    main()