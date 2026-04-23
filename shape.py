import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from shape import final_shape_migong as migong
from shape import final_shape_sym as sym
from shape import final_shape_circle as circle

plt.rcParams["font.family"] = ["SimHei"]  # 适配不同系统的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

def read_image(path: str, flags=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def to_display_image(img: np.ndarray) -> np.ndarray:
    """
    将 OpenCV 读入的图像转换成适合 matplotlib 显示的格式
    """
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
    """
    运行方形迷宫程序
    """
    maze_image = "data/34migong.png"
    maze_out_dir = "./output_maze/shape_maze"
    migong.extract_maze(maze_image, maze_out_dir)
    return maze_out_dir


def run_sym():
    """
    运行对称图程序
    """
    sym_image = "data/35duichen.png"
    sym_out_dir = "./output_sym/shape_sym"
    sym.extract_symmetry(
        image_path=sym_image,
        out_dir=sym_out_dir,
        ignore_side_width=190,
        expected_width=1201,
        expected_height=1601,
        ignore_mode="white"
    )
    return sym_out_dir


def run_circle():
    """
    运行圆形迷宫程序
    """
    circle_image = "data/36circle.png"
    circle_out_dir = "./output_circle/shape_circle"
    circle.extract_maze(circle_image, circle_out_dir)
    return circle_out_dir


def main():
    # 1. 先自动运行三个程序
    maze_out_dir = run_maze()
    sym_out_dir = run_sym()
    circle_out_dir = run_circle()

    # 2. 读取第一组图片:maze_roi.png 与 maze_mask.png
    # maze_roi_path = os.path.join(maze_out_dir, "maze_roi.png")
    maze_roi_path = "data/34migong.png"
    maze_mask_path = os.path.join(maze_out_dir, "maze_mask.png")

    maze_roi = read_image(maze_roi_path)
    maze_mask = read_image(maze_mask_path)

    # 3. 读取第二组图片:sym_preprocessed_white / sym_blue_mask / sym_helper_mask_completed
    # sym_preprocessed_white_path = os.path.join(sym_out_dir, "sym_preprocessed_white.png")
    sym_preprocessed_white_path = "data/35duichen.png"
    sym_blue_mask_path = os.path.join(sym_out_dir, "sym_blue_mask.png")
    sym_helper_mask_completed_path = os.path.join(sym_out_dir, "sym_helper_mask_completed.png")

    sym_preprocessed_white = read_image(sym_preprocessed_white_path)
    sym_blue_mask = read_image(sym_blue_mask_path)
    sym_helper_mask_completed = read_image(sym_helper_mask_completed_path)

    # 4. 读取第三组图片:circle_roi.png 与 circle_mask.png
    # circle_roi_path = os.path.join(circle_out_dir, "circle_roi.png")
    circle_roi_path = "data/36circle.png"
    circle_mask_path = os.path.join(circle_out_dir, "circle_mask.png")

    circle_roi = read_image(circle_roi_path)
    circle_mask = read_image(circle_mask_path)

    # 5. 按要求展示
    show_group(
        images=[maze_roi, maze_mask],
        titles=["原始迷宫", "迷宫线条"],
        fig_title="方形迷宫图形",
        figsize=(12, 6)
    )

    show_group(
        images=[sym_preprocessed_white, sym_blue_mask, sym_helper_mask_completed],
        titles=[
            "原始对称游戏",
            "对称图形",
            "辅助线框"
        ],
        fig_title="对称图形",
        figsize=(18, 6)
    )

    show_group(
        images=[circle_roi, circle_mask],
        titles=["原始圆形迷宫", "圆形迷宫线条"],
        fig_title="圆形迷宫图形",
        figsize=(12, 6)
    )


if __name__ == "__main__":
    main()