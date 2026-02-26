import os
import cv2
import shutil
import numpy as np


# =========================
# 工具函数
# =========================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_directory(path: str):
    """
    清空输出目录下的所有文件和子目录；如果目录不存在则创建
    """
    if os.path.exists(path):
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full) or os.path.islink(full):
                os.remove(full)
            elif os.path.isdir(full):
                shutil.rmtree(full)
    else:
        os.makedirs(path)


def save_mask(mask: np.ndarray, path: str):
    cv2.imwrite(path, mask)


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full_like(image, 255)
    result[mask > 0] = image[mask > 0]
    return result


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def keep_largest_rectangle_contour(binary_mask: np.ndarray, min_area=50000):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if 4 <= len(approx) <= 8 and area > best_area:
            best_area = area
            best = cnt

    if best is None:
        return None, None

    x, y, w, h = cv2.boundingRect(best)
    return best, (x, y, w, h)


def crop_with_rect(image: np.ndarray, rect):
    x, y, w, h = rect
    return image[y:y+h, x:x+w].copy()


def put_roi_back(full_shape, roi_mask, rect):
    x, y, w, h = rect
    full = np.zeros(full_shape[:2], dtype=np.uint8)
    full[y:y+h, x:x+w] = roi_mask
    return full


# =========================
# 核心：基于 blackhat 的迷宫提取
# =========================

def extract_maze_lines_blackhat(maze_roi: np.ndarray, debug_dir=None) -> np.ndarray:
    gray = cv2.cvtColor(maze_roi, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # 2. Blackhat（主通道）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, kernel)

    # 3. blackhat 阈值
    _, mask_bh = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)

    # 4. 弱灰度补充
    _, mask_gray = cv2.threshold(gray_eq, 145, 255, cv2.THRESH_BINARY_INV)

    merged = cv2.bitwise_or(mask_bh, mask_gray)

    # 5. 轻微定向补线
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

    merged_h = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, h_kernel)
    merged_v = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, v_kernel)
    merged = cv2.bitwise_or(merged_h, merged_v)

    # 6. 去极小噪点
    merged = remove_small_components(merged, min_area=12)

    if debug_dir:
        ensure_dir(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "1_gray_eq.png"), gray_eq)
        cv2.imwrite(os.path.join(debug_dir, "2_blackhat.png"), blackhat)
        cv2.imwrite(os.path.join(debug_dir, "3_mask_bh.png"), mask_bh)
        cv2.imwrite(os.path.join(debug_dir, "4_mask_gray.png"), mask_gray)
        cv2.imwrite(os.path.join(debug_dir, "5_merged.png"), merged)

    return merged


# =========================
# 角落装饰清理
# =========================

def clean_corner_decorations(mask: np.ndarray, debug_dir=None) -> np.ndarray:
    """
    只清理右上角小猪和左下角房子的残留，不碰中间迷宫主体
    """
    h, w = mask.shape
    cleaned = mask.copy()

    # 两个角落区域
    # 右上角：小猪
    # 需要放大区域 84 16
    tr_x1, tr_y1 = int(0.88 * w), 0
    tr_x2, tr_y2 = w, int(0.09 * h)

    # 左下角：房子
    # 18 86
    bl_x1, bl_y1 = 0, int(0.91 * h)
    bl_x2, bl_y2 = int(0.1555 * w), h

    corners = [
        ("top_right", tr_x1, tr_y1, tr_x2, tr_y2),
        ("bottom_left", bl_x1, bl_y1, bl_x2, bl_y2),
    ]

    for name, x1, y1, x2, y2 in corners:
        roi = cleaned[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        roi_out = roi.copy()

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            ww = stats[i, cv2.CC_STAT_WIDTH]
            hh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            box_area = max(ww * hh, 1)
            fill_ratio = area / box_area
            aspect = max(ww / max(hh, 1), hh / max(ww, 1))

            # 判断是否像迷宫线：
            # 迷宫线通常是细长、水平/垂直、填充率不高
            is_line_like = (
                (ww >= 18 and hh <= 5) or
                (hh >= 18 and ww <= 5) or
                (aspect >= 4.0 and fill_ratio <= 0.55)
            )

            # 块状/碎片状更像装饰
            is_blob_like = (
                (fill_ratio >= 0.50 and area >= 20) or
                (ww >= 8 and hh >= 8 and aspect <= 2.5)
            )

            # 在角落里，默认更严格
            remove = False
            if is_blob_like and not is_line_like:
                remove = True

            # 特别小的碎片也删掉
            if area < 10:
                remove = True

            if remove:
                roi_out[labels == i] = 0

        cleaned[y1:y2, x1:x2] = roi_out

        if debug_dir:
            ensure_dir(debug_dir)
            cv2.imwrite(os.path.join(debug_dir, f"6_corner_{name}_cleaned.png"), roi_out)

    return cleaned


# =========================
# 主函数
# =========================

def extract_maze(image_path: str, out_dir: str):
    # 运行前清空输出目录
    clear_directory(out_dir)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: 找迷宫外框
    _, dark_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.dilate(
        dark_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    cnt, rect = keep_largest_rectangle_contour(dark_mask)
    if rect is None:
        raise RuntimeError("没有找到迷宫框")

    # Step 2: 裁出 ROI
    maze_roi = crop_with_rect(img, rect)
    cv2.imwrite(os.path.join(out_dir, "maze_roi.png"), maze_roi)

    # Step 3: 提取迷宫线
    debug_dir = os.path.join(out_dir, "debug")
    maze_mask_roi = extract_maze_lines_blackhat(maze_roi, debug_dir=debug_dir)

    # Step 4: 清除角落装饰
    maze_mask_roi = clean_corner_decorations(maze_mask_roi, debug_dir=debug_dir)
    cv2.imwrite(os.path.join(debug_dir, "7_after_corner_cleanup.png"), maze_mask_roi)

    # Step 5: 内边界限制
    margin = 6
    h, w = maze_mask_roi.shape
    inner = np.zeros_like(maze_mask_roi)
    cv2.rectangle(inner, (margin, margin), (w - margin, h - margin), 255, -1)
    maze_mask_roi = cv2.bitwise_and(maze_mask_roi, inner)

    # margin = 0
    # h, w = maze_mask_roi.shape
    # inner = np.zeros_like(maze_mask_roi)
    # cv2.rectangle(inner, (margin, margin), (w - 1 - margin, h - 1 - margin), 255, -1)
    # maze_mask_roi = cv2.bitwise_and(maze_mask_roi, inner)

    # Step 6: 回填到原图
    maze_mask_full = put_roi_back(img.shape, maze_mask_roi, rect)

    save_mask(maze_mask_full, os.path.join(out_dir, "maze_mask.png"))

    maze_only = apply_mask_to_image(img, maze_mask_full)
    cv2.imwrite(os.path.join(out_dir, "maze_only.png"), maze_only)

    print("✅ 迷宫提取完成")


# =========================
# 入口
# =========================

def main():
    maze_image = "data/34migong.png"
    output_dir = "./maze_output/3"
    extract_maze(maze_image, output_dir)


if __name__ == "__main__":
    main()