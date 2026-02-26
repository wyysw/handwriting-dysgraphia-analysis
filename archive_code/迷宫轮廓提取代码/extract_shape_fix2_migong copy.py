import os
import cv2
import numpy as np


# =========================
# 工具函数
# =========================

def ensure_dir(path: str):
    if not os.path.exists(path):
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
# ⭐ 核心：基于 blackhat 的迷宫提取（改进版）
# =========================

def extract_maze_lines_blackhat(maze_roi: np.ndarray, debug_dir=None) -> np.ndarray:
    gray = cv2.cvtColor(maze_roi, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE（轻量增强）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # 2. Blackhat（核心）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, kernel)

    # ⭐ 关键：只对 blackhat 做阈值
    _, mask_bh = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)

    # 3. 辅助（弱）灰度阈值，仅用于补缺口
    _, mask_gray = cv2.threshold(gray_eq, 145, 255, cv2.THRESH_BINARY_INV)

    # ⚠️ 不再全量 OR，只允许 small merge
    merged = cv2.bitwise_or(mask_bh, mask_gray)

    # 4. 定向补线（非常关键）
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

    merged_h = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, h_kernel)
    merged_v = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, v_kernel)

    merged = cv2.bitwise_or(merged_h, merged_v)

    # 5. 去小噪点（稍微放宽）
    merged = remove_small_components(merged, min_area=12)

    # 调试输出
    if debug_dir:
        ensure_dir(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "1_gray_eq.png"), gray_eq)
        cv2.imwrite(os.path.join(debug_dir, "2_blackhat.png"), blackhat)
        cv2.imwrite(os.path.join(debug_dir, "3_mask_bh.png"), mask_bh)
        cv2.imwrite(os.path.join(debug_dir, "4_mask_gray.png"), mask_gray)
        cv2.imwrite(os.path.join(debug_dir, "5_merged.png"), merged)

    return merged


# =========================
# 主函数
# =========================

def extract_maze(image_path: str, out_dir: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    ensure_dir(out_dir)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: 找外框
    _, dark_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    dark_mask = cv2.dilate(
        dark_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    cnt, rect = keep_largest_rectangle_contour(dark_mask)

    if rect is None:
        raise RuntimeError("没有找到迷宫框")

    # Step 2: 裁剪 ROI
    maze_roi = crop_with_rect(img, rect)
    cv2.imwrite(os.path.join(out_dir, "maze_roi.png"), maze_roi)

    # Step 3: 提取迷宫线（新方法）
    maze_mask_roi = extract_maze_lines_blackhat(
        maze_roi,
        debug_dir=os.path.join(out_dir, "debug")
    )

    # Step 4: 内边界限制
    margin = 6
    h, w = maze_mask_roi.shape
    inner = np.zeros_like(maze_mask_roi)
    cv2.rectangle(inner, (margin, margin), (w-margin, h-margin), 255, -1)
    maze_mask_roi = cv2.bitwise_and(maze_mask_roi, inner)

    # Step 5: 回填到原图
    maze_mask_full = put_roi_back(img.shape, maze_mask_roi, rect)

    save_mask(maze_mask_full, os.path.join(out_dir, "maze_mask.png"))

    maze_only = apply_mask_to_image(img, maze_mask_full)
    cv2.imwrite(os.path.join(out_dir, "maze_only.png"), maze_only)

    print("✅ 迷宫提取完成（blackhat优化版）")


# =========================
# 入口
# =========================

def main():
    maze_image = "data/34migong.png"
    output_dir = "./maze_output/2"

    extract_maze(maze_image, output_dir)


if __name__ == "__main__":
    main()