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


def crop_with_rect(image: np.ndarray, rect):
    x, y, w, h = rect
    return image[y:y+h, x:x+w].copy()


def put_roi_back(full_shape, roi_mask, rect):
    x, y, w, h = rect
    full = np.zeros(full_shape[:2], dtype=np.uint8)
    full[y:y+h, x:x+w] = roi_mask
    return full


# =========================
# 白色画布定位（取代矩形外框定位）
# =========================

def locate_white_canvas(img: np.ndarray, white_thresh: int = 240,
                        close_ksize: int = 15, min_area: int = 50000):
    """
    在整张图中寻找白色画布区域。
    由于圆形迷宫图外围为黑色/透明背景，画布是一块白色矩形，
    因此不再用"最大暗色矩形"，而是用"最大白色连通域"作为ROI。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    white_mask = (gray >= white_thresh).astype(np.uint8) * 255
    # 闭运算把白色画布内部因迷宫线造成的小断裂填平
    white_closed = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_closed, connectivity=8)

    best_idx = -1
    best_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx == -1:
        return None

    x = stats[best_idx, cv2.CC_STAT_LEFT]
    y = stats[best_idx, cv2.CC_STAT_TOP]
    w = stats[best_idx, cv2.CC_STAT_WIDTH]
    h = stats[best_idx, cv2.CC_STAT_HEIGHT]
    return (x, y, w, h)


# =========================
# 核心：圆形迷宫线条提取
# =========================

def build_colorful_mask(bgr_roi: np.ndarray, diff_thresh: int = 25) -> np.ndarray:
    """
    识别彩色像素（出入口箭头）。
    黑白灰像素的 R/G/B 三通道接近，彩色像素三通道差异明显。
    """
    b = bgr_roi[:, :, 0].astype(np.int16)
    g = bgr_roi[:, :, 1].astype(np.int16)
    r = bgr_roi[:, :, 2].astype(np.int16)

    diff = np.abs(r - g) + np.abs(r - b) + np.abs(g - b)
    colorful = (diff > diff_thresh).astype(np.uint8) * 255
    return colorful


def extract_circle_maze_lines(maze_roi: np.ndarray, debug_dir=None) -> np.ndarray:
    """
    在白色画布 ROI 中提取黑色迷宫线条。
    流程与方形迷宫的 blackhat 思路不同：
    这里线条对比强烈（黑线+白底），用阈值法即可；
    同时识别并扣除彩色箭头。
    """
    gray = cv2.cvtColor(maze_roi, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE 稍微增强对比度（对细线条有帮助）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # 2. 双阈值：强黑线 + 弱灰线
    _, mask_dark = cv2.threshold(gray_eq, 120, 255, cv2.THRESH_BINARY_INV)
    _, mask_soft = cv2.threshold(gray_eq, 180, 255, cv2.THRESH_BINARY_INV)
    merged = cv2.bitwise_or(mask_dark, mask_soft)

    # 3. 识别彩色箭头并从线条掩码中扣除
    colorful = build_colorful_mask(maze_roi, diff_thresh=25)
    # 把彩色区域稍微膨胀一点，以覆盖箭头边缘过渡的灰色像素
    colorful_dilated = cv2.dilate(
        colorful,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    merged[colorful_dilated > 0] = 0

    # 4. 轻微连通补线（保持曲线的连续性）
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    merged_h = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, h_kernel)
    merged_v = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, v_kernel)
    merged = cv2.bitwise_or(merged_h, merged_v)

    # 5. 去极小噪点
    merged = remove_small_components(merged, min_area=15)

    if debug_dir:
        ensure_dir(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "1_gray_eq.png"), gray_eq)
        cv2.imwrite(os.path.join(debug_dir, "2_mask_dark.png"), mask_dark)
        cv2.imwrite(os.path.join(debug_dir, "3_mask_soft.png"), mask_soft)
        cv2.imwrite(os.path.join(debug_dir, "4_colorful.png"), colorful)
        cv2.imwrite(os.path.join(debug_dir, "5_colorful_dilated.png"), colorful_dilated)
        cv2.imwrite(os.path.join(debug_dir, "6_merged.png"), merged)

    return merged


# =========================
# 画布边缘清理
# =========================

def clean_canvas_border(mask: np.ndarray, border_thickness: int = 2) -> np.ndarray:
    """
    白色画布的矩形外缘常常会被阈值法误识别成一圈细线。
    因为圆形迷宫本身不存在矩形边框，所以这里直接抹掉最外一圈。
    """
    h, w = mask.shape
    cleaned = mask.copy()
    cleaned[:border_thickness, :] = 0
    cleaned[h - border_thickness:, :] = 0
    cleaned[:, :border_thickness] = 0
    cleaned[:, w - border_thickness:] = 0
    return cleaned


# =========================
# 删除孤立短线
# =========================

def remove_isolated_short_segments(mask: np.ndarray, max_len: int = 10, debug_dir=None) -> np.ndarray:
    """
    删除长度不超过 max_len 像素的孤立短线。
    按连通域处理，删除满足以下条件的小组件：
    - 外接框长边 <= max_len
    - 细线状
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    removed = np.zeros_like(mask)

    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        long_side = max(w, h)
        short_side = min(w, h)
        box_area = max(w * h, 1)
        fill_ratio = area / box_area

        is_short_segment = (
            long_side <= max_len and
            short_side <= 3 and
            area <= max_len * 2 and
            fill_ratio >= 0.2
        )

        if is_short_segment:
            out[labels == i] = 0
            removed[labels == i] = 255

    if debug_dir:
        ensure_dir(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "8_removed_short_segments.png"), removed)
        cv2.imwrite(os.path.join(debug_dir, "9_after_remove_short_segments.png"), out)

    return out


# =========================
# 主函数
# =========================

def extract_maze(image_path: str, out_dir: str):
    clear_directory(out_dir)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # 如果带透明通道，先把透明区域填充成黑色（与当前视觉一致），再转3通道
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3].copy()
        bgr[alpha == 0] = (0, 0, 0)
        img = bgr

    # Step 1: 定位白色画布（取代矩形暗框定位）
    rect = locate_white_canvas(img, white_thresh=240, close_ksize=15, min_area=50000)
    if rect is None:
        raise RuntimeError("没有找到白色画布区域")

    # Step 2: 裁出 ROI
    maze_roi = crop_with_rect(img, rect)
    cv2.imwrite(os.path.join(out_dir, "circle_roi.png"), maze_roi)

    debug_dir = os.path.join(out_dir, "debug")

    # Step 3: 提取迷宫线条 + 扣除彩色箭头
    maze_mask_roi = extract_circle_maze_lines(maze_roi, debug_dir=debug_dir)

    # Step 4: 去掉画布最外一圈的矩形边缘残留
    maze_mask_roi = clean_canvas_border(maze_mask_roi, border_thickness=2)
    cv2.imwrite(os.path.join(debug_dir, "7_after_border_cleanup.png"), maze_mask_roi)

    # Step 5: 删除孤立短线
    maze_mask_roi = remove_isolated_short_segments(
        maze_mask_roi,
        max_len=10,
        debug_dir=debug_dir
    )

    # Step 6: 回填到原图（保持 1201x1601 输出尺寸）
    maze_mask_full = put_roi_back(img.shape, maze_mask_roi, rect)

    save_mask(maze_mask_full, os.path.join(out_dir, "circle_mask.png"))

    maze_only = apply_mask_to_image(img, maze_mask_full)
    cv2.imwrite(os.path.join(out_dir, "circle_only.png"), maze_only)

    print("✅ 圆形迷宫提取完成")


# =========================
# 入口
# =========================

def main():
    maze_image = "data/raw/36circle.png"
    output_dir = "./output_circle/shape_circle"
    extract_maze(maze_image, output_dir)


if __name__ == "__main__":
    main()