import os
import cv2
import numpy as np


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_mask(mask: np.ndarray, path: str):
    cv2.imwrite(path, mask)


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full_like(image, 255)
    result[mask > 0] = image[mask > 0]
    return result


def keep_largest_rectangle_contour(binary_mask: np.ndarray, min_area=10000):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if 4 <= len(approx) <= 10:
            if area > best_area:
                best_area = area
                best = cnt

    if best is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(best)
    box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
    return best, box, (x, y, w, h)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def crop_with_rect(image: np.ndarray, rect):
    x, y, w, h = rect
    return image[y:y + h, x:x + w].copy()


def put_roi_back(full_shape, roi_mask, rect):
    x, y, w, h = rect
    full = np.zeros(full_shape[:2], dtype=np.uint8)
    full[y:y + h, x:x + w] = roi_mask
    return full


def enhance_gray_for_thin_dark_lines(gray: np.ndarray) -> np.ndarray:
    """
    针对细浅黑线做局部对比度增强
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def find_maze_outer_rect(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 外框通常仍然较深，用较稳妥的阈值先找
    _, dark_mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

    dark_mask = cv2.dilate(
        dark_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    _, _, rect = keep_largest_rectangle_contour(dark_mask, min_area=50000)
    return rect


def extract_maze_lines_from_roi(maze_roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(maze_roi, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_gray_for_thin_dark_lines(gray)

    # ---- 路线1：提取较明显黑线 ----
    _, global_dark = cv2.threshold(enhanced, 135, 255, cv2.THRESH_BINARY_INV)

    # ---- 路线2：自适应阈值，补细浅线 ----
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        6
    )

    # 合并两路结果
    combined = cv2.bitwise_or(global_dark, adaptive)

    # 轻微开运算去噪
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    # 闭运算连接细小断裂，尤其适合浅细线
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    # 再去小连通域，阈值调低一点，避免误删细线
    combined = remove_small_components(combined, min_area=20)

    return combined


def extract_maze(image_path: str, out_dir: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    ensure_dir(out_dir)

    rect = find_maze_outer_rect(img)
    if rect is None:
        raise RuntimeError("没有找到迷宫主框，请检查阈值。")

    x, y, w, h = rect
    maze_roi = crop_with_rect(img, rect)
    cv2.imwrite(os.path.join(out_dir, "maze_roi.png"), maze_roi)

    maze_line_mask_roi = extract_maze_lines_from_roi(maze_roi)

    save_mask(maze_line_mask_roi, os.path.join(out_dir, "maze_lines_mask_roi.png"))

    maze_line_mask_full = put_roi_back(img.shape, maze_line_mask_roi, rect)
    save_mask(maze_line_mask_full, os.path.join(out_dir, "maze_lines_mask.png"))

    maze_only = apply_mask_to_image(img, maze_line_mask_full)
    cv2.imwrite(os.path.join(out_dir, "maze_lines_only.png"), maze_only)

    outer_box_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(outer_box_mask, (x, y), (x + w, y + h), 255, 2)
    cv2.imwrite(os.path.join(out_dir, "maze_outer_box_mask.png"), outer_box_mask)

    print(f"[迷宫] 提取完成，结果保存在: {out_dir}")


def main():
    maze_image = "data/34migong(1).png"
    maze_out_dir = "./output_maze"
    extract_maze(maze_image, maze_out_dir)


if __name__ == "__main__":
    main()