import os
import cv2
import numpy as np


# =========================
# 基础工具函数
# =========================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_mask(mask: np.ndarray, path: str):
    """
    保存二值 mask，保持原图尺寸
    """
    cv2.imwrite(path, mask)


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    用 mask 从原图中抠出目标，尺寸不变
    """
    result = np.full_like(image, 255)
    result[mask > 0] = image[mask > 0]
    return result


def keep_largest_rectangle_contour(binary_mask: np.ndarray, min_area=10000):
    """
    在二值图中找近似矩形的最大轮廓
    返回: contour, box_points, rect(x,y,w,h)
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        fill_ratio = area / (rect_area + 1e-6)

        # 接近矩形：4~8个顶点都可接受，且面积较大
        if len(approx) >= 4 and len(approx) <= 8:
            if area > best_area:
                best_area = area
                best = cnt

    if best is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(best)
    box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
    return best, box, (x, y, w, h)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    去掉小连通域
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def extract_dark_lines(gray: np.ndarray, thresh=80) -> np.ndarray:
    """
    提取近黑色线条
    """
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return mask


def extract_light_gray_lines(gray: np.ndarray, low=180, high=245) -> np.ndarray:
    """
    提取浅灰色线
    """
    mask = cv2.inRange(gray, low, high)
    return mask


def horizontal_line_enhance(mask: np.ndarray, ksize=25) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, 1))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return out


def vertical_line_enhance(mask: np.ndarray, ksize=25) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return out


def crop_with_rect(image: np.ndarray, rect):
    x, y, w, h = rect
    return image[y:y+h, x:x+w].copy()


def put_roi_back(full_shape, roi_mask, rect):
    """
    把 ROI 内的 mask 放回原图坐标系
    """
    x, y, w, h = rect
    full = np.zeros(full_shape[:2], dtype=np.uint8)
    full[y:y+h, x:x+w] = roi_mask
    return full


# =========================
# 1. 迷宫图提取
# =========================

def extract_maze(image_path: str, out_dir: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    ensure_dir(out_dir)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: 提取较深色内容，优先找迷宫大外框
    dark_mask = extract_dark_lines(gray, thresh=90)

    # 适当膨胀，让外框更连续
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask_dilated = cv2.dilate(dark_mask, kernel, iterations=1)

    cnt, box, rect = keep_largest_rectangle_contour(dark_mask_dilated, min_area=50000)
    if rect is None:
        raise RuntimeError("没有找到迷宫主框，请检查阈值或图片。")

    x, y, rw, rh = rect

    # 裁出迷宫主区域
    maze_roi = crop_with_rect(img, rect)
    maze_gray = cv2.cvtColor(maze_roi, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(os.path.join(out_dir, "maze_roi.png"), maze_roi)

    # Step 2: 在 ROI 内提取黑色线条
    maze_line_mask = extract_dark_lines(maze_gray, thresh=100)

    # Step 3: 去掉彩色装饰（小猪、房子）
    # 因为它们通常不够“黑”，但如果残留，进一步用连通域和形态学清洗
    maze_line_mask = cv2.morphologyEx(
        maze_line_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    maze_line_mask = remove_small_components(maze_line_mask, min_area=40)

    # Step 4: 保存 ROI 内 mask
    save_mask(maze_line_mask, os.path.join(out_dir, "maze_lines_mask_roi.png"))

    # Step 5: 放回原图坐标系
    maze_line_mask_full = put_roi_back(img.shape, maze_line_mask, rect)
    save_mask(maze_line_mask_full, os.path.join(out_dir, "maze_lines_mask.png"))

    maze_only = apply_mask_to_image(img, maze_line_mask_full)
    cv2.imwrite(os.path.join(out_dir, "maze_lines_only.png"), maze_only)

    # 额外输出：外框框线
    outer_box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(outer_box_mask, (x, y), (x + rw, y + rh), 255, 2)
    cv2.imwrite(os.path.join(out_dir, "maze_outer_box_mask.png"), outer_box_mask)

    print(f"[迷宫] 提取完成，结果保存在: {out_dir}")


# =========================
# 2. 对称图提取
# =========================

def extract_blue_polyline(img: np.ndarray) -> np.ndarray:
    """
    提取蓝色折线
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝线阈值，可按实际情况微调
    lower_blue = np.array([90, 60, 50], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 轻微清理
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    blue_mask = remove_small_components(blue_mask, min_area=20)

    return blue_mask


def extract_outer_box(img: np.ndarray) -> tuple:
    """
    提取黑色外框，返回 mask, rect
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_mask = extract_dark_lines(gray, thresh=100)

    dark_mask = cv2.dilate(
        dark_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    cnt, box, rect = keep_largest_rectangle_contour(dark_mask, min_area=50000)
    if rect is None:
        raise RuntimeError("没有找到对称图的大外框。")

    x, y, w, h = rect
    outer_box_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.rectangle(outer_box_mask, (x, y), (x + w, y + h), 255, 2)
    return outer_box_mask, rect


def extract_grid_and_dashed(img: np.ndarray, rect) -> tuple:
    """
    在大外框内提取：
    - 灰色网格
    - 中间虚线
    """
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 先取浅灰线
    light_mask = extract_light_gray_lines(gray, low=190, high=245)

    # 去掉蓝线、黑框影响
    dark_mask = extract_dark_lines(gray, thresh=120)
    light_mask[dark_mask > 0] = 0

    # 横线和竖线增强
    hori = horizontal_line_enhance(light_mask, ksize=25)
    vert = vertical_line_enhance(light_mask, ksize=25)

    grid_mask_roi = cv2.bitwise_or(hori, vert)
    grid_mask_roi = remove_small_components(grid_mask_roi, min_area=30)

    # ---- 提取虚线 ----
    # 思路：虚线是某一水平高度上的多个短横段
    # 先从 light_mask 中增强短横线
    dashed_candidate = horizontal_line_enhance(light_mask, ksize=9)

    # 用霍夫变换可进一步加强，但这里尽量保持纯形态学思路
    # 然后按行统计，找横向分布比较集中的行
    row_sum = np.sum(dashed_candidate > 0, axis=1)

    # 找“有不少短横段”的行
    # 这里阈值可按图调整
    candidate_rows = np.where(row_sum > max(20, w * 0.03))[0]

    dashed_mask_roi = np.zeros_like(dashed_candidate)
    if len(candidate_rows) > 0:
        # 找最可能的虚线区域：选择中位附近的候选行
        target_row = int(np.median(candidate_rows))

        band_top = max(0, target_row - 3)
        band_bottom = min(h, target_row + 4)
        dashed_band = dashed_candidate[band_top:band_bottom, :].copy()

        # 限制为细横段
        dashed_band = cv2.morphologyEx(
            dashed_band,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        )

        dashed_mask_roi[band_top:band_bottom, :] = dashed_band
        dashed_mask_roi = remove_small_components(dashed_mask_roi, min_area=5)

    # 网格中去掉虚线，避免重复
    grid_wo_dashed_roi = grid_mask_roi.copy()
    grid_wo_dashed_roi[dashed_mask_roi > 0] = 0

    # 放回原图坐标系
    grid_mask_full = put_roi_back(img.shape, grid_wo_dashed_roi, rect)
    dashed_mask_full = put_roi_back(img.shape, dashed_mask_roi, rect)

    return grid_mask_full, dashed_mask_full


def extract_symmetry(image_path: str, out_dir: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    ensure_dir(out_dir)

    # Step 1: 提取蓝色折线
    blue_mask = extract_blue_polyline(img)
    save_mask(blue_mask, os.path.join(out_dir, "sym_blue_mask.png"))
    blue_only = apply_mask_to_image(img, blue_mask)
    cv2.imwrite(os.path.join(out_dir, "sym_blue_only.png"), blue_only)

    # Step 2: 提取外框
    outer_box_mask, rect = extract_outer_box(img)
    save_mask(outer_box_mask, os.path.join(out_dir, "sym_outer_box_mask.png"))

    # Step 3: 提取网格和虚线
    grid_mask, dashed_mask = extract_grid_and_dashed(img, rect)
    save_mask(grid_mask, os.path.join(out_dir, "sym_grid_mask.png"))
    save_mask(dashed_mask, os.path.join(out_dir, "sym_dashed_mask.png"))

    # Step 4: 辅助线框总 mask = 外框 + 网格 + 虚线
    helper_mask = cv2.bitwise_or(outer_box_mask, grid_mask)
    helper_mask = cv2.bitwise_or(helper_mask, dashed_mask)
    save_mask(helper_mask, os.path.join(out_dir, "sym_helper_mask.png"))

    helper_only = apply_mask_to_image(img, helper_mask)
    cv2.imwrite(os.path.join(out_dir, "sym_helper_only.png"), helper_only)

    print(f"[对称图] 提取完成，结果保存在: {out_dir}")


# =========================
# 主程序
# =========================

def main():
    maze_image = "data/34migong.png"
    sym_image = "data/35duichen(1).png"

    maze_out_dir = "./output_maze"
    sym_out_dir = "./output_symmetry"

    extract_maze(maze_image, maze_out_dir)
    extract_symmetry(sym_image, sym_out_dir)

    print("全部处理完成。")


if __name__ == "__main__":
    main()