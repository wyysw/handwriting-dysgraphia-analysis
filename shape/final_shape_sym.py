import os
import shutil
import cv2
import numpy as np

EXPECTED_WIDTH = 1201
EXPECTED_HEIGHT = 1601
IGNORE_SIDE_WIDTH = 190
DEFAULT_IGNORE_MODE = "white"   # "white" or "transparent"

# 线提取参数
DARK_THRESH = 100
GRID_GRAY_LOW = 190
GRID_GRAY_HIGH = 245

# 网格补全参数
MIN_VERTICAL_COL_SUM_RATIO = 0.10
MIN_HORIZONTAL_ROW_SUM_RATIO = 0.10
CLUSTER_GAP = 6
GRID_LINE_THICKNESS = 1
OUTER_BORDER_INSET = 1

# 新增：边缘辅助线删除参数
BORDER_NEAR_LINE_RATIO = 0.75
# 含义：若“边缘到首条线”的距离 < 正常间距均值 * 0.75，则删掉这条边缘线

# 新增：中轴线补线参数
MID_AXIS_GAP_RATIO = 1.6
# 含义：若某个相邻间距 > 正常间距中位数 * 1.6，则在该间距中点补一条线


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_directory(dir_path: str):
    """删除目录中的所有内容，但保留目录本身。"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)


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


def extract_dark_lines(gray: np.ndarray, thresh=100) -> np.ndarray:
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return mask


def extract_light_gray_lines(gray: np.ndarray, low=190, high=245) -> np.ndarray:
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


def put_roi_back(full_shape, roi_mask, rect):
    x, y, w, h = rect
    full = np.zeros(full_shape[:2], dtype=np.uint8)
    full[y:y + h, x:x + w] = roi_mask
    return full


def build_ignore_mask(image_shape, side_width=190) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    side_width = max(0, min(side_width, w // 2))
    if side_width > 0:
        mask[:, :side_width] = 255
        mask[:, w - side_width:] = 255

    return mask


def preprocess_ignore_side_regions(
    img: np.ndarray,
    side_width: int = 190,
    expected_width: int = 1201,
    expected_height: int = 1601,
    fill_mode: str = "white"
):
    h, w = img.shape[:2]

    if (w, h) != (expected_width, expected_height):
        print(
            f"[警告] 当前图片尺寸为 {w}x{h}，"
            f"与预期 {expected_width}x{expected_height} 不一致。"
            f"仍将按左右各 {side_width}px 进行忽略处理。"
        )

    ignored_mask = build_ignore_mask(img.shape, side_width=side_width)

    processed_bgr = img.copy()
    processed_bgr[ignored_mask > 0] = (255, 255, 255)

    if fill_mode == "transparent":
        preview_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        preview_rgba[ignored_mask > 0, 3] = 0
    else:
        preview_rgba = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2BGRA)

    return processed_bgr, ignored_mask, preview_rgba


def extract_blue_polyline(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 50], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    blue_mask = remove_small_components(blue_mask, min_area=20)
    return blue_mask


def extract_outer_box(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_mask = extract_dark_lines(gray, thresh=DARK_THRESH)

    dark_mask = cv2.dilate(
        dark_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    _, _, rect = keep_largest_rectangle_contour(dark_mask, min_area=50000)
    if rect is None:
        raise RuntimeError("没有找到对称图的大外框。")

    x, y, w, h = rect
    outer_box_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.rectangle(outer_box_mask, (x, y), (x + w, y + h), 255, 2)
    return outer_box_mask, rect


def extract_grid_and_dashed(img: np.ndarray, rect):
    """
    - grid_mask_full: 原始提取的网格（未补全）
    - dashed_mask_full: 虚线
    - light_mask_full: 原始浅灰线候选（便于调试）
    """
    x, y, w, h = rect
    roi = img[y:y + h, x:x + w].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    light_mask = extract_light_gray_lines(gray, low=GRID_GRAY_LOW, high=GRID_GRAY_HIGH)

    dark_mask = extract_dark_lines(gray, thresh=120)
    light_mask[dark_mask > 0] = 0

    hori = horizontal_line_enhance(light_mask, ksize=25)
    vert = vertical_line_enhance(light_mask, ksize=25)

    grid_mask_roi = cv2.bitwise_or(hori, vert)
    grid_mask_roi = remove_small_components(grid_mask_roi, min_area=30)

    dashed_candidate = horizontal_line_enhance(light_mask, ksize=9)

    row_sum = np.sum(dashed_candidate > 0, axis=1)
    candidate_rows = np.where(row_sum > max(20, w * 0.03))[0]

    dashed_mask_roi = np.zeros_like(dashed_candidate)
    if len(candidate_rows) > 0:
        target_row = int(np.median(candidate_rows))

        band_top = max(0, target_row - 3)
        band_bottom = min(h, target_row + 4)
        dashed_band = dashed_candidate[band_top:band_bottom, :].copy()

        dashed_band = cv2.morphologyEx(
            dashed_band,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        )

        dashed_mask_roi[band_top:band_bottom, :] = dashed_band
        dashed_mask_roi = remove_small_components(dashed_mask_roi, min_area=5)

    grid_wo_dashed_roi = grid_mask_roi.copy()
    grid_wo_dashed_roi[dashed_mask_roi > 0] = 0

    grid_mask_full = put_roi_back(img.shape, grid_wo_dashed_roi, rect)
    dashed_mask_full = put_roi_back(img.shape, dashed_mask_roi, rect)
    light_mask_full = put_roi_back(img.shape, light_mask, rect)

    return grid_mask_full, dashed_mask_full, light_mask_full


def cluster_positions(positions, max_gap=6):
    if len(positions) == 0:
        return []

    positions = sorted(int(p) for p in positions)
    groups = [[positions[0]]]

    for p in positions[1:]:
        if p - groups[-1][-1] <= max_gap:
            groups[-1].append(p)
        else:
            groups.append([p])

    centers = [int(round(np.mean(g))) for g in groups]
    return centers


def mean_internal_spacing(positions):
    if len(positions) < 2:
        return None
    gaps = np.diff(sorted(positions))
    if len(gaps) == 0:
        return None
    return float(np.mean(gaps))


def median_internal_spacing(positions):
    if len(positions) < 2:
        return None
    gaps = np.diff(sorted(positions))
    if len(gaps) == 0:
        return None
    return float(np.median(gaps))


def remove_border_adjacent_lines(positions, extent_len, border_ratio=0.75):
    """
    删除与边框过近的首尾线。
    """
    positions = sorted(int(p) for p in positions)
    if len(positions) < 3:
        return positions, {"removed_start": False, "removed_end": False}

    changed = {"removed_start": False, "removed_end": False}

    # 先处理首条
    if len(positions) >= 3:
        normal_mean = mean_internal_spacing(positions[1:])  # 排除首条后看内部间距
        if normal_mean is not None:
            start_gap = positions[0]
            if start_gap < normal_mean * border_ratio:
                positions = positions[1:]
                changed["removed_start"] = True

    # 再处理末条
    if len(positions) >= 3:
        normal_mean = mean_internal_spacing(positions[:-1])  # 排除末条后看内部间距
        if normal_mean is not None:
            end_gap = (extent_len - 1) - positions[-1]
            if end_gap < normal_mean * border_ratio:
                positions = positions[:-1]
                changed["removed_end"] = True

    return positions, changed


def insert_mid_axis_if_needed(positions, gap_ratio=1.6):
    """
    如果某个相邻间距明显大于正常间距，则在中间补一条线。
    例如 500 和 610 之间会补 555。
    """
    positions = sorted(int(p) for p in positions)
    if len(positions) < 2:
        return positions, None

    typical_gap = median_internal_spacing(positions)
    if typical_gap is None or typical_gap <= 0:
        return positions, None

    gaps = np.diff(positions)
    max_idx = int(np.argmax(gaps))
    max_gap = gaps[max_idx]

    if max_gap > typical_gap * gap_ratio:
        left_p = positions[max_idx]
        right_p = positions[max_idx + 1]
        mid_p = int(round((left_p + right_p) / 2.0))

        if mid_p not in positions:
            positions = sorted(positions + [mid_p])
            return positions, {
                "left": left_p,
                "right": right_p,
                "mid": mid_p,
                "typical_gap": typical_gap,
                "max_gap": int(max_gap)
            }

    return positions, None


def complete_grid_inside_outer_box(
    grid_mask_full: np.ndarray,
    rect,
    min_vertical_col_sum_ratio=MIN_VERTICAL_COL_SUM_RATIO,
    min_horizontal_row_sum_ratio=MIN_HORIZONTAL_ROW_SUM_RATIO,
    cluster_gap=CLUSTER_GAP,
    line_thickness=GRID_LINE_THICKNESS,
    inset=OUTER_BORDER_INSET,
    border_ratio=BORDER_NEAR_LINE_RATIO,
    mid_axis_gap_ratio=MID_AXIS_GAP_RATIO
):
    """
    返回：
    - completed_full: 全图坐标下的补全后网格mask
    - vertical_xs_raw: 原始识别竖线位置
    - horizontal_ys_raw: 原始识别横线位置
    - vertical_xs: 清理后的竖线位置
    - horizontal_ys: 清理并补中轴后的横线位置
    - debug_info: 调试信息
    """
    x, y, w, h = rect

    roi = grid_mask_full[y:y + h, x:x + w].copy()
    roi_bin = (roi > 0).astype(np.uint8)

    col_sum = np.sum(roi_bin, axis=0)
    row_sum = np.sum(roi_bin, axis=1)

    min_col_sum = max(10, int(h * min_vertical_col_sum_ratio))
    min_row_sum = max(10, int(w * min_horizontal_row_sum_ratio))

    candidate_x = np.where(col_sum >= min_col_sum)[0]
    candidate_y = np.where(row_sum >= min_row_sum)[0]

    vertical_xs_raw = cluster_positions(candidate_x.tolist(), max_gap=cluster_gap)
    horizontal_ys_raw = cluster_positions(candidate_y.tolist(), max_gap=cluster_gap)

    # 1) 删除与边框过近的首尾辅助线
    vertical_xs, vertical_border_debug = remove_border_adjacent_lines(
        vertical_xs_raw, extent_len=w, border_ratio=border_ratio
    )
    horizontal_ys, horizontal_border_debug = remove_border_adjacent_lines(
        horizontal_ys_raw, extent_len=h, border_ratio=border_ratio
    )

    # 2) 横线中若存在异常大间距，则补一条中轴线
    horizontal_ys, horizontal_mid_debug = insert_mid_axis_if_needed(
        horizontal_ys, gap_ratio=mid_axis_gap_ratio
    )

    completed_roi = np.zeros_like(roi)

    x0 = inset
    x1 = max(inset, w - 1 - inset)
    y0 = inset
    y1 = max(inset, h - 1 - inset)

    for xx in vertical_xs:
        xx = int(np.clip(xx, 0, w - 1))
        cv2.line(completed_roi, (xx, y0), (xx, y1), 255, thickness=line_thickness)

    for yy in horizontal_ys:
        yy = int(np.clip(yy, 0, h - 1))
        cv2.line(completed_roi, (x0, yy), (x1, yy), 255, thickness=line_thickness)

    completed_full = np.zeros_like(grid_mask_full)
    completed_full[y:y + h, x:x + w] = completed_roi

    debug_info = {
        "vertical_border_debug": vertical_border_debug,
        "horizontal_border_debug": horizontal_border_debug,
        "horizontal_mid_debug": horizontal_mid_debug,
        "min_col_sum": min_col_sum,
        "min_row_sum": min_row_sum,
    }

    return (
        completed_full,
        vertical_xs_raw,
        horizontal_ys_raw,
        vertical_xs,
        horizontal_ys,
        debug_info
    )


def draw_grid_keypoints_preview(img_shape, rect, vertical_xs, horizontal_ys):
    x, y, w, h = rect
    canvas = np.zeros(img_shape[:2], dtype=np.uint8)

    for xx in vertical_xs:
        gx = x + xx
        cv2.line(canvas, (gx, y), (gx, y + h - 1), 255, 1)

    for yy in horizontal_ys:
        gy = y + yy
        cv2.line(canvas, (x, gy), (x + w - 1, gy), 255, 1)

    return canvas


def extract_symmetry(
    image_path: str,
    out_dir: str,
    ignore_side_width: int = IGNORE_SIDE_WIDTH,
    expected_width: int = EXPECTED_WIDTH,
    expected_height: int = EXPECTED_HEIGHT,
    ignore_mode: str = DEFAULT_IGNORE_MODE
):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    clear_directory(out_dir)

    # 1) 左右忽略区预处理
    processed_img, ignored_mask, preview_rgba = preprocess_ignore_side_regions(
        img,
        side_width=ignore_side_width,
        expected_width=expected_width,
        expected_height=expected_height,
        fill_mode=ignore_mode
    )

    save_mask(ignored_mask, os.path.join(out_dir, "sym_ignored_side_mask.png"))
    cv2.imwrite(os.path.join(out_dir, "sym_preprocessed_white.png"), processed_img)
    cv2.imwrite(os.path.join(out_dir, "sym_preprocessed_preview_rgba.png"), preview_rgba)

    # 2) 蓝色折线
    blue_mask = extract_blue_polyline(processed_img)
    save_mask(blue_mask, os.path.join(out_dir, "sym_blue_mask.png"))
    blue_only = apply_mask_to_image(processed_img, blue_mask)
    cv2.imwrite(os.path.join(out_dir, "sym_blue_only.png"), blue_only)

    # 3) 外框
    outer_box_mask, rect = extract_outer_box(processed_img)
    save_mask(outer_box_mask, os.path.join(out_dir, "sym_outer_box_mask.png"))

    # 4) 原始网格 + 虚线
    grid_mask_raw, dashed_mask, light_mask = extract_grid_and_dashed(processed_img, rect)
    save_mask(light_mask, os.path.join(out_dir, "sym_light_gray_mask.png"))
    save_mask(grid_mask_raw, os.path.join(out_dir, "sym_grid_mask_raw.png"))
    save_mask(dashed_mask, os.path.join(out_dir, "sym_dashed_mask.png"))

    # 5) 补全网格（含边缘线删除 + 中轴线补充）
    (
        grid_mask_completed,
        vertical_xs_raw,
        horizontal_ys_raw,
        vertical_xs,
        horizontal_ys,
        debug_info
    ) = complete_grid_inside_outer_box(
        grid_mask_raw,
        rect=rect,
        min_vertical_col_sum_ratio=MIN_VERTICAL_COL_SUM_RATIO,
        min_horizontal_row_sum_ratio=MIN_HORIZONTAL_ROW_SUM_RATIO,
        cluster_gap=CLUSTER_GAP,
        line_thickness=GRID_LINE_THICKNESS,
        inset=OUTER_BORDER_INSET,
        border_ratio=BORDER_NEAR_LINE_RATIO,
        mid_axis_gap_ratio=MID_AXIS_GAP_RATIO
    )

    save_mask(grid_mask_completed, os.path.join(out_dir, "sym_grid_completed_mask.png"))

    # 调试图：原始检测位置
    grid_key_preview_raw = draw_grid_keypoints_preview(
        processed_img.shape, rect, vertical_xs_raw, horizontal_ys_raw
    )
    save_mask(grid_key_preview_raw, os.path.join(out_dir, "sym_grid_keypoints_preview_raw.png"))

    # 调试图：最终位置
    grid_key_preview_final = draw_grid_keypoints_preview(
        processed_img.shape, rect, vertical_xs, horizontal_ys
    )
    save_mask(grid_key_preview_final, os.path.join(out_dir, "sym_grid_keypoints_preview_final.png"))

    # 6) 叠回原图查看
    grid_completed_only = apply_mask_to_image(processed_img, grid_mask_completed)
    cv2.imwrite(os.path.join(out_dir, "sym_grid_completed_only.png"), grid_completed_only)

    # 7) helper：外框 + 补全网格 + 虚线
    helper_mask_completed = cv2.bitwise_or(outer_box_mask, grid_mask_completed)
    helper_mask_completed = cv2.bitwise_or(helper_mask_completed, dashed_mask)
    save_mask(helper_mask_completed, os.path.join(out_dir, "sym_helper_mask.png"))

    helper_only_completed = apply_mask_to_image(processed_img, helper_mask_completed)
    cv2.imwrite(os.path.join(out_dir, "sym_helper_only_completed.png"), helper_only_completed)

    # 旧版 helper 保留
    helper_mask_raw = cv2.bitwise_or(outer_box_mask, grid_mask_raw)
    helper_mask_raw = cv2.bitwise_or(helper_mask_raw, dashed_mask)
    save_mask(helper_mask_raw, os.path.join(out_dir, "sym_helper_mask_raw.png"))

    helper_only_raw = apply_mask_to_image(processed_img, helper_mask_raw)
    cv2.imwrite(os.path.join(out_dir, "sym_helper_only_raw.png"), helper_only_raw)

    print(f"[对称图] 提取完成，结果保存在: {out_dir}")
    print(f"[对称图] 外框 rect = {rect}")

    print("\n=== 原始检测到的线位置（ROI内）===")
    print(f"竖线 raw: {vertical_xs_raw}")
    print(f"横线 raw: {horizontal_ys_raw}")

    print("\n=== 边缘线清理后的线位置（ROI内）===")
    print(f"竖线 final: {vertical_xs}")
    print(f"横线 final: {horizontal_ys}")

    print("\n=== 调试信息 ===")
    print(f"vertical_border_debug: {debug_info['vertical_border_debug']}")
    print(f"horizontal_border_debug: {debug_info['horizontal_border_debug']}")
    print(f"horizontal_mid_debug: {debug_info['horizontal_mid_debug']}")
    print(f"min_col_sum: {debug_info['min_col_sum']}, min_row_sum: {debug_info['min_row_sum']}")

    if debug_info["horizontal_mid_debug"] is not None:
        info = debug_info["horizontal_mid_debug"]
        print(
            f"[中轴补线] 在横线 ROI 坐标 {info['left']} 与 {info['right']} 之间，"
            f"补入中轴线 {info['mid']}"
        )


def main():
    # 修改成你的实际输入图片路径
    sym_image = "data/raw/35duichen.png"
    sym_out_dir = "./output_sym/shape_sym"

    extract_symmetry(
        image_path=sym_image,
        out_dir=sym_out_dir,
        ignore_side_width=190,
        expected_width=1201,
        expected_height=1601,
        ignore_mode="white"
    )


if __name__ == "__main__":
    main()