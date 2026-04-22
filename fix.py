import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pen import analyze
from pen import pen_trajectory_plotter

SKIP_ROWS = 3
TXT_PATH = 'data/samples/maze/l4.txt'
REF_PNG_PATH = 'data/samples/maze/l4.png'
CANVAS_PATH = 'data/maze_mask.png'
OUT_PNG_PATH = 'data/son.png'
OUT_SCRIPT_INFO = 'data/on.txt'


def get_visible_bbox_from_reference_png(ref_png_path):
    """从 s3.png 中提取可见轨迹的包围盒。优先使用 alpha 通道。"""
    img = Image.open(ref_png_path).convert('RGBA')
    arr = np.array(img)
    alpha = arr[..., 3]

    visible = alpha > 0
    if not np.any(visible):
        rgb = arr[..., :3]
        visible = np.any(rgb < 250, axis=2)

    ys, xs = np.where(visible)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError('未能从参考图中提取到可见轨迹 bbox。')

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def build_affine_from_data_to_bbox(strokes, target_bbox):
    """基于落笔点范围，构建到参考 bbox 的线性映射。"""
    all_x = np.concatenate([s['x'] for s in strokes])
    all_y = np.concatenate([s['y'] for s in strokes])

    sx0, sy0 = np.min(all_x), np.min(all_y)
    sx1, sy1 = np.max(all_x), np.max(all_y)
    tx0, ty0, tx1, ty1 = target_bbox

    scale_x = (tx1 - tx0) / (sx1 - sx0)
    scale_y = (ty1 - ty0) / (sy1 - sy0)

    def transform(stroke):
        return {
            'x': (stroke['x'] - sx0) * scale_x + tx0,
            'y': (stroke['y'] - sy0) * scale_y + ty0,
            'pressure': stroke['pressure'].copy(),
        }

    return transform, {
        'source_bbox': (float(sx0), float(sy0), float(sx1), float(sy1)),
        'target_bbox': target_bbox,
        'scale_x': float(scale_x),
        'scale_y': float(scale_y),
    }


def main():
    raw = analyze.load_trajectory_data(TXT_PATH, skip_rows=SKIP_ROWS)
    if raw is None:
        raise RuntimeError('轨迹数据加载失败。')

    strokes = analyze.split_into_strokes_simple(raw)
    if not strokes:
        raise RuntimeError('未分割出任何落笔笔画。')

    ref_bbox = get_visible_bbox_from_reference_png(REF_PNG_PATH)
    transform, info = build_affine_from_data_to_bbox(strokes, ref_bbox)
    transformed_strokes = [transform(s) for s in strokes]

    bg = Image.open(CANVAS_PATH).convert('RGB')
    bg_arr = np.array(bg)
    bg_w, bg_h = bg.size

    fig_w = bg_w / 100.0
    fig_h = bg_h / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(bg_arr, extent=(0, bg_w - 1, bg_h - 1, 0))

    xlim = (0, bg_w - 1)
    ylim = (0, bg_h - 1)

    for idx, stroke in enumerate(transformed_strokes):
        pen_trajectory_plotter.plot_stroke(
            stroke_data=stroke,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            fig=fig,
            stroke_index=str(idx),
            color='red',
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT_PNG_PATH, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    with open(OUT_SCRIPT_INFO, 'w', encoding='utf-8') as f:
        f.write('s3.txt 落笔笔画数: %d\n' % len(strokes))
        f.write('source_bbox: %s\n' % (info['source_bbox'],))
        f.write('target_bbox(from s3.png): %s\n' % (info['target_bbox'],))
        f.write('scale_x: %.8f\n' % info['scale_x'])
        f.write('scale_y: %.8f\n' % info['scale_y'])

    print('Saved:', OUT_PNG_PATH)
    print('Saved:', OUT_SCRIPT_INFO)
    print('stroke_count:', len(strokes))
    print('target_bbox:', info['target_bbox'])


if __name__ == '__main__':
    main()
