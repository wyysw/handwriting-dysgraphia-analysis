"""
三组图层叠加脚本

组1: data/raw/35duichen.png  +  data/raw/sym/{id}.png   → data/mark/sym/{id}.png
组2: data/raw/34migong.png   +  data/raw/maze/{id}.png  → data/mark/maze/{id}.png
组3: data/raw/36circle.png   +  data/raw/circle/{id}.png→ data/mark/circle/{id}.png
"""

from pathlib import Path
from PIL import Image


GROUPS = [
    {
        "name": "sym",
        "background": "data/raw/35duichen.png",
        "samples_dir": "data/raw/sym",
        "output_dir": "data/mark/sym",
    },
    {
        "name": "maze",
        "background": "data/raw/34migong.png",
        "samples_dir": "data/raw/maze",
        "output_dir": "data/mark/maze",
    },
    {
        "name": "circle",
        "background": "data/raw/36circle.png",
        "samples_dir": "data/raw/circle",
        "output_dir": "data/mark/circle",
    },
]


def process_group(name: str, background: str, samples_dir: str, output_dir: str) -> None:
    bg_path = Path(background)
    samples_path = Path(samples_dir)
    out_path = Path(output_dir)

    print(f"{'='*50}")
    print(f"[组: {name}]")

    if not bg_path.exists():
        print(f"  ✗ 背景图不存在，跳过：{bg_path}\n")
        return

    if not samples_path.exists():
        print(f"  ✗ 样本目录不存在，跳过：{samples_path}\n")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    png_files = sorted(samples_path.glob("*.png"))
    if not png_files:
        print(f"  ✗ 未找到任何 PNG 文件：{samples_path}\n")
        return

    background_orig = Image.open(bg_path).convert("RGBA")
    print(f"  背景图：{bg_path}  尺寸：{background_orig.size}")
    print(f"  样本数：{len(png_files)}")

    for png_file in png_files:
        sample_id = png_file.stem
        background = background_orig.copy()
        overlay = Image.open(png_file).convert("RGBA")

        if overlay.size != background.size:
            print(f"  ! [{sample_id}] 尺寸不匹配（overlay={overlay.size}, bg={background.size}），自动缩放上层图。")
            overlay = overlay.resize(background.size, Image.LANCZOS)

        background.paste(overlay, (0, 0), mask=overlay)

        output_file = out_path / f"{sample_id}.png"
        background.convert("RGB").save(output_file, "PNG")
        print(f"  ✓ {sample_id}.png")

    print(f"  完成！输出目录：{out_path}\n")


def main() -> None:
    for group in GROUPS:
        process_group(**group)
    print("全部三组处理完毕。")


if __name__ == "__main__":
    main()