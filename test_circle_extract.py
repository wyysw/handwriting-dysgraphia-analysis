import subprocess
from pathlib import Path

ids = ['c7', 'c9', 'c1']  # 或自动扫描 data/samples/circle/*.txt

for id in ids:
    cmd = [
        'python', 'features/maze_feature_extractor.py',
        '--txt', f'data/samples/circle/{id}.txt',
        '--png', f'data/samples/circle/{id}.png',
        '--mask', 'output_circle/shape_circle/circle_mask.png',
        '--out', f'output_circle/extract/{id}.json',
        '--vis_dir', f'output_circle/extract/vis_{id}',
        '--sample_id', id
    ]
    subprocess.run(cmd)