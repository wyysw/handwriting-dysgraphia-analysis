import subprocess
from pathlib import Path

ids = ['l3', 'l4', 'l1']  # 或自动扫描 data/samples/maze/*.txt

for id in ids:
    cmd = [
        'python', 'features/maze_feature_extractor.py',
        '--txt', f'data/samples/maze/{id}.txt',
        '--png', f'data/samples/maze/{id}.png',
        '--mask', 'output_maze/shape_maze/maze_mask.png',
        '--out', f'output_maze/extract/{id}.json',
        '--vis_dir', f'output_maze/extract/vis_{id}',
        '--sample_id', id
    ]
    subprocess.run(cmd)