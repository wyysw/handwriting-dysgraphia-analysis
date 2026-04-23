import subprocess
from pathlib import Path

import os
import glob

ids = [os.path.splitext(os.path.basename(f))[0] 
       for f in glob.glob('data/samples/maze/*.txt')]

# ids = ['l3', 'l4', 'l1']

for id in ids:
    cmd = [
        'python', 'features/maze_feature_extractor.py',
        '--txt', f'data/samples/maze/{id}.txt',
        '--png', f'data/samples/maze/{id}.png',
        '--game', 'maze',
        '--mask', 'output_maze/shape_maze/maze_mask.png',
        '--out', f'output_maze/extract/{id}.json',
        '--vis_dir', f'output_maze/extract/vis_{id}',
        '--sample_id', id
    ]
    subprocess.run(cmd)