import subprocess
from pathlib import Path

import os
import glob

ids = [os.path.splitext(os.path.basename(f))[0] 
       for f in glob.glob('data/samples/circle/*.txt')]

# ids = ['c11','c1','c7','c9']

for id in ids:
    cmd = [
        'python', 'features/maze_feature_extractor.py',
        '--txt', f'data/samples/circle/{id}.txt',
        '--png', f'data/samples/circle/{id}.png',
        '--game', 'circle',
        '--mask', 'output_circle/shape_circle/circle_mask.png',
        '--out', f'output_circle/extract/{id}.json',
        '--vis_dir', f'output_circle/extract/vis_{id}',
        '--sample_id', id
    ]
    subprocess.run(cmd)