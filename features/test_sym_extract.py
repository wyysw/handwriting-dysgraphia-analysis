import subprocess
from pathlib import Path

import os
import glob

ids = [os.path.splitext(os.path.basename(f))[0] 
       for f in glob.glob('data/raw/sym/*.txt')]

# ids = ['s1','s2','s3','s4']

for id in ids:
    cmd = [
        'python', 'features/sym_feature_extractor.py',
        '--txt', f'data/raw/sym/{id}.txt',
        '--png', f'data/raw/sym/{id}.png',
        '--blue', 'data/shape_out/sym_blue_mask',
        '--helper', 'data/shape_out/sym_helper_mask.png',
        '--out', f'output_sym/extract/{id}.json',
        '--vis', f'output_sym/extract/vis_{id}',
        '--sample_id', id
    ]
    subprocess.run(cmd)