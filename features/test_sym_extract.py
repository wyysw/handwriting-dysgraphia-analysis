import subprocess
from pathlib import Path

import os
import glob

ids = [os.path.splitext(os.path.basename(f))[0] 
       for f in glob.glob('data/samples/sym/*.txt')]

# ids = ['s1','s2','s3','s4']

for id in ids:
    cmd = [
        'python', 'features/sym_feature_extractor.py',
        '--txt', f'data/samples/sym/{id}.txt',
        '--png', f'data/samples/sym/{id}.png',
        '--blue', 'output_sym/shape_sym/sym_blue_mask.png',
        '--helper', 'output_sym/shape_sym/sym_helper_mask_completed.png',
        '--out', f'output_sym/extract/{id}.json',
        '--vis', f'output_sym/extract/vis_{id}',
        '--sample_id', id
    ]
    subprocess.run(cmd)