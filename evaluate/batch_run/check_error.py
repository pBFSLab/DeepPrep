import os
from pathlib import Path
import shutil

if __name__ == '__main__':
    data_path = Path('/mnt/nfs')

    subj_paths = list(data_path.glob('I*'))
    for subj_path in subj_paths:
        if (subj_path / 'scripts' / 'recon-all.error').exists():
            shutil.rmtree(subj_path)
            print(f'delete: {subj_path}')
    pass
