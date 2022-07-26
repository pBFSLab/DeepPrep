import os
from pathlib import Path


def make_mirror_dir(root_dir, target_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_ in files:
            source_file = Path(root) / file_
            target_file = Path(root.replace(root_dir, target_dir)) / file_
            target_parent = target_file.parent
            if not target_parent.exists():
                target_parent.mkdir(parents=True, exist_ok=True)
            if not target_file.exists():
                target_file.symlink_to(source_file)


if __name__ == '__main__':
    # s_dir = '/mnt/ngshare/DeepPrep/MSC'
    # t_dir = '/mnt/ngshare/Data_Mirror/DeepPrep/MSC'

    s_dir = '/mnt/ngshare/SurfReg/Data_Processing/NAMIC'
    t_dir = '/mnt/ngshare/SurfReg/Data_Processing/NAMIC_mirror'
    make_mirror_dir(s_dir, t_dir)
