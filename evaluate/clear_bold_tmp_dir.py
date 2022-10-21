import os
from pathlib import Path


def clear_subject_bold_tmp_dir(bold_preprocess_dir: Path, subject_ids: list, task: str):
    for subject_id in subject_ids:
        tmp_dir = bold_preprocess_dir / subject_id / 'tmp' / f'task-{task}'
        if tmp_dir.exists():
            print(f'rm -r {tmp_dir}')
            os.system(f'rm -r {tmp_dir}')


def main():
    bold_preprocess_dir = Path('/mnt/ngshare2/DeepPrep_UKB/UKB_BoldPreprocess')
    subjects_ids = os.listdir(bold_preprocess_dir)
    clear_subject_bold_tmp_dir(bold_preprocess_dir, subjects_ids, task='rest')


if __name__ == '__main__':
    main()
