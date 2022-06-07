import os
from pathlib import Path
import bids

from bids_validator import BIDSValidator

validator = BIDSValidator()
filepaths = ["/sub-01/anat/sub-01_rec-CSD_T1w.nii.gz",
             "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.exe",  # wrong extension
             "home/username/my_dataset/participants.tsv",  # not relative to root
             "/participants.tsv"]
for filepath in filepaths:
    print(validator.is_bids(filepath))

if __name__ == '__main__':
    data_path = Path('/home/weiwei/workdata/DeepPrep/BoldPipeline/TestData')

    validator = bids.BIDSValidator()
    validator.is_bids(str(data_path))
    pass
