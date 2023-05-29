import nibabel as nib
import numpy as np


def convert_to_ras(input_path, output_path):
    img = nib.load(input_path)
    orig_ornt = nib.orientations.io_orientation(img.header.get_sform())
    RAS_ornt = nib.orientations.axcodes2ornt('RAS')
    if np.array_equal(orig_ornt, RAS_ornt) is True:
        print(f"{input_path} is already in RAS orientation. Copying to {output_path}.")
        nib.save(img, output_path)
    else:
        newimg = img.as_reoriented(orig_ornt)
        nib.save(newimg, output_path)
        print(f"Successfully converted {input_path} to RAS orientation and saved to {output_path}.")


if __name__ == '__main__':
    nifti_file = '/mnt/ngshare/Workspace/sub-pbfs11_ses-001_task-reading_run-01_bold.nii.gz'
    output_file = '/mnt/ngshare/Workspace/sub-pbfs11_ses-001_task-reading_run-02_bold.nii.gz'
    convert_to_ras(nifti_file, output_file)
