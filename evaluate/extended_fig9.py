import nibabel as nib

def create_contours_image(mni152_aparc_file, output_file):
    data = nib.load(mni152_aparc_file).get_data()
    for x in len(data):


if __name__ == '__main__':
    mni152_seg_file = '/usr/local/MNI152_T1_1mm/mri/aparc+aseg.mgz'
    contours_MNI152 = 'contours_MNI152.nii.gz'
    create_contours_image(mni152_seg_file, contours_MNI152)
