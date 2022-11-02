from app.surface_projection import surface_projection as sp
import bids
from pathlib import Path
from interface.run import set_envrion
import os


def pipeline(subject_ids, task, data_path, bold_path):

    layout = bids.BIDSLayout(str(data_path), derivatives=False)
    for subject_id in subject_ids:
        subj = subject_id.split('-')[1]
        bids_bolds = layout.get(subject=subj, task=task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            entities = dict(bids_bold.entities)
            for hemi in ['lh', 'rh']:
                sub_ses = subject_id + '-ses-' + entities['session']
                subj_surf_path = Path(bold_path) / sub_ses / f"ses-{entities['session']}"/ 'surf'
                subj_func_path = Path(bold_path) / sub_ses / f"ses-{entities['session']}"/ 'func'
                subj_surf_path.mkdir(parents=True, exist_ok=True)
                file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
                dst_resid_file = subj_func_path / f'{file_prefix}_mc.nii.gz'
                dst_reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
                fs6_path = sp.indi_to_fs6(subj_surf_path, subject_id , dst_resid_file, dst_reg_file,
                                          hemi)
                sm6_path = sp.smooth_fs6(fs6_path, hemi)
                sp.downsample_fs6_to_fs4(sm6_path, hemi)

if __name__ == '__main__':
    set_envrion()
    os.environ['SUBJECTS_DIR'] = f'/mnt/ngshare2/BOAI/data_Recon'
    data_path = f'/mnt/ngshare2/BOAI/data_bids'
    bold_path = f'/mnt/ngshare2/BOAI/data_result'
    subject_ids = os.listdir(data_path)
    subject_ids.remove('dataset_description.json')
    task = 'motor'  # motor or task

    pipeline(subject_ids, task, data_path, bold_path)

