#! /usr/bin/env python3
from pathlib import Path
import argparse
import os


# import bids
# from niworkflows.interfaces.bids import DerivativesDataSink
# layout = bids.BIDSLayout(root=bids_dir, validate=False)
# info = layout.parse_file_entities(bold_space_orig_file)
# dsink = DerivativesDataSink(base_directory=output_dir, check_hdr=False)
# dsink.inputs.in_file = str(aseg_downsample_mgz)
# dsink.inputs.source_file = str(bold_space_t1w_file)
# dsink.inputs.desc = 'dseg'
# dsink.out_path_base = os.path.join(output_dir, 'tmp', 'anat2bold_t1w')
# res = dsink.run()
# res.outputs.out_file


def anat2bold_t1w(aseg_mgz: str, brainmask_mgz: str, bold_space_t1w_file: str,
                  aseg, wm, vent, csf, brainmask, brainmask_bin
                  ):

    cmd = f'mri_convert -rl {bold_space_t1w_file} {aseg_mgz} {aseg}'
    os.system(cmd)
    assert os.path.exists(aseg)

    cmd = f'mri_convert -rl {bold_space_t1w_file} {brainmask_mgz} {brainmask}'
    os.system(cmd)
    assert os.path.exists(brainmask)

    shargs = [
        '--i', aseg,
        '--wm',
        '--erode', '1',
        '--o', wm]
    os.system('mri_binarize ' + ' '.join(shargs))
    assert os.path.exists(wm)

    shargs = [
        '--i', aseg,
        '--min', '24',
        '--max', '24',
        '--o', csf]
    os.system('mri_binarize ' + ' '.join(shargs))
    assert os.path.exists(csf)

    shargs = [
        '--i', aseg,
        '--ventricles',
        '--o', vent]
    os.system('mri_binarize ' + ' '.join(shargs))
    assert os.path.exists(vent)

    shargs = [
        '--i', brainmask,
        '--o', brainmask_bin,
        '--min', '0.0001']
    os.system('mri_binarize ' + ' '.join(shargs))
    assert os.path.exists(brainmask_bin)


def anat2bold_by_bbregister(subj_func_dir: Path, mc: Path, bbregister_dat: Path, subjects_dir: Path, subject_id: str):
    mov = mc
    reg = bbregister_dat

    # project aparc+aseg to mc
    seg = Path(subjects_dir) / subject_id / 'mri' / 'aparc+aseg.mgz'  # Recon
    func = subj_func_dir / mc.name.replace('bold.nii.gz', 'desc-aparcaseg_dseg.nii.gz')
    wm = subj_func_dir / mc.name.replace('bold.nii.gz', 'desc-wm_mask.nii.gz')
    vent = subj_func_dir / mc.name.replace('bold.nii.gz', 'desc-ventricles_mask.nii.gz')
    csf = subj_func_dir / mc.name.replace('bold.nii.gz', 'desc-csf_mask.nii.gz')
    # project brainmask.mgz to mc
    targ = Path(subjects_dir) / subject_id / 'mri' / 'brainmask.mgz'  # Recon
    mask = subj_func_dir / mc.name.replace('bold.nii.gz', 'desc-brain_mask.nii.gz')
    binmask = subj_func_dir / mc.name.replace('bold.nii.gz', 'desc-brain_maskbin.nii.gz')

    shargs = [
        '--seg', seg,
        '--temp', mov,
        '--reg', reg,
        '--o', func]
    os.system('mri_label2vol ' + ' '.join(shargs))

    shargs = [
        '--i', func,
        '--wm',
        '--erode', '1',
        '--o', wm]
    os.system('mri_binarize ' + ' '.join(shargs))

    shargs = [
        '--i', func,
        '--min', '24',
        '--max', '24',
        '--o', csf]
    os.system('mri_binarize ' + ' '.join(shargs))

    shargs = [
        '--i', func,
        '--ventricles',
        '--o', vent]
    os.system('mri_binarize ' + ' '.join(shargs))

    shargs = [
        '--reg', reg,
        '--targ', targ,
        '--mov', mov,
        '--inv',
        '--o', mask]
    os.system('mri_vol2vol ' + ' '.join(shargs))

    shargs = [
        '--i', mask,
        '--o', binmask,
        '--min', '0.0001']
    os.system('mri_binarize ' + ' '.join(shargs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- MKbrainmask"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--mc", required=True)
    parser.add_argument("--bbregister_dat", required=True)
    # parser.add_argument("--aparaseg", required=True)
    parser.add_argument("--bold_id", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    mc_file = subj_func_dir / os.path.basename(args.mc)
    bbregister_dat = subj_func_dir / os.path.basename(args.bbregister_dat)
    anat2bold_by_bbregister(subj_func_dir, mc_file, bbregister_dat, args.subjects_dir, args.subject_id)
