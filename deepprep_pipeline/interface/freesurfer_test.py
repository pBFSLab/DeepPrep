import os
from freesurfer import OrigAndRawavg, WhitePreaparc, WhitePialThickness
from pathlib import Path
from freesurfer import Brainmask, InflatedSphere, Curvstats, Cortribbon, Parcstats, Pctsurfcon, Hyporelabel
from nipype import Node
from run import set_envrion


def OrigAndRawavg_test():
    set_envrion(1)
    subjects_dir = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon'
    os.environ['SUBJECTS_DIR'] = subjects_dir  # 设置FreeSurfer的subjects_dir

    subject_id = 'OrigAndRawavg_test1'
    t1w_files = [
        f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
    ]
    origandrawavg_node = Node(OrigAndRawavg(), f'origandrawavg_node')
    origandrawavg_node.inputs.t1w_files = t1w_files
    origandrawavg_node.inputs.subjects_dir = subjects_dir
    origandrawavg_node.inputs.subject_id = subject_id
    origandrawavg_node.inputs.threads = 1
    origandrawavg_node.run()

    subject_id = 'OrigAndRawavg_test2'
    t1w_files = [
        f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
        f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
    ]
    origandrawavg_node = Node(OrigAndRawavg(), f'origandrawavg_node')
    origandrawavg_node.inputs.t1w_files = t1w_files
    origandrawavg_node.inputs.subjects_dir = subjects_dir
    origandrawavg_node.inputs.subject_id = subject_id
    origandrawavg_node.inputs.threads = 1
    origandrawavg_node.run()


def Brainmask_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    brainmask_node = Node(Brainmask(), name='brainmask_node')
    brainmask_node.inputs.subjects_dir = subjects_dir
    brainmask_node.inputs.subject_id = subject_id
    brainmask_node.inputs.need_t1 = True
    brainmask_node.inputs.nu_file = subjects_dir / subject_id / 'mri' / 'nu.mgz'
    brainmask_node.inputs.mask_file = subjects_dir / subject_id / 'mri' / 'mask.mgz'
    brainmask_node.inputs.T1_file = subjects_dir / subject_id / 'mri' / 'T1.mgz'
    brainmask_node.inputs.brainmask_file = subjects_dir / subject_id / 'mri' / 'brainmask.mgz'
    brainmask_node.inputs.norm_file = subjects_dir / subject_id / 'mri' / 'norm.mgz'
    brainmask_node.run()


def white_preaparc_test():
    fswhitepreaparc = False
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject = "sub-001"
    hemi = "rh"
    threads = 8

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    white_preaparc = Node(WhitePreaparc(output_dir=subjects_dir, threads=threads), name="white_preaparc")
    white_preaparc.inputs.fswhitepreaparc = fswhitepreaparc
    white_preaparc.inputs.subject = subject
    white_preaparc.inputs.hemi = hemi

    white_preaparc.run()


def InflatedSphere_test():
    subjects_dir = Path("/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon")
    subject_id = "sub-001"
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    for hemi in ['lh', 'rh']:
        os.environ['SUBJECTS_DIR'] = str(subjects_dir)
        white_preaparc_dir = subjects_dir / subject_id / "surf" / f"{hemi}.white.preaparc"
        smoothwm_dir = subjects_dir / subject_id / "surf" / f"{hemi}.smoothwm"
        inflated_dir = subjects_dir / subject_id / "surf" / f"{hemi}.inflated"
        sulc_dir = subjects_dir / subject_id / "surf" / f"{hemi}.sulc"
        Inflated_Sphere_node = Node(InflatedSphere(), f'Inflated_Sphere_node')
        Inflated_Sphere_node.inputs.hemi = hemi
        Inflated_Sphere_node.inputs.threads = 8
        Inflated_Sphere_node.inputs.subject = subject_id
        Inflated_Sphere_node.inputs.white_preaparc_file = white_preaparc_dir
        Inflated_Sphere_node.inputs.smoothwm_file = smoothwm_dir
        Inflated_Sphere_node.inputs.inflated_file = inflated_dir
        Inflated_Sphere_node.inputs.sulc_file = sulc_dir

        Inflated_Sphere_node.run()


def Curvstats_test():
    set_envrion()
    subject_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_stats_dir = subject_dir / subject_id / 'stats'
    subject_surf_dir = subject_dir / subject_id / 'surf'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon'
    for hemi in ['lh', 'rh']:
        Curvstats_node = Node(Curvstats(), name='Curvstats_node')
        Curvstats_node.inputs.subject_dir = subject_dir
        Curvstats_node.inputs.subject_id = subject_id
        Curvstats_node.inputs.hemi = hemi
        Curvstats_node.inputs.hemi_smoothwm_file = subject_surf_dir / f'{hemi}.smoothwm'
        Curvstats_node.inputs.hemi_curv_file = subject_surf_dir / f'{hemi}.curv'
        Curvstats_node.inputs.hemi_sulc_file = subject_surf_dir / f'{hemi}.sulc'
        Curvstats_node.inputs.threads = threads

        Curvstats_node.inputs.hemi_curv_stats_file = subject_stats_dir / f'{hemi}.curv.stats'

        Curvstats_node.run()


def white_pial_thickness_test():
    fswhitepial = True
    subject_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject = "sub-765"
    hemi = "rh"
    threads = 8

    os.environ['SUBJECTS_DIR'] = str(subject_dir)

    white_pial_thickness = Node(WhitePialThickness(output_dir=subject_dir, threads=threads),
                                name="white_pial_thickness")
    white_pial_thickness.inputs.fswhitepial = fswhitepial
    white_pial_thickness.inputs.subject = subject
    white_pial_thickness.inputs.hemi = hemi

    white_pial_thickness.run()


def Cortribbon_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon'
    for hemi in ['lh', 'rh']:
        Cortribbon_node = Node(Cortribbon(), name='Cortribbon_node')
        Cortribbon_node.inputs.subjects_dir = subjects_dir
        Cortribbon_node.inputs.subject_id = subject_id
        Cortribbon_node.inputs.hemi = hemi
        Cortribbon_node.inputs.threads = threads
        Cortribbon_node.inputs.aseg_presurf_file = subject_mri_dir / 'aseg.presurf.mgz'
        Cortribbon_node.inputs.hemi_white = subject_surf_dir / f'{hemi}.white'
        Cortribbon_node.inputs.hemi_pial = subject_surf_dir / f'{hemi}.pial'

        Cortribbon_node.inputs.hemi_ribbon = subject_mri_dir / f'{hemi}.ribbon.mgz'
        Cortribbon_node.inputs.ribbon = subject_mri_dir / 'ribbon.mgz'
        Cortribbon_node.run()


def Pctsurfcon_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    subject_stats_dir = subjects_dir / subject_id / 'stats'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon'
    for hemi in ['lh', 'rh']:
        Parcstats_node = Node(Parcstats(), name='Parcstats_node')
        Parcstats_node.inputs.subjects_dir = subjects_dir
        Parcstats_node.inputs.subject_id = subject_id
        Parcstats_node.inputs.hemi = hemi
        Parcstats_node.inputs.threads = threads
        Parcstats_node.inputs.hemi_aparc_annot_file = subject_label_dir / f'{hemi}.aparc.annot'
        Parcstats_node.inputs.wm_file = subject_mri_dir / 'wm.mgz'
        Parcstats_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
        Parcstats_node.inputs.hemi_white_file = subject_surf_dir / f'{hemi}.white'
        Parcstats_node.inputs.hemi_pial_file = subject_surf_dir / f'{hemi}.pial'
        Parcstats_node.inputs.hemi_thickness_file = subject_surf_dir / f'{hemi}.thickness'
        Parcstats_node.inputs.hemi_aparc_stats_file = subject_stats_dir / f'{hemi}.aparc.stats'
        Parcstats_node.inputs.hemi_aparc_pial_stats_file = subject_stats_dir / f'{hemi}.aparc.pial.stats'
        Parcstats_node.inputs.aparc_annot_ctab_file = subject_label_dir / 'aparc.annot.ctab'
        Parcstats_node.run()

def Pctsurfcon_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    subject_stats_dir = subjects_dir / subject_id / 'stats'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon'
    for hemi in ['lh', 'rh']:
        Pctsurfcon_node = Node(Pctsurfcon(), name='Pctsurfcon_node')
        Pctsurfcon_node.inputs.subjects_dir = subjects_dir
        Pctsurfcon_node.inputs.subject_id = subject_id
        Pctsurfcon_node.inputs.hemi = hemi
        Pctsurfcon_node.inputs.threads = threads
        Pctsurfcon_node.inputs.rawavg_file = subject_mri_dir / 'rawavg.mgz'
        Pctsurfcon_node.inputs.orig_file = subject_mri_dir / 'orig.mgz'
        Pctsurfcon_node.inputs.hemi_cortex_label_file = subject_label_dir / f'{hemi}.cortex.label'
        Pctsurfcon_node.inputs.hemi_white_file = subject_surf_dir / f'{hemi}.white'
        Pctsurfcon_node.inputs.hemi_wg_pct_mgh_file = subject_surf_dir / f'{hemi}.w-g.pct.mgh'
        Pctsurfcon_node.inputs.hemi_wg_pct_stats_file = subject_stats_dir / f'{hemi}.w-g.pct.stats'
        Pctsurfcon_node.run()

def Hyporelabel_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon'
    for hemi in ['lh', 'rh']:
        Hyporelabel_node = Node(Hyporelabel(), name='Hyporelabel_node')
        Hyporelabel_node.inputs.subjects_dir = subjects_dir
        Hyporelabel_node.inputs.subject_id = subject_id
        Hyporelabel_node.inputs.hemi = hemi
        Hyporelabel_node.inputs.threads = threads
        Hyporelabel_node.inputs.aseg_presurf_file = subject_mri_dir / 'aseg.presurf.mgz'
        Hyporelabel_node.inputs.hemi_white_file = subject_surf_dir / f'{hemi}.white'
        Hyporelabel_node.inputs.aseg_presurf_hypos_file = subject_mri_dir / 'aseg.presurf.hypos.mgz'
        Hyporelabel_node.run()
if __name__ == '__main__':
    # OrigAndRawavg_test()
    # Brainmask_test()
    # Inflated_Sphere_test()
    # white_preaparc_test()

    set_envrion()
    white_pial_thickness_test()
