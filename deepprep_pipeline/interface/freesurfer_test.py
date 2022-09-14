import os
from freesurfer import OrigAndRawavg, WhitePreaparc, WhitePialThickness1, WhitePialThickness2
from pathlib import Path
from freesurfer import Brainmask, InflatedSphere, Curvstats, Cortribbon, Parcstats, Pctsurfcon, Hyporelabel, JacobianAvgcurvCortparc, Segstats, Aseg7
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


def white_pial_thickness1_test():
    subject_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject = "sub-765"
    hemi = "lh"
    threads = 8

    os.environ['SUBJECTS_DIR'] = str(subject_dir)

    white_pial_thickness = Node(WhitePialThickness1(output_dir=subject_dir, threads=threads),
                                name="white_pial_thickness1")
    white_pial_thickness.inputs.subject = subject
    white_pial_thickness.inputs.hemi = hemi

    white_pial_thickness.run()

def white_pial_thickness2_test():
    subject_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject = "sub-765"
    hemi = "lh"
    threads = 8

    os.environ['SUBJECTS_DIR'] = str(subject_dir)

    autodet_gw_stats_hemi_dat = subject_dir / subject / f"surf/autodet.gw.stats.{hemi}.dat"
    aseg_presurf = subject_dir / subject / "mri/aseg.presurf.mgz"
    wm_file = subject_dir / subject / "mri/wm.mgz"
    brain_finalsurfs = subject_dir / subject / "mri/brain.finalsurfs.mgz"
    hemi_white_preaparc = subject_dir / subject / f"surf/{hemi}.white.preaparc"
    hemi_white = subject_dir / subject / f"surf/{hemi}.white"
    hemi_cortex_label = subject_dir / subject / f"label/{hemi}.cortex.label"
    hemi_aparc_DKTatlas_mapped_annot = subject_dir / subject / f"label/{hemi}.aparc.DKTatlas.mapped.annot"
    hemi_pial_t1 = subject_dir / subject / f"surf/{hemi}.pial.T1"

    white_pial_thickness = Node(WhitePialThickness1(output_dir=subject_dir, threads=threads),
                                name="white_pial_thickness2")
    white_pial_thickness.inputs.subject = subject
    white_pial_thickness.inputs.hemi = hemi
    white_pial_thickness.inputs.autodet_gw_stats_hemi_dat = autodet_gw_stats_hemi_dat
    white_pial_thickness.inputs.aseg_presurf = aseg_presurf
    white_pial_thickness.inputs.wm_file = wm_file
    white_pial_thickness.inputs.brain_finalsurfs = brain_finalsurfs
    white_pial_thickness.inputs.hemi_white_preaparc = hemi_white_preaparc
    white_pial_thickness.inputs.hemi_white = hemi_white
    white_pial_thickness.inputs.hemi_cortex_label = hemi_cortex_label
    white_pial_thickness.inputs.hemi_aparc_DKTatlas_mapped_annot = hemi_aparc_DKTatlas_mapped_annot
    white_pial_thickness.inputs.hemi_pial_t1 = hemi_pial_t1

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


def JacobianAvgcurvCortparc_test():
    subjects_dir = Path("/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon")
    subject_id = "sub-001"
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    for hemi in ['lh', 'rh']:
        os.environ['SUBJECTS_DIR'] = str(subjects_dir)
        white_preaparc_dir = subjects_dir / subject_id / "surf" / f"{hemi}.white.preaparc"
        sphere_reg_dir = subjects_dir / subject_id / "surf" / f"{hemi}.sphere.reg"
        jacobian_white_dir = subjects_dir / subject_id / "surf" / f"{hemi}.jacobian_white"
        avg_curv_dir = subjects_dir / subject_id / "surf" / f"{hemi}.avg_curv"
        aseg_presurf_dir = subjects_dir / subject_id / "mri" / "aseg.presurf.mgz"
        cortex_label_dir = subjects_dir / subject_id / "label" / f"{hemi}.cortex.label"
        aparc_annot_dir = subjects_dir / subject_id / "label" / f"{hemi}.aparc.annot"
        JacobianAvgcurvCortparc_node = Node(JacobianAvgcurvCortparc(), f'JacobianAvgcurvCortparc_node')
        JacobianAvgcurvCortparc_node.inputs.hemi = hemi
        JacobianAvgcurvCortparc_node.inputs.threads = 8
        JacobianAvgcurvCortparc_node.inputs.subject = subject_id
        JacobianAvgcurvCortparc_node.inputs.white_preaparc_file = white_preaparc_dir
        JacobianAvgcurvCortparc_node.inputs.sphere_reg_file = sphere_reg_dir
        JacobianAvgcurvCortparc_node.inputs.jacobian_white_file = jacobian_white_dir
        JacobianAvgcurvCortparc_node.inputs.avg_curv_file = avg_curv_dir
        JacobianAvgcurvCortparc_node.inputs.aseg_presurf_file = aseg_presurf_dir
        JacobianAvgcurvCortparc_node.inputs.cortex_label_file = cortex_label_dir
        JacobianAvgcurvCortparc_node.inputs.aparc_annot_file = aparc_annot_dir

        JacobianAvgcurvCortparc_node.run()
def Segstats_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_stats_dir = subjects_dir / subject_id / 'stats'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon'
    for hemi in ['lh', 'rh']:
        Segstats_node = Node(Segstats(), name='Segstats_node')
        Segstats_node.inputs.subjects_dir = subjects_dir
        Segstats_node.inputs.subject_id = subject_id
        Segstats_node.inputs.threads = threads
        Segstats_node.inputs.hemi = hemi
        Segstats_node.inputs.brainmask_file = subject_mri_dir / 'brainmask.mgz'
        Segstats_node.inputs.norm_file = subject_mri_dir / 'norm.mgz'
        Segstats_node.inputs.aseg_file = subject_mri_dir / 'aseg.mgz'
        Segstats_node.inputs.aseg_presurf_file = subject_mri_dir / 'aseg.presurf.mgz'
        Segstats_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
        Segstats_node.inputs.hemi_orig_nofix_file = subject_surf_dir / f'{hemi}.orig.premesh'
        Segstats_node.inputs.hemi_white_file = subject_surf_dir / f'{hemi}.white'
        Segstats_node.inputs.hemi_pial_file = subject_surf_dir / f'{hemi}.pial'

        Segstats_node.inputs.aseg_stats_file = subject_stats_dir / 'aseg.stats'
        Segstats_node.run()
def Aseg7_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    subject_id = 'sub-765'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir =  subjects_dir / subject_id / 'label'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon'
    Aseg7_node = Node(Aseg7(), name='Aseg7_node')
    Aseg7_node.inputs.subjects_dir = subjects_dir
    Aseg7_node.inputs.subject_id = subject_id
    Aseg7_node.inputs.threads = threads
    Aseg7_node.inputs.subject_mri_dir = subject_mri_dir
    Aseg7_node.inputs.aseg_presurf_hypos_file = subject_mri_dir / 'aseg.presurf.hypos.mgz'
    Aseg7_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
    Aseg7_node.inputs.lh_cortex_label_file = subject_label_dir / 'lh.cortex.label'
    Aseg7_node.inputs.lh_white_file = subject_surf_dir / 'lh.white'
    Aseg7_node.inputs.lh_pial_file = subject_surf_dir / 'lh.pial'
    Aseg7_node.inputs.lh_aparc_annot_file = subject_label_dir / 'lh.aparc.annot'
    Aseg7_node.inputs.rh_cortex_label_file = subject_label_dir / 'rh.cortex.label'
    Aseg7_node.inputs.rh_white_file = subject_surf_dir / 'rh.white'
    Aseg7_node.inputs.rh_pial_file = subject_surf_dir / 'rh.pial'
    Aseg7_node.inputs.rh_aparc_annot_file = subject_label_dir / 'rh.aparc.annot'

    Aseg7_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc+aseg.mgz'
    Aseg7_node.run()
if __name__ == '__main__':
    # OrigAndRawavg_test()
    # Brainmask_test()
    # Inflated_Sphere_test()
    # white_preaparc_test()

    set_envrion()

    # JacobianAvgcurvCortparc_test()
    Aseg7_test()
