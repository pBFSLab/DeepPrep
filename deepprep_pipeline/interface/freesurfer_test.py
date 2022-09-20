import os
from freesurfer import OrigAndRawavg, WhitePreaparc1, WhitePialThickness1, WhitePialThickness2, Aseg7, Aseg7ToAseg
from pathlib import Path
from freesurfer import Brainmask, Filled, InflatedSphere, Curvstats, Cortribbon, Parcstats, Pctsurfcon, Hyporelabel, \
    JacobianAvgcurvCortparc, Segstats, BalabelsMult

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


def filled_test():
    set_envrion()
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-001"
    threads = 8

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    filled_node = Node(Filled(), name='filled_node')
    filled_node.inputs.subjects_dir = subjects_dir
    filled_node.inputs.subject_id = subject_id
    filled_node.inputs.threads = threads

    filled_node.inputs.aseg_auto_file = subjects_dir / subject_id / 'mri/aseg.auto.mgz'
    filled_node.inputs.norm_file = subjects_dir / subject_id / 'mri/norm.mgz'
    filled_node.inputs.brainmask_file = subjects_dir / subject_id / 'mri/brainmask.mgz'
    filled_node.inputs.talairach_file = subjects_dir / subject_id / 'mri/transforms/talairach.lta'

    filled_node.run()



def white_preaparc1_test():
    fswhitepreaparc = False
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-002"
    threads = 8

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    white_preaparc1 = Node(WhitePreaparc1(), name="white_preaparc1_node")
    white_preaparc1.inputs.subjects_dir = subjects_dir
    white_preaparc1.inputs.subject_id = subject_id
    white_preaparc1.inputs.threads = threads

    white_preaparc1.inputs.aseg_presurf = subjects_dir / subject_id / 'mri' / 'aseg.presurf.mgz'
    white_preaparc1.inputs.brain_finalsurfs = subjects_dir / subject_id / 'mri' / 'brain.finalsurfs.mgz'
    white_preaparc1.inputs.wm_file = subjects_dir / subject_id / 'mri' / 'wm.mgz'
    white_preaparc1.inputs.filled_file = subjects_dir / subject_id / 'mri' / 'filled.mgz'
    white_preaparc1.inputs.lh_orig = subjects_dir / subject_id / 'surf' / 'lh.orig'
    white_preaparc1.inputs.rh_orig = subjects_dir / subject_id / 'surf' / 'rh.orig'

    white_preaparc1.run()


def InflatedSphere_test():
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-002"
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    lh_white_preaparc_file = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    rh_white_preaparc_file = subjects_dir / subject_id / "surf" / "rh.white.preaparc"

    Inflated_Sphere_node = Node(InflatedSphere(), f'Inflated_Sphere_node')
    Inflated_Sphere_node.inputs.threads = 8
    Inflated_Sphere_node.inputs.subjects_dir = subjects_dir
    Inflated_Sphere_node.inputs.subject_id = subject_id
    Inflated_Sphere_node.inputs.lh_white_preaparc_file = lh_white_preaparc_file
    Inflated_Sphere_node.inputs.rh_white_preaparc_file = rh_white_preaparc_file

    Inflated_Sphere_node.run()


def Curvstats_test():
    set_envrion()
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-002"
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    subject_stats_dir = subjects_dir / subject_id / 'stats'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    threads = 8

    Curvstats_node = Node(Curvstats(), name='Curvstats_node')
    Curvstats_node.inputs.subjects_dir = subjects_dir
    Curvstats_node.inputs.subject_id = subject_id

    Curvstats_node.inputs.lh_smoothwm = subject_surf_dir / f'lh.smoothwm'
    Curvstats_node.inputs.rh_smoothwm = subject_surf_dir / f'rh.smoothwm'
    Curvstats_node.inputs.lh_curv = subject_surf_dir / f'lh.curv'
    Curvstats_node.inputs.rh_curv = subject_surf_dir / f'rh.curv'
    Curvstats_node.inputs.lh_sulc = subject_surf_dir / f'lh.sulc'
    Curvstats_node.inputs.rh_sulc = subject_surf_dir / f'rh.sulc'
    Curvstats_node.inputs.threads = threads

    Curvstats_node.run()


def white_pial_thickness1_test():
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-002"
    threads = 8

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    white_pial_thickness1 = Node(WhitePialThickness1(), name="white_pial_thickness1")
    white_pial_thickness1.inputs.subjects_dir = subjects_dir
    white_pial_thickness1.inputs.subject_id = subject_id
    white_pial_thickness1.inputs.threads = threads

    white_pial_thickness1.inputs.lh_white_preaparc = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    white_pial_thickness1.inputs.rh_white_preaparc = subjects_dir / subject_id / "surf" / "rh.white.preaparc"
    white_pial_thickness1.inputs.aseg_presurf = subjects_dir / subject_id / "mri" / "aseg.presurf.mgz"
    white_pial_thickness1.inputs.brain_finalsurfs = subjects_dir / subject_id / "mri" / "brain.finalsurfs.mgz"
    white_pial_thickness1.inputs.wm_file = subjects_dir / subject_id / "mri" / "wm.mgz"
    white_pial_thickness1.inputs.lh_aparc_annot = subjects_dir / subject_id / "label" / "lh.aparc.annot"
    white_pial_thickness1.inputs.rh_aparc_annot = subjects_dir / subject_id / "label" / "rh.aparc.annot"
    white_pial_thickness1.inputs.lh_cortex_hipamyg_label = subjects_dir / subject_id / "label" / "lh.cortex+hipamyg.label"
    white_pial_thickness1.inputs.rh_cortex_hipamyg_label = subjects_dir / subject_id / "label" / "rh.cortex+hipamyg.label"
    white_pial_thickness1.inputs.lh_cortex_label = subjects_dir / subject_id / "label" / "lh.cortex.label"
    white_pial_thickness1.inputs.rh_cortex_label = subjects_dir / subject_id / "label" / "rh.cortex.label"

    white_pial_thickness1.inputs.lh_aparc_DKTatlas_mapped_annot = subjects_dir / subject_id / "label" / "lh.aparc.DKTatlas.mapped.annot"
    white_pial_thickness1.inputs.rh_aparc_DKTatlas_mapped_annot = subjects_dir / subject_id / "label" / "rh.aparc.DKTatlas.mapped.annot"
    white_pial_thickness1.inputs.lh_white = subjects_dir / subject_id / "surf" / "lh.white"
    white_pial_thickness1.inputs.rh_white = subjects_dir / subject_id / "surf" / "rh.white"

    white_pial_thickness1.run()


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
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-002"
    threads = 8

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'

    Cortribbon_node = Node(Cortribbon(), name='Cortribbon_node')
    Cortribbon_node.inputs.subjects_dir = subjects_dir
    Cortribbon_node.inputs.subject_id = subject_id
    Cortribbon_node.inputs.threads = threads

    Cortribbon_node.inputs.aseg_presurf_file = subject_mri_dir / 'aseg.presurf.mgz'
    Cortribbon_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Cortribbon_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Cortribbon_node.inputs.lh_pial = subject_surf_dir / f'lh.pial'
    Cortribbon_node.inputs.rh_pial = subject_surf_dir / f'rh.pial'

    Cortribbon_node.inputs.lh_ribbon = subject_mri_dir / f'lh.ribbon.mgz'
    Cortribbon_node.inputs.rh_ribbon = subject_mri_dir / f'rh.ribbon.mgz'
    Cortribbon_node.inputs.ribbon = subject_mri_dir / 'ribbon.mgz'
    Cortribbon_node.run()


def Parcstats_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_id = 'sub-170'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    subject_stats_dir = subjects_dir / subject_id / 'stats'
    threads = 8
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    Parcstats_node = Node(Parcstats(), name='Parcstats_node')
    Parcstats_node.inputs.subjects_dir = subjects_dir
    Parcstats_node.inputs.subject_id = subject_id
    Parcstats_node.inputs.threads = threads

    Parcstats_node.inputs.lh_aparc_annot = subject_label_dir / f'lh.aparc.annot'
    Parcstats_node.inputs.rh_aparc_annot = subject_label_dir / f'rh.aparc.annot'
    Parcstats_node.inputs.wm_file = subject_mri_dir / 'wm.mgz'
    Parcstats_node.inputs.aseg_file = subject_mri_dir / 'aseg.mgz'
    Parcstats_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
    Parcstats_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Parcstats_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Parcstats_node.inputs.lh_pial = subject_surf_dir / f'lh.pial'
    Parcstats_node.inputs.rh_pial = subject_surf_dir / f'rh.pial'
    Parcstats_node.inputs.lh_thickness = subject_surf_dir / f'lh.thickness'
    Parcstats_node.inputs.rh_thickness = subject_surf_dir / f'rh.thickness'

    Parcstats_node.inputs.lh_aparc_stats = subject_stats_dir / f'lh.aparc.stats'
    Parcstats_node.inputs.rh_aparc_stats = subject_stats_dir / f'rh.aparc.stats'
    Parcstats_node.inputs.lh_aparc_pial_stats = subject_stats_dir / f'lh.aparc.pial.stats'
    Parcstats_node.inputs.rh_aparc_pial_stats = subject_stats_dir / f'rh.aparc.pial.stats'
    Parcstats_node.inputs.aparc_annot_ctab = subject_label_dir / 'aparc.annot.ctab'
    Parcstats_node.run()


def Pctsurfcon_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_id = 'sub-170'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    subject_stats_dir = subjects_dir / subject_id / 'stats'
    threads = 8
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    Pctsurfcon_node = Node(Pctsurfcon(), name='Pctsurfcon_node')
    Pctsurfcon_node.inputs.subjects_dir = subjects_dir
    Pctsurfcon_node.inputs.subject_id = subject_id
    Pctsurfcon_node.inputs.threads = threads

    Pctsurfcon_node.inputs.rawavg_file = subject_mri_dir / 'rawavg.mgz'
    Pctsurfcon_node.inputs.orig_file = subject_mri_dir / 'orig.mgz'
    Pctsurfcon_node.inputs.lh_cortex_label = subject_label_dir / f'lh.cortex.label'
    Pctsurfcon_node.inputs.rh_cortex_label = subject_label_dir / f'rh.cortex.label'
    Pctsurfcon_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Pctsurfcon_node.inputs.rh_white = subject_surf_dir / f'rh.white'

    Pctsurfcon_node.inputs.lh_wg_pct_mgh = subject_surf_dir / f'lh.w-g.pct.mgh'
    Pctsurfcon_node.inputs.rh_wg_pct_mgh = subject_surf_dir / f'rh.w-g.pct.mgh'
    Pctsurfcon_node.inputs.lh_wg_pct_stats = subject_stats_dir / f'lh.w-g.pct.stats'
    Pctsurfcon_node.inputs.rh_wg_pct_stats = subject_stats_dir / f'rh.w-g.pct.stats'
    Pctsurfcon_node.run()


def Hyporelabel_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_id = 'sub-170'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    threads = 8
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    Hyporelabel_node = Node(Hyporelabel(), name='Hyporelabel_node')
    Hyporelabel_node.inputs.subjects_dir = subjects_dir
    Hyporelabel_node.inputs.subject_id = subject_id
    Hyporelabel_node.inputs.threads = threads
    Hyporelabel_node.inputs.aseg_presurf = subject_mri_dir / 'aseg.presurf.mgz'
    Hyporelabel_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Hyporelabel_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Hyporelabel_node.inputs.aseg_presurf_hypos = subject_mri_dir / 'aseg.presurf.hypos.mgz'
    Hyporelabel_node.run()


def JacobianAvgcurvCortparc_test():
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-002"
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    lh_white_preaparc = subjects_dir / subject_id / "surf" / f"lh.white.preaparc"
    rh_white_preaparc = subjects_dir / subject_id / "surf" / f"rh.white.preaparc"
    lh_sphere_reg = subjects_dir / subject_id / "surf" / f"lh.sphere.reg"
    rh_sphere_reg = subjects_dir / subject_id / "surf" / f"rh.sphere.reg"
    lh_jacobian_white = subjects_dir / subject_id / "surf" / f"lh.jacobian_white"
    rh_jacobian_white = subjects_dir / subject_id / "surf" / f"rh.jacobian_white"
    lh_avg_curv = subjects_dir / subject_id / "surf" / f"lh.avg_curv"
    rh_avg_curv = subjects_dir / subject_id / "surf" / f"rh.avg_curv"
    aseg_presurf_dir = subjects_dir / subject_id / "mri" / "aseg.presurf.mgz"
    lh_cortex_label = subjects_dir / subject_id / "label" / f"lh.cortex.label"
    rh_cortex_label = subjects_dir / subject_id / "label" / f"rh.cortex.label"
    lh_aparc_annot = subjects_dir / subject_id / "label" / f"lh.aparc.annot"
    rh_aparc_annot = subjects_dir / subject_id / "label" / f"rh.aparc.annot"

    JacobianAvgcurvCortparc_node = Node(JacobianAvgcurvCortparc(), f'JacobianAvgcurvCortparc_node')
    JacobianAvgcurvCortparc_node.inputs.subjects_dir = subjects_dir
    JacobianAvgcurvCortparc_node.inputs.subject_id = subject_id
    JacobianAvgcurvCortparc_node.inputs.lh_white_preaparc = lh_white_preaparc
    JacobianAvgcurvCortparc_node.inputs.rh_white_preaparc = rh_white_preaparc
    JacobianAvgcurvCortparc_node.inputs.lh_sphere_reg = lh_sphere_reg
    JacobianAvgcurvCortparc_node.inputs.rh_sphere_reg = rh_sphere_reg
    JacobianAvgcurvCortparc_node.inputs.lh_jacobian_white = lh_jacobian_white
    JacobianAvgcurvCortparc_node.inputs.rh_jacobian_white = rh_jacobian_white
    JacobianAvgcurvCortparc_node.inputs.lh_avg_curv = lh_avg_curv
    JacobianAvgcurvCortparc_node.inputs.rh_avg_curv = rh_avg_curv
    JacobianAvgcurvCortparc_node.inputs.aseg_presurf_file = aseg_presurf_dir
    JacobianAvgcurvCortparc_node.inputs.lh_cortex_label = lh_cortex_label
    JacobianAvgcurvCortparc_node.inputs.rh_cortex_label = rh_cortex_label

    JacobianAvgcurvCortparc_node.inputs.lh_aparc_annot = lh_aparc_annot
    JacobianAvgcurvCortparc_node.inputs.rh_aparc_annot = rh_aparc_annot
    JacobianAvgcurvCortparc_node.inputs.threads = 8


    JacobianAvgcurvCortparc_node.run()


def Segstats_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-170"
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_stats_dir = subjects_dir / subject_id / 'stats'
    threads = 8
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    Segstats_node = Node(Segstats(), name='Segstats_node')
    Segstats_node.inputs.subjects_dir = subjects_dir
    Segstats_node.inputs.subject_id = subject_id
    Segstats_node.inputs.threads = threads

    Segstats_node.inputs.brainmask_file = subject_mri_dir / 'brainmask.mgz'
    Segstats_node.inputs.norm_file = subject_mri_dir / 'norm.mgz'
    Segstats_node.inputs.aseg_file = subject_mri_dir / 'aseg.mgz'
    Segstats_node.inputs.aseg_presurf = subject_mri_dir / 'aseg.presurf.mgz'
    Segstats_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
    Segstats_node.inputs.lh_orig_premesh = subject_surf_dir / f'lh.orig.premesh'
    Segstats_node.inputs.rh_orig_premesh = subject_surf_dir / f'rh.orig.premesh'
    Segstats_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Segstats_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Segstats_node.inputs.lh_pial = subject_surf_dir / f'lh.pial'
    Segstats_node.inputs.rh_pial = subject_surf_dir / f'rh.pial'

    Segstats_node.inputs.aseg_stats = subject_stats_dir / 'aseg.stats'
    Segstats_node.run()


def Aseg7_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    subject_id = 'sub-765'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon'
    Aseg7_node = Node(Aseg7(), name='Aseg7_node')
    Aseg7_node.inputs.subjects_dir = subjects_dir
    Aseg7_node.inputs.subject_id = subject_id
    Aseg7_node.inputs.threads = threads
    Aseg7_node.inputs.subject_mri_dir = subject_mri_dir
    Aseg7_node.inputs.aseg_presurf_hypos_file = subject_mri_dir / 'aseg.presurf.hypos.mgz'
    # Aseg7_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
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


def Aseg7ToAseg_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    subject_id = 'sub-765'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    threads = 8
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon'
    Aseg7ToAseg_node = Node(Aseg7ToAseg(), name='Aseg7_node')
    Aseg7ToAseg_node.inputs.subjects_dir = subjects_dir
    Aseg7ToAseg_node.inputs.subject_id = subject_id
    Aseg7ToAseg_node.inputs.threads = threads
    # Aseg7ToAseg_node.inputs.aseg_presurf_hypos_file = subject_mri_dir / 'aseg.presurf.hypos.mgz'
    # Aseg7ToAseg_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
    Aseg7ToAseg_node.inputs.lh_cortex_label_file = subject_label_dir / 'lh.cortex.label'
    Aseg7ToAseg_node.inputs.lh_white_file = subject_surf_dir / 'lh.white'
    Aseg7ToAseg_node.inputs.lh_pial_file = subject_surf_dir / 'lh.pial'
    Aseg7ToAseg_node.inputs.rh_cortex_label_file = subject_label_dir / 'rh.cortex.label'
    Aseg7ToAseg_node.inputs.rh_white_file = subject_surf_dir / 'rh.white'
    Aseg7ToAseg_node.inputs.rh_pial_file = subject_surf_dir / 'rh.pial'

    Aseg7ToAseg_node.inputs.aseg_file = subject_mri_dir / 'aseg.mgz'
    Aseg7ToAseg_node.run()


def BalabelsMult_test():
    set_envrion()
    subjects_dir = Path(f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    subject_stats_dir = subjects_dir / subject_id / 'stats'
    threads = 10
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon'

    for hemi in ['lh', 'rh']:
        BalabelsMult_node = Node(BalabelsMult(), name='BalabelsMult_node')
        BalabelsMult_node.inputs.subjects_dir = subjects_dir
        BalabelsMult_node.inputs.subject_id = subject_id
        BalabelsMult_node.inputs.hemi = hemi
        BalabelsMult_node.inputs.threads = threads
        BalabelsMult_node.inputs.sub_label_dir = subject_label_dir
        BalabelsMult_node.inputs.sub_stats_dir = subject_stats_dir
        BalabelsMult_node.inputs.freesurfer_dir = os.environ['FREESURFER']
        BalabelsMult_node.inputs.hemi_sphere_file = subject_surf_dir / f'{hemi}.sphere.reg'
        BalabelsMult_node.inputs.fsaverage_label_dir = subjects_dir / 'fsaverage' / 'label'
        BalabelsMult_node.inputs.hemi_BA45_exvivo_file = subject_label_dir / f'{hemi}.BA45_exvivo.label'
        BalabelsMult_node.inputs.BA_exvivo_thresh_file = subject_label_dir / 'BA_exvivo.thresh.ctab'
        BalabelsMult_node.inputs.hemi_perirhinal_exvivo_file = subject_label_dir / f'{hemi}.perirhinal_exvivo.label'
        BalabelsMult_node.inputs.hemi_entorhinal_exvivo_file = subject_label_dir / f'{hemi}.entorhinal_exvivo.label'
        BalabelsMult_node.run()


if __name__ == '__main__':
    set_envrion()

    # OrigAndRawavg_test()

    # Brainmask_test()

    # filled_test()

    # InflatedSphere_test()

    # white_preaparc1_test()


    # JacobianAvgcurvCortparc_test()

    # white_pial_thickness1_test()

    # Curvstats_test()

    # Cortribbon_test()

    # Parcstats_test()

    # Pctsurfcon_test()

    # Hyporelabel_test()

    Segstats_test()

    # Aseg7_test()
    # Aseg7ToAseg_test()