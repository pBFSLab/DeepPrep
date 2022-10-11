from interface.create_node import *


# from interface.run import *

def set_envrion(threads: int = 1):
    # FreeSurfer recon-all env
    freesurfer_home = '/usr/local/freesurfer720'
    os.environ['FREESURFER_HOME'] = f'{freesurfer_home}'
    os.environ['FREESURFER'] = f'{freesurfer_home}'
    os.environ['SUBJECTS_DIR'] = f'{freesurfer_home}/subjects'
    os.environ['PATH'] = f'{freesurfer_home}/bin:/usr/local/freesurfer/mni/bin:/usr/local/freesurfer/tktools:' + \
                         f'{freesurfer_home}/fsfast/bin:' + os.environ['PATH']
    os.environ['MINC_BIN_DIR'] = f'{freesurfer_home}/mni/bin'
    os.environ['MINC_LIB_DIR'] = f'{freesurfer_home}/mni/lib'
    os.environ['PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
    os.environ['MNI_PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
    # FreeSurfer fsfast env
    os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

    # FastCSR
    os.environ[
        'LD_LIBRARY_PATH'] = '/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server:'

    # FSL
    os.environ['PATH'] = '/usr/local/fsl/bin:' + os.environ['PATH']

    # set threads
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads)

    # subjects dir
    os.environ['SUBJECTS_DIR'] = "/mnt/ngshare2/DeepPrep_UKB/UKB_Recon"
    os.environ['WORKFLOW_CACHED_DIR'] = "/mnt/ngshare2/DeepPrep_UKB/UKB_Workflow"
    os.environ['FASTSURFER_HOME'] = "/home/youjia/workspace/DeepPrep/deepprep_pipeline/FastSurfer"
    os.environ['FASTCSR_HOME'] = "/home/youjia/workspace/DeepPrep/deepprep_pipeline/FastCSR"
    os.environ['FEATREG_HOME'] = "/home/youjia/workspace/DeepPrep/deepprep_pipeline/FeatReg"


if __name__ == '__main__':
    set_envrion()

    subject_id = "sub-1002026-ses-02"
    t1w_files = ["/mnt/ngshare2/UKB/BIDS/sub-1002026/ses-02/anat/sub-1002026_ses-02_T1w.nii.gz"]

    ######### origandrawavg_node & segment_node ##########
    # node = create_origandrawavg_node(subject_id, t1w_files)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    ########## segment_node & noccseg_node ##########
    # node = create_Segment_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    ########## noccseg_node & N4BiasCorrect_node ##########
    # node = create_Noccseg_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    ########## N4BiasCorrect_node & TalairachAndNu_node ##########
    # node = create_N4BiasCorrect_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    #
    ########## TalairachAndNu_node & Brainmask_node ##########
    # node = create_TalairachAndNu_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    #
    ########## Brainmask_node & UpdateAseg_node ##########
    # node = create_Brainmask_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    ########## UpdateAseg_node & Filled_node ##########
    # node = create_UpdateAseg_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    #
    ########## Filled_node & FastCSR_node ##########
    # node = create_Filled_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    ######### FastCSR_node & WhitePreaparc1_node ##########
    # node = create_FastCSR_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    ######### WhitePreaparc1_node & SampleSegmentationToSurfave_node & InflatedSphere_node ##########
    # node = create_WhitePreaparc1_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # for n in sub_node:
    #     n.run()

    create_SampleSegmentationToSurfave_node(subject_id).run()
    exit()
    # ######### InflatedSphere_node & FeatReg_node ##########
    # node = create_InflatedSphere_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    # ######### FeatReg_node & JacobianAvgcurvCortparc_node ##########
    # node = create_FeatReg_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    #
    # ######### JacobianAvgcurvCortparc_node & WhitePialThickness1_node ##########
    # node = create_JacobianAvgcurvCortparc_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    #
    # ######### WhitePialThickness1_node & Curvstats_node & BalabelsMult_node & Cortribbon_node ##########
    # node = create_WhitePialThickness1_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # for n in sub_node:
    #     n.run()
    #
    # ######### Cortribbon_node & Parcstats_node ##########
    # node = create_Cortribbon_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    #
    # ######### Parcstats_node & Aseg7_node ##########
    # node = create_Parcstats_node(subject_id)
    # node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()