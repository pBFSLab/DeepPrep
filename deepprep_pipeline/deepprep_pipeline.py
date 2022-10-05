from pathlib import Path
from nipype import config, logging
from deepprep_structure_subwf import init_structure_part1_wf, init_structure_part2_wf, init_structure_part3_wf, \
    init_structure_part4_1_wf, init_structure_part4_2_wf, init_structure_part5_wf, init_structure_part6_wf, \
    init_structure_part7_wf, clear_is_running
from deepprep_bold_subwf import init_bold_part1_wf, init_bold_part2_wf, init_bold_part3_wf, init_bold_part4_wf, \
    init_bold_part5_wf, clear_subject_bold_tmp_dir
from interface.run import set_envrion


def pipeline():
    pwd = Path.cwd()

    # ############### Structure
    fastsurfer_home = pwd / "FastSurfer"
    freesurfer_home = Path('/usr/local/freesurfer720')
    fastcsr_home = pwd.parent / "deepprep_pipeline/FastCSR"
    featreg_home = pwd.parent / "deepprep_pipeline/FeatReg"

    # ############### BOLD
    mni152_brain_mask = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
    vxm_model_path = pwd / 'model' / 'voxelmorph'
    resource_dir = pwd / 'resource'
    atlas_type = 'MNI152_T1_2mm'
    task = 'rest'  # 'motor' or 'rest'
    preprocess_method = 'rest'  # 'task' or 'rest'

    # ############### Common
    # python_interpret = Path('/home/youjia/anaconda3/envs/3.8/bin/python3')
    python_interpret = Path('/home/anning/miniconda3/envs/3.8/bin/python3')

    data_path = Path("/mnt/ngshare2/UKB/BIDS")  # Raw Data BIDS dir
    subjects_dir = Path("/mnt/ngshare2/DeepPrep_UKB/UKB_Recon")  # Recon result dir
    derivative_deepprep_path = Path("/mnt/ngshare2/DeepPrep_UKB/UKB_BoldPreprocess")  # BOLD result dir
    workflow_cache_dir = Path("/mnt/ngshare2/DeepPrep_UKB/UKB_Workflow")  # workflow tmp cache dir

    subjects_dir.mkdir(parents=True, exist_ok=True)
    derivative_deepprep_path.mkdir(parents=True, exist_ok=True)
    workflow_cache_dir.mkdir(parents=True, exist_ok=True)

    layout = bids.BIDSLayout(str(data_path), derivatives=False)

    t1w_filess_all = list()
    subject_ids_all = list()
    for t1w_file in layout.get(return_type='filename', suffix="T1w"):
        sub_info = layout.parse_file_entities(t1w_file)
        subject_id = f"sub-{sub_info['subject']}"
        if 'session' in sub_info:
            subject_id = subject_id + f"-ses-{sub_info['session']}"
        t1w_filess_all.append([t1w_file])
        subject_ids_all.append(subject_id)

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    batch_size = 16

    for epoch in range(len(subject_ids_all) + 1):
        try:
            t1w_filess = t1w_filess_all[epoch * batch_size: (epoch + 1) * batch_size]
            subject_ids = subject_ids_all[epoch * batch_size: (epoch + 1) * batch_size]

            # 设置log目录位置
            log_dir = workflow_cache_dir / 'log' / f'batchsize_{batch_size:03d}_epoch_{epoch:03d}'
            log_dir.mkdir(parents=True, exist_ok=True)
            config.update_config({'logging': {'log_directory': log_dir,
                                              'log_to_file': True}})
            logging.update_logging(config)

            # ################################## STRUCTURE ###############################

            clear_is_running(subjects_dir=subjects_dir,
                             subject_ids=subject_ids)

            structure_part1_wf = init_structure_part1_wf(t1w_filess=t1w_filess,
                                                         subjects_dir=subjects_dir,
                                                         subject_ids=subject_ids)
            structure_part1_wf.base_dir = workflow_cache_dir
            structure_part1_wf.run('MultiProc', plugin_args={'n_procs': 30})

            structure_part2_wf = init_structure_part2_wf(subjects_dir=subjects_dir,
                                                         subject_ids=subject_ids,
                                                         python_interpret=python_interpret,
                                                         fastsurfer_home=fastsurfer_home)
            structure_part2_wf.base_dir = workflow_cache_dir
            structure_part2_wf.run('MultiProc', plugin_args={'n_procs': 2})

            structure_part3_wf = init_structure_part3_wf(subjects_dir=subjects_dir,
                                                         subject_ids=subject_ids,
                                                         python_interpret=python_interpret,
                                                         fastsurfer_home=fastsurfer_home,
                                                         freesurfer_home=freesurfer_home)
            structure_part3_wf.base_dir = workflow_cache_dir
            structure_part3_wf.run('MultiProc', plugin_args={'n_procs': 30})

            structure_part4_1_wf = init_structure_part4_1_wf(subjects_dir=subjects_dir,
                                                             subject_ids=subject_ids,
                                                             python_interpret=python_interpret,
                                                             fastcsr_home=fastcsr_home)
            structure_part4_1_wf.base_dir = workflow_cache_dir
            structure_part4_1_wf.run('MultiProc', plugin_args={'n_procs': 3})

            structure_part4_2_wf = init_structure_part4_2_wf(subjects_dir=subjects_dir,
                                                             subject_ids=subject_ids,
                                                             python_interpret=python_interpret,
                                                             fastcsr_home=fastcsr_home)
            structure_part4_2_wf.base_dir = workflow_cache_dir
            structure_part4_2_wf.run('MultiProc', plugin_args={'n_procs': 15})

            structure_part5_wf = init_structure_part5_wf(subjects_dir=subjects_dir,
                                                         subject_ids=subject_ids,
                                                         python_interpret=python_interpret,
                                                         fastsurfer_home=fastsurfer_home,
                                                         freesurfer_home=freesurfer_home
                                                         )
            structure_part5_wf.base_dir = workflow_cache_dir
            structure_part5_wf.run('MultiProc', plugin_args={'n_procs': 12})

            structure_part6_wf = init_structure_part6_wf(subjects_dir=subjects_dir,
                                                         subject_ids=subject_ids,
                                                         python_interpret=python_interpret,
                                                         freesurfer_home=freesurfer_home,
                                                         featreg_home=featreg_home)
            structure_part6_wf.base_dir = workflow_cache_dir
            structure_part6_wf.run('MultiProc', plugin_args={'n_procs': 3})
            #
            structure_part7_wf = init_structure_part7_wf(subjects_dir=subjects_dir,
                                                         subject_ids=subject_ids)
            structure_part7_wf.base_dir = workflow_cache_dir
            structure_part7_wf.run('MultiProc', plugin_args={'n_procs': 30})

            # ################################## BOLD ###############################

            bold_part1_wf = init_bold_part1_wf(subject_ids=subject_ids,
                                               data_path=data_path,
                                               vxm_model_path=vxm_model_path,
                                               atlas_type=atlas_type,
                                               subjects_dir=subjects_dir,
                                               derivative_deepprep_path=derivative_deepprep_path)
            bold_part1_wf.base_dir = workflow_cache_dir
            bold_part1_wf.run('MultiProc', plugin_args={'n_procs': 8})

            bold_part2_wf = init_bold_part2_wf(subject_ids=subject_ids,
                                               task=task,
                                               data_path=data_path,
                                               subjects_dir=subjects_dir,
                                               derivative_deepprep_path=derivative_deepprep_path)
            bold_part2_wf.base_dir = workflow_cache_dir
            bold_part2_wf.run('MultiProc', plugin_args={'n_procs': 8})

            if task == 'rest':
                bold_part3_wf = init_bold_part3_wf(subject_ids=subject_ids,
                                                   task=task,
                                                   data_path=data_path,
                                                   subjects_dir=subjects_dir,
                                                   derivative_deepprep_path=derivative_deepprep_path)
                bold_part3_wf.base_dir = workflow_cache_dir
                bold_part3_wf.run('MultiProc', plugin_args={'n_procs': 3})

            bold_part4_wf = init_bold_part4_wf(subject_ids=subject_ids,
                                               task=task,
                                               data_path=data_path,
                                               subjects_dir=subjects_dir,
                                               preprocess_method=preprocess_method,
                                               vxm_model_path=vxm_model_path,
                                               atlas_type=atlas_type,
                                               resource_dir=resource_dir,
                                               derivative_deepprep_path=derivative_deepprep_path)
            bold_part4_wf.base_dir = workflow_cache_dir
            bold_part4_wf.run('MultiProc', plugin_args={'n_procs': 4})

            bold_part5_wf = init_bold_part5_wf(subject_ids=subject_ids,
                                               task=task,
                                               data_path=data_path,
                                               preprocess_method=preprocess_method,
                                               mni152_brain_mask=mni152_brain_mask,
                                               derivative_deepprep_path=derivative_deepprep_path)
            bold_part5_wf.base_dir = workflow_cache_dir
            bold_part5_wf.run('MultiProc', plugin_args={'n_procs': 8})

            clear_subject_bold_tmp_dir(derivative_deepprep_path, subject_ids, task)
        except:
            pass
        # if epoch == 1:
        #     break
    # wf.write_graph(graph2use='flat', simple_form=False)


if __name__ == '__main__':
    import os
    import bids

    set_envrion()

    pipeline()
