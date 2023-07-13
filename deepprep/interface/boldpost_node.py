from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, File, TraitedSpec, Directory, Str
from deepprep.interface.run import multipool_run, multipool_BidsBolds, multipool_BidsBolds_2
import nibabel as nib
import numpy as np
from pathlib import Path
import bids
import os
import ants
import shutil
from app.filters.filters import bandpass_nifti


class RestGaussInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc="subject id", mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    data_path = Directory(exists=True, desc='data path', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class RestGaussOutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc="subject id")


class RestGauss(BaseInterface):
    input_spec = RestGaussInputSpec
    output_spec = RestGaussOutputSpec

    time = 11 / 60  # 运行时间：分钟
    cpu = 0  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(RestGauss, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task

        for run in runs:
            RestGauss_output_files = [f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(RestGauss_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def cmd(self, run):
        from app.filters.filters import gauss_nifti

        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        mc = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        fcmri_dir = Path(preprocess_dir) / self.inputs.subject_id / 'fcmri'
        Path(fcmri_dir).mkdir(parents=True, exist_ok=True)
        gauss_nifti(str(mc), 1000000000)

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        multipool_run(self.cmd, runs, Multi_Num=8)

        self.check_output(runs)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self, settings):
        from interface.create_node_boldpost import create_RestBandpass_node
        node = create_RestBandpass_node(self.inputs.subject_id,
                                        self.inputs.task,
                                        self.inputs.atlas_type,
                                        self.inputs.preprocess_method,
                                        settings)

        return node


class RestBandpassInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    data_path = Directory(exists=True, desc='data path', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class RestBandpassOutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc='subject')
    data_path = Directory(exists=True, desc='data path')


class RestBandpass(BaseInterface):
    input_spec = RestBandpassInputSpec
    output_spec = RestBandpassOutputSpec

    time = 120 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(RestBandpass, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task

        for run in runs:
            RestBandpass_output_files = [f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(RestBandpass_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def cmd(self, idx, bids_entities, bids_path):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        entities = dict(bids_entities)
        if 'RepetitionTime' in entities:
            TR = entities['RepetitionTime']
        else:
            bold = ants.image_read(bids_path)
            TR = bold.spacing[3]
        run = f'{idx + 1:03}'
        gauss_path = preprocess_dir / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz'
        bandpass_nifti(str(gauss_path), TR)

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj = self.inputs.subject_id.split('-')[1:]
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        all_idx = []
        all_bids_entities = []
        all_bids_path = []
        for idx, bids_bold in enumerate(bids_bolds):
            all_idx.append(idx)
            all_bids_entities.append(bids_bold.entities)
            all_bids_path.append(bids_bold.path)
        multipool_BidsBolds(self.cmd, all_idx, all_bids_entities, all_bids_path, Multi_Num=8)
        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])

        self.check_output(runs)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs

    def create_sub_node(self, settings):
        from interface.create_node_boldpost import create_RestRegression_node
        node = create_RestRegression_node(self.inputs.subject_id,
                                          self.inputs.task,
                                          self.inputs.atlas_type,
                                          self.inputs.preprocess_method,
                                          settings)

        return node


class RestRegressionInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class RestRegressionOutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc='subject')
    data_path = Directory(exists=True, desc='data_path')


class RestRegression(BaseInterface):
    input_spec = RestRegressionInputSpec
    output_spec = RestRegressionOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB
    def __init__(self):
        super(RestRegression, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task

        for run in runs:
            RestRegression_bold_output_files = [
                f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_snr.nii.gz',
                f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_sd1.nii.gz',
                f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz']
            bold_output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_bold_result = set(RestRegression_bold_output_files) <= set(bold_output_list)

            RestRegression_fcmri_output_files = [f"{sub}_bld{run}_mov_regressor.dat",
                                                 f"{sub}_bld{run}_pca_regressor_dt.dat",
                                                 f"{sub}_bld{run}_regressors.dat",
                                                 f"{sub}_bld{run}_ventricles_regressor_dt.dat",
                                                 f"{sub}_bld{run}_vent_wm_dt.dat",
                                                 f"{sub}_bld{run}_WB_regressor_dt.dat",
                                                 f"{sub}_bld{run}_wm_regressor_dt.dat"]
            fcmri_output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'fcmri')
            check_fcmri_result = set(RestRegression_fcmri_output_files) <= set(fcmri_output_list)

            RestRegression_movement_output_files = [f"{sub}_bld{run}_rest_reorient_skip_faln_mc.dat",
                                                    f"{sub}_bld{run}_rest_reorient_skip_faln_mc.ddat",
                                                    f"{sub}_bld{run}_rest_reorient_skip_faln_mc.par",
                                                    f"{sub}_bld{run}_rest_reorient_skip_faln_mc.rdat",
                                                    f"{sub}_bld{run}_rest_reorient_skip_faln_mc.rddat"]
            movement_output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'movement')
            check_movement_result = set(RestRegression_movement_output_files) <= set(movement_output_list)

        if not check_bold_result and not check_fcmri_result and not check_movement_result:
            return FileExistsError

    def setenv_smooth_downsampling(self):
        subjects_dir = Path(self.inputs.subjects_dir)
        fsaverage6_dir = subjects_dir / 'fsaverage6'
        if not fsaverage6_dir.exists():
            src_fsaverage6_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage6'
            try:
                os.symlink(src_fsaverage6_dir, fsaverage6_dir)
            except FileExistsError:
                pass

        fsaverage5_dir = subjects_dir / 'fsaverage5'
        if not fsaverage5_dir.exists():
            src_fsaverage5_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage5'
            try:
                os.symlink(src_fsaverage5_dir, fsaverage5_dir)
            except FileExistsError:
                pass

        fsaverage4_dir = subjects_dir / 'fsaverage4'
        if not fsaverage4_dir.exists():
            src_fsaverage4_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage4'
            try:
                os.symlink(src_fsaverage4_dir, fsaverage4_dir)
            except FileExistsError:
                pass

    # smooth_downsampling
    # def cmd(self, hemi, subj_surf_path, dst_resid_file, dst_reg_file):
    #     from deepprep.app.surface_projection import surface_projection as sp
    #     fs6_path = sp.indi_to_fs6(subj_surf_path, f'{self.inputs.subject_id}', dst_resid_file, dst_reg_file,
    #                               hemi)
    #     sm6_path = sp.smooth_fs6(fs6_path, hemi)
    #     sp.downsample_fs6_to_fs4(sm6_path, hemi)
    def _run_interface(self, runtime):
        from app.regressors.regressors import compile_regressors, regression
        from app.surface_projection import surface_projection as sp

        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'

        deepprep_subj_path = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        fcmri_dir = preprocess_dir / self.inputs.subject_id / 'fcmri'
        bold_dir = preprocess_dir / self.inputs.subject_id / 'bold'
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj = self.inputs.subject_id.split('-')[1]
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            run = f'{idx + 1:03}'
            bpss_path = f'{preprocess_dir}/{self.inputs.subject_id}/bold/{run}/{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz'

            all_regressors = compile_regressors(Path(preprocess_dir), Path(bold_dir), run, self.inputs.subject_id,
                                                Path(fcmri_dir), bpss_path)
            regression(bpss_path, all_regressors)

        self.setenv_smooth_downsampling()
        deepprep_subj_path = Path(deepprep_subj_path)

        subj_bold_dir = Path(preprocess_dir) / f'{self.inputs.subject_id}' / 'bold'
        for idx, bids_bold in enumerate(bids_bolds):
            run = f"{idx + 1:03}"
            entities = dict(bids_bold.entities)
            # subj = entities['subject']
            file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
            if 'session' in entities:
                subj_func_path = deepprep_subj_path / f"ses-{entities['session']}" / 'func'
                subj_surf_path = deepprep_subj_path / f"ses-{entities['session']}" / 'surf'
            else:
                subj_func_path = deepprep_subj_path / 'func'
                subj_surf_path = deepprep_subj_path / 'surf'
            subj_surf_path.mkdir(exist_ok=True)
            src_resid_file = subj_bold_dir / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
            dst_resid_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
            shutil.copy(src_resid_file, dst_resid_file)

            dst_reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
            # hemi = ['lh','rh']
            # multiregressionpool(self.cmd, hemi, subj_surf_path, dst_resid_file, dst_reg_file, Multi_Num=2)
            for hemi in ['lh', 'rh']:
                fs6_path = sp.indi_to_fs6(subj_surf_path, f'{self.inputs.subject_id}', dst_resid_file, dst_reg_file,
                                          hemi)
                sm6_path = sp.smooth_fs6(fs6_path, hemi)
                sp.downsample_fs6_to_fs4(sm6_path, hemi)

            runs = sorted(
                [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
            self.check_output(runs)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs

    def create_sub_node(self, settings):
        from interface.create_node_bold import create_VxmRegNormMNI152_node
        node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                            self.inputs.task,
                                            self.inputs.atlas_type,
                                            self.inputs.preprocess_method,
                                            settings)

        return node


class SmoothInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    MNI152_T1_2mm_brain_mask = File(exists=True, desc='MNI152 brain mask path', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class SmoothOutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc='subject')
    data_path = Directory(exists=True, desc='data_path')


class Smooth(BaseInterface):
    input_spec = SmoothInputSpec
    output_spec = SmoothOutputSpec

    time = 68 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(Smooth, self).__init__()

    def check_output(self, subj_func_path, file_prefix):
        sub = self.inputs.subject_id

        Smooth_output_files = [f'{file_prefix}_resid_MNI2mm_sm6.nii.gz']  # TODO MNI2mm 要不要优化
        output_list = os.listdir(subj_func_path)
        check_result = set(Smooth_output_files) <= set(output_list)
        if not check_result:
            return FileExistsError

    def save_bold(self, warped_img, temp_file, bold_file, save_file):
        ants.image_write(warped_img, str(temp_file))
        bold_info = nib.load(bold_file)
        affine_info = nib.load(temp_file)
        bold2 = nib.Nifti1Image(warped_img.numpy(), affine=affine_info.affine, header=bold_info.header)
        del bold_info
        del affine_info
        os.remove(temp_file)
        nib.save(bold2, save_file)

    def bold_smooth_6_ants(self, t12mm: str, t12mm_sm6_file: Path,
                           temp_file: Path, bold_file: Path, verbose=False):

        # mask file
        # MNI152_T1_2mm_brain_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
        brain_mask = ants.image_read(self.inputs.MNI152_T1_2mm_brain_mask)

        if isinstance(t12mm, str):
            bold_img = ants.image_read(t12mm)
        else:
            bold_img = t12mm

        bold_origin = bold_img.origin
        bold_spacing = bold_img.spacing
        bold_direction = bold_img.direction.copy()

        # smooth
        smoothed_img = ants.from_numpy(bold_img.numpy(), bold_origin[:3], bold_spacing[:3],
                                       bold_direction[:3, :3].copy(), has_components=True)
        # mask
        smoothed_np = ants.smooth_image(smoothed_img, sigma=6, FWHM=True).numpy()
        del smoothed_img
        mask_np = brain_mask.numpy()
        masked_np = np.zeros(smoothed_np.shape, dtype=np.float32)
        idx = mask_np == 1
        masked_np[idx, :] = smoothed_np[idx, :]
        del smoothed_np
        masked_img = ants.from_numpy(masked_np, bold_origin, bold_spacing, bold_direction)
        del masked_np
        if verbose:
            # save
            self.save_bold(masked_img, temp_file, bold_file, t12mm_sm6_file)
            # ants.image_write(masked_img, str(t12mm_sm6_file))
        return masked_img

    def cmd(self, bids_entities, bids_path):
        entities = dict(bids_entities)
        file_prefix = Path(bids_path).name.replace('.nii.gz', '')
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        derivative_deepprep_path = Path(self.inputs.derivative_deepprep_path)
        deepprep_subj_path = derivative_deepprep_path / self.inputs.subject_id
        if 'session' in entities:
            ses = entities['session']
            subj_func_path = Path(deepprep_subj_path) / f'ses-{ses}' / 'func'
        else:
            subj_func_path = Path(deepprep_subj_path) / 'func'
        if self.inputs.preprocess_method == 'rest':
            bold_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
            save_file = subj_func_path / f'{file_prefix}_resid_MIN2mm_sm6.nii.gz'
        else:
            bold_file = subj_func_path / f'{file_prefix}_mc.nii.gz'
            save_file = subj_func_path / f'{file_prefix}_mc_MIN2mm.nii.gz'
        if self.inputs.preprocess_method == 'rest':
            temp_file = Path(preprocess_dir) / f'{file_prefix}_MNI2mm_sm6_temp.nii.gz'
            warped_file = subj_func_path / f'{self.inputs.subject_id}_MNI2mm.nii.gz'
            warped_img = ants.image_read(str(warped_file))
            self.bold_smooth_6_ants(warped_img, save_file, temp_file, bold_file, verbose=True)
        else:
            temp_file = Path(preprocess_dir) / f'{file_prefix}_MNI2mm_temp.nii.gz'
            warped_file = subj_func_path / f'{self.inputs.subject_id}_MNI2mm.nii.gz'
            warped_img = ants.image_read(str(warped_file))
            self.save_bold(warped_img, temp_file, bold_file, save_file)

    def _run_interface(self, runtime):
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj = self.inputs.subject_id.split('-')[1]
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        bids_entities = []
        bids_path = []
        for bids_bold in bids_bolds:
            bids_entities.append(bids_bold.entities)
            bids_path.append(bids_bold.path)
        multipool_BidsBolds_2(self.cmd, bids_entities, bids_path, Multi_Num=8)

        derivative_deepprep_path = Path(self.inputs.derivative_deepprep_path)
        deepprep_subj_path = derivative_deepprep_path / self.inputs.subject_id
        for i in range(len(bids_entities)):
            entities = dict(bids_entities[i])
            file_prefix = Path(bids_path[i]).name.replace('.nii.gz', '')
            if 'session' in entities:
                ses = entities['session']
                subj_func_path = Path(deepprep_subj_path) / f'ses-{ses}' / 'func'
            else:
                subj_func_path = Path(deepprep_subj_path) / 'func'
            self.check_output(subj_func_path, file_prefix)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs
