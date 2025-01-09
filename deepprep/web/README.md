```shell
/opt/DeepPrep/deepprep/deeprep.sh \
/mnt/ngshare/DeepPrep/test_sample/ds004192 \
/mnt/ngshare/DeepPrep/test_sample/ds004192_preprocess \
participant

/opt/DeepPrep/deepprep/web/pages/quickqc.sh \
/mnt/ngshare/DeepPrep/test_sample/ds004192 \
/mnt/ngshare/DeepPrep/test_sample/ds004192_quickqc \
participant

/opt/DeepPrep/deepprep/web/pages/postprocess.sh \
/mnt/ngshare/DeepPrep/test_sample/ds004192_cifti/BOLD \
/mnt/ngshare/DeepPrep/test_sample/ds004192_cifti_postprocess \
participant \
--task_id "rest motor" \
--subjects_id "01 02"  \
--space "fsaverage6 MNI152NLin6Asym" \
--confounds_index_file /mnt/ngshare2/anning/workspace/DeepPrep/deepprep/rest/denoise/12motion_6param_10eCompCor.txt \
--surface_fwhm 6 \
--volume_fwhm 6
```