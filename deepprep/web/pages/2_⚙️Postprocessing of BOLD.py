#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import streamlit as st
import subprocess
import os

st.markdown(f'# ⚙️Postprocessing of BOLD')
st.write(
    """
At present, this program is designed to process resting-state functional magnetic resonance imaging (rs-fMRI) data. The processed data can be utilized for calculating functional connectivity (FC) maps, individualized brain functional parcellation, or other relevant analyses.

For task-based functional magnetic resonance imaging (task-fMRI) data, it is recommended to employ alternative tools for subsequent analysis, such as Statistical Parametric Mapping (SPM).

#### processing steps
Surface space: bandpass -> regression -> smooth (optional)

Volume space:  bandpass -> regression -> smooth (optional)

-----------------
"""
)

deepprep_cmd = ''
commond_error = False

bids_dir = st.text_input("Preprocessing Result Path:", help="refers to the directory containing the preprocessed BOLD results by DeepPrep, which should be in BIDS format.")
if not bids_dir:
    st.error("The Preprocessing Result Path must be input!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not bids_dir.startswith('/'):
    st.error("The path must be an absolute path that starts with '/'.")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not os.path.exists(bids_dir):
    st.error("The Preprocessing Result Path does not exist!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {bids_dir}'

output_dir = st.text_input("Output Path:", help="refers to the output directory for the postprocessed BOLD data.")
if not output_dir:
    st.error("The Output Path must be input!")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
elif not output_dir.startswith('/'):
    st.error("The path must be an absolute path that starts with '/'.")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {output_dir}'
deepprep_cmd += f' participant'

freesurfer_license_file = st.text_input("FreeSurfer license file path", value='/opt/freesurfer/license.txt', help="FreeSurfer license file path. It is highly recommended to replace the license.txt path with your own FreeSurfer license! You can get it for free from https://surfer.nmr.mgh.harvard.edu/registration.html")
if not freesurfer_license_file.startswith('/'):
    st.error("The path must be an absolute path that starts with '/'.")
    commond_error = True
elif not os.path.exists(freesurfer_license_file):
    st.error("The FreeSurfer license file path does not exist!")
    commond_error = True
else:
    deepprep_cmd += f' --fs_license_file {freesurfer_license_file}'

confounds_file = st.text_input("Confounds File Path", value='/opt/DeepPrep/deepprep/rest/denoise/12motion_6param_10bCompCor.txt', help="The path to the text file that contains all the confound names needed for regression.")
if not confounds_file.startswith('/'):
    st.error("The path must be an absolute path that starts with '/'.")
    commond_error = True
elif not os.path.exists(confounds_file):
    st.error("The Confounds File Path does not exist!")
    commond_error = True
else:
    deepprep_cmd += f' --confounds_index_file {confounds_file}'

bold_task_type = st.text_input("BOLD task type", placeholder="i.e. rest, motor, 'rest motor'", help="The task label of BOLD images. If there are multiple tasks, separate them with spaces.")
if bold_task_type:
    if 'task-' in bold_task_type:
        bold_task_type = bold_task_type.replace('task-', '')
    bold_task_type = bold_task_type.replace("'", "")
    bold_task_type = bold_task_type.replace('"', "")
    deepprep_cmd += f" --task_id '{bold_task_type}'"
else:
    st.error("The BOLD task type must be input!")
    commond_error = True

spaces = st.multiselect("required output spaces (optional)", ["T1w", "MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsnative", "fsaverage6", "fsaverage5", "fsaverage4", "fsaverage3"], help="select the output spaces you need")
if spaces:
    spaces = ' '.join(spaces)
    deepprep_cmd += f" --space '{spaces}'"

if "fsnative" in spaces:
    subjects_dir = st.text_input("Recon Result Path",
                                 help=" the directory containing the recon files.")
    if not subjects_dir:
        st.error("The Recon Result Path must be input!")
        commond_error = True
    elif not subjects_dir.startswith('/'):
        st.error("The path must be an absolute path that starts with '/'.")
        commond_error = True
    elif not os.path.exists(subjects_dir):
        st.error("The Recon Result Path does not exist!")
        commond_error = True
    else:
        deepprep_cmd += f' --subjects_dir {subjects_dir}'

bold_skip_frame = st.text_input("skip n frames of BOLD data", value="2", help="skip n frames of BOLD fMRI; the default is `2`.")
if not bold_skip_frame:
    deepprep_cmd += f' --skip_frame 2'
else:
    deepprep_cmd += f' --skip_frame {bold_skip_frame}'

bold_fwhm = st.text_input("fwhm", value="6", help="smooth by fwhm mm; the default is `6`.")
if not bold_skip_frame:
    deepprep_cmd += f' --surface_fwhm 6 --volume_fwhm 6'
else:
    deepprep_cmd += f' --surface_fwhm {bold_fwhm} --volume_fwhm {bold_fwhm}'

bold_bandpass = st.text_input("bandpass filter", value="0.01-0.08", help="the default range is `0.01-0.08`.")
if not bold_bandpass:
    deepprep_cmd += f' --bandpass 0.01-0.08'
else:
    assert len(bold_bandpass.split('-')) == 2
    deepprep_cmd += f' --bandpass {bold_bandpass}'

participant_label = st.text_input("the subject IDs (optional)",
                                  help="Identify the subjects you'd like to process by their IDs, i.e. '001 002'.")
if participant_label:
    if 'sub-' in participant_label:
        participant_label = participant_label.replace('sub-', '')
    participant_label = participant_label.replace("'", "")
    participant_label = participant_label.replace('"', "")
    deepprep_cmd += f" --subject_id '{participant_label}'"

col1, col2, col3 = st.columns(3)

with col1:
    skip_bids_validation = st.checkbox("skip_bids_validation", value=True, help="with this flag, the BIDS format validation step of the input dataset will be skipped.")
    if skip_bids_validation:
        deepprep_cmd += ' --skip_bids_validation'
with col2:
    ignore_error = st.checkbox("ignore_error", help="ignores the errors that occurred during processing.")
    if ignore_error:
        deepprep_cmd += ' --ignore_error'
with col3:
    resume = st.checkbox("resume", value=True, help="allows the DeepPrep pipeline to start from the last exit point.")
    if resume:
        deepprep_cmd += ' --resume'


def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )

    while True:
        output = process.stdout.readline()
        print(output)
        if output == "" and process.poll() is not None:
            break
        if output:
            yield output + '\n'

    process.wait()


st.write(f'-----------  ------------')
st.write(f'postprocess {deepprep_cmd}')
if st.button("Run", disabled=commond_error):
    with st.spinner('Waiting for the process to finish, please do not leave this page...'):
        command = [f"/opt/DeepPrep/deepprep/web/pages/postprocess.sh {deepprep_cmd}"]
        with st.expander("------------ running log ------------"):
            st.write_stream(run_command(command))
        import time
        time.sleep(2)
    st.success("Done!")
