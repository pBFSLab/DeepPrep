#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import streamlit as st
import subprocess
import os

st.markdown(f'# 🚀Preprocessing of T1w & BOLD')
st.markdown(
    """
DeepPrep is a preprocessing pipeline that can flexibly handle anatomical and functional MRI data end-to-end.
It accommodates various sizes, from a single participant to LARGE-scale datasets, achieving a 10-fold acceleration compared to the state-of-the-art pipeline.

Both the anatomical and functional parts can be run separately. However, preprocessed Recon is a mandatory prerequisite for executing the functional process.

The DeepPrep workflow takes the directory of the dataset to be processed as input, which is required to be in a valid BIDS format.

-----------------
"""
)

selected_option = st.radio("select a process: ", ("All", "T1w only", "BOLD only"), horizontal=True, help="'All' preprocess both T1w and BOLD data, 'T1w only' preprocess T1w data, 'BOLD only' preprocess BOLD data")
device = st.radio("select a device: ", ("auto", "GPU", "CPU"), horizontal=True, help="Specifies the device. The default is auto, which automatically selects a device.")

st.write(f"Preprocess '{selected_option}'", f"on the '{device}' device.")

deepprep_cmd = ''
docker_cmd = 'docker run -it --rm'
commond_error = False

bids_dir = st.text_input("BIDS Path:", help="refers to the directory of the input dataset, which is required to be in BIDS format.")
if not bids_dir:
    st.error("The BIDS Path must be input!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not bids_dir.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not os.path.exists(bids_dir):
    st.error("The BIDS Path does not exist!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {bids_dir}'

output_dir = st.text_input("Output Path:", help="refers to the directory to save the DeepPrep outputs.")
if not output_dir:
    st.error("The Output Path must be input!")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
elif not output_dir.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {output_dir}'
deepprep_cmd += f' participant'

if selected_option == "BOLD only":
    subjects_dir = st.text_input("Recon Result Path",
                                 help=" the directory of Recon files.")
    if not subjects_dir:
        st.error("The Recon Result Path must be input!")
        commond_error = True
    elif not os.path.exists(subjects_dir):
        st.error("The Recon Result Path does not exist!")
        commond_error = True
else:
    subjects_dir = st.text_input("Recon Result Path (optional)", help="the output directory for the Recon files, default is <output_dir>/Recon.")

if subjects_dir:
    if not subjects_dir.startswith('/'):
        st.error("The path must be an absolute path starts with '/'.")
        commond_error = True
    elif not os.path.exists(subjects_dir):
        st.error("The Recon Result Path does not exist!")
        commond_error = True
    else:
        deepprep_cmd += f' --subjects_dir {subjects_dir}'

freesurfer_license_file = st.text_input("FreeSurfer license file path", value='/opt/freesurfer/license.txt', help="FreeSurfer license file path. It is highly recommended to replace the license.txt path with your own FreeSurfer license! You can get it for free from https://surfer.nmr.mgh.harvard.edu/registration.html")
if not freesurfer_license_file.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    commond_error = True
elif not os.path.exists(freesurfer_license_file):
    st.error("The FreeSurfer license file Path does not exist!")
    commond_error = True
else:
    deepprep_cmd += f' --fs_license_file {freesurfer_license_file}'

if selected_option != "T1w only":
    bold_task_type = st.text_input("BOLD task type", placeholder="i.e. rest, motor, 'rest motor'", help="the task label of BOLD images (i.e. rest, motor, 'rest motor').")
    if not bold_task_type:
        st.error("The BOLD task type must be input!")
        commond_error = True
    else:
        bold_task_type.replace("'", "")
        bold_task_type.replace('"', "")
        deepprep_cmd += f" --bold_task_type '{bold_task_type}'"

    bold_cifti = st.checkbox("CIFTI format", value=False, help="whether to output cifti format files.")
    # if cifti: do not support to select the spaces
    if bold_cifti:
        deepprep_cmd += ' --bold_cifti'
    else:
        surface_spaces = st.multiselect("select the surface spaces: (optional)",
            ["fsnative", "fsaverage6", "fsaverage5", "fsaverage4", "fsaverage3"],
            ["fsaverage6"],
            help="select the surface spaces from FreeSurfer"
        )
        if surface_spaces:
            surface_spaces = ' '.join(surface_spaces)
            deepprep_cmd += f" --bold_surface_spaces '{surface_spaces}'"

        bold_volume_space = st.selectbox("select a normalized volume space: (optional)", ("MNI152NLin6Asym", "MNI152NLin2009cAsym", "None"), help="select a volumetric space from TemplateFlow")
        deepprep_cmd += f' --bold_volume_space {bold_volume_space} --bold_volume_res 02'

    bold_skip_frame = st.text_input("skip n frames of BOLD data", value="2", help="skip n frames of BOLD fMRI; the default is `2`.")
    if not bold_skip_frame:
        deepprep_cmd += f' --bold_skip_frame 2'
    else:
        deepprep_cmd += f' --bold_skip_frame {bold_skip_frame}'

    bold_bandpass = st.text_input("Bandpass filter", value="0.01-0.08", help="the default range is `0.01-0.08`.")
    if not bold_bandpass:
        deepprep_cmd += f' --bold_bandpass 0.01-0.08'
    else:
        assert len(bold_bandpass.split('-')) == 2
        deepprep_cmd += f' --bold_bandpass {bold_bandpass}'

    col4, col5, col6 = st.columns(3)
    with col4:
        bold_sdc = st.checkbox("bold_sdc", value=True,
                               help="applies susceptibility distortion correction (SDC), default is True.")
        if bold_sdc:
            deepprep_cmd += ' --bold_sdc'
    with col5:
        bold_confounds = st.checkbox("bold_confounds", value=True,
                                     help="generates confounds derived from BOLD fMRI, such as head motion variables and global signals, default is True.")
        if bold_confounds:
            deepprep_cmd += ' --bold_confounds'

participant_label = st.text_input("the subject IDs (optional)", placeholder="001 002",
                                  help="Identify the subjects you'd like to process by their IDs, i.e. 'sub-001 sub-002'.")
if participant_label:
    participant_label.replace("'", "")
    participant_label.replace('"', "")
    deepprep_cmd += f" --participant_label '{participant_label}'"

if device == "GPU":
    deepprep_cmd += f' --device GPU'
    docker_cmd += ' --gpus all'
elif device == "CPU":
    deepprep_cmd += f' --device CPU'
else:
    deepprep_cmd += f' --device auto'

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

if selected_option == "BOLD only":
    deepprep_cmd += ' --bold_only'
elif selected_option == "T1w only":
    deepprep_cmd += ' --anat_only'


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
st.write(f'{docker_cmd} pbfslab/deepprep {deepprep_cmd}')
if st.button("Run", disabled=commond_error):
    with st.spinner('Waiting for the process to finish, please do not leave this page...'):
        command = [f"/opt/DeepPrep/deepprep/deepprep.sh {deepprep_cmd}"]
        with st.expander("------------ running log ------------"):
            st.write_stream(run_command(command))
        import time
        time.sleep(2)
    st.success("Done!")
