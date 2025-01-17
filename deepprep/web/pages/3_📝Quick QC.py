#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import streamlit as st
import subprocess
import os

st.markdown(f'# 📝Quick QC')
st.markdown(
"""
    
This page allows you to quickly perform quality control (QC) on your BOLD data.

Input the path first, and then click the 'Run' button. Once the process is complete, click 'Show' to view the results. 

More QC functions will be online, stay tuned!

-----------------
"""
)

deepprep_cmd = ''
commond_error = False

bids_dir = st.text_input("BIDS directory:", help="The directory of the BIDS dataset.")
if not bids_dir:
    st.error("The BIDS directory must be input!")
    commond_error = True
elif not bids_dir.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    commond_error = True
elif not os.path.exists(bids_dir):
    st.error("The BIDS directory does not exist!")
    commond_error = True
else:
    deepprep_cmd += f' {bids_dir}'

output_dir = st.text_input("Output directory:", help="The directory to save the QC outputs.")
if not output_dir:
    st.error("The output directory must be input!")
    commond_error = True
elif not output_dir.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    commond_error = True
else:
    deepprep_cmd += f' {output_dir}'
deepprep_cmd += f' participant'

freesurfer_license_file = st.text_input("FreeSurfer license file path", value='/opt/freesurfer/license.txt', help="FreeSurfer license file path. It is highly recommended to replace the license.txt path with your own FreeSurfer license! You can get it for free from https://surfer.nmr.mgh.harvard.edu/registration.html")
if not freesurfer_license_file.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    commond_error = True
elif not os.path.exists(freesurfer_license_file):
    st.error("The FreeSurfer license file Path does not exist!")
    commond_error = True
else:
    deepprep_cmd += f' --fs_license_file {freesurfer_license_file}'

# subjects_dir = st.text_input("Subjects directory:", help="The directory of the subjects.")
# if not subjects_dir:
#     st.error("The subjects directory must be input!")
#     commond_error = True
# elif not subjects_dir.startswith('/'):
#     st.error("The path must be an absolute path starts with '/'.")
#     commond_error = True
# elif not os.path.exists(subjects_dir):
#     st.error("The subjects directory does not exist!")
#     commond_error = True
# else:
#     deepprep_cmd += f' --subjects_dir {subjects_dir}'

task_id = st.text_input("Task ID (optional):", help="The task ID of BOLD data. (i.e. rest, motor, 'rest motor')")
if task_id:
    deepprep_cmd += f' --task_id {task_id}'

subject_id = st.text_input("Subject ID (optional):", help="Identify the subjects you'd like to process by their IDs. (i.e. '001 002')")
if subject_id:
    if 'sub-' in subject_id:
        subject_id = subject_id.replace('sub-', '')
    deepprep_cmd += f' --subject_id {subject_id}'

# bold_only = st.checkbox("BOLD only", value=False, help="Process BOLD data only.")
# if bold_only:
#     deepprep_cmd += ' --bold_only'

# freesurfer_home = st.text_input("FreeSurfer home (optional):", help="The FreeSurfer home directory.")
# if freesurfer_home:
#     deepprep_cmd += f' --freesurfer_home {freesurfer_home}'
#
# fsl_home = st.text_input("FSL home (optional):", help="The FSL home directory.")
# if fsl_home:
#     deepprep_cmd += f' --fsl_home {fsl_home}'

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
st.write(f'qc {deepprep_cmd}')
if st.button("Run", disabled=commond_error):
    os.makedirs(output_dir, exist_ok=True)
    with st.spinner('Waiting for the process to finish, please do not leave this page...'):
        command = [f"/opt/DeepPrep/deepprep/web/pages/quickqc.sh {deepprep_cmd}"]
        with st.expander("------------ running log ------------"):
            st.write_stream(run_command(command))
        import time
        time.sleep(2)
    st.success("Done!")
st.write(f'-----------  ------------')
st.write(f"Result : {os.path.join(output_dir, 'QuickQC')}")
if st.button(f"Show", disabled=commond_error):
    from bids import BIDSLayout
    quickqc_dir = os.path.join(output_dir, 'QuickQC')
    layout_bids = BIDSLayout(output_dir, validate=False, reset_database=True)

    orig_entities = {
        'suffix': 'fd',
        'extension': ['.csv']
    }
    if task_id:
        orig_entities['task'] = task_id
    if subject_id:
        orig_entities['subject'] = subject_id

    bold_orig_files = layout_bids.get(**orig_entities)

    if not bold_orig_files:
        st.warning('No result found!')

    import numpy as np

    fd_data = []
    for bold_orig_file in bold_orig_files:
        data = np.loadtxt(bold_orig_file.path, delimiter=',', skiprows=1, usecols=0)

        # Calculate the maximum value, mean, and standard deviation
        max_value = np.max(data)
        mean_value = np.mean(data)
        std_value = np.std(data)
        d = {
            'file': bold_orig_file.path.replace(os.path.join(output_dir, 'QuickQC'), ''),
            'max': max_value,
            'mean': mean_value,
            'std': std_value
        }
        fd_data.append(d)
    import pandas as pd
    fd_df = pd.DataFrame(fd_data)

    st.write(fd_df)

    st.success("Done!")