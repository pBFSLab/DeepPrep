import streamlit as st
import subprocess
import os

st.markdown(f'# 🚀Preprocessing of T1 & BOLD')
st.write(
    """
    DeepPrep is able to end-to-end preprocess anatomical and functional MRI data for different data size ranging from a single participant to a HUGE dataset. It is also flexible to run the anatomical part or functional part that requires a complete Recon folder to be specified. The DeepPrep workflow takes the directory of the dataset that is to be processed as the input, which is required to be in the valid BIDS format.
"""
)

selected_option = st.radio("请选择一个选项:", ("All", "Recon only", "BOLD only"), horizontal=True)

st.write("您选择的是:", selected_option)

deepprep_cmd = ''
docker_cmd = 'docker run -it --rm'
commond_error = False

bids_dir = st.text_input("BIDS Path:", help="refers to the directory of the input dataset, which should be in BIDS format.")
if not bids_dir:
    st.error("必须输入 BIDS Path!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not bids_dir.startswith('/'):
    st.error("必须输入以 '/' 开头的绝对路径!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not os.path.exists(bids_dir):
    st.error("BIDS Path 路径不存在!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {bids_dir}'

output_dir = st.text_input("output Path:", help="refers to the directory for the outputs of DeepPrep.")
if not output_dir:
    st.error("必须输入 output Path!")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
elif not output_dir.startswith('/'):
    st.error("必须输入以 '/' 开头的绝对路径!")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {output_dir}'
deepprep_cmd += f' participant'

if selected_option == "BOLD only":
    subjects_dir = st.text_input("Recon result Path",
                                 help=" the output directory of Recon files, default is <output_dir>/Recon.")
    if not subjects_dir:
        st.error("必须输入已存在的 Recon result Path!")
        commond_error = True
else:
    subjects_dir = st.text_input("Recon result Path (optional)", help=" the output directory of Recon files, default is <output_dir>/Recon.")

if subjects_dir:
    if not subjects_dir.startswith('/'):
        st.error("必须输入以 '/' 开头的绝对路径!")
        commond_error = True
    elif not os.path.exists(subjects_dir):
        st.error("Recon result Path 路径不存在!")
        commond_error = True
    else:
        deepprep_cmd += f' --subjects_dir {subjects_dir}'

freesurfer_license_file = st.text_input("FreeSurfer license file path", value='/opt/freesurfer/license.txt', help="the file of a valid FreeSurfer License.")
deepprep_cmd += f' --fs_license_file {freesurfer_license_file}'

if selected_option != "Recon only":
    bold_task_type = st.text_input("BOLD task type", value='rest', help="the task label of BOLD images (i.e. rest, motor).")
    if not bold_task_type:
        st.error("必须输入 BOLD task type!")
        commond_error = True
    else:
        deepprep_cmd += f' --bold_task_type {bold_task_type}'

    bold_surface_spaces = st.text_input("BOLD surface spaces", value="fsaverage6", help="specifies surface template spaces, i.e. 'fsnative fsaverage fsaverage[3-6]', default is 'fsaverage6'.")
    if not bold_surface_spaces:
        deepprep_cmd += f" --bold_surface_spaces 'fsaverage6'"
    else:
        deepprep_cmd += f" --bold_surface_spaces '{bold_surface_spaces}'"

    bold_volume_space = st.text_input("BOLD volume space", value="MNI152NLin6Asym", help="specifies an available volumetric space from TemplateFlow, default is MNI152NLin6Asym.")
    if not bold_volume_space:
        deepprep_cmd += f' --bold_volume_space MNI152NLin6Asym'
    else:
        deepprep_cmd += f' --bold_volume_space {bold_volume_space}'

    bold_volume_res = st.text_input("BOLD volume space resolution", value="02", help="specifies the spatial resolution of the corresponding template space from TemplateFlow, default is 02.")
    if not bold_volume_res:
        deepprep_cmd += f' --bold_volume_res 02'
    else:
        deepprep_cmd += f' --bold_volume_res {bold_volume_res}'

participant_label = st.text_input("the subject IDs (optional)",
                                  help="the subject ID you want to process, i.e. 'sub-001 sub-002'.")
if participant_label:
    deepprep_cmd += f" --participant_label '{participant_label}'"

device = st.radio("select a device:", ("auto", "GPU", "CPU"), horizontal=True, help="specifies the device. The default is auto, which automatically selects a GPU")
if device == "GPU":
    deepprep_cmd += f' --device GPU'
    docker_cmd += ' --gpus all'
elif device == "CPU":
    deepprep_cmd += f' --device CPU'
else:
    deepprep_cmd += f' --device auto'

col1, col2, col3 = st.columns(3)

with col1:
    skip_bids_validation = st.checkbox("skip_bids_validation", value=True, help="Skip bids validation")
    if skip_bids_validation:
        deepprep_cmd += ' --skip_bids_validation'
with col2:
    ignore_error = st.checkbox("ignore_error")
    if ignore_error:
        deepprep_cmd += ' --ignore_error'
with col3:
    resume = st.checkbox("resume", value=True)
    if resume:
        deepprep_cmd += ' --resume'

col4, col5 = st.columns(2)
with col4:
    bold_sdc = st.checkbox("bold_sdc", value=True)
    if bold_sdc:
        deepprep_cmd += ' --bold_sdc'
with col5:
    bold_confounds = st.checkbox("bold_confounds", value=True)
    if bold_confounds:
        deepprep_cmd += ' --bold_confounds'


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
    with st.spinner('Wait for it...'):
        command = [f"../deepprep.sh {deepprep_cmd}"]
        run_command(command)
        with st.expander("------------ running log ------------"):
            st.write_stream(run_command(command))
        import time
        time.sleep(2)
    st.success("Done!")
