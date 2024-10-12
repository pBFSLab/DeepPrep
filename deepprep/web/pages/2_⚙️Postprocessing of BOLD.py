import streamlit as st
import subprocess
import os

st.markdown(f'# ⚙️Postprocessing of BOLD')
st.write(
    """
#### processing steps
Surface space： bandpass -> regression -> smooth(optional)

Volume space：  bandpass -> regression -> smooth(optional)

-----------------
"""
)

deepprep_cmd = ''
commond_error = False

bids_dir = st.text_input("preprocessing result Path:", help="refers to the directory of the DeepPrep BOLD result, which should be in BIDS format.")
if not bids_dir:
    st.error("The preprocessing result Path must be input!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not bids_dir.startswith('/'):
    st.error("It must be an absolute path that starts with '/'.")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not os.path.exists(bids_dir):
    st.error("The preprocessing result Path does not exist!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {bids_dir}'

output_dir = st.text_input("output Path:", help="refers to the directory for the outputs of postprocessing of BOLD.")
if not output_dir:
    st.error("The output Path must be input!")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
elif not output_dir.startswith('/'):
    st.error("It must be an absolute path that starts with '/'.")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {output_dir}'
deepprep_cmd += f' participant'



freesurfer_license_file = st.text_input("FreeSurfer license file path", value='/opt/freesurfer/license.txt', help="FreeSurfer license file path.You should replace license.txt path with your own FreeSurfer license! You can get your license file for free from https://surfer.nmr.mgh.harvard.edu/registration.html")
if not freesurfer_license_file.startswith('/'):
    st.error("It must be an absolute path that starts with '/'.")
    commond_error = True
elif not os.path.exists(freesurfer_license_file):
    st.error("The FreeSurfer license file Path does not exist!")
    commond_error = True
else:
    deepprep_cmd += f' --fs_license_file {freesurfer_license_file}'

bold_task_type = st.text_input("BOLD task type", value='rest', help="the task label of BOLD images (i.e. rest, motor, 'rest motor').")
if not bold_task_type:
    st.error("The BOLD task type must be input!")
    commond_error = True
else:
    bold_task_type.replace("'", "")
    bold_task_type.replace('"', "")
    deepprep_cmd += f"--bold_task_type '{bold_task_type}'"

bold_volume_space = st.selectbox("select a space", ("MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsnative", "fsaverage6", "fsaverage5", "fsaverage4"), help="select a space")
deepprep_cmd += f' --bold_volume_space {bold_volume_space} --bold_volume_res 02'

if bold_volume_space == "fanative":
    subjects_dir = st.text_input("Recon result Path",
                                 help=" the directory of Recon files.")
    if not subjects_dir:
        st.error("The Recon result Path must be input!")
        commond_error = True
    elif not subjects_dir.startswith('/'):
        st.error("It must be an absolute path that starts with '/'.")
        commond_error = True
    elif not os.path.exists(subjects_dir):
        st.error("The Recon result Path does not exist!")
        commond_error = True
    else:
        deepprep_cmd += f' --subjects_dir {subjects_dir}'

bold_skip_frame = st.text_input("skip n frames of BOLD data", value="2", help="skip n frames of BOLD fMRI; the default is `2`.")
if not bold_skip_frame:
    deepprep_cmd += f' --bold_skip_frame 2'
else:
    deepprep_cmd += f' --bold_skip_frame {bold_skip_frame}'

participant_label = st.text_input("the subject IDs (optional)",
                                  help="the subject ID you want to process, i.e. 'sub-001 sub-002'.")
if participant_label:
    participant_label.replace("'", "")
    participant_label.replace('"', "")
    deepprep_cmd += f" --participant_label '{participant_label}'"

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
# st.write(f'{docker_cmd} pbfslab/deepprep {deepprep_cmd}')
if st.button("Run", disabled=commond_error):
    with st.spinner('Wait for it...'):
        command = [f"../deepprep.sh {deepprep_cmd}"]
        run_command(command)
        with st.expander("------------ running log ------------"):
            st.write_stream(run_command(command))
        import time
        time.sleep(2)
    st.success("Done!")
