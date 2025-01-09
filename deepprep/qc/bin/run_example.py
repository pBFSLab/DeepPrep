import subprocess
import os
import json

def run_bold_get_bold_file_in_bids():
    # Define the paths for the test
    bids_dir = '/mnt/ngshare/DeepPrep/test_sample/ds004192'
    task_id = 'rest'
    output_dir = '/mnt/ngshare/DeepPrep/test_sample/ds004192_quickqc'
    work_dir = '/mnt/ngshare/DeepPrep/test_sample/ds004192_quickqc/WorkDir'

    # Ensure the output and work directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Command to run bold_get_bold_file_in_bids.py
    cmd = [
        'python3', 'bold_get_bold_file_in_bids.py',
        '--bids_dir', bids_dir,
        '--task_id', task_id,
        '--output_dir', output_dir,
        '--work_dir', work_dir
    ]
    print("Running command:", ' '.join(cmd))
    subprocess.run(cmd, check=True)

    # Assuming the JSON file created by bold_get_bold_file_in_bids.py is in the output directory
    json_dir = os.path.dirname(os.path.abspath(__file__))
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("No JSON files found in the output directory")

    bold_json_path = os.path.join(json_dir, json_files[0])
    return bold_json_path

def run_bold_head_motion(bold_json_path):
    # Define the paths for the test
    output_dir = '/mnt/ngshare/DeepPrep/test_sample/ds004192_quickqc'
    work_dir = '/mnt/ngshare/DeepPrep/test_sample/ds004192_quickqc/WorkDir'

    # Ensure the output and work directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Command to run bold_head_motion.py
    cmd = [
        'python3', 'bold_head_motion.py',
        '--bold_json', bold_json_path,
        '--output_dir', output_dir,
        '--work_dir', work_dir
    ]
    print("Running command:", ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    bold_json_path = run_bold_get_bold_file_in_bids()
    run_bold_head_motion(bold_json_path)