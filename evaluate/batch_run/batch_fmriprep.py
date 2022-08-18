import os
from pathlib import Path
import multiprocessing as mp
import docker


# --fs-subjects-dir
# --participant-label
# -t, --task-id
def decker_run():
    client = docker.from_env()
    client.containers.run('ubuntu:20.04', 'echo hello world')
    Path.parent
    pass


if __name__ == '__main__':
    cmds = list()
    for i in range(10):
        cmd = f'docker run --rm -v /home/weiwei/workdata/DeepPrep/workdir/ds000224/:/home/fmriprep/input -v /home/weiwei/workdata/DeepPrep/fmriprep_output5/rest:/home/fmriprep/output -v /home/weiwei/workdata/DeepPrep/fmriprep_output/MSC/FreeSurfer:/subjects ' + \
              f'-v /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt nipreps/fmriprep:21.0.2 /home/fmriprep/input /home/fmriprep/output participant --skip_bids_validation --fs-subjects-dir /subjects --participant-label MSC{i + 1:02} -t rest --output-spaces MNI152NLin6Asym:res-02 > MSC{i + 1:02}_rest.log'
        cmds.append(cmd)
        cmd = f'docker run --rm -v /home/weiwei/workdata/DeepPrep/workdir/ds000224/:/home/fmriprep/input -v /home/weiwei/workdata/DeepPrep/fmriprep_output5/motor:/home/fmriprep/output -v /home/weiwei/workdata/DeepPrep/fmriprep_output/MSC/FreeSurfer:/subjects ' + \
              f'-v /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt nipreps/fmriprep:21.0.2 /home/fmriprep/input /home/fmriprep/output participant --skip_bids_validation --fs-subjects-dir /subjects --participant-label MSC{i + 1:02} -t motor --output-spaces MNI152NLin6Asym:res-02 > MSC{i + 1:02}_motor.log'
        cmds.append(cmd)
    with mp.Pool(processes=2) as pool:
        pool.map(os.system, cmds)
