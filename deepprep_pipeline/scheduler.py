import os
import argparse
import sys
import time

import bids
from pathlib import Path
from multiprocessing import Pool
from collections import OrderedDict
from nipype import config, logging
from interface.run import set_envrion
from interface.create_node import create_origandrawavg_node
from interface.node_source import Source


logging_wf = logging.getLogger("nipype.workflow")


class SubjectQueue:
    def __init__(self, subject_id: str, t1w_files: list):
        self.subject_id = subject_id
        self.t1w_files = t1w_files

        self.nodes_ready = []  # 待执行的Node
        self.nodes_running = []  # 待执行的Node
        self.nodes_error = []  # 执行错误的Node

    def init_first_node(self):
        node = create_origandrawavg_node(subject_id=self.subject_id, t1w_files=self.t1w_files)
        self.nodes_ready.append(node)


# class SubjectNode:
#     """
#     这个Node是Interface的Node
#     """
#     def __init__(self):
#         self.source = Source()
#
#     def postprocess(self, subject: SubjectQueue):
#         if self.node_run_success:
#             self.create_sub_node(subject.node_ready)
#         else:
#             self.interp(subject.node_error)
#
#     def node_run_success(self):
#         """
#         在执行node.run()以后，判断node是否完整运行
#         """
#         return True or False
#
#     def create_sub_node(self):
#         return node
#
#     def last_node(self):
#         """
#         在Queue中删除自己的subject
#         """
#         pass


class Queue:
    def __init__(self, subject_ids: list, t1w_filess: list):

        self.subjects = OrderedDict()

        for subject_id, t1w_files in zip(subject_ids, t1w_filess):
            subject = SubjectQueue(subject_id, t1w_files)
            subject.init_first_node()
            self.subjects[subject_id] = subject


class Scheduler:
    def __init__(self, subject_ids: list, t1w_filess: list):
        self.source_res = Source(36, 24000, 60000, 150, 450)

        self.queue = Queue(subject_ids, t1w_filess)

        self.pool = Pool()

        self.subjects_success = []
        self.subjects_error = []

    def check_node_source(self, source: Source):
        source_result = self.source_res - source
        for i in source_result:
            if i < 0:
                return False
        return True

    def run_node(self, subject_queue: SubjectQueue, node):
        try:
            subject_queue.nodes_running.append(node)
            node.run()
        except Exception as why:
            logging_wf.error(f'Run_Node_Error : {why}')
            subject_queue.nodes_running.remove(node)
            subject_queue.nodes_error.append(node)
        else:
            try:
                sub_nodes = node.interface.create_sub_node()
                if isinstance(sub_nodes, list):
                    subject_queue.nodes_ready.extend(sub_nodes)
                else:
                    subject_queue.nodes_ready.append(sub_nodes)
            except Exception as why:
                logging_wf.error(f'Run_Node_Error : {why}')
        finally:
            self.source_res += node.source  # 回收资源
            logging_wf.info(f'ADD    {node.name}    {self.source_res}')
            # 判断是否为最后一个node
            if 'Smooth_node' in node.name:
                self.queue.subjects.pop(subject_queue.subject_id)
            # 判断subject_queue是否运行完成
            elif len(subject_queue.nodes_running) == 0 and len(subject_queue.nodes_ready) == 0:
                self.queue.subjects.pop(subject_queue.subject_id)

    def run_queue(self):
        """
        1. node执行完毕
        2. 有新的node进入队列
        """
        queue = self.queue

        run_new_node = False
        for subject_id, subject_queue in queue.subjects.items():
            for node in subject_queue.nodes_ready:
                if self.check_node_source(node.source):
                    subject_queue.nodes_ready.remove(node)
                    self.source_res -= node.source
                    logging_wf.info(f'SUB    {node.name}    {self.source_res}')
                    self.pool.apply_async(node)  # 子进程 run
                    self.run_node(subject_queue, node)
                    # node.run()  # 子进程 run
                    # sub_nodes = node.interface.create_sub_node()
                    # if isinstance(sub_nodes, list):
                    #     subject_queue.nodes_ready.extend(sub_nodes)
                    # else:
                    #     subject_queue.nodes_ready.append(sub_nodes)
                    run_new_node = True
                if run_new_node:
                    break
            if run_new_node:
                break
        return run_new_node

    def run(self):
        while True:
            run_new_node = self.run_queue()
            if not run_new_node:
                time.sleep(5)


def parse_args():
    """
python3 deepprep_pipeline.py
--bids_dir
/mnt/ngshare2/UKB/BIDS
--recon_output_dir
/mnt/ngshare2/DeepPrep_UKB/UKB_Recon
--bold_output_dir
/mnt/ngshare2/DeepPrep_UKB/UKB_BoldPreprocess
--cache_dir
/mnt/ngshare2/DeepPrep_UKB/UKB_Workflow
"""

    parser = argparse.ArgumentParser(
        description="DeepPrep: sMRI and fMRI PreProcessing workflows"
    )

    parser.add_argument("--bids_dir", help="directory of BIDS type: /mnt/ngshare2/UKB/BIDS", required=True)
    parser.add_argument("--recon_output_dir", help="structure data Recon output directory: /mnt/ngshare2/DeepPrep_UKB/UKB_Recon", required=True)
    parser.add_argument("--bold_output_dir", help="BOLD data Preprocess output directory: /mnt/ngshare2/DeepPrep_UKB/UKB_BoldPreprocess", required=True)
    parser.add_argument("--cache_dir", help="workflow cache dir: /mnt/ngshare2/DeepPrep_UKB/UKB_Workflow", required=True)
    args = parser.parse_args()

    return args


def main():
    set_envrion(threads=1)
    args = parse_args()
    data_path = Path(args.bids_dir)
    subjects_dir = Path(args.recon_output_dir)
    bold_preprocess_dir = Path(args.bold_output_dir)
    workflow_cached_dir = Path(args.cache_dir)

    # ############### Structure
    pwd = Path.cwd()
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
    python_interpret = Path(sys.executable)  # 获取当前的Python解析器地址

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    os.environ['BOLD_PREPROCESS_DIR'] = str(bold_preprocess_dir)
    os.environ['WORKFLOW_CACHED_DIR'] = str(workflow_cached_dir)
    os.environ['FASTSURFER_HOME'] = str(fastsurfer_home)
    os.environ['FREESURFER_HOME'] = str(freesurfer_home)
    os.environ['FASTCSR_HOME'] = str(fastcsr_home)
    os.environ['FEATREG_HOME'] = str(featreg_home)

    subjects_dir.mkdir(parents=True, exist_ok=True)
    bold_preprocess_dir.mkdir(parents=True, exist_ok=True)
    workflow_cached_dir.mkdir(parents=True, exist_ok=True)

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

    batch_size = 16

    for epoch in range(len(subject_ids_all) + 1):
        # try:
        t1w_files = t1w_filess_all[epoch * batch_size: (epoch + 1) * batch_size]
        subject_ids = subject_ids_all[epoch * batch_size: (epoch + 1) * batch_size]

        # 设置log目录位置
        log_dir = workflow_cached_dir / 'log' / f'batchsize_{batch_size:03d}_epoch_{epoch:03d}'
        log_dir.mkdir(parents=True, exist_ok=True)
        config.update_config({'logging': {'log_directory': log_dir,
                                          'log_to_file': True}})
        logging.update_logging(config)

        scheduler = Scheduler(subject_ids, t1w_files)
        scheduler.run()
        # except Exception as why:
        #     print(f'Exception : {why}')


if __name__ == '__main__':
    main()
