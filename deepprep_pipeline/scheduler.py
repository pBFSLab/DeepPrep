import os
import argparse
import time

import bids
from pathlib import Path
from multiprocessing import Lock, Manager, Process
from nipype import config, logging
from interface.run import set_envrion
from interface.node_source import Source
from interface.create_node_structure import create_origandrawavg_node

logging_wf = logging.getLogger("nipype.workflow")


def clear_is_running(subjects_dir: Path, subject_ids: list):
    for subject_id in subject_ids:
        is_running_file = subjects_dir / subject_id / 'scripts' / 'IsRunning.lh+rh'
        if is_running_file.exists():
            os.remove(is_running_file)


class Scheduler:
    def __init__(self, share_manager: Manager, subject_ids: list):
        self.source_res = Source(36, 24000, 60000, 150, 450)

        # Queue
        self.queue_subject = share_manager.list()  # 队列中的subject
        self.subject_success = share_manager.list()  # 运行成功的subject
        self.subject_error = share_manager.list()  # 运行出错的subject
        self.queue_subject.extend(subject_ids)

        # Node_name
        self.nodes_ready = list()  # 待执行的Node_name
        self.s_nodes_running = share_manager.list()  # 正在执行的Node_name
        self.s_nodes_done = share_manager.list()  # 执行完毕的Node_name
        self.s_nodes_success = share_manager.list()  # 执行正确的Node_name
        self.s_nodes_error = share_manager.list()  # 执行错误的Node_name

        # Node
        self.node_all = dict()

    def check_node_source(self, source: Source):
        """
        检查资源是否满足现在的node
        """
        source_result = self.source_res - source
        for i in source_result:
            if i < 0:
                return False
        return True

    def node_all_add_nodes(self, nodes):
        """
        将node信息加入node_all
        """
        if isinstance(nodes, list):
            for node in nodes:
                self.node_all[node.name] = node
        else:
            self.node_all[nodes.name] = nodes

    def get_subject_ready_nodes(self, subject_id: str):
        """
        根据subject_id获取这个人全部准备好的Node
        """
        nodes = list()
        if len(self.nodes_ready) <= 0:
            return nodes
        for node_name in self.nodes_ready:
            if subject_id in node_name:
                nodes.append(self.node_all[node_name])
        return nodes

    def check_run_success(self, subject_id):
        """
        检查一个subject_id的所有node是否都运行成功
        如果有这个subject_id对应的Node在self.nodes_error中存在
        则说明这个subject_id运行错误
        """
        for node_name in self.s_nodes_error:
            if subject_id in node_name:
                return False
        return True

    @staticmethod
    def run_node(node, node_error: list, node_success: list, node_running: list, node_done: list, lock: Lock):
        try:
            print(f'run start {node.name}')
            node.run()
            print(f'run over {node.name}')
        except Exception as why:
            logging_wf.error(f'Run_Node_Error : {why}')
            if lock.acquire():
                node_error.append(node.name)
                lock.release()
        else:
            if lock.acquire():
                node_success.append(node.name)
                lock.release()
        finally:
            if lock.acquire():
                node_done.append(node.name)
                node_running.remove(node.name)
                lock.release()

    def run_queue(self, lock):
        """
        1. node执行完毕
        2. 有新的node进入队列
        """
        print('Start run queue =================== ==================')
        lock.acquire()
        print(f'nodes_ready    : {len(self.nodes_ready):3d}', self.nodes_ready)
        print(f'nodes_success  : {len(self.s_nodes_success):3d}', self.s_nodes_success)
        print(f'nodes_error    : {len(self.s_nodes_error):3d}', self.s_nodes_error)
        print(f'nodes_running  : {len(self.s_nodes_running):3d}', self.s_nodes_running)
        print(f'nodes_done     : {len(self.s_nodes_done):3d}', self.s_nodes_done)
        print(f'Source Res     : task_running {len(self.s_nodes_running):3d}', self.source_res)
        print()
        print(f'subjects_success     : {len(self.subject_success):3d}', self.subject_success)
        print(f'subjects_error       : {len(self.subject_error):3d}', self.subject_error)
        print('                =================== ==================')
        # ############# Start deal nodes_done =================== ==================')
        for node_name in self.s_nodes_done:
            node = self.node_all.pop(node_name)
            self.source_res += node.source
            self.s_nodes_done.remove(node_name)
            print(f'Source ADD +    {node.name} :   {self.source_res}')
            if node_name in self.s_nodes_success:
                try:
                    last_node = getattr(node, 'last_name')
                except AttributeError:
                    last_node = False
                if last_node:
                    self.queue_subject.remove(node.subject_id)
                    if self.check_run_success(node.subject_id):
                        self.subject_success.append(node.subject_id)
                    else:
                        self.subject_error.append(node.subject_id)
                else:
                    try:
                        sub_nodes = node.interface.create_sub_node()
                        if isinstance(sub_nodes, list):
                            self.nodes_ready.extend([i.name for i in sub_nodes])
                        else:
                            self.nodes_ready.append(sub_nodes.name)
                        self.node_all_add_nodes(sub_nodes)
                    except Exception as why:
                        logging_wf.error(f'Create_Sub_Node_Error {node_name}: {why}')
        # ############# End deal nodes_done =================== ==================')

        # ############# Start deal nodes_ready =================== ==================')
        # run node in nodes_ready
        run_new_node = False
        for subject_id in self.queue_subject:
            nodes = self.get_subject_ready_nodes(subject_id)  # get nodes from self.node_all and self.node_ready
            if len(nodes) <= 0:
                continue
            for node in nodes:
                if self.check_node_source(node.source):
                    self.nodes_ready.remove(node.name)
                    self.s_nodes_running.append(node.name)
                    self.source_res -= node.source
                    print(f'Source SUB -    {node.name} :   {self.source_res}')
                    # 不能用pool
                    # self.pool.apply_async(run_node, (node, self.s_nodes_error, self.s_nodes_success,
                    #                                  self.s_nodes_done, lock))  # 子进程 run
                    Process(target=self.run_node, args=(node, self.s_nodes_error, self.s_nodes_success,
                                                        self.s_nodes_running, self.s_nodes_done, lock)).start()
                    run_new_node = True
                if run_new_node:
                    break
            if run_new_node:
                break
        # ############# End deal nodes_ready =================== ==================')
        lock.release()
        return run_new_node

    def run(self, lock):
        while True:
            run_new_node = self.run_queue(lock)
            if not run_new_node:
                print('no new node, sleep 3s')
                time.sleep(3)


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
    parser.add_argument("--recon_output_dir",
                        help="structure data Recon output directory: /mnt/ngshare2/DeepPrep_UKB/UKB_Recon",
                        required=True)
    parser.add_argument("--bold_output_dir",
                        help="BOLD data Preprocess output directory: /mnt/ngshare2/DeepPrep_UKB/UKB_BoldPreprocess",
                        required=True)
    parser.add_argument("--cache_dir", help="workflow cache dir: /mnt/ngshare2/DeepPrep_UKB/UKB_Workflow",
                        required=True)
    args = parser.parse_args()

    return args


def main():
    set_envrion(threads=1)
    args = parse_args()
    bids_data_path = Path(args.bids_dir)
    subjects_dir = Path(args.recon_output_dir)
    bold_preprocess_dir = Path(args.bold_output_dir)
    workflow_cached_dir = Path(args.cache_dir)

    # ############### Structure
    pwd = Path.cwd()  # deepprep_pipeline/
    fastsurfer_home = pwd / "FastSurfer"
    freesurfer_home = Path('/usr/local/freesurfer720')
    fastcsr_home = pwd / "FastCSR"
    featreg_home = pwd / "FeatReg"

    # ############### BOLD
    mni152_brain_mask = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
    vxm_model_path = pwd / 'model' / 'voxelmorph'
    resource_dir = pwd / 'resource'
    atlas_type = 'MNI152_T1_2mm'
    task = 'rest'  # 'motor' or 'rest'
    preprocess_method = 'rest'  # 'task' or 'rest'

    # ############### Common
    # python_interpret = Path(sys.executable)  # 获取当前的Python解析器地址

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    os.environ['BOLD_PREPROCESS_DIR'] = str(bold_preprocess_dir)
    os.environ['WORKFLOW_CACHED_DIR'] = str(workflow_cached_dir)
    os.environ['FASTSURFER_HOME'] = str(fastsurfer_home)
    os.environ['FREESURFER_HOME'] = str(freesurfer_home)
    os.environ['FASTCSR_HOME'] = str(fastcsr_home)
    os.environ['FEATREG_HOME'] = str(featreg_home)
    os.environ['BIDS_DIR'] = str(bids_data_path)
    os.environ['VXM_MODEL_PATH'] = str(vxm_model_path)
    os.environ['MNI152_BRAIN_MASK'] = str(mni152_brain_mask)
    os.environ['RESOURCE_DIR'] = str(resource_dir)

    os.environ['DEEPPREP_ATLAS_TYPE'] = atlas_type
    os.environ['DEEPPREP_TASK'] = task
    os.environ['DEEPPREP_PREPROCESS_METHOD'] = preprocess_method

    subjects_dir.mkdir(parents=True, exist_ok=True)
    bold_preprocess_dir.mkdir(parents=True, exist_ok=True)
    workflow_cached_dir.mkdir(parents=True, exist_ok=True)

    layout = bids.BIDSLayout(str(bids_data_path), derivatives=False)

    t1w_filess_all = list()
    subject_ids_all = list()
    for t1w_file in layout.get(return_type='filename', suffix="T1w"):
        sub_info = layout.parse_file_entities(t1w_file)
        subject_id = f"sub-{sub_info['subject']}"
        if 'session' in sub_info:
            subject_id = subject_id + f"-ses-{sub_info['session']}"
        t1w_filess_all.append([t1w_file])
        subject_ids_all.append(subject_id)

    batch_size = 20

    for epoch in range(len(subject_ids_all) + 1):
        # try:
        t1w_filess = t1w_filess_all[epoch * batch_size: (epoch + 1) * batch_size]
        subject_ids = subject_ids_all[epoch * batch_size: (epoch + 1) * batch_size]

        # 设置log目录位置
        log_dir = workflow_cached_dir / 'log' / f'batchsize_{batch_size:03d}_epoch_{epoch:03d}'
        log_dir.mkdir(parents=True, exist_ok=True)
        config.update_config({'logging': {'log_directory': log_dir,
                                          'log_to_file': True}})
        logging.update_logging(config)

        clear_is_running(subjects_dir=subjects_dir,
                         subject_ids=subject_ids)

        lock = Lock()
        with Manager() as share_manager:
            scheduler = Scheduler(share_manager, subject_ids)
            for subject_id, t1w_files in zip(subject_ids, t1w_filess):
                node = create_origandrawavg_node(subject_id=subject_id, t1w_files=t1w_files)
                scheduler.node_all[node.name] = node
                scheduler.nodes_ready.append(node.name)
            scheduler.run(lock)
            logging_wf.info(f'subject_success: {scheduler.subject_success}')
            logging_wf.info(f'nodes_error: {scheduler.s_nodes_error}')
            logging_wf.info(f'subject_error: {scheduler.subject_error}')
        # except Exception as why:
        #     print(f'Exception : {why}')


if __name__ == '__main__':
    main()
