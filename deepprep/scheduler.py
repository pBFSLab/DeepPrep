import os
import argparse
import time
from datetime import datetime

import bids
from pathlib import Path
from multiprocessing import Lock, Manager, Process
from nipype import config, logging
from interface.run import set_envrion
from interface.node_source import Source
from interface.create_node_structure import create_OrigAndRawavg_node

logging_wf = logging.getLogger("nipype.workflow")  # TODO: 日志应该拆分为多个 scheduler和subject


def clear_is_running(subjects_dir: Path, subject_ids: list):
    for subject_id in subject_ids:
        is_running_file = subjects_dir / subject_id / 'scripts' / 'IsRunning.lh+rh'
        if is_running_file.exists():
            os.remove(is_running_file)


def clear_subject_bold_tmp_dir(bold_preprocess_dir: Path, subject_ids: list, task: str):
    for subject_id in subject_ids:
        tmp_dir = bold_preprocess_dir / subject_id / 'tmp' / f'task-{task}'
        if tmp_dir.exists():
            os.system(f'rm -r {tmp_dir}')


class Scheduler:
    def __init__(self, share_manager: Manager, subject_ids: list, last_node_name=None, auto_schedule=True,
                 settings=None):
        self.last_node_name = last_node_name
        self.auto_schedule = auto_schedule  # 是否开启自动调度
        self.settings = settings
        self.source_res = Source(CPU_n=settings.CPU_NUM, GPU_MB=settings.GPU_MB, RAM_MB=settings.RAM_MB,
                                 IO_write_MB=settings.IO_WRITE_MB, IO_read_MB=settings.IO_READ_MB)

        self.start_datetime = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # Queue
        self.queue_subject = share_manager.list()  # 队列中的subject
        self.subject_success = share_manager.list()  # 运行成功的subject
        self.subject_success_datetime = share_manager.list()  # 运行成功的datetime
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

        # others
        self.iter_count = 0

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
    def run_node_single(node, node_error: list, node_success: list, node_running: list, node_done: list,
                        subject_error: list):
        try:
            print(f'run start {node.name}')
            node.run()
            print(f'run over {node.name}')
        except Exception as why:
            logging_wf.error(f'Run_Node_Error : {why}')
            node_error.append(node.name)
            subject_error.append(node.inputs.subject_id)
        else:
            node_success.append(node.name)
        finally:
            node_done.append(node.name)
            node_running.remove(node.name)

    @staticmethod
    def run_node(node, node_error: list, node_success: list, node_running: list, node_done: list, subject_error: list,
                 lock: Lock):
        try:
            print(f'run start {node.name}')
            node.run()
            print(f'run over {node.name}')
        except Exception as why:
            logging_wf.error(f'Run_Node_Error : {why}')
            if lock.acquire():
                node_error.append(node.name)
                subject_error.append(node.inputs.subject_id)
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
        lock.acquire()
        if self.iter_count % 6 == 0:  # TODO CHG 如果是debug模式，才会在运行时输出以下内容
            print('Start run queue =================== ==================')
            print(f'start_datetime : {self.start_datetime}')
            print(f'nodes_running  : {len(self.s_nodes_running):3d}', self.s_nodes_running)
            print(f'nodes_done     : {len(self.s_nodes_done):3d}', self.s_nodes_done)
            print(f'nodes_ready    : {len(self.nodes_ready):3d}', self.nodes_ready)
            print(f'nodes_success  : {len(self.s_nodes_success):3d}', self.s_nodes_success[-25:])
            print(f'nodes_error    : {len(self.s_nodes_error):3d}', self.s_nodes_error)
            print(f'Source Res     : task_running {len(self.s_nodes_running):3d}', self.source_res)
            print()
            print(f'subjects_success     : {len(self.subject_success):3d}', self.subject_success)
            print(f'subjects_datetime    : {len(self.subject_success_datetime):3d}', self.subject_success_datetime)
            print(f'subjects_error       : {len(self.subject_error):3d}', self.subject_error)
            print()
            print('                =================== =================')
        # ############# Start deal nodes_done =================== ==================')
        for node_name in self.s_nodes_done:
            node = self.node_all.pop(node_name)
            self.source_res += node.source
            self.s_nodes_done.remove(node_name)
            print(f'Source ADD +    {node.name} :   {self.source_res}')
            if node_name in self.s_nodes_success:
                if self.last_node_name is not None and self.last_node_name in node_name:
                    self.queue_subject.remove(node.inputs.subject_id)
                    if self.check_run_success(node.inputs.subject_id):
                        self.subject_success.append(node.inputs.subject_id)
                        self.subject_success_datetime.append(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    else:
                        self.subject_error.append(node.inputs.subject_id)
                else:
                    try:
                        sub_nodes = node.interface.create_sub_node(self.settings)
                        if isinstance(sub_nodes, list):
                            self.nodes_ready.extend([i.name for i in sub_nodes])
                        else:
                            self.nodes_ready.append(sub_nodes.name)
                        self.node_all_add_nodes(sub_nodes)
                    except Exception as why:
                        logging_wf.error(f'Create_Sub_Node_Error {node_name}: {why}')

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
                    # 不能用pool,只能用Process
                    # self.pool.apply_async(run_node, (node, self.s_nodes_error, self.s_nodes_success,
                    #                                  self.s_nodes_done, lock))  # 子进程 run
                    if self.auto_schedule:
                        Process(target=self.run_node, args=(node, self.s_nodes_error, self.s_nodes_success,
                                                            self.s_nodes_running, self.s_nodes_done, self.subject_error,
                                                            lock)).start()
                    else:
                        self.run_node_single(node, self.s_nodes_error, self.s_nodes_success,
                                             self.s_nodes_running, self.s_nodes_done, self.subject_error)
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
                lock.acquire()
                if len(self.s_nodes_running) == 0 and len(self.s_nodes_done) == 0:
                    logging_wf.info('All task node Done!')
                    break
                lock.release()
                print('No new node, wait 10s')
                time.sleep(10)
            self.iter_count += 1


def parse_args(settings):
    """
    python3 deepprep.py
    --bids_dir
    /mnt/ngshare2/UKB/BIDS
    --recon_output_dir
    /mnt/ngshare2/DeepPrep_UKB/UKB_Recon
    --bold_output_dir
    /mnt/ngshare2/DeepPrep_UKB/UKB_BoldPreprocess
    --cache_dir
    /mnt/ngshare2/DeepPrep_UKB/UKB_Workflow

    --bids_dir
    /mnt/ngshare2/MSC_all/MSC
    --recon_output_dir
    /mnt/ngshare2/MSC_all/MSC_Recon
    --bold_output_dir
    /mnt/ngshare2/MSC_all/MSC_BoldPreprocess
    --cache_dir
    /mnt/ngshare2/MSC_all/MSC_Workflow
    --bold_task_type
    motor
    --bold_preprocess_method
    task
    --bold_only
    True
    --single_sub_multi_t1
    True

    """

    parser = argparse.ArgumentParser(
        description="DeepPrep: sMRI and fMRI PreProcessing workflows"
    )

    parser.add_argument("--bids-dir", help="directory of BIDS type: /mnt/ngshare2/UKB/BIDS", required=True)
    parser.add_argument("--recon-output-dir",
                        help="structure data Recon output directory: /mnt/ngshare2/DeepPrep_UKB/UKB_Recon",
                        required=True)
    parser.add_argument("--bold-output-dir",
                        help="BOLD data Preprocess output directory: /mnt/ngshare2/DeepPrep_UKB/UKB_BoldPreprocess",
                        required=True)
    parser.add_argument("--cache-dir", help="workflow cache dir: /mnt/ngshare2/DeepPrep_UKB/UKB_Workflow",
                        required=True)

    parser.add_argument("--bold-atlas-type", help="bold使用的MNI模板类型", default='MNI152_T1_2mm', required=False)
    parser.add_argument("--bold-task-type", help="跑的task类型example:motor、rest", default='rest', required=False)
    parser.add_argument("--bold-preprocess-method", help='使用的bold处理方法 rest or task', default=None,
                        required=False)

    parser.add_argument("--bold-only", help='跳过Recon', dest='bold_only', action='store_true', required=False)
    parser.add_argument("--recon-only", help='跳过BOLD', dest='recon_only', action='store_true', required=False)
    parser.set_defaults(bold_only=False)
    parser.set_defaults(recon_only=False)

    parser.add_argument("--no-rawavg-t1", help='是否平均多个T1。如果不平均T1，那么只运行Recon预处理，不运行BOLD',
                        dest='rawavg_t1', action='store_false', required=False)
    parser.set_defaults(rawavg_t1=True)

    parser.add_argument("--subject-filter", help='通过subject_id过滤, file of subject_id or subject id list',
                        required=False, nargs='+')

    parser.add_argument("--source-CPU-NUM", help="设置资源池-使用CPU数量", type=int, required=False)
    parser.add_argument("--source-GPU-MB", help="设置资源池-使用GPU", type=int, required=False)
    parser.add_argument("--source-RAM-MB", help="设置资源池-使用CPU数量", type=int, required=False)
    parser.add_argument("--source-IO-WRITE-MB", help="设置资源池-使用硬盘IO", type=int, required=False)
    parser.add_argument("--source-IO-READ-MB", help="设置资源池-使用硬盘IO", type=int, required=False)

    args = parser.parse_args()

    settings.BIDS_DIR = args.bids_dir
    settings.SUBJECTS_DIR = args.recon_output_dir
    settings.BOLD_PREPROCESS_DIR = args.bold_output_dir
    settings.WORKFLOW_CACHED_DIR = args.cache_dir

    settings.RECON_ONLY = args.recon_only
    settings.BOLD_ONLY = args.bold_only

    settings.SMRI.RAWAVG = args.rawavg_t1
    # Warning：如果不平均T1，那么只运行sMRI的Recon流程，不运行fMRI流程
    # 原因为：fMRI流程与Recon结果相关，需要固定一个Recon结果。目前使用的是与subject_id相同的Recon结果（无 ses 和 run 的信息）。
    if not args.rawavg_t1:
        settings.RECON_ONLY = True

    if args.bold_atlas_type is not None:
        settings.FMRI.ATLAS_TYPE = args.bold_atlas_type
    if args.bold_task_type is not None:
        settings.FMRI.TASK = args.bold_task_type
    if args.bold_preprocess_method is not None:
        settings.FMRI.PREPROCESS_TYPE = args.bold_preprocess_method

    settings.SUBJECT_FILTER = args.subject_filter

    if args.source_CPU_NUM is not None:
        settings.CPU_NUM = args.source_CPU_NUM
    if args.source_GPU_MB is not None:
        settings.GPU_MB = args.source_GPU_MB
    if args.source_RAM_MB is not None:
        settings.RAM_MB = args.source_RAM_MB
    if args.source_IO_WRITE_MB is not None:
        settings.IO_WRITE_MB = args.source_IO_WRITE_MB
    if args.source_IO_READ_MB is not None:
        settings.IO_READ_MB = args.source_IO_READ_MB
    return settings


def main(settings):

    parse_args(settings)
    bids_data_path = Path(settings.BIDS_DIR)
    subjects_dir = Path(settings.SUBJECTS_DIR)
    bold_preprocess_dir = Path(settings.BOLD_PREPROCESS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    # ############### BOLD
    atlas_type = settings.FMRI.ATLAS_TYPE
    task = settings.FMRI.TASK  # 'motor' or 'rest' or '...'
    if settings.FMRI.PREPROCESS_TYPE is None:
        if task == 'rest':
            preprocess_method = 'rest'
        else:
            preprocess_method = 'task'
    else:
        preprocess_method = settings.FMRI.PREPROCESS_TYPE  # 'task' or 'rest'

    # ############### Common
    last_node_name = 'VxmRegNormMNI152_node'  # workflow的最后一个node的名字,VxmRegNormMNI152_node or Smooth_node or ...
    last_node_name_bold = 'VxmRegNormMNI152_node'
    last_node_name_recon = 'Aseg7_node'
    auto_schedule = settings.AUTO_SCHEDULE  # 是否开启自动调度

    set_envrion(
        freesurfer_home=settings.FREESURFER_HOME,
        java_home=settings.JAVA_HOME,
        fsl_home=settings.FSL_HOME,
        subjects_dir=str(subjects_dir),
        threads=settings.THREADS
    )

    # ############### filter subjects by subjects_filter_file
    if settings.SUBJECT_FILTER is not None:
        if len(settings.SUBJECT_FILTER) == 1 and os.path.isfile(settings.SUBJECT_FILTER[0]):
            with open(settings.SUBJECT_FILTER[0], 'r') as f:
                subject_filter_ids = f.readlines()
                subject_filter_ids = [i.strip() for i in subject_filter_ids]
                subject_filter_ids = set(subject_filter_ids)
        else:
            subject_filter_ids = settings.SUBJECT_FILTER
    else:
        subject_filter_ids = None

    subjects_dir.mkdir(parents=True, exist_ok=True)
    bold_preprocess_dir.mkdir(parents=True, exist_ok=True)
    workflow_cached_dir.mkdir(parents=True, exist_ok=True)

    layout = bids.BIDSLayout(str(bids_data_path), derivatives=False)

    # TODO: bold文件应该提前集中获取，而不是在每个Bold Node都去使用bids.layout获取一次
    t1w_filess = list()
    subject_ids = list()
    subject_dict = {}
    for t1w_file in layout.get(return_type='filename', suffix="T1w", extension='.nii.gz'):
        sub_info = layout.parse_file_entities(t1w_file)
        subject_id = f"sub-{sub_info['subject']}"
        # filter subjects by subjects_filter_file
        if (subject_filter_ids is not None) and (subject_id not in subject_filter_ids):
            continue
        if not settings.SMRI.RAWAVG:
            if 'session' in sub_info:
                subject_id = subject_id + f"-ses-{sub_info['session']}"
            if 'run' in sub_info:
                subject_id = subject_id + f"-run-{sub_info['run']}"
            t1w_filess.append([t1w_file])
            subject_ids.append(subject_id)
        else:
            # 合并多个T1跑Recon
            subject_dict.setdefault(subject_id, []).append(t1w_file)
            subject_ids = list(subject_dict.keys())
            t1w_filess = list(subject_dict.values())

    if len(t1w_filess) <= 0 or len(subject_ids) <= 0:
        logging_wf.warning(f'len(subject_ids == 0)')
        return

    # 设置log目录位置  # TODO 增加到 settings [log] 下
    log_dir = workflow_cached_dir / f'log_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    log_dir.mkdir(parents=True, exist_ok=True)
    config.update_config({'logging': {'log_directory': log_dir,
                                      'log_to_file': True}})
    logging.update_logging(config)

    # TODO force_recon 如果开启这个参数，强制重新跑recon结果，清理 'IsRunning.lh+rh' 文件
    force_recon = settings.DANGER.FORCE_RECON
    if force_recon:
        clear_is_running(subjects_dir=subjects_dir,
                         subject_ids=subject_ids)

    lock = Lock()
    with Manager() as share_manager:
        scheduler = Scheduler(share_manager, subject_ids,
                              last_node_name=last_node_name,
                              auto_schedule=auto_schedule,
                              settings=settings)

        # TODO 初始化 node 的逻辑梳理为一个函数放到 create_node.py 中
        if settings.BOLD_ONLY:
            # TODO： 如果 BOLD_ONLY，需要检测Recon结果是否存在。如果不存在，那么启动后无法完整运行，直接终止。
            scheduler.last_node_name = last_node_name_bold
            from interface.create_node_bold_new import create_BoldSkipReorient_node
            for subject_id in subject_ids:
                node = create_BoldSkipReorient_node(subject_id=subject_id, task=task, atlas_type=atlas_type,
                                                    preprocess_method=preprocess_method, settings=settings)
                scheduler.node_all[node.name] = node
                scheduler.nodes_ready.append(node.name)

        elif settings.RECON_ONLY:
            scheduler.last_node_name = last_node_name_recon
            for subject_id, t1w_files in zip(subject_ids, t1w_filess):
                node = create_OrigAndRawavg_node(subject_id=subject_id, t1w_files=t1w_files, settings=settings)
                scheduler.node_all[node.name] = node
                scheduler.nodes_ready.append(node.name)
        else:
            from interface.create_node_bold_new import create_BoldSkipReorient_node
            for subject_id, t1w_files in zip(subject_ids, t1w_filess):
                node = create_OrigAndRawavg_node(subject_id=subject_id, t1w_files=t1w_files, settings=settings)
                scheduler.node_all[node.name] = node
                scheduler.nodes_ready.append(node.name)
                node = create_BoldSkipReorient_node(subject_id, task, atlas_type, preprocess_method, settings=settings)
                scheduler.node_all[node.name] = node
                scheduler.nodes_ready.append(node.name)

        scheduler.run(lock)
        logging_wf.info(f'subject_success {len(scheduler.subject_success)}: {scheduler.subject_success}')
        logging_wf.info(f'subject_start_datetime  {scheduler.start_datetime}')
        logging_wf.info(f'subject_success_datetime {len(scheduler.subject_success_datetime)}:'
                        f' {scheduler.subject_success_datetime}')
        logging_wf.error(f'nodes_error {len(scheduler.s_nodes_error)}: {scheduler.s_nodes_error}')
        logging_wf.error(f'subject_error {len(scheduler.subject_error)}: {scheduler.subject_error}')
        if settings.DANGER.CHEAR_BOLD_CACHE_DIR:
            clear_subject_bold_tmp_dir(bold_preprocess_dir, subject_ids, task)


if __name__ == '__main__':
    from config import settings as main_settings

    main(main_settings)
