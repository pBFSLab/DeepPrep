import os
from pathlib import Path
import multiprocessing
from collections import OrderedDict
from nipype import Node
from interface.freesurfer_node import OrigAndRawavg
from interface.create_node import create_origandrawavg_node


class SubjectQueue:
    def __init__(self, subject_id, subjects_dir):
        self.subject_id = subject_id
        self.subjects_dir = subjects_dir

        self.node_ready = []  # 待执行的Node
        self.node_error = []  # 执行错误的Node

    def init_first_node(self):
        node = create_origandrawavg_node(subject_id, subjects_dir)
        self.node_ready.append(node)


class SubjectNode:
    """
    这个Node是Interface的Node
    """
    def __init__(self):
        self.source = Source()

    def postprocess(self, subject: SubjectQueue):
        if self.node_run_success:
            self.create_sub_node(subject.node_ready)
        else:
            self.interp(subject.node_error)

    def node_run_success(self):
        """
        在执行node.run()以后，判断node是否完整运行
        """
        return True or False

    def create_sub_node(self):
        return node

    def last_node(self):
        """
        在Queue中删除自己的subject
        # TODO 这个逻辑放到哪不清楚
        """
        pass


class Queue:
    def __init__(self, subject_ids):

        self.subjects = OrderedDict()

        for subject_id in subject_ids:
            subject = SubjectQueue(subject_id)
            subject.set_first_node()
            self.subjects[subject_id] = SubjectQueue(subject_id)


class Source:

    def __init__(self, CPU_n, GPU_MB, RAM_MB, IO_write_MB, IO_read_MB):
        self.CPU_n = 36
        self.GPU_MB = 24000
        self.RAM_MB = 60000
        self.IO_write_MB = 150
        self.IO_read_MB = 450

    def add(self, source):
        # TODO

        return Source(*add_result)

    def sub(self, source):
        # TODO
        return Source(*sub_result)


class Scheduler:
    def __init__(self, subject_ids: list):
        self.source_res = Source(36, 24000, 60000, 150, 450)

        self.queue = Queue(subject_ids)

        self.pool = Pool()

    def check_node(self, node: Node):
        source_result = self.source_res - node.source
        for i in source_result:
            if source_result[i] < 0:
                return False
        return True

    def run_node(self):
        """
        1. node执行完毕
        2. 有新的node进入队列
        """
        queue = self.queue

        for subject_queue in queue.subjects:
            for node in subject_queue.nodes_ready:
                if self.check_node(node):
                    subject_queue.node_ready.pop(node)
                    self.source_res -= node.source
                    self.pool.sync_run(node)  # 子进程 run


if __name__ == '__main__':
    scheduler = Scheduler(subjects_ids)
    scheduler.run_node()


