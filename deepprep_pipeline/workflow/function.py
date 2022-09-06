from nipype import Node, Function


class Common:
    """
    https://mqhfidmks7.feishu.cn/file/boxcnyx3CsXzwFI99hPI6pAeA2d?office_edit=1
    """
    def __init__(self, cpu: int, gpu: bool, rewrite=True):
        self.rewrite = rewrite
        self.cpu = cpu
        self.gpu = gpu
        self.inputs = {}
        self.outputs = {}
        self.outputs_node = None
        self.time = 0  # self.run大概需要的时间,单位 m
        self.cpu_single = 0  # self.run大概需要的时间
        self.parallel = 0  # 并行数量，由self.cpu和self.cpu_single计算得到

    def input_exist_check(self):
        """检查输入文件是否都存在
        all 存在返回True
        否则返回False
        """
        self.inputs = {}
        if all_check():
            return True
        else:
            return False

    def output_exist_check(self):
        """检查输出文件是否都存在
        all 存在返回True
        否则返回False
        """
        self.outputs = {}
        if all_check():
            return True
        else:
            return False

    def run(self):
        """核心计算流程
        """
        pass

    def set_config(self):
        """根据设置的CPU数量、GPU是否可用、本身是否支持线程设置、是否可以并行
        计算cpu使用量、GPU使用量
        """
        pass

    def outputnode_data_select(self):
        """如果需要，设置outputsnode属性值，下一个流程使用
        """

    def stop_and_raise_check_error(self, filelist):
        raise FileExistsError(f'期望的文件不存在：{filelist}')

    def interface(self):
        if self.output_exist_check():
            if self.rewrite:
                self.process()
            else:
                self.output_exist_check()
        else:
            self.process()

    def process(self):
        if self.input_exist_check():
            self.set_config()
            self.run()
            if not self.output_exist_check():
                self.stop_and_raise_check_error(self.outputs)
        else:
            self.stop_and_raise_check_error(self.inputs)
