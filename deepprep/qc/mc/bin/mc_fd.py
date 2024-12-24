from nipype.algorithms.confounds import FramewiseDisplacement
from nipype import Node
from pathlib import Path

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class FramewiseDisplacementNode:
    def __call__(self, mcpar, base_dir):
        FramewiseDisplacement_node = Node(FramewiseDisplacement(), f'FramewiseDisplacement_node')
        FramewiseDisplacement_node.inputs.in_file = mcpar
        FramewiseDisplacement_node.inputs.parameter_source = 'FSL'
        FramewiseDisplacement_node.base_dir = base_dir
        result = FramewiseDisplacement_node.run()
        return os.path.abspath(result.outputs.out_file)


def plot_motion_params(data, output_image):
    # 提取运动参数
    timepoints = np.arange(data.shape[0])
    translation_x = data[:, 0]

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制平移参数
    plt.subplot(3, 1, 1)
    plt.plot(timepoints, translation_x, label='FD')
    plt.xlabel('Timepoint')
    plt.ylabel('FD')
    plt.legend()
    plt.grid(True)


    # 保存图像
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()


if __name__ == '__main__':
    sub_id = '*'
    mcpar_file = './*.par'
    workdir = Path('./')
    plot_dir = Path('./')

    sub_workdir = workdir / sub_id
    sub_workdir.mkdir(exist_ok=True, parents=True)

    output_image = plot_dir / os.path.basename(mcpar_file).replace('par', 'png')
    framewisedisplacement = FramewiseDisplacementNode()

    fd_path = framewisedisplacement(mcpar_file, sub_workdir)
    fd = pd.read_csv(fd_path, sep='\t', encoding='utf-8').values
    plot_motion_params(fd, output_image)
    mcfs_mean = np.mean(fd)
    print(mcfs_mean)


