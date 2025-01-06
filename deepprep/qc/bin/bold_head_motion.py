import os
import subprocess
import shutil
import numpy as np
from nipype.algorithms.confounds import FramewiseDisplacement
from pathlib import Path
import argparse
from bids import BIDSLayout


def calculate_mcflirt_motion_parameters(input_nii_path, output_dir=None):
    """
    使用FSL的MCFLIRT工具计算头动参数，并在完成后删除由 -out 生成的文件。

    参数:
    input_nii_path (str): 输入的BIDS格式的NIfTI文件路径。
    output_dir (str): 输出目录路径，默认为输入文件所在目录。

    返回:
    str: 头动参数文件的路径。
    """
    # 确保输入文件存在
    if not os.path.exists(input_nii_path):
        raise FileNotFoundError(f"输入文件 {input_nii_path} 不存在")

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_nii_path)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 构建输出文件路径
    input_filename = Path(input_nii_path).stem.replace('.nii', '')
    output_prefix = os.path.join(output_dir, f"{input_filename}_mcf")

    # 构建MCFLIRT命令
    mcflirt_cmd = [
        'mcflirt',
        '-in', input_nii_path,
        '-out', output_prefix,
        '-plots',  # 保存变换参数
        '-report'  # 报告进度
    ]

    # 执行MCFLIRT命令
    try:
        subprocess.run(mcflirt_cmd, check=True)
        print(f"MCFLIRT成功执行，结果保存在 {output_prefix}")
    except subprocess.CalledProcessError as e:
        print(f"MCFLIRT执行失败: {e}")
        return None

    # 头动参数文件路径
    motion_params_file = f"{output_prefix}.par"

    # 清理由 -out 生成的文件，只保留 .par 文件
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if file_path.startswith(output_prefix) and not file_path.endswith(".par"):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    print(f"已清理由 -out 生成的文件，只保留 {motion_params_file}")

    return motion_params_file


def calculate_framewise_displacement(par_file, output_dir=None):
    """
    使用Nipype计算帧位移（FD）并保存结果，包括绘图。

    参数:
    par_file (str): MCFLIRT生成的头动参数文件路径。
    output_dir (str): 输出目录路径，默认为输入文件所在目录。

    返回:
    str: FD结果文件的路径。
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(par_file)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 构建输出文件路径
    output_fd_file = os.path.join(output_dir, os.path.basename(par_file).replace('.par', '_fd.txt'))
    output_figure = os.path.join(output_dir, os.path.basename(par_file).replace('.par', '_fd_plot.png'))

    # 使用Nipype计算FD
    fd_calculator = FramewiseDisplacement()
    fd_calculator.inputs.in_file = par_file
    fd_calculator.inputs.parameter_source = 'FSL'  # 指定参数来源为FSL
    fd_calculator.inputs.save_plot = True  # 开启绘图
    fd_calculator.inputs.out_figure = output_figure  # 设置绘图文件路径
    fd_calculator.inputs.out_file = output_fd_file

    # 运行FD计算
    fd_result = fd_calculator.run()

    print(f"FD计算完成，结果保存在 {output_fd_file}")
    print(f"绘图结果保存在 {output_figure}")
    return output_fd_file


def get_bids_output_dir(input_nii_path, base_output_dir):
    """
    根据输入的 BOLD 文件路径构建 BIDS 格式的输出目录。

    参数:
    input_nii_path (str): 输入的 BOLD 文件路径。
    base_output_dir (str): 基础输出目录。

    返回:
    str: BIDS 格式的输出目录路径。
    """
    # 解析输入文件的 BIDS 实体
    layout = BIDSLayout(os.path.dirname(input_nii_path), validate=False)
    bids_file = layout.get_file(input_nii_path)
    relpath = bids_file.relpath  # 获取相对路径

    # 构建 BIDS 格式的输出目录
    output_dir = os.path.join(base_output_dir, os.path.dirname(relpath))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing Workflows -- Head Motion Calculation\n"
                    "This script calculates head motion parameters and framewise displacement (FD) for BOLD fMRI data."
    )

    parser.add_argument("--bold_series", required=True, nargs='+',
                        help="Path to the file containing the BOLD fMRI file path.")
    parser.add_argument("--output_dir", required=True,
                        help="Base path to the output directory where results will be saved.")
    parser.add_argument("--work_dir", required=True,
                        help="Path to the working directory for intermediate files.")
    parser.add_argument("--freesurfer_home", required=False, default="/opt/freesurfer")
    parser.add_argument("--fsl_home", required=False, default="/opt/fsl")
    args = parser.parse_args()

    def set_environ(freesurfer_home, fsl_home):
        # FreeSurfer
        os.environ['FREESURFER_HOME'] = freesurfer_home
        os.environ['PATH'] = f'{freesurfer_home}/bin:' + f'{fsl_home}/bin:' + os.environ['PATH']
    if args.freesurfer and args.workbench:
        set_environ(args.freesurfer_home, args.fsl_home)

    # 从文件中读取 BOLD 文件路径
    with open(args.bold_series[0], 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    input_nii_path = data[1]  # 假设 BOLD 文件路径在文件的第二行

    # 设置工作目录
    os.makedirs(args.work_dir, exist_ok=True)

    # 根据输入文件构建 BIDS 格式的输出目录
    bids_output_dir = get_bids_output_dir(input_nii_path, args.output_dir)

    # 计算头动参数
    motion_params_file = calculate_mcflirt_motion_parameters(input_nii_path, args.work_dir)

    if motion_params_file:
        print(f"头动参数文件已生成: {motion_params_file}")
        # 计算帧位移（FD）
        fd_file = calculate_framewise_displacement(motion_params_file, bids_output_dir)
        if fd_file:
            print(f"FD结果文件已生成: {fd_file}")
        else:
            print("FD计算失败")
    else:
        print("头动参数文件生成失败")


if __name__ == "__main__":
    main()
