import argparse
import shutil

import numpy as np
import nibabel as nib
import os
from pathlib import Path
import base64
from wand.image import Image

svg_img_head = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" viewBox="0 0 {view_height} {view_width}" preserveAspectRatio="xMidYMid meet">'

svg_img_style = '''<style type="text/css">
@keyframes flickerAnimation5f434eec-e6f9-4af9-84b5-cc416fafbe79 { 0% {opacity: 1;} 100% { opacity: 0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite paused flickerAnimation5f434eec-e6f9-4af9-84b5-cc416fafbe79;}
.foreground-svg:hover { animation-play-state: running;}
</style>
'''

svg_img_body_1 = '''    <g class="background-svg">
      <g transform="translate(0, 0) scale(1.0 1.0) ">
      <image xlink:href="{back_encode}" />
      </g>
    </g>
'''

svg_img_tail = '</svg>'


def set_environ(freesurfer_home):
    # FreeSurfer
    os.environ['FREESURFER_HOME'] = freesurfer_home
    os.environ['SUBJECTS_DIR'] = f'{freesurfer_home}/subjects'
    os.environ['PATH'] = f'{freesurfer_home}/bin:' + os.environ['PATH']


def parse_args():
    parser = argparse.ArgumentParser(description="plot subject aseg fig")
    parser.add_argument('--subject_id', help='输入的subjects id', required=True)
    parser.add_argument('--subjects_dir', help='输入的subjects_dir文件', required=True)
    parser.add_argument('--dlabel_info', help='aseg 转换分区的color info', required=True)
    parser.add_argument('--scene_file', help='画图所需要的scene文件', required=True)
    parser.add_argument('--svg_outpath', help='输出的sav图片保存路径', required=True)
    parser.add_argument('--freesurfer_home', help='freesurfer 的环境变量', default="/usr/local/freesurfer720",
                        required=False)
    args = parser.parse_args()

    return args


def mgz2nii(mgz_file, nii_file):
    cmd = f'mri_convert {mgz_file} {nii_file}'
    os.system(cmd)


def asge_nii2dlabel(aparc_asge_nii, dlabel_info, aparc_asge_cifti):
    cmd = f'wb_command -volume-label-import {aparc_asge_nii} {dlabel_info} {aparc_asge_cifti} -drop-unused-labels'
    os.system(cmd)


def scene_plot(scene_file, savepath, length, width):
    cmd = f'wb_command -show-scene {scene_file} 1  {savepath} {length} {width}'
    os.system(cmd)


def write_single_svg(svg, png_back, view_height, view_width):
    pngdata_back = encode_png(png_back)

    with open(svg, 'w') as f:
        f.write(svg_img_head.format(view_height=view_height, view_width=view_width))
        f.write(svg_img_style)
        f.write(svg_img_body_1.format(back_encode=pngdata_back))
        f.write(svg_img_tail)


def encode_png(png_img):
    img = Image(filename=png_img)
    image_data = img.make_blob(format='png')
    encoded = base64.b64encode(image_data).decode()
    pngdata = 'data:image/png;base64,{}'.format(encoded)
    return pngdata


if __name__ == '__main__':
    args = parse_args()

    subject_id = args.subject_id
    subjects_dir = args.subjects_dir
    dlabel_info_txt = args.dlabel_info
    scene_file = args.scene_file
    savepath_svg = args.svg_outpath
    freesurfer_home = args.freesurfer_home
    set_environ(freesurfer_home)

    subject_resultdir = Path(savepath_svg).parent
    if subject_resultdir.exists() is False:
        subject_resultdir.mkdir(parents=True, exist_ok=True)
    subject_workdir = Path(subject_resultdir) / 'aseg_aparc'
    subject_workdir.mkdir(parents=True, exist_ok=True)

    norm_mgz = Path(subjects_dir) / subject_id / 'mri' / 'norm.mgz'
    aseg_mgz = Path(subjects_dir) / subject_id / 'mri' / 'aparc+aseg.mgz'
    norm_nii = subject_workdir / 'norm.nii.gz'
    aparc_asge_nii = subject_workdir / 'aparc+aseg.nii.gz'
    mgz2nii(norm_mgz, norm_nii)
    mgz2nii(aseg_mgz, aparc_asge_nii)
    aparc_asge_cifti = subject_workdir / 'aparc+aseg_dlabel.nii.gz'
    asge_nii2dlabel(aparc_asge_nii, dlabel_info_txt, aparc_asge_cifti)
    png_savepath = subject_workdir / 'Volume_parc.png'
    Volume_parc_scene = subject_workdir / 'Volume_parc.scene'
    if Volume_parc_scene.exists() is False:
        shutil.copyfile(scene_file, Volume_parc_scene)
    scene_plot(Volume_parc_scene, png_savepath, 2400, 1000)
    Volume_parc_savepath_svg = subject_resultdir / 'Volume_parc.svg'
    write_single_svg(Volume_parc_savepath_svg, png_savepath, 2400, 1000)
    shutil.rmtree(subject_workdir)