#! /usr/bin/env python3
import argparse
import shutil

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


def label_nii2gii(nii_label, nii_white, gii_label):
    cmd = f'mris_convert --annot {nii_label} {nii_white} {gii_label}'
    os.system(cmd)


def surf_nii2gii(nii_surf, gii_surf):
    cmd = f'mris_convert {nii_surf} {gii_surf}'
    os.system(cmd)


def surface_apply_affine(giisurf_file, affine_mat):
    cmd = f'wb_command -surface-apply-affine {giisurf_file} {affine_mat} {giisurf_file}'
    os.system(cmd)


def scene_plot(scene_file, savepath, length, width):
    cmd = f'wb_command -show-scene {scene_file} 1  {savepath} {length} {width}'
    os.system(cmd)


def rewrite_affine(surf_file, src_affine_mat):
    _, _, header_info = nib.freesurfer.read_geometry(surf_file, read_metadata=True)
    c_ras = header_info['cras']

    affine_list = [[1, 0, 0, c_ras[0]], [0, 1, 0, c_ras[1]], [0, 0, 1, c_ras[2]], [0, 0, 0, 1]]

    with open(src_affine_mat, 'w') as f:
        for line in affine_list:
            f.write(str(line)[1:-1].replace(',', '    ') + '\n')

    f.close()


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
    parser = argparse.ArgumentParser(description="plot subject surf parc fig")
    parser.add_argument('--subject_id', help='输入的subjects id', required=True)
    parser.add_argument('--subjects_dir', help='输入的subjects dir文件', required=True)
    parser.add_argument('--qc_result_path', help='QC result path', required=True)
    parser.add_argument('--affine_mat', help='surf转换格式配准的affine', required=True)
    parser.add_argument('--scene_file', help='画图所需要的scene文件', required=True)
    parser.add_argument('--svg_outpath', help='输出的svg图片保存路径', required=True)
    parser.add_argument('--freesurfer_home', help='freesurfer 的环境变量', default="/usr/local/freesurfer720",
                        required=False)
    args = parser.parse_args()

    subject_id = args.subject_id
    subjects_dir = args.subjects_dir
    affine_mat_atlas = args.affine_mat
    scene_file = args.scene_file
    freesurfer_home = args.freesurfer_home
    set_environ(freesurfer_home)

    subject_resultdir = Path(args.qc_result_path) / subject_id / 'figures'
    subject_resultdir.mkdir(parents=True, exist_ok=True)
    subject_workdir = Path(subject_resultdir) / 'surfparc'
    subject_workdir.mkdir(parents=True, exist_ok=True)
    affine_mat = subject_workdir / 'affine.mat'
    shutil.copyfile(affine_mat_atlas, affine_mat)

    lh_white = Path(subjects_dir) / subject_id / 'surf' / 'lh.white'
    rh_white = Path(subjects_dir) / subject_id / 'surf' / 'rh.white'
    lh_pial = Path(subjects_dir) / subject_id / 'surf' / 'lh.pial'
    rh_pial = Path(subjects_dir) / subject_id / 'surf' / 'rh.pial'
    lh_white_gii = subject_workdir / 'lh.white.surf.gii'
    rh_white_gii = subject_workdir / 'rh.white.surf.gii'
    surf_nii2gii(lh_white, lh_white_gii)
    surf_nii2gii(rh_white, rh_white_gii)
    lh_pial_gii = subject_workdir / 'lh.pial.surf.gii'
    rh_pial_gii = subject_workdir / 'rh.pial.surf.gii'
    surf_nii2gii(lh_pial, lh_pial_gii)
    surf_nii2gii(rh_pial, rh_pial_gii)
    rewrite_affine(lh_white, affine_mat)
    surface_apply_affine(lh_white_gii, affine_mat)
    surface_apply_affine(rh_white_gii, affine_mat)
    surface_apply_affine(lh_pial_gii, affine_mat)
    surface_apply_affine(rh_pial_gii, affine_mat)

    lh_annot_label = Path(subjects_dir) / subject_id / 'label' / 'lh.aparc.annot'
    rh_annot_label = Path(subjects_dir) / subject_id / 'label' / 'rh.aparc.annot'
    lh_white = Path(subjects_dir) / subject_id / 'surf' / 'lh.white'
    rh_white = Path(subjects_dir) / subject_id / 'surf' / 'rh.white'
    lh_annot_label_gii = subject_workdir / 'lh.aparc.label.gii'
    rh_annot_label_gii = subject_workdir / 'rh.aparc.label.gii'
    label_nii2gii(lh_annot_label, lh_white, lh_annot_label_gii)
    label_nii2gii(rh_annot_label, rh_white, rh_annot_label_gii)

    Surface_parc_savepath = subject_workdir / 'Surface_parc.png'
    Surface_parc_scene = subject_workdir / 'Surface_parc.scene'
    if Surface_parc_scene.exists() is False:
        shutil.copyfile(scene_file, Surface_parc_scene)
    scene_plot(Surface_parc_scene, Surface_parc_savepath, 2400, 1000)
    Surface_parc_savepath_svg = subject_resultdir / f'{subject_id}_desc-surfparc_T1w.svg'
    print(f'>>> {Surface_parc_savepath_svg}')
    write_single_svg(Surface_parc_savepath_svg, Surface_parc_savepath, 2400, 1000)
    shutil.rmtree(subject_workdir)