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


def parse_args():
    parser = argparse.ArgumentParser(description="plot subject aseg fig")
    parser.add_argument('--subject_id', help='输入的subjects id', required=True)
    parser.add_argument('--subjects_dir', help='输入的subjects_dir文件', required=True)
    parser.add_argument('--affine_mat', help='surf转换格式配准的affine', required=True)
    parser.add_argument('--scene_file', help='画图所需要的scene文件', required=True)
    parser.add_argument('--svg_outpath', help='输出的sav图片保存路径', required=True)
    parser.add_argument('--freesurfer_home', help='freesurfer 的环境变量', default="/usr/local/freesurfer720",
                        required=False)
    args = parser.parse_args()

    return args


def mgz2nii(mgz_file, nii_file):
    cmd = f'mri_convert {mgz_file} {nii_file}'
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


def rewrite_affine(surf_file, src_affine_mat):
    _, _, header_info = nib.freesurfer.read_geometry(surf_file, read_metadata=True)
    c_ras = header_info['cras']

    affine_list = [[1, 0, 0, c_ras[0]], [0, 1, 0, c_ras[1]], [0, 0, 1, c_ras[2]], [0, 0, 0, 1]]

    with open(src_affine_mat, 'w') as f:
        for line in affine_list:
            f.write(str(line)[1:-1].replace(',', '    ') + '\n')

    f.close()


if __name__ == '__main__':
    args = parse_args()

    subject_id = args.subject_id
    subjects_dir = args.subjects_dir
    affine_mat = args.affine_mat
    scene_file = args.scene_file
    savepath_svg = args.svg_outpath
    freesurfer_home = args.freesurfer_home
    set_environ(freesurfer_home)

    subject_resultdir = Path(savepath_svg).parent
    if subject_resultdir.exists() is False:
        subject_resultdir.mkdir(parents=True, exist_ok=True)
    subject_workdir = Path(subject_resultdir) / 'volsurf'
    subject_workdir.mkdir(parents=True, exist_ok=True)

    T1_mgz = Path(subjects_dir) / subject_id / 'mri' / 'T1.mgz'
    T1_nii = subject_workdir / 'T1.nii.gz'
    mgz2nii(T1_mgz, T1_nii)
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

    Vol_Surface_savepath = subject_workdir / 'Vol_Surface.png'
    Vol_Surface_scene = subject_workdir / 'Vol_Surface.scene'
    if Vol_Surface_scene.exists() is False:
         shutil.copyfile(scene_file, Vol_Surface_scene)
    scene_plot(Vol_Surface_scene, Vol_Surface_savepath, 2400, 1000)
    Vol_Surface_savepath_svg = subject_resultdir / 'Vol_Surface.svg'
    write_single_svg(Vol_Surface_savepath_svg, Vol_Surface_savepath, 2400, 1000)
    shutil.rmtree(subject_workdir)