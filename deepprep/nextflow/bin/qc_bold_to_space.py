import argparse
import shutil

import os
from pathlib import Path
import base64
from wand.image import Image
import nibabel as nib

svg_img_head = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" viewBox="0 0 {view_height} {view_width}" preserveAspectRatio="xMidYMid meet">'

svg_img_style = '''<style type="text/css">
@keyframes flickerAnimation5f434eec-e6f9-4af9-84b5-cc416fafbe79 { 0% {opacity: 1;} 100% { opacity: 0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite paused flickerAnimation5f434eec-e6f9-4af9-84b5-cc416fafbe79;}
.foreground-svg:hover { animation-play-state: running;}
</style>
'''

svg_img_body_2 = '''    <g class="background-svg">
      <g transform="translate(0, 0) scale(1.0 1.0) ">
      <image xlink:href="{back_encode}" />      
       <text x="10" y="40" font-family="Arial" font-size="50" fill="white">moved</text>
      </g>
    </g>
    <g class="foreground-svg">
      <g transform="translate(0, 0) scale(1.0 1.0) ">
      <image xlink:href="{fore_encode}" />
       <text x="10" y="40" font-family="Arial" font-size="50" fill="white">fixed</text>
      </g>
    </g>
'''
svg_img_tail = '</svg>'


def parse_args():
    parser = argparse.ArgumentParser(description="plot subject bold space fig")
    parser.add_argument('--subject_id', help='输入的subjects id', required=True)
    parser.add_argument('--bold_id', help='输入的bold id', required=True)
    parser.add_argument('--fs_native_space', help='是否投到个体T1空间', required=True)
    parser.add_argument('--subjects_dir', help='subjects_dir', required=True)
    parser.add_argument('--bold_preprocess_path', help='bold preprocess', required=True)
    parser.add_argument('--qc_tool_package', help='qc画图的辅助文件包', required=True)
    parser.add_argument('--svg_outpath', help='输出的svg图片保存路径', required=True)
    parser.add_argument('--freesurfer_home', help='freesurfer home', required=True)
    args = parser.parse_args()

    return args


def scene_plot(scene_file, savepath, length, width):
    cmd = f'wb_command -show-scene {scene_file} 1  {savepath} {length} {width}'
    os.system(cmd)


def write_combine_svg(svg, png_back, png_fore, view_height, view_width):
    pngdata_back = encode_png(png_back)
    pngdata_fore = encode_png(png_fore)

    with open(svg, 'w') as f:
        f.write(svg_img_head.format(view_height=view_height, view_width=view_width))
        f.write(svg_img_style)
        f.write(svg_img_body_2.format(back_encode=pngdata_back, fore_encode=pngdata_fore))
        f.write(svg_img_tail)


def encode_png(png_img):
    img = Image(filename=png_img)
    image_data = img.make_blob(format='png')
    encoded = base64.b64encode(image_data).decode()
    pngdata = 'data:image/png;base64,{}'.format(encoded)
    return pngdata

def mgz2nii(mgz_file, nii_file):
    cmd = f'mri_convert {mgz_file} {nii_file}'
    os.system(cmd)


def surf_nii2gii(nii_surf, gii_surf):
    cmd = f'mris_convert {nii_surf} {gii_surf}'
    os.system(cmd)


def surface_apply_affine(giisurf_file, affine_mat):
    cmd = f'wb_command -surface-apply-affine {giisurf_file} {affine_mat} {giisurf_file}'
    os.system(cmd)

def rewrite_affine(surf_file, src_affine_mat):
    _, _, header_info = nib.freesurfer.read_geometry(surf_file, read_metadata=True)
    c_ras = header_info['cras']

    affine_list = [[1, 0, 0, c_ras[0]], [0, 1, 0, c_ras[1]], [0, 0, 1, c_ras[2]], [0, 0, 0, 1]]

    with open(src_affine_mat, 'w') as f:
        for line in affine_list:
            f.write(str(line)[1:-1].replace(',', '    ') + '\n')

    f.close()


def set_environ(freesurfer_home):
    # FreeSurfer
    os.environ['FREESURFER_HOME'] = freesurfer_home
    os.environ['SUBJECTS_DIR'] = f'{freesurfer_home}/subjects'
    os.environ['PATH'] = f'{freesurfer_home}/bin:' + os.environ['PATH']


if __name__ == '__main__':
    args = parse_args()

    subject_id = args.subject_id
    bold_id = args.bold_id
    fs_native_space = args.fs_native_space
    subjects_dir = args.subjects_dir
    bold_preprocess_dir = args.bold_preprocess_path
    qc_tool_package = args.qc_tool_package
    svg_outpath = args.svg_outpath
    freesurfer_home = args.freesurfer_home
    set_environ(freesurfer_home)

    subject_resultdir = Path(svg_outpath).parent
    if subject_resultdir.exists() is False:
        subject_resultdir.mkdir(parents=True, exist_ok=True)
    subject_bold2mni152_workdir = Path(subject_resultdir) / f'{bold_id}_bold2min152'
    subject_bold2mni152_workdir.mkdir(parents=True, exist_ok=True)

    lh_MNI152_white_gii = Path(qc_tool_package) / 'lh.MNI152.white.surf.gii'
    rh_MNI152_white_gii = Path(qc_tool_package) / 'rh.MNI152.white.surf.gii'
    lh_MNI152_pial_gii = Path(qc_tool_package) / 'lh.MNI152.pial.surf.gii'
    rh_MNI152_pial_gii = Path(qc_tool_package) / 'rh.MNI152.pial.surf.gii'
    mni152_norm = Path(qc_tool_package) / 'MNI152_norm.nii.gz'
    mni152_scene = Path(qc_tool_package) / 'plot_MNI152.scene'
    bold2mni152_scene = Path(qc_tool_package) / 'plot_bold2MNI152.scene'

    lh_MNI152_white_gii_tmp = subject_bold2mni152_workdir / 'lh.MNI152_tmp.white.surf.gii'
    rh_MNI152_white_gii_tmp = subject_bold2mni152_workdir / 'rh.MNI152_tmp.white.surf.gii'
    lh_MNI152_pial_gii_tmp = subject_bold2mni152_workdir / 'lh.MNI152_tmp.pial.surf.gii'
    rh_MNI152_pial_gii_tmp = subject_bold2mni152_workdir / 'rh.MNI152_tmp.pial.surf.gii'
    MNI152_norm_tmp = subject_bold2mni152_workdir / 'MNI152_norm.nii.gz'
    mni152_scene_tmp = subject_bold2mni152_workdir / 'plot_MNI152_tmp.scene'
    bold2mni152_scene_tmp = subject_bold2mni152_workdir / 'plot_bold2MNI152_tmp.scene'

    shutil.copyfile(lh_MNI152_white_gii, lh_MNI152_white_gii_tmp)
    shutil.copyfile(rh_MNI152_white_gii, rh_MNI152_white_gii_tmp)
    shutil.copyfile(lh_MNI152_pial_gii, lh_MNI152_pial_gii_tmp)
    shutil.copyfile(rh_MNI152_pial_gii, rh_MNI152_pial_gii_tmp)
    shutil.copyfile(mni152_norm, MNI152_norm_tmp)
    shutil.copyfile(mni152_scene, mni152_scene_tmp)
    shutil.copyfile(bold2mni152_scene, bold2mni152_scene_tmp)

    bold2mni152_src = Path(bold_preprocess_dir) / subject_id / 'func' / f'{bold_id}_skip_reorient_stc_mc_space-MNI152_T1_2mm_fframe.nii.gz'
    bold2mni152_trg = subject_bold2mni152_workdir / f'bold2MNI152_tmp_bold.nii.gz'
    shutil.copyfile(bold2mni152_src, bold2mni152_trg)

    bold2MNI152_savepath = subject_bold2mni152_workdir / f'{bold_id}_bold_to_MNI152_moved.png'
    MNI152_savepath = subject_bold2mni152_workdir / f'{bold_id}_MNI152_atlas_fixed.png'

    scene_plot(bold2mni152_scene_tmp, bold2MNI152_savepath, 2400, 1000)
    scene_plot(mni152_scene_tmp, MNI152_savepath, 2400, 1000)
    combine_svg_savepath = subject_resultdir / f'{bold_id}_desc-reg2MNI152_bold.svg'
    write_combine_svg(combine_svg_savepath, bold2MNI152_savepath, MNI152_savepath, 2400,
                      1000)
    shutil.rmtree(subject_bold2mni152_workdir)

    if fs_native_space == 'True':
        subject_bold2T1_workdir = Path(subject_resultdir) / f'{bold_id}_bold2T1'
        subject_bold2T1_workdir.mkdir(parents=True, exist_ok=True)

        affine_mat_atlas = Path(qc_tool_package) / 'affine.mat'
        affine_mat = subject_bold2T1_workdir / 'affine.mat'
        shutil.copyfile(affine_mat_atlas, affine_mat)
        T1_mgz = Path(subjects_dir) / subject_id / 'mri' / 'norm.mgz'
        T1_nii = subject_bold2T1_workdir / 'norm.nii.gz'
        mgz2nii(T1_mgz, T1_nii)
        lh_white = Path(subjects_dir) / subject_id / 'surf' / 'lh.white'
        rh_white = Path(subjects_dir) / subject_id / 'surf' / 'rh.white'
        lh_pial = Path(subjects_dir) / subject_id / 'surf' / 'lh.pial'
        rh_pial = Path(subjects_dir) / subject_id / 'surf' / 'rh.pial'

        lh_white_gii = subject_bold2T1_workdir / 'lh.white.surf.gii'
        rh_white_gii = subject_bold2T1_workdir / 'rh.white.surf.gii'
        surf_nii2gii(lh_white, lh_white_gii)
        surf_nii2gii(rh_white, rh_white_gii)
        lh_pial_gii = subject_bold2T1_workdir / 'lh.pial.surf.gii'
        rh_pial_gii = subject_bold2T1_workdir / 'rh.pial.surf.gii'
        surf_nii2gii(lh_pial, lh_pial_gii)
        surf_nii2gii(rh_pial, rh_pial_gii)
        rewrite_affine(lh_white, affine_mat)
        surface_apply_affine(lh_white_gii, affine_mat)
        surface_apply_affine(rh_white_gii, affine_mat)
        surface_apply_affine(lh_pial_gii, affine_mat)
        surface_apply_affine(rh_pial_gii, affine_mat)

        bold2T1_scene = Path(qc_tool_package) / 'plot_boldT1.scene'
        T1_scene = Path(qc_tool_package) / 'plot_T1.scene'
        bold2T1_scene_tmp = subject_bold2T1_workdir / 'plot_boldT1_tmp.scene'
        T1_scene_tmp = subject_bold2T1_workdir / 'plot_T1_tmp.scene'
        shutil.copyfile(bold2T1_scene, bold2T1_scene_tmp)
        shutil.copyfile(T1_scene, T1_scene_tmp)

        bold2mni152_src = Path(bold_preprocess_dir) / subject_id / 'func' / f'{bold_id}_skip_reorient_stc_mc_bbregister_space-native_2mm_fframe.nii.gz'
        bold2mni152_trg = subject_bold2T1_workdir / f'bold2T1_tmp_bold.nii.gz'
        shutil.copyfile(bold2mni152_src, bold2mni152_trg)

        bold2T1_savepath = subject_bold2T1_workdir / f'{bold_id}_bold_to_T1_moved.png'
        T1_savepath = subject_bold2T1_workdir / f'{bold_id}_T1_atlas_fixed.png'

        scene_plot(bold2T1_scene_tmp, bold2T1_savepath, 2400, 1000)
        scene_plot(T1_scene_tmp, T1_savepath, 2400, 1000)
        combine_svg_savepath = subject_resultdir / f'{bold_id}_desc-reg2native_bold.svg'
        write_combine_svg(combine_svg_savepath, bold2T1_savepath, T1_savepath, 2400,
                          1000)
        shutil.rmtree(subject_bold2T1_workdir)
