import argparse
import shutil
import os
from pathlib import Path
import base64
from wand.image import Image
import nibabel as nib
from nipype.algorithms.confounds import TSNR
from nipype import Node
import numpy as np
from PIL import Image as image_plt


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


def TSNR_test(inputs_file, output_file):
    TSNR_node = Node(TSNR(), 'TSNR_node')

    TSNR_node.inputs.in_file = inputs_file
    TSNR_node.inputs.tsnr_file = output_file

    TSNR_node.run()


def rewrite_tsnr(output_path, mc_brainmask):
    src_tsnr_data = nib.load(output_path).get_fdata()
    brainmask_data = nib.load(mc_brainmask).get_fdata()

    src_tsnr_data[brainmask_data == 0] = 0
    trg_tsnr_img = nib.Nifti1Image(src_tsnr_data, affine=np.eye(4))

    nib.save(trg_tsnr_img, output_path)


def set_environ(freesurfer_home, subjects_dir):
    # FreeSurfer
    os.environ['FREESURFER_HOME'] = freesurfer_home
    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['PATH'] = f'{freesurfer_home}/bin:' + '/usr/local/workbench/bin_linux64:' + os.environ['PATH']


def vol2surf(vol_file, reg_dat, hemi, out_file, interp_type):
    cmd = f'mri_vol2surf --mov {vol_file} --reg {reg_dat} --hemi {hemi} --projfrac 0.5 --o {out_file} --interp {interp_type}'
    os.system(cmd)


def surf2surf(hemi, subject_id, native_mgh, fs6_mgh):
    cmd = f'mri_surf2surf --hemi {hemi} --srcsubject {subject_id} --srcsurfval {native_mgh} --src_type curv --trgsubject fsaverage6 --trgsurfval {fs6_mgh} --trg_type curv'
    os.system(cmd)


def surf_nii2gii(nii_surf, gii_surf):
    cmd = f'mris_convert {nii_surf} {gii_surf}'
    os.system(cmd)


def nii2gii(nii_file, gii_file):
    cmd = f'mri_convert {nii_file} {gii_file}'
    os.system(cmd)


def combine_bar(image_png, color_bar):
    image1 = image_plt.open(image_png)
    image2 = image_plt.open(color_bar)

    width, height = image1.size
    _, height_ = image2.size
    new_image = image_plt.new('RGB', (width, height + height_))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, height))

    image1.close()
    image2.close()

    return new_image, width, height + height_


def combine_png(image_1, image_2, color_bar, combine_png_path):
    image_1_color, width_1, height = combine_bar(image_1, color_bar)
    image_2_color, width_2, _ = combine_bar(image_2, color_bar)

    new_width = width_1 + width_2
    new_height = height
    new_image = image_plt.new('RGB', (new_width, new_height))
    new_image.paste(image_1_color, (0, 0))
    new_image.paste(image_2_color, (width_1, 0))
    new_image.save(combine_png_path)

    image_1_color.close()
    image_2_color.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="plot subject mc tsnr surf fig")
    parser.add_argument('--subject_id', help='输入的subjects id', required=True)
    parser.add_argument('--bold_id', help='输入的bold id', required=True)
    parser.add_argument('--subjects_dir', help='subjects dir', required=True)
    parser.add_argument('--bold_preprocess_path', help='bold preprocess', required=True)
    parser.add_argument('--qc_result_path', help='QC result path', required=True)
    parser.add_argument('--fs6_scene_file', help='画图所需要的scene文件', required=True)
    parser.add_argument('--native_scene_file', help='画图所需要的scene文件', required=True)
    parser.add_argument('--color_bar', help='画图所需要的color bar png', required=True)
    parser.add_argument('--svg_outpath', help='输出的svg图片保存路径', required=True)
    parser.add_argument('--freesurfer_home', help='freesurfer home', required=True)
    args = parser.parse_args()

    subject_id = args.subject_id
    bold_id = args.bold_id
    subjects_dir = args.subjects_dir
    bold_preprocess_dir = args.bold_preprocess_path
    fs6_scene_file = args.fs6_scene_file
    native_scene_file = args.native_scene_file
    color_bar = args.color_bar
    savepath_svg = args.svg_outpath
    freesurfer_home = args.freesurfer_home
    set_environ(freesurfer_home, subjects_dir)

    subject_resultdir = Path(args.qc_result_path) / subject_id / 'figures'
    subject_resultdir.mkdir(parents=True, exist_ok=True)
    subject_workdir = Path(subject_resultdir) / f'{bold_id}_mctsnr_surf'
    subject_workdir.mkdir(parents=True, exist_ok=True)

    cur_path = os.getcwd()

    bold_mc = Path(cur_path) / bold_preprocess_dir / subject_id / 'func' / f'{bold_id}_skip_reorient_stc_mc.nii.gz'
    bbreg_dat = Path(cur_path) / bold_preprocess_dir / subject_id / 'func' / f'{bold_id}_skip_reorient_stc_mc_from_mc_to_fsnative_bbregister_rigid.dat'
    mc_tsnr_path = Path(cur_path) / str(subject_workdir) / 'mc_tsnr_vol.nii.gz'
    TSNR_test(bold_mc, mc_tsnr_path)
    fs6_path = Path(freesurfer_home) / 'subjects' / 'fsaverage6'
    subject_workdir_fs6 = Path(subjects_dir) / 'fsaverage6'
    if subject_workdir_fs6.exists() is False:
        os.symlink(fs6_path, subject_workdir_fs6)

    hemis = ['lh', 'rh']
    for hemi in hemis:
        out_fs6_surf_file = subject_workdir / f'{hemi}.mctsnr_fs6.mgh'
        out_native_surf_file = subject_workdir / f'{hemi}.mctsnr_native.mgh'
        vol2surf(mc_tsnr_path, subject_id, subject_id, hemi, out_native_surf_file)
        vol2surf(mc_tsnr_path, bbreg_dat, hemi, out_native_surf_file, 'trilinear')
        surf2surf(hemi, subject_id, out_native_surf_file, out_fs6_surf_file)
        out_fs6_surf_gii_file = subject_workdir / f'{hemi}.mctsnr_fs6.shape.gii'
        out_native_surf_gii_file = subject_workdir / f'{hemi}.mctsnr_native.shape.gii'
        nii2gii(out_fs6_surf_file, out_fs6_surf_gii_file)
        nii2gii(out_native_surf_file, out_native_surf_gii_file)

    lh_fs6_pial = Path(subjects_dir) / 'fsaverage6' / 'surf' / 'lh.pial'
    rh_fs6_pial = Path(subjects_dir) / 'fsaverage6' / 'surf' / 'rh.pial'
    lh_fs6_pial_gii = subject_workdir / 'lh.pial_fs6.surf.gii'
    rh_fs6_pial_gii = subject_workdir / 'rh.pial_fs6.surf.gii'
    surf_nii2gii(lh_fs6_pial, lh_fs6_pial_gii)
    surf_nii2gii(rh_fs6_pial, rh_fs6_pial_gii)

    lh_native_pial = Path(subjects_dir) / subject_id / 'surf' / 'lh.pial'
    rh_native_pial = Path(subjects_dir) / subject_id / 'surf' / 'rh.pial'
    lh_native_pial_gii = subject_workdir / 'lh.pial_native.surf.gii'
    rh_native_pial_gii = subject_workdir / 'rh.pial_native.surf.gii'
    surf_nii2gii(lh_native_pial, lh_native_pial_gii)
    surf_nii2gii(rh_native_pial, rh_native_pial_gii)

    output_tsnr_surf_fs6_savepath = subject_workdir / f'{bold_id}_mc_tsnr_surf_fs6.png'
    mctsnr_surf_fs6_scene = subject_workdir / 'plot_mctsnr_surf_fs6.scene'
    if mctsnr_surf_fs6_scene.exists() is False:
        shutil.copyfile(fs6_scene_file, mctsnr_surf_fs6_scene)
    scene_plot(mctsnr_surf_fs6_scene, output_tsnr_surf_fs6_savepath, 1000, 800)

    output_tsnr_surf_native_savepath = subject_workdir / f'{bold_id}_mc_tsnr_surf_native.png'
    mctsnr_surf_native_scene = subject_workdir / 'plot_mctsnr_surf_native.scene'
    if mctsnr_surf_native_scene.exists() is False:
        shutil.copyfile(native_scene_file, mctsnr_surf_native_scene)
    scene_plot(mctsnr_surf_native_scene, output_tsnr_surf_native_savepath, 1000, 800)

    combine_png_path = subject_workdir / f'{bold_id}_mc_tsnr_surf_combine.png'
    combine_png(str(output_tsnr_surf_native_savepath), str(output_tsnr_surf_fs6_savepath), color_bar, str(combine_png_path))

    mctsnr_surf_savepath_svg = subject_resultdir / f'{bold_id}_desc-tsnr2surf_bold.svg'
    write_single_svg(mctsnr_surf_savepath_svg, combine_png_path, 2400, 1000)
    shutil.rmtree(subject_workdir)
