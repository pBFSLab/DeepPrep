#! /usr/bin/env python3
import argparse
import shutil

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


def get_space_t1w_bold(bids_orig, bids_preproc, bold_orig_file, space_template):
    from bids import BIDSLayout
    layout_orig = BIDSLayout(bids_orig, validate=False)
    layout_preproc = BIDSLayout(bids_preproc, validate=False)
    info = layout_orig.parse_file_entities(bold_orig_file)

    space_template_t1w_info = info.copy()
    space_template_t1w_info['suffix'] = 'boldref'
    space_template_t1w_info['space'] = space_template
    space_template_t1w_info['extension'] = 'nii.gz'
    space_template_file = layout_preproc.get(**space_template_t1w_info)[0]

    return space_template_file.path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="plot subject bold space fig")
    parser.add_argument('--subject_id', help='输入的subjects id', required=True)
    parser.add_argument('--bold_id', help='输入的bold id', required=True)
    parser.add_argument('--bids_dir', required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument('--bold_preprocess_path', required=True)
    parser.add_argument('--space_template', help='space_mni152_bold_path', required=True)
    parser.add_argument('--qc_result_path', help='QC result path', required=True)
    parser.add_argument('--qc_tool_package', help='qc画图的辅助文件包', required=True)
    args = parser.parse_args()

    subject_id = args.subject_id
    bold_id = args.bold_id
    qc_tool_package = args.qc_tool_package
    space_template = args.space_template

    subject_resultdir = Path(args.qc_result_path) / subject_id / 'figures'
    subject_resultdir.mkdir(parents=True, exist_ok=True)
    subject_bold2mni152_workdir = Path(subject_resultdir) / f'{bold_id}_bold2{space_template}'
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

    with open(args.bold_file, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_orig_file = data[1]

    bold_space_template_file = get_space_t1w_bold(args.bids_dir, args.bold_preprocess_dir, bold_orig_file,
                                                  space_template)

    bold2mni152_trg = subject_bold2mni152_workdir / f'bold2MNI152_tmp_bold.nii.gz'
    shutil.copyfile(bold_space_template_file, bold2mni152_trg)

    bold2MNI152_savepath = subject_bold2mni152_workdir / f'{bold_id}_bold_to_MNI152_moved.png'
    MNI152_savepath = subject_bold2mni152_workdir / f'{bold_id}_MNI152_atlas_fixed.png'

    scene_plot(bold2mni152_scene_tmp, bold2MNI152_savepath, 2400, 1000)
    scene_plot(mni152_scene_tmp, MNI152_savepath, 2400, 1000)
    combine_svg_savepath = subject_resultdir / f'{bold_id}_desc-reg2MNI152_bold.svg'
    print(f'>>> {combine_svg_savepath}')
    write_combine_svg(combine_svg_savepath, bold2MNI152_savepath, MNI152_savepath, 2400,
                      1000)
    shutil.rmtree(subject_bold2mni152_workdir)
