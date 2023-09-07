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


def parse_args():
    parser = argparse.ArgumentParser(description="plot subject norm to mni152 fig")
    parser.add_argument('--subject_id', help='输入的subjects id', required=True)
    parser.add_argument('--bold_preprocess_path', help='bold preprocess', required=True)
    parser.add_argument('--scene_file', help='画图所需要的scene文件', required=True)
    parser.add_argument('--mni152_norm_png', help='模板MNI152的norm png图片', required=True)
    parser.add_argument('--svg_outpath', help='输出的svg图片保存路径', required=True)
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


if __name__ == '__main__':
    args = parse_args()

    subject_id = args.subject_id
    bold_preprocess_dir = args.bold_preprocess_path
    scene_file = args.scene_file
    mni152_norm_png = args.mni152_norm_png
    savepath_svg = args.svg_outpath

    subject_resultdir = Path(savepath_svg).parent
    if subject_resultdir.exists() is False:
        subject_resultdir.mkdir(parents=True, exist_ok=True)
    subject_workdir = Path(subject_resultdir) / 'norm2mni152'
    subject_workdir.mkdir(parents=True, exist_ok=True)

    norm_to_mni152nii = Path(
        bold_preprocess_dir) / subject_id / 'anat' / f'{subject_id}_norm_space-MNI152_T1_2mm.nii.gz'
    norm_to_mni152nii_tmp = subject_workdir / 'norm_to_mni152nii.nii.gz'
    shutil.copyfile(norm_to_mni152nii, norm_to_mni152nii_tmp)

    NormtoMNI152_savepath = subject_workdir / 'NormtoMNI152.png'
    NormtoMNI152_scene = subject_workdir / 'NormtoMNI152.scene'
    if NormtoMNI152_scene.exists() is False:
        shutil.copyfile(scene_file, NormtoMNI152_scene)
    scene_plot(NormtoMNI152_scene, NormtoMNI152_savepath, 2400, 1000)
    combine_svg_savepath = subject_resultdir / f'{subject_id}_desc-T1toMNI152_combine.svg'
    write_combine_svg(combine_svg_savepath, mni152_norm_png, NormtoMNI152_savepath, 2400, 1000)
    if subject_workdir.exists():
        shutil.rmtree(subject_workdir)
