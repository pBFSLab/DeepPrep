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


def parse_args():
    parser = argparse.ArgumentParser(description="plot subject mc tsnr fig")
    parser.add_argument('--subject_id', help='输入的subjects id', required=True)
    parser.add_argument('--bold_id', help='输入的bold id', required=True)
    parser.add_argument('--bold_preprocess_path', help='bold preprocess', required=True)
    parser.add_argument('--scene_file', help='画图所需要的scene文件', required=True)
    parser.add_argument('--color_bar_png', help='画图所需要的color bar png', required=True)
    parser.add_argument('--svg_outpath', help='输出的svg图片保存路径', required=True)
    args = parser.parse_args()

    return args


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


def combine_bar(output_tsnr_savepath, color_bar_png):
    image1 = image_plt.open(output_tsnr_savepath)
    image2 = image_plt.open(color_bar_png)

    width, height = image1.size
    _, height_ = image2.size

    new_image = image_plt.new('RGB', (width, height + height_))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, height))
    new_image.save(output_tsnr_savepath)

    image1.close()
    image2.close()


if __name__ == '__main__':
    args = parse_args()

    subject_id = args.subject_id
    bold_id = args.bold_id
    bold_preprocess_dir = args.bold_preprocess_path
    scene_file = args.scene_file
    color_bar_png = args.color_bar_png
    savepath_svg = args.svg_outpath

    subject_resultdir = Path(savepath_svg).parent
    if subject_resultdir.exists() is False:
        subject_resultdir.mkdir(parents=True, exist_ok=True)
    subject_workdir = Path(subject_resultdir) / f'{bold_id}_mctsnr'
    subject_workdir.mkdir(parents=True, exist_ok=True)

    cur_path = os.getcwd()

    bold_mc = Path(cur_path) / bold_preprocess_dir / subject_id / 'func' / f'{bold_id}_skip_reorient_stc_mc.nii.gz'
    mc_tsnr_path = Path(cur_path) / str(subject_workdir) / 'mc_tsnr.nii.gz'
    TSNR_test(bold_mc, mc_tsnr_path)
    mc_brainmask = str(bold_mc).replace('.nii.gz', '.anat.brainmask.nii.gz')
    rewrite_tsnr(mc_tsnr_path, mc_brainmask)
    output_tsnr_savepath = subject_workdir / f'{bold_id}_mc_tsnr.png'
    McTSNR_scene = subject_workdir / 'McTSNR.scene'
    if McTSNR_scene.exists() is False:
        shutil.copyfile(scene_file, McTSNR_scene)
    scene_plot(McTSNR_scene, output_tsnr_savepath, 2400, 1000)

    combine_bar(str(output_tsnr_savepath), str(color_bar_png))
    Surface_parc_savepath_svg = subject_resultdir / f'{bold_id}_desc-tsnr_bold.svg'
    write_single_svg(Surface_parc_savepath_svg, output_tsnr_savepath, 2400, 1000)
    shutil.rmtree(subject_workdir)
