import cv2
from pathlib import Path


def read_image(img):
    """
    如是路径，使用opencv读取image
    如果已经是cv.img，直接返回
    """
    if isinstance(img, Path):
        img = str(img)
    if isinstance(img, str):
        print(img)
        img = cv2.imread(img)
        print(f"{img.shape}")
    return img


def concat_horizontal(images: list, img_save: str = None):
    """
    在水平方向对多张图像进行拼接，返回拼接的cv.img
    如果img_save is not None，则保存图像
    """
    imgs = [read_image(i) for i in images]
    img = cv2.hconcat(imgs)
    if img_save is not None:
        cv2.imwrite(img_save, img)
        print(f'>>> save image : {img_save}')
    return img


def concat_vertical(images: list, img_save: str = None):
    """
    在垂直方向对多张图像进行拼接，返回拼接的cv.img
    如果img_save is not None，则保存图像
    """
    imgs = [read_image(i) for i in images]
    img = cv2.vconcat(imgs)
    if img_save is not None:
        cv2.imwrite(img_save, img)
        print(f'>>> save image : {img_save}')
    return img
