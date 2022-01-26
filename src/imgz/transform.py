import cv2
import numpy as np

from . import D_TYPE


def normalize(img, dtype=D_TYPE):
    return img.astype(dtype) / 255


def mask_dims(img):
    return np.expand_dims(img, axis=2) if len(img.shape) == 2 else img[..., :1]


def color_dims(img):
    return img if len(img.shape) == 3 else img[..., :3]


def concatenate(img, mask):
    return np.concatenate((color_dims(img), mask_dims(mask)), axis=2)


def daub(img, mask, bg_color=(144, 238, 144)):
    img = color_dims(img)
    mask = normalize(mask_dims(mask))

    bg_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(3):
        bg_img[..., i].fill(bg_color[i])

    result = img * mask + bg_img * (1 - mask)

    return result


def blend(img, mask, bg=None, norm=True, otype=np.uint8):
    mask = mask_dims(mask)

    if norm:
        mask = normalize(mask)

    output = img * mask if bg is None else img * mask + bg * (1 - mask)

    return output.astype(otype)


def smooth(img, mask, ksize=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=ksize)
    gt_mask = cv2.dilate(mask, kernel, iterations=1)
    blur = cv2.blur(img, ksize=ksize)
    output = blend(blur, gt_mask, bg=img)

    output = blend(img, mask, bg=output)

    return output
