import cv2
import numpy as np

D_TYPE = np.float16


def found_alpha_channel(img):
    return np.expand_dims(img, axis=2) if len(img.shape) == 2 else img[..., :1]


def bounding_rect_area(chn_mask):
    sum_area = 0
    contours, hierarchy = cv2.findContours(chn_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for _, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        sum_area += w * h

    return sum_area


def calc_flaws_area(chn_mask):
    sum_area = 0
    contours, hierarchy = cv2.findContours(chn_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for _, cnt in enumerate(contours):
        sum_area += cv2.contourArea(cnt)

    return sum_area


def norm_to_heatmap(chn_norm):
    out_hvs = np.zeros((*(chn_norm.shape if len(chn_norm.shape) == 2 else chn_norm.shape[:2]), 3), np.uint8)
    out_hvs[..., 0] = (chn_norm * 120)[..., 0].astype(np.uint8)
    out_hvs[..., 1] = 172
    out_hvs[..., 2] = 230

    out_bgr = cv2.cvtColor(out_hvs, cv2.COLOR_HSV2BGR)

    return out_bgr


def eval_flaws_norm(gt, mark):
    chn_gt = found_alpha_channel(gt).astype(D_TYPE)
    chn_mask = found_alpha_channel(mark).astype(D_TYPE)

    chn_min = np.full(chn_mask.shape, dtype=D_TYPE, fill_value=-255)
    chn_max = np.full(chn_gt.shape, dtype=D_TYPE, fill_value=510)

    # 0 ~ 1
    chn_hvs = ((chn_gt - chn_mask) - chn_min) / chn_max

    return chn_hvs


def eval_flaws_mask(gt, mark):
    chn_gt = found_alpha_channel(gt).astype(D_TYPE)
    chn_mask = found_alpha_channel(mark).astype(D_TYPE)

    # -1 ~ 1
    img_flaws = chn_gt - chn_mask

    img_flaws[np.where(img_flaws < 0)] *= -1
    img_flaws[np.where(img_flaws > 1)] = 255
    # img_flaws = np.expand_dims(img_flaws, axis=2)

    gt_loss = calc_flaws_area(img_flaws) / bounding_rect_area(chn_mask)

    return gt_loss, img_flaws.astype(np.uint8)
