import torch
import numpy as np


def whiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0 / (float(x.numel()) ** 0.5))
    y = (x - mean) / std_adj
    return y


def fixed_normalize(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def im_resample(img, sz):
    out_shape = (sz[0], sz[1])
    im_data = torch.nn.functional.interpolate(img, size=out_shape, mode="area")
    return im_data


def generate_bounding_box(reg, probs, scale, thresh):
    stride = 2
    cell_size = 12

    mask = probs >= thresh
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask.nonzero().float().flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cell_size - 1 + 1) / scale).floor()
    bounding_box = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)

    return bounding_box


def bb_reg(bounding_box, reg):
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = bounding_box[:, 2] - bounding_box[:, 0] + 1
    h = bounding_box[:, 3] - bounding_box[:, 1] + 1
    b1 = bounding_box[:, 0] + reg[:, 0] * w
    b2 = bounding_box[:, 1] + reg[:, 1] * h
    b3 = bounding_box[:, 2] + reg[:, 2] * w
    b4 = bounding_box[:, 3] + reg[:, 3] * h
    bounding_box[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return bounding_box


def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def re_rec(bbox_a):
    h = bbox_a[:, 3] - bbox_a[:, 1]
    w = bbox_a[:, 2] - bbox_a[:, 0]
    l = np.maximum(w, h)
    bbox_a[:, 0] = bbox_a[:, 0] + w * 0.5 - l * 0.5
    bbox_a[:, 1] = bbox_a[:, 1] + h * 0.5 - l * 0.5
    bbox_a[:, 2:4] = bbox_a[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bbox_a


def pad(total_boxes, w, h):
    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    e_x = total_boxes[:, 2].copy().astype(np.int32)
    e_y = total_boxes[:, 3].copy().astype(np.int32)

    x[np.where(x < 1)] = 1
    y[np.where(y < 1)] = 1
    e_x[np.where(e_x > w)] = w
    e_y[np.where(e_y > h)] = h

    return y, e_y, x, e_x
