import os
import numpy as np
import torch
from torchvision.transforms import functional as F
from collections.abc import Iterable
from utils import *
from PIL import Image


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]

    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img.size()[0])),
        int(min(box[3] + margin[1] / 2, img.size()[1])),
    ]
    # TODO: optimize
    face = Image.fromarray(img.numpy()).crop(box).resize((image_size, image_size), 2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_args = {"compress_level": 0} if ".png" in save_path else {}
        face.save(save_path, **save_args)

    face = F.to_tensor(np.float32(face))
    return face


def detect_face(imgs,  pnet, rnet, onet, threshold = [0.6, 0.7, 0.7], factor=0.709, device='cpu',minsize=160,):
    if not isinstance(imgs, Iterable):
        imgs = [imgs]

    imgs = [torch.as_tensor(np.uint8(img)).float().to(device) for img in imgs]
    imgs = torch.stack(imgs).permute(0, 3, 1, 2)

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    total_boxes_all = [np.empty((0, 9)) for i in range(batch_size)]
    scale = m
    while minl >= 12:
        hs = int(h * scale + 1)
        ws = int(w * scale + 1)
        im_data = im_resample(imgs, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)


        for b_i in range(batch_size):
            boxes = generate_bounding_box(reg[b_i], probs[b_i, 1], scale, threshold[0]).detach().numpy()

            # inter-scale nms
            pick = nms(boxes, 0.5, "Union")
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes_all[b_i] = np.append(total_boxes_all[b_i], boxes, axis=0)

        scale = scale * factor
        minl = minl * factor



    batch_boxes = []
    batch_points = []
    for img, total_boxes in zip(imgs, total_boxes_all):
        points = np.zeros((2, 5, 0))
        numbox = total_boxes.shape[0]
        if numbox > 0:
            pick = nms(total_boxes, 0.7, "Union")
            total_boxes = total_boxes[pick, :]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = re_rec(total_boxes)
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            y, ey, x, ex = pad(total_boxes, w, h)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            im_data = []
            for k in range(0, numbox):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[:, (y[k] - 1): ey[k], (x[k] - 1): ex[k]].unsqueeze(0)
                    im_data.append(im_resample(img_k, (24, 24)))
            im_data = torch.cat(im_data, 0)
            im_data = (im_data - 127.5) * 0.0078125
            out = rnet(im_data)

            out0 = np.transpose(out[0].detach().numpy())
            out1 = np.transpose(out[1].detach().numpy())
            score = out1[1, :]
            ipass = np.where(score > threshold[1])
            total_boxes = np.hstack(
                [total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)]
            )
            mv = out0[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                pick = nms(total_boxes, 0.7, "Union")
                total_boxes = total_boxes[pick, :]
                total_boxes = bb_reg(total_boxes.copy(), np.transpose(mv[:, pick]))
                total_boxes = re_rec(total_boxes.copy())


        numbox = total_boxes.shape[0]

        if numbox > 0:
            total_boxes = np.fix(total_boxes).astype(np.int32)
            y, ey, x, ex = pad(total_boxes.copy(), w, h)
            im_data = []
            for k in range(0, numbox):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[:, (y[k] - 1): ey[k], (x[k] - 1): ex[k]].unsqueeze(0)
                    im_data.append(im_resample(img_k, (48, 48)))
            im_data = torch.cat(im_data, 0)
            im_data = (im_data - 127.5) * 0.0078125
            out = onet(im_data)

            out0 = np.transpose(out[0].detach().numpy())
            out1 = np.transpose(out[1].detach().numpy())
            out2 = np.transpose(out[2].detach().numpy())
            score = out2[1, :]
            points = out1
            ipass = np.where(score > threshold[2])
            points = points[:, ipass[0]]
            total_boxes = np.hstack(
                [total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)]
            )
            mv = out0[:, ipass[0]]

            w_i = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h_i = total_boxes[:, 3] - total_boxes[:, 1] + 1
            points_x = (
                    np.tile(w_i, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
            )
            points_y = (
                    np.tile(h_i, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
            )
            points = np.stack((points_x, points_y), axis=0)
            if total_boxes.shape[0] > 0:
                total_boxes = bb_reg(total_boxes, np.transpose(mv))
                pick = nms(total_boxes, 0.7, "Min")
                total_boxes = total_boxes[pick, :]
                points = points[:, :, pick]

        batch_boxes.append(total_boxes)
        batch_points.append(np.transpose(points))

    return np.array(batch_boxes), np.array(batch_points)

def image_to_tensor(img):
    img = img[np.newaxis, :]
    img = torch.from_numpy(img)
    return img


import cv2
from pnet import *
from rnet import *
from onet import *
pnet = PNet()
rnet = RNet()
onet = ONet()

device = torch.device('cpu')
img = cv2.imread('./photo.jpg')

res =detect_face( image_to_tensor(img) , pnet, rnet, onet)
print(res)


