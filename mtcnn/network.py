import torch
from torch import nn
import numpy as np
import os
from collections.abc import Iterable
from facerec.mtcnn import PNet, ONet, RNet
from facerec.mtcnn.face import detect_face, extract_face
from facerec.mtcnn.utils import fixed_normalize


class MTCNN(nn.Module):
    def __init__(
            self, image_size=160, margin=0, min_face_size=20,
            thresholds=None, factor=0.709, post_process=True,
            select_largest=True, keep_all=False, device=None
    ):
        super().__init__()

        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, img, save_path=None, return_prob=False):
        with torch.no_grad():
            batch_boxes, batch_probs = self.detect(img)

        batch_mode = True
        if not isinstance(img, Iterable):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_probs = [batch_probs]
            batch_mode = False

        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]

        faces, probs, face_boxes = [], [], []
        for im, box_im, prob_im, path_im in zip(img, batch_boxes, batch_probs, save_path):
            if box_im is None:
                faces.append(None)
                probs.append([None] if self.keep_all else None)
                face_boxes.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                face = extract_face(im, box, self.image_size, self.margin, face_path)
                if self.post_process:
                    face = fixed_normalize(face)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]
                prob_im = prob_im[0]

            faces.append(faces_im)
            probs.append(prob_im)
            face_boxes.append(box_im)

        if not batch_mode:
            faces = faces[0]
            probs = probs[0]

        if return_prob:
            return faces, probs, face_boxes
        else:
            return faces

    def detect(self, img, landmarks=False):
        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes)
        probs = np.array(probs)
        points = np.array(points)

        if not isinstance(img, Iterable):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]

        if landmarks:
            return boxes, probs, points

        return boxes, probs
