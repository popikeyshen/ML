import torch
import numpy as np
from facerec.mtcnn.network import MTCNN
#from facerec.recognition.inception import InceptionResnetV1
from enum import Enum
import cv2


class PersonType(Enum):
    VIP = 0
    SUSPECT = 1


class PersonInfo:
    def __init__(self, name: str, person_type: PersonType, frame: torch.Tensor, embedding: torch.Tensor):
        self.name = name
        self.frame = frame.clone()
        self.person_type = person_type
        self.embedding = embedding.clone()

    def __str__(self):
        return self.name + ' ' + str(self.person_type)


class FaceRecognizer:
    def __init__(self, device=torch.device('cuda:0'), distance_threshold=1.0):
        self.device = device
        self.distance_threshold = distance_threshold
        self.mtcnn = MTCNN(
            image_size=150, margin=0, min_face_size=30,
            thresholds=[0.5, 0.6, 0.9], factor=0.709, post_process=True,
            device=device
        )
        #self.resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
        self.persons = []

    def get_faces_from_frame(self, frame: torch.Tensor):
        face_aligned, probs, face_boxes = self.mtcnn(frame, return_prob=True)
        return [], face_boxes
        print(face_boxes)
        if face_aligned is not None and face_aligned[0] is not None:

            #print(face_aligned[0].shape)
            #print(face_aligned[0].cpu().detach().numpy())
            #face = face_aligned[0].cpu().detach().numpy()
            #face = np.transpose(face, ( 2, 1, 0))
            #print(face.shape)
            #cv2.imshow('face_aligned', face)
            #cv2.waitKey(0)


            face_aligned = torch.stack(face_aligned).to(self.device)
            embeddings = []  #self.resnet(face_aligned).detach().cpu()
            return embeddings, face_boxes
        else:
            return [], []

    def add_person(self, name: str, frame: torch.Tensor, person_type: PersonType):
        embeddings, boxes = self.get_faces_from_frame(frame)
        if len(boxes) == 1 and boxes[0] is not None:
            person = PersonInfo(name=name, person_type=person_type, frame=frame, embedding=embeddings[0])
            self.persons.append(person)

    def classify_person(self, embedding: torch.Tensor):
        min_dist = 1000000000   # 1*10^9
        result_person = None
        for person in self.persons:
            dist = ((person.embedding - embedding)).norm()  # **2
            print(dist)
            if dist < min_dist:
                min_dist = dist
                result_person = person

        if self.distance_threshold >= min_dist:
            return min_dist, result_person
        else:
            return None, None


def image_to_tensor(img):
    img = img[np.newaxis, :]
    img = torch.from_numpy(img)
    return img
