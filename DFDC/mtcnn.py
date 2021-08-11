

import cv2
import torch
from facerec import FaceRecognizer, image_to_tensor, PersonType, PersonInfo
from utils.images import hconcat_resize_min, get_int_rect

import imutils
import matplotlib.pyplot as plt


VIDEO_PATH="/media/popikeyshen/30c5a789-895a-4cc2-910a-3c678cc563d7/deepfake/train_sample_videos/alvgwypubw.mp4"



### init torch device
device = torch.device('cuda:0')

### init face detector
distance_threshold = 1.0
recognizer = FaceRecognizer(device=device, distance_threshold=distance_threshold)


### init video capture
cap = cv2.VideoCapture(VIDEO_PATH)
while(1):

	### read and resize frame
	ret, im = cap.read()
	im = imutils.resize(im, height=800)

	### detect faces
	embeddings, boxes = recognizer.get_faces_from_frame(image_to_tensor(im))
	for box in  boxes:

			### draw box around face
			min_x, min_y, max_x, max_y = get_int_rect(box)
			cv2.rectangle(im, (min_x, min_y), (max_x, max_y), (0,255,0), 1)

			### crop face
			cropped_face = im[ min_y:max_y, min_x:max_x]
			cv2.imshow("cropped_face",cropped_face)
			k = cv2.waitKey(1)

			### calc some info about out face
			histr = cv2.calcHist([cropped_face],[0],None,[256],[0,256]) 
  
			# show the plotting graph of an image 
			plt.plot(histr) 
			plt.show() 
			

	cv2.imshow("im",im)
	k = cv2.waitKey(0)

	if k == 'q':
		break

