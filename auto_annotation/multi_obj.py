
# Dataset annotation tool
# Autor: Viacheslav Popika

# How to install
# sudo apt install python-opencv
# sudo pip install opencv-contrib-python

# How to run
# python multi_obj.py --tracker csrt  --video "05 feet MG Weapon Away From Body.MP4"

# How to use
# Put "s" button to show region with mouse
# Put "c" to stop cropping this region
# Put "q" to quit

# you change save folder and name in line 104




# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

import json

f = open("data.json","w+")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# loop over frames from the video stream
image_num=0

#data = {}
#data['_via_img_metadata'] = []

settings = ' {"_via_settings":{"ui":{"annotation_editor_height":25,"annotation_editor_fontsize":0.8,"leftsidebar_width":18,"image_grid":{"img_height":80,"rshape_fill":"none","rshape_fill_opacity":0.3,"rshape_stroke":"yellow","rshape_stroke_width":2,"show_region_shape":true,"show_image_policy":"all"},"image":{"region_label":"__via_region_id__","region_color":"__via_default_region_color__","region_label_font":"10px Sans","on_image_annotation_editor_placement":"NEAR_REGION"}},"core":{"buffer_size":18,"filepath":{},"default_filepath":""},"project":{"name":"via_project_16Jul2019_12h15m"}}'
f.write(settings)

while True:
	try:

		# grab the current frame, then handle if we are using a
		# VideoStream or VideoCapture object
		frame = vs.read()
		frame = frame[1] if args.get("video", False) else frame

		# check to see if we have reached the end of the stream
		if frame is None:
			break

		# resize the frame (so we can process it faster)
		#frame = imutils.resize(frame, width=500)

		# grab the updated bounding box coordinates (if any) for each
		# object that is being tracked
		(success, boxes) = trackers.update(frame)

		# loop over the bounding boxes and draw then on the frame
		for box in boxes:
			(x, y, w, h) = [int(v) for v in box]

			height, width, channels = frame.shape 
			if(x + w +2> width or y + h +2>height):
				break

			if(h>w):
				crop_img = frame[y : y +h , x : x + h ]#+ w]
			if(w>h):
				crop_img = frame[y : y +w , x : x + w ]#+ w]

			#cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
			cv2.imshow("faces",crop_img)



# Save folder and name 
# you can chenge this 
			
			name     =  args["video"]+str(image_num)+ ".jpg"
			cv2.imwrite("./save/" + name , frame)    # cropped 
			#cv2.imwrite("./save/1" + str(image_num) + ".jpg", crop_img)    # cropped 
			#cv2.imwrite("./save/1" + str(image_num) + ".jpg", frame)       # not cropped
			size     =  str( os.stat("./save/" + name).st_size)

			image_num+=1
			cv2.rectangle(frame, (x-1, y-1), (x + w +2, y + h +2), (0, 255, 0), 1)

			# write as txt file
			name       = '"'+name+size+'"'
			filename   = '"filename":' + name
			size       = '"size":'     + size

			regions    = '[{"shape_attributes":{"name":"rect","x":'+str(x)+',"y":'+str(y)+',"width":'+str(w)+',"height":'+str(h)+'},"region_attributes":{}}]'
			region     = '"regions":'+regions 
			att        = '"file_attributes":{}}'

			line = ","+name+":{"+filename+","+size+","+region+","+att
			f.write(line)



			#write as json file for Darknet
			#filename =  args["video"]+str(size)
			#regions  =  [{"shape_attributes":{"name":"rect","x":str(x),"y":str(y),"width":str(w),"height":str(h)},"region_attributes":{"objectname":"guntest"}}]
			#line =     {   name:{ "filename":filename,"size":size,"regions":regions,"file_attributes":{} }  }
			#data['_via_img_metadata'].append( line )


		cv2.putText(frame,"S - mark, C - cancel, Q - quit", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),1)
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(5) & 0xFF



		# if the 's' key is selected, we are going to "select" a bounding
		# box to track
		if key == ord("s"):
			try:
				# select the bounding box of the object we want to track (make
				# sure you press ENTER or SPACE after selecting the ROI)
				box = cv2.selectROI("Frame", frame, fromCenter=False,
					showCrosshair=True)

				# create a new object tracker for the bounding box and add it
				# to our multi-object tracker
				tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
				trackers.add(tracker, frame, box)

			except:
				print("maybe box too big")	
		

		if key == ord("c"):

			trackers = cv2.MultiTracker_create()

		# if the `q` key was pressed, break from the loop
		elif key == ord("q"):
			break
	except Exception as e:
		print("errror :{}".format(e))

f.write("}")

#with open('data.json',"w+") as outfile:
#     json.dump(data, outfile)

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
