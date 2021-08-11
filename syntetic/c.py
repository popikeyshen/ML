import os
import cv2
import numpy as np
from random import randint
import uuid





def txt_to_xy(img, rects):
	img_h, img_w, _ = img.shape
	for  r in rects:

		box=[]
		box.append( float( r.split(' ')[1] ) )
		box.append( float( r.split(' ')[2] ) )
		box.append( float( r.split(' ')[3] ) )

		box.append( float( r.split(' ')[4].split('\n')[0] ) )
		x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
		x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)

	return x1,x2,y1,y2



def rotate_image(image, angle, image_center):


  #image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


  
  return result


import uuid
import os

def save_augmented_data(img, x1,x2,y1,y2, cl, folder):
		if not os.path.exists(folder):
			os.makedirs(folder)

		b = str( uuid.uuid4().hex )

		cv2.imwrite(folder + b + ".jpg" , img)  
 
		x=x2
		y=y2
		w=x1-x2
		h=y1-y2

		height,width,c = img.shape

		x1 =str( float(x+w/2)/width  )+" "
		y1 =str( float(y+h/2)/height )+" "
		w1 =str( float(w)/width  )+" "
		h1 =str( float(h)/height )+"\n"


		s = open(folder+b+".txt", "w")
		s.write(cl+x1+y1+w1+h1) 
		s.close() 

def make_flip(img, x1,x2,y1,y2):
	img_h, img_w, _ = img.shape

	img = cv2.flip(img, 1)

	print(img_w, x1)
	flipped_x1 = img_w-x1
	flipped_x2 = img_w-x2

	return img, flipped_x1,flipped_x2,y1,y2


bckg = "./bckg/"
save_location = "./done/"
folder = "./data/"



files = os.listdir(folder)
b_files = os.listdir(bckg)
b_len = len(b_files)
for ff in files:
    #if ".jpg" in ff:
	b_files = np.random.permutation(b_files)
	image = cv2.imread(folder + ff)

	#img = rotate_image(img, 30)

	#rects = open("./txt/"+ff.replace("jpg","txt"), "r")
	rects = open("./txt/a1.txt", "r")

	# resize
	for res in range(5,10,1):

		image  = cv2.resize(image, (0,0) ,interpolation = cv2.INTER_NEAREST, fx=res/10.0, fy=res/10.0)
		x1,x2,y1,y2 = txt_to_xy(image,rects)



		x=int( (x1+x2)/2 )
		y=int( (y1+y2)/2 )
		image_center = x,y
		


		row, col, chn = image.shape
		img_rand = randint(0, b_len - 1)

		background = cv2.imread(bckg + b_files[img_rand])


		for nx in range(0,100,50):
			for ny in range(50,100,50):
				for angle in range(-80,-20,20):
					img = rotate_image(image, angle, image_center)



					res_img = background.copy()

					shift_w = nx #400
					shift_h = ny #200

					for r in range(row):
						for c in range(col):
							if sum(img[r,c]) >= 1+1+1:
								res_img[r+shift_h,c+shift_w] = img[r,c]

					#bk_img = apply_brightness_contrast(bk_img,0.5,40)
					#bk_img =  cv2.blur( bk_img,(3,3))


					
					x1s=x1+shift_w
					x2s=x2+shift_w
					y1s=y1+shift_h
					y2s=y2+shift_h

					#name = str( uuid.uuid4().hex )
					#cv2.imwrite(location + name + ".jpg", bk_img)



					save_augmented_data(res_img, x1s,x2s,y1s,y2s, "6 ",save_location+"id6/")

					res_img, x1s,x2s,y1s,y2s = make_flip(res_img, x1s,x2s,y1s,y2s)
					save_augmented_data(res_img, x1s,x2s,y1s,y2s, "8 ",save_location+"id8/")

					cv2.rectangle(res_img,(x1s,y1s),(x2s,y2s),(0,255,0),3)
					cv2.imshow("img",res_img)
					cv2.waitKey(0)











