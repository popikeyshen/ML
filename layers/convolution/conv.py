

from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn as nn

# scipy conv
def scipy_3x3_conv():

	ascent = misc.ascent()
	scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
		           [-10+0j, 0+ 0j, +10 +0j],
		           [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
	grad = signal.convolve2d(ascent, scharr,  mode='same') # boundary='symm',




	fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))
	ax_orig.imshow(ascent, cmap='gray')
	ax_orig.set_title('Original')
	ax_orig.set_axis_off()
	ax_mag.imshow(np.absolute(grad), cmap='gray')
	ax_mag.set_title('Gradient magnitude')
	ax_mag.set_axis_off()
	ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles
	ax_ang.set_title('Gradient orientation')
	ax_ang.set_axis_off()
	fig.show()

def scipy_big_conv(img, kernel):
	#print(img.shape, kernel.shape)
	grad = signal.convolve2d(img, kernel,  mode='same') # boundary='symm',




def torch_3x3_conv(img,kernel):

	### create 2d kernel 
	conv = nn.Conv2d( 1, 1, 3, stride=1, padding=1)

	kernel = torch.Tensor(kernel)
	kernel = torch.nn.Parameter( kernel )
	conv.weight = kernel
	#print(conv, conv.weight)

	img = torch.Tensor(img)#.unsqueeze_(0)
	r = conv(img)
	res = r.detach().numpy()[0][0]

	#print(res.shape)
	plt.imshow(res)
	plt.show()

### fast big size kernel
def torch_100x100_conv(image,kern):
	img    = image.copy()
	kernel = kern.copy()

	### create 2d kernel 
	conv = nn.Conv2d( 1, 1, 50, stride=1, padding=50)

	kernel = torch.Tensor(kernel)
	kernel = torch.nn.Parameter( kernel )
	conv.weight = kernel
	#print(conv, conv.weight)

	img = torch.Tensor(img)#.unsqueeze_(0)
	r = conv(img)
	res = r.detach().numpy()[0][0]

	#print(res.shape)
	#plt.imshow(res)
	#plt.show()

def torch_100x100_conv_cuda(image,kern):
	img    = image.copy()
	kernel = kern.copy()

	### create 2d kernel 
	conv = nn.Conv2d( 1, 1, 50, stride=1, padding=50)

	kernel = torch.Tensor(kernel)
	kernel = torch.nn.Parameter( kernel )
	conv.weight = kernel
	#print(conv, conv.weight)

	img = torch.Tensor(img)#.unsqueeze_(0)
	r = conv(img).cuda()
	res = r.detach().cpu().numpy()[0][0]

	#print(res.shape)
	#plt.imshow(res)
	#plt.show()


import time
if __name__ == "__main__":

	#scipy_3x3_conv()

	kernel =  	[[[[-1 ,-1, -1],
			   [-1,  8 ,-1], 
			   [-1, -1 ,-1]]]]

	kernel =  	[[[[-1, -1, -1],
			   [ 2,  3,  2], 
			   [-1, -1, -1]]]]

	kernel =  	[[[[0,  0, 0],
			   [2,  1, 2], 
			   [0,  0, 0]]]]

	print(torch.cuda.is_available())

	img = np.ones((1,1,50,50), np.uint8)
	img[0,0,10,10] = 200
	img[0,0,11,10] = 200
	img[0,0,20,20] = 200
	img[0,0,30,30] = 200

	torch_3x3_conv(img,kernel)

	kernel = np.ones((1,1,50,50), np.uint8)
	kernel[0,0,10,10] = 200
	kernel[0,0,11,10] = 200
	kernel[0,0,20,20] = 200
	kernel[0,0,25,25] = 200

	t0 = time.time()
	torch_100x100_conv(img,kernel)
	t1 = time.time()
	print(t1-t0)

	#t0 = time.time()
	#torch_100x100_conv_cuda(img,kernel)
	#t1 = time.time()
	#print(t1-t0)


	img = np.ones((50,50), np.uint8)
	img[10,10] = 200
	img[11,10] = 200
	img[20,20] = 200
	img[30,30] = 200

	t0 = time.time()
	scipy_big_conv(img,img)
	t1 = time.time()
	print(t1-t0)

	sizes=[200,100,50]
	time1=[0.57,0.15,0.051]
	time2=[2.05,0.14,0.013]

	plt.plot(sizes,time1,label='pytorch')
	plt.plot(sizes,time2,label='scipy')
	plt.show()
