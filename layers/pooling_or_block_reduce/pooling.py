import cv2
import numpy as np
import skimage.measure
#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce


def skimage_reduce(img):

	h,w,c = img.shape
	block_size= (150,200,1)
	#block_size= (40,40,1)
	#block_size= (h,w,1)
	print(block_size)

	#reduced = skimage.measure.block_reduce(img, block_size, np.max)
	#reduced = skimage.measure.block_reduce(img, block_size, np.average)
	#reduced = skimage.measure.block_reduce(img, block_size, np.median)
	reduced = skimage.measure.block_reduce(img, block_size, np.median)


	# 'numpy.float64' object cannot be interpreted as an integer
	reduced = reduced.astype(np.uint8)


	reduced = cv2.resize(reduced,(w,h), interpolation = cv2.INTER_NEAREST)
	cv2.imshow("reduced",reduced)
	cv2.imshow("img",img)
	cv2.waitKey(0)

	#res = img-reduced.astype(np.float32)

	return reduced


def my_reduce(img):
	block_size=5
	kernel_size = 5


	h,w,c = img.shape
	print(h,w)	

	#height = int(h/block_size)
	#width  = int(w/block_size)
	#res = np.zeros((height,width,3), np.uint8)
	res = img.copy()


	for y in range(0,h,block_size)[:-1]:
		for x in range(0,w,block_size)[:-1]:	


			crop = img[y:y+block_size,x:x+block_size].copy()

			### median
			#median = np.median(crop,axis=(0,1))
			#median = np.array([[median]]).astype(np.uint8)
			#median = cv2.resize(median,(block_size,block_size), interpolation = cv2.INTER_NEAREST)
			#res[y:y+block_size,x:x+block_size]=median


			### mode
			crop = crop.reshape(-1, 3)
			unique, counts = np.unique(crop, return_counts = True, axis = 0 )
			mode = unique[ counts.argmax()]
			res[y:y+block_size,x:x+block_size] = mode

	cv2.imshow("res",res)
	cv2.waitKey(0)


if __name__ == "__main__":

	img = cv2.imread("cat.jpg",1)

	#skimage_reduce(img)
	my_reduce(img)

