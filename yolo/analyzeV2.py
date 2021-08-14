
## можна рахувати відмінність по кількості задіяних нейронів

import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision

#from matplotlib import pyplot
import matplotlib.pyplot as plt

activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output#.detach()
	#return hook


from utils.utils import *

activations = {}
def get_activations(name):
	def hook(model, input, output):
		activations = output #.detach()

		print("hook")
		print( type(activations) )
		print( len(activations))

		if type(activations) is tuple :
			print( len(activations) )
			print( "tuple1", activations[0].shape ) 
		else:
			activations = activations.cpu().detach().numpy()
			#print( "activations", activations.shape, activations[0,0,0,0],activations[0,0,75,75],
#activations[0,254,0,0],activations[0,254,75,75]  )  ## 0 75 ...

			#print( "activations", activations.shape, activations[0,0,0,0],activations[0,0,1,0],
#activations[0,0,0,1],activations[0,0,1,1]  )


			print("lay105 activation 0,4,35,34 (1,255,76,76)", activations[0,4,35,34] )

			#device = torch_utils.select_device(device='cpu')
			#t =  np.array([1.3453815])
			#t = torch.from_numpy(t).to(device)
			#print(torch.sigmoid(t))


			'''
			plt.figure(figsize=(8,8))
			columns = 8
			for i in range(32): #activations.shape[1]
				plt.subplot( 32 / columns + 1, columns, i + 1)
				plt.imshow( activations[0, i,:,:] )
			'''


			#for i in range(255):
			#	activations[0, i,:,:]

			#plt.figure(name)
			#plt.show()

	return hook


layer104 = 0

activations = {}
def get_activations_221(name):
	global layer104
	def hook(model, input, output):
		global layer104

		activations = output #.detach()

		print("hook")
		print( type(activations) )
		print( len(activations))

		if type(activations) is tuple :
			print( len(activations) )
			print( "tuple1", activations[0].shape ) 
		else:
			activations = activations.cpu().detach().numpy()

			print("activations221", activations.shape , " 35,34 ",activations[:,:,35,34])

			layer104 = activations


	return hook


layer103 = 0
activations = {}
def get_activations_220(name):
	global layer103
	def hook(model, input, output):
		global layer103
		
		activations = output #.detach()
		print(activations.shape)

		print("hook")
		print( type(activations) )
		print( len(activations))

		if type(activations) is tuple :
			print( len(activations) )
			print( "tuple1", activations[0].shape ) 
		else:
			activations = activations.cpu().detach().numpy()

			print("activations220", activations.shape )

			layer103 = activations


	return hook

layer102 = 0
activations = {}
def get_activations_219(name):
	global layer102
	def hook(model, input, output):
		global layer102

		activations = output #.detach()

		print("hook")
		print( type(activations) )
		print( len(activations))

		if type(activations) is tuple :
			print( len(activations) )
			print( "tuple1", activations[0].shape ) 
		else:
			activations = activations.cpu().detach().numpy()

			print("activations219", activations.shape )

			layer102 = activations


	return hook


frame =0


### about fuckin yolo

#                 img ->  3x608x608
#   conv 1 32x3x3x3   -> 32x608x808
#   conv 2 64x32x3x3  -> 64x608x608
#   conv 3 32x64x3x3  -> 32x608x608
#
#



# debug to measure difference
# and other

# 1x1 conv + b
def conv(weights, layer, b):

	# w221 torch.Size([255, 256, 1, 1])
	# lay104 (1, 256, 76, 76)
	# lay220 (255)

	out = np.zeros((255, 76, 76), dtype=float)

	#for i in range(0,255):
	i = 4


	a = []
	for j in range(0,256):

			### TO DO automaticaly
			#for x in range(34,36):
			#	for y in range(33,35):
					y = 34
					x = 35

					out[i,x,y] +=  weights[i,j,0,0] * layer[0,j,x,y]
					
					#print(j, weights[i,j,0,0] * layer[0,j,x,y])

					a.append(weights[i,j,0,0] * layer[0,j,x,y])

			#for x in range(0,76):
			#	for y in range(0,76):
			#		out[i,0,0] += weights[i,j,0,0] * layer[0,j,0,0]
	print(out[4,35,34])

	for i in range(0,255):
		out[i] = out[i]+b[i]

	print("result ",out[4,35,34])


	return a

from torch.autograd.variable import Variable

# 1x1 conv + b
def conv1x1(weights, layer, b,b2,   w425,w426,w427,w428,w429):

	print("conv1x1" )
	print(weights.shape, layer.shape, b.shape)  # (128, 256, 1, 1) (1, 256, 76, 76) (128,)
	print("b2",b2.shape) # (128,)
	# w221 torch.Size([255, 256, 1, 1])
	# lay104 (1, 256, 76, 76)

	out = np.zeros((128, 76, 76), dtype=float)




	a = []


	i = 4
	for i in range(0,128):
		for j in range(0,256):

			### TO DO automaticaly
			for x in range(0,76):
				for y in range(0,76):
					#y = 34
					#x = 35

					out[i,x,y] +=  weights[i,j,0,0] * layer[0,j,x,y]
					
					#print(j, weights[i,j,0,0] * layer[0,j,x,y])

					a.append(weights[i,j,0,0] * layer[0,j,x,y])

			#for x in range(0,76):
			#	for y in range(0,76):
			#		out[i,0,0] += weights[i,j,0,0] * layer[0,j,0,0]
	print(out[4,35,34])

	
	#for i in range(0,128):
	#	out[i] = out[i]+b[i]
	conv = out	

	print("lay103 result 1x1",out[0,0,0],out[0,0,1],out[0,0,2],out[0,0,3])  

	print("add batch norm")


	#m = nn.BatchNorm2d(128, affine=False)
	#input = torch.randn(1, 128, 76, 76)
	#output = m(input)
	#print(input.shape, output.shape)

	### end of conv 1x1
	### start of batchnorm https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
	###  https://medium.com/@shahkaran76/yolo-object-detection-algorithm-in-tensorflow-e080a58fa79b
	### https://blog.paperspace.com/improving-yolo/

	bn = nn.BatchNorm2d(128, momentum=0.1) #, weights = b2)
	print("batchnorm", bn)

	out = np.array([out])
	x   = Variable(torch.from_numpy(out) )
	print("x ",x.shape)


	w0 = w425.cpu().detach().numpy()
	w1 = w426.cpu().detach().numpy()
	w2 = w427.cpu().detach().numpy()
	w3 = w428.cpu().detach().numpy()
	w4 = w429.cpu().detach().numpy()

	print("biases and ",w1.shape,w2.shape,w3.shape)	

	
	w426 = torch.from_numpy( w1 )
	w427 = torch.from_numpy( w2 )
	w428 = torch.from_numpy( w3 )
	w429 = torch.from_numpy( w4 )

	

	bn.weight.data 	= Variable( w426 )
	bn.bias.data 	= Variable( w427 )
	bn.running_mean = Variable( w428 )
	bn.running_var 	= Variable( w429 )
	x  = bn( x.float() )




	gamma = bn.weight
	mu   = bn.running_mean
	var  = bn.running_var
	beta  = bn.bias
	eps   = bn.eps    
	k = gamma / torch.sqrt(var + eps)

	#print( k.shape )
	#print( x.shape )
	#print( beta.shape )
	#out = k* x + beta.data


	# https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
	#mu = np.mean(X, axis=0)
	#var = np.var(X, axis=0)

	k     = k.cpu().detach().numpy()
	x     = x.cpu().detach().numpy()
	beta  = beta.data.cpu().detach().numpy()
	gamma = gamma.data.cpu().detach().numpy()
	mu  = mu.data.cpu().detach().numpy()
	var = var.data.cpu().detach().numpy()


	print(w425[0][0][0][0],      gamma[0],beta[0])
	print(weights[0][0][0][0],   b[0],    b2[0]  )


	mu = np.mean(x, axis=(0,2,3))
	var = np.var(x, axis=(0,2,3))
	#print("mu var",mu,var)
	#print("mu var",mu.shape,var.shape)



	print("shapes ",x.shape,mu.shape,var.shape,gamma.shape,beta.shape)
	out = np.zeros((128, 76, 76), dtype=float)
	X_norm = np.zeros((128, 76, 76), dtype=float)
	for i in range(0,128):
		for a in range(0,76):
			for b in range(0,76):

				#X_norm[i][a][b] = (x[0][i][a][b] - mu[i]) / np.sqrt(var[i] + 1e-8)
				#out[i][a][b] = gamma[i] * X_norm[i][a][b] + beta[i]

				X_norm[i][a][b] = (x[0][i][a][b] -mu[i]) / np.sqrt(var[i]+ 1e-8) 
				out[i][a][b] = gamma[i] * X_norm[i][a][b] + beta[i]	
	

	print("batch norm ",out[0,0,0],out[0,0,1],out[0,0,2],out[0,0,3])   


	batched = np.array([out])
	batched   = Variable(torch.from_numpy(batched) )

	activation = nn.LeakyReLU(0.1, inplace=True)
	batched = activation(batched).cpu().detach().numpy()


	return a



### conv 3x3
def conv3x3(weights, layer, b, b2):
	print("conv3x3 ",weights.shape, layer.shape, b.shape)

	out = np.zeros((256, 76, 76), dtype=float)
	for i in range(0,2):#256):
		for j in range(0,128):#,128):

			#print("first for")
			for x in range(0,76-2):
				for y in range(0,76-2):
					x=0
					y=0

					### TO DO - crop all <0
					### conv 3x3 weights
                                        
					for c1 in range(0,2):
						for c2 in range(0,2):
							#c1=0
							#c2=0
							#print(c1,x+c1) 

							out[i,x,y] +=  weights[i,j,c1,c2] * layer[0,j,x+c1,y+c2]
							#print(weights[i,j,c1,c2] * layer[0,j,x+c1,y+c2])
							
	print("after conv 3x3",out[0,0,0],out[1,0,0],out[1,35,34],out[2,35,34])
        

	for i in range(0,256):
		out[i] = out[i]+b[i]
	print("Afer conv +b[]",out.shape, out[0,0,0], out[1,0,0])

	### end of conv 3x3
	### start of batchnorm https://wiseodd.github.io/techblog/2016/07/04/batchnorm/

	mu = np.mean(out, axis=1)
	mu = np.mean(mu,  axis=1)
	print("mu",mu.shape)

	var = np.var(out, axis=1)
	var = np.var(var, axis=1)
	print("var",var.shape)

	sqr = np.sqrt(var + 1e-8)
	print(sqr.shape)

	X_norm = np.zeros((256,76,76), dtype=float)
	#X_norm = (out - mu) 
	for i in range(0,256):
		X_norm[i] = ( out[i]-mu[i] ) / sqr[i]
	print("Xnorm", X_norm.shape)

	#X_norm = X_norm 
	#print("Xnorm", X_norm.shape)

	for i in range(0,256):
		out[i] =  0.0001*X_norm[i] + b2[i]

	#conv = out
	#out = torch_utils.fuse_conv_and_bn(conv, b2)
	

	print("conv3x3 result ", out[0,0,0],out[1,0,0],out[2,35,34], out[255,35,34])

	return 0



from opencv_visualizer import visualizer
from opencv_visualizer import visualizer_numpy


def analyze(model,img):
	global layer104, layer103, layer102
	global frame,activations
	activations = {}

	print("---analyzing----")

	## the layers by one line
	#pred = model(img)[0]
	#print(pred)
	#print(pred.shape)

	#img = img[:,:1,:,:]
	print("img", img.shape )

	pred = model(img)[1]
	#print("layer -2 =",len(pred))
	#print(pred[0].shape)
	#print(pred[1].shape)
	#print(pred[2].shape)


	out1 = np.zeros((19,19,3), np.uint8)
	out2 = np.zeros((38,38,3), np.uint8)
	out3 = np.zeros((76,76,3), np.uint8)





	yolo1 = pred[0].cpu().detach().numpy()
	#print(yolo1.shape)
	#print(yolo1[0,0,0,0,:])
	for i in range(19):
		for j in range(19):
			for c in range(3):

				conf = yolo1[0,c,i,j,4]    # [1,3,19,19,85]
				if(conf>0):

					print("19x19",i,j,c)
					out1[i,j,c] = conf*255



	yolo2 = pred[1].cpu().detach().numpy()
	#print(yolo2[0,0,0,0,:])
	for i in range(38):
		for j in range(38):
			for c in range(3):
				conf = yolo2[0,c,i,j,4]    # [1,3,38,38,85]
				if(conf>0):
					print("38x38",i,j,c)
					out2[i,j,c] = conf*255

	activation_i=0
	activation_j=0
	activation_c=0


	yolo3 = pred[2].cpu().detach().numpy()

	for i in range(76):
		for j in range(76):
			for c in range(3):
				conf = yolo3[0,c,i,j,4]    # [1,3,76,76,85]
				if(conf>0):
					print("76x76",i,j,c)
					print( "conf", conf )
					out3[i,j,c] = conf*255

					activation_i=0
					activation_j=0
					activation_c=0
	
					#print("conf",conf)

	print("76 more",yolo3[0,0,0,0,0],yolo3[0,0,35,34,4])

	out1 = cv2.resize(out1, (0,0) ,interpolation = cv2.INTER_NEAREST, fx=4, fy=4)  
	out2 = cv2.resize(out2, (0,0) ,interpolation = cv2.INTER_NEAREST, fx=4, fy=4)  
	out3 = cv2.resize(out3, (0,0) ,interpolation = cv2.INTER_NEAREST, fx=4, fy=4)  


	cv2.imshow("out1",out1)
	cv2.imshow("out2",out2)
	cv2.imshow("out3",out3)
	frame +=1
	#print(frame)


	w222 = 0
	w221 = 0
	w220 = 0
	w219 = 0
	w218 = 0

	w217 = 0
	w216 = 0
	w215 = 0

	w425=0
	w426=0
	w427=0
	w428=0
	w429=0

	num = 0
	for d in model.state_dict():
		num +=1
		print(num,d)
		a= model.state_dict()[d]
		print(a.shape)

		if num ==425:
			w425=a

		if num ==426:
			w426=a

		if num ==427:
			w427=a

		if num ==428:
			w428=a

		if num ==429:
			w429=a


	num = 0
	for weights in model.parameters():
		num +=1
		print("params",num, weights.shape)

		if num == 222:
			print('222w',weights.shape)
			w222 = weights
			
		if num == 221:
			print('221w',weights.shape)
			w221 = weights

		if num == 220:
			print('220w',weights.shape)
			w220 = weights

		if num == 219:
			print('219w',weights.shape)
			w219 = weights

		if num == 218:
			print('218w',weights.shape)
			w218 = weights

		if num == 217:
			print('217w',weights.shape)
			w217 = weights


		if num == 216:
			print('216w',weights.shape)
			w216 = weights


		if num == 215:
			print('215w',weights.shape)
			w215 = weights


			#for i in range(255):
			#	if weights[i]>0:
			#		print(i,weights[i])




	# 1. 'module_list.105' is the last activation of anchor 3
	# 2. 
	#for i in model.

	### register and show hooks 
	hooks = {}
	for name, module in model.named_modules():
		print(name)
		#hooks[name] = module.register_forward_hook(get_activations())
		#hooks[name] = module.register_forward_hook(get_activation(name))

		#if name == 'module_list.3.activation':
		#	module.register_forward_hook(get_activations(name))
		#if name == 'module_list.10.activation':
		#	module.register_forward_hook(get_activations(name))

		#if name == 'module_list.35.activation':
		#	module.register_forward_hook(get_activations(name))
		#if name == 'module_list.60.Conv2d':
		#	module.register_forward_hook(get_activations(name))

		if name == 'module_list.102':
			module.register_forward_hook(get_activations_219(name))

		if name == 'module_list.103':
			module.register_forward_hook(get_activations_220(name))

		if name == 'module_list.104':
			module.register_forward_hook(get_activations_221(name))
		if name == 'module_list.105':
			module.register_forward_hook(get_activations(name))


	print('w222',w222.shape)
	print('w221',w221.shape)

	try:

		print('lay102',layer102.shape, layer102[0,0,0,0])
		print('lay103',layer103.shape, layer103[0,0,0,0],layer103[0,0,0,1],layer103[0,0,0,2],layer103[0,0,0,3])
		print('lay104',layer104.shape, layer104[0,0,0,0],layer104[0,1,0,0],layer104[0,1,35,34],layer104[0,2,35,34])

	
		w221 = w221.cpu().detach().numpy()
		w222 = w222.cpu().detach().numpy()

		layer105 = conv( w221,layer104, w222 )

		
		w218 = w218.cpu().detach().numpy()
		w219 = w219.cpu().detach().numpy()
		w220 = w220.cpu().detach().numpy()
		
		#layer104 = conv3x3( w218,layer103, w219,w220 )


		w217 = w217.cpu().detach().numpy()  # beta
		w216 = w216.cpu().detach().numpy()  # gamma
		w215 = w215.cpu().detach().numpy()
		
		layer103 = conv1x1( w215,layer102, w216,w217,   w425,w426,w427,w428,w429 )


		print("\n  vis \n")
		visualizer_numpy(layer105)

		

	except Exception as e:
		print('\n\n\n some error {}'.format(e))

	#plt.show()





	#cv2.waitKey(1)
	print("---analyzing----")



