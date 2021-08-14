

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



activations = {}
def get_activations(name):
	def hook(model, input, output):
		activations = output #.detach()
		#print( type(activations) )
		#print( len(activations))

		if type(activations) is tuple :
			print( len(activations) )
		else:
			activations = activations.cpu().detach().numpy()
			print( activations.shape  )



			plt.figure(figsize=(8,8))
			columns = 8
			for i in range(64): #activations.shape[1]
				plt.subplot( 64 / columns + 1, columns, i + 1)
				plt.imshow( activations[0,i,:,:] )
			#plt.figure(name)
			#plt.show()

	return hook



frame =0

def analyze(model,img):
	global frame,activations
	activations = {}

	print("---analyzing----")

	## the layers by one line
	#pred = model(img)[0]
	#print(pred)
	#print(pred.shape)


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
					out1[i,j,c] = conf*255



	yolo2 = pred[1].cpu().detach().numpy()
	#print(yolo2[0,0,0,0,:])
	for i in range(38):
		for j in range(38):
			for c in range(3):
				conf = yolo2[0,c,i,j,4]    # [1,3,38,38,85]
				if(conf>0):
					out2[i,j,c] = conf*255


	yolo3 = pred[2].cpu().detach().numpy()
	#print(yolo3[0,0,0,0,:])
	for i in range(76):
		for j in range(76):
			for c in range(3):
				conf = yolo3[0,c,i,j,4]    # [1,3,76,76,85]
				if(conf>0):
					out3[i,j,c] = conf*255

					#print("conf",conf)

	out1 = cv2.resize(out1, (0,0) ,interpolation = cv2.INTER_NEAREST, fx=4, fy=4)  
	out2 = cv2.resize(out2, (0,0) ,interpolation = cv2.INTER_NEAREST, fx=4, fy=4)  
	out3 = cv2.resize(out3, (0,0) ,interpolation = cv2.INTER_NEAREST, fx=4, fy=4)  


	cv2.imshow("out1",out1)
	cv2.imshow("out2",out2)
	cv2.imshow("out3",out3)
	frame +=1
	#print(frame)



	### test 1
	'''
	### convert from second array to detections
	pred1 = model(img)[1][0].cpu().detach().numpy()[0]
	pred1 = pred1.T.reshape(85,-1)
	pred1 = np.swapaxes(pred1,0,1)
	print("1shape=",pred1.shape)

	pred2 = model(img)[1][1].cpu().detach().numpy()[0]
	pred2 = pred2.T.reshape(85,-1)
	pred2 = np.swapaxes(pred2,0,1)
	print("2shape=",pred2.shape)

	pred3 = model(img)[1][2].cpu().detach().numpy()[0]
	pred3 = pred3.T.reshape(85,-1)
	pred3 = np.swapaxes(pred3,0,1)
	print("3shape=",pred3.shape)

	pred = pred1 #np.concatenate((pred3, pred2), axis=0)
	#pred = np.concatenate((pred, pred1), axis=0)

	#activations
	#print(pred[:][0])

	pred = model(img)[0].cpu().detach().numpy()
	print(pred.shape)
	print(pred[0,0,:])
	'''

	#pred = np.array([pred])
	#pred = torch.from_numpy(pred).to(device)
	#print("shape=",pred.shape)
	
	'''
	print("hook")
	### hooks 
	hooks = {}
	for name, module in model.named_modules():
		print(name)
		#hooks[name] = module.register_forward_hook(get_activations())
		#hooks[name] = module.register_forward_hook(get_activation(name))

		#if name == 'module_list.3.activation':
		#	module.register_forward_hook(get_activations(name))
		#if name == 'module_list.10.activation':
		#	module.register_forward_hook(get_activations(name))

		if name == 'module_list.35.activation':
			module.register_forward_hook(get_activations(name))
		if name == 'module_list.60.activation':
			module.register_forward_hook(get_activations(name))
		if name == 'module_list.73.activation':
			module.register_forward_hook(get_activations(name))
	plt.show()
	'''


	#print("hook")
	

	#print(hooks['module_list.0.activation'])

	#act = activation['module_list.0.activation'][0]
	#fig, axarr = plt.subplots(act.size(0))
	#for idx in range(act.size(0)):
	#	axarr[idx].imshow(act[idx])


	#cv2.waitKey(1)
	print("---analyzing----")



