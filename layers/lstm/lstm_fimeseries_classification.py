
def v2():
	import numpy as np

	## https://github.com/xieyulai/LSTM-for-Human-Activity-Recognition-using-2D-Pose_Pytorch/blob/master/lstm.ipynb

	# Useful Constants
	# Output classes to learn how to classify
	LABELS = [    
	    "JUMPING",
	    "JUMPING_JACKS",
	    "BOXING",
	    "WAVING_2HANDS",
	    "WAVING_1HAND",
	    "CLAPPING_HANDS"

	] 
	#DATASET_PATH = "data/HAR_pose_activities/database/"
	DATASET_PATH = "/home/popikeyshen/RNN-HAR-2D-Pose-database/"

	X_train_path = DATASET_PATH + "X_train.txt"
	X_test_path = DATASET_PATH + "X_test.txt"

	y_train_path = DATASET_PATH + "Y_train.txt"
	y_test_path = DATASET_PATH + "Y_test.txt"

	n_steps = 32 # 32 timesteps per series
	n_categories = len(LABELS)

	# Load the networks inputs

	def load_X(X_path):
	    file = open(X_path, 'r')
	    X_ = np.array(
		[elem for elem in [
		    row.split(',') for row in file
		]], 
		dtype=np.float32
	    )
	    file.close()
	    blocks = int(len(X_) / n_steps)
	    
	    X_ = np.array(np.split(X_,blocks))

	    return X_ 

	# Load the networks outputs
	def load_y(y_path):
	    file = open(y_path, 'r')
	    y_ = np.array(
		[elem for elem in [
		    row.replace('  ', ' ').strip().split(' ') for row in file
		]], 
		dtype=np.int32
	    )
	    file.close()
	    
	    # for 0-based indexing 
	    return y_ - 1

	X_train = load_X(X_train_path)
	X_test = load_X(X_test_path)


	# [ j0_x, j0_y, j1_x, j1_y , j2_x, j2_y, j3_x, j3_y, j4_x, j4_y, j5_x, j5_y, j6_x, j6_y, j7_x, j7_y, j8_x, j8_y, j9_x, j9_y, j10_x, j10_y, j11_x, j11_y, j12_x, j12_y, j13_x, j13_y, j14_x, j14_y, j15_x, j15_y, j16_x, j16_y, j17_x, j17_y ]

	print(X_train.shape)
	from matplotlib import pyplot as plt
	plt.plot(X_train[100][2][:], X_train[100][3][:])
	plt.show()

	y_train = load_y(y_train_path)
	y_test = load_y(y_test_path)

	import torch


	tensor_X_test = torch.from_numpy(X_test)
	print('test_data_size:',tensor_X_test.size())
	tensor_y_test = torch.from_numpy(y_test)
	print('test_label_size:',tensor_y_test.size())
	n_data_size_test = tensor_X_test.size()[0]
	print('n_data_size_test:',n_data_size_test)

	tensor_X_train = torch.from_numpy(X_train)
	print('train_data_size:',tensor_X_train.size())
	tensor_y_train = torch.from_numpy(y_train)
	print('train_label_size:',tensor_y_train.size())
	n_data_size_train = tensor_X_train.size()[0]
	print('n_data_size_train:',n_data_size_train)


	import torch.nn as nn

	#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	device = torch.device('cpu')



	class LSTM(nn.Module):
	    
		def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
			super(LSTM,self).__init__()
			self.hidden_dim = hidden_dim
			self.output_dim = output_dim
			self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
			self.fc = torch.nn.Linear(hidden_dim,output_dim)
			self.bn = nn.BatchNorm1d(32)
			
		def forward(self,inputs):
			x = self.bn(inputs)
			lstm_out,(hn,cn) = self.lstm(x)
			out = self.fc(lstm_out[:,-1,:])
			return out


	n_hidden = 200   ## hidden layer can be big or small
	n_joints = 18*2  ## input shape x's y's                   ## or rgbn for other task
	n_categories = 6
	n_layer = 3
	rnn = LSTM(n_joints,n_hidden,n_categories,n_layer)
	rnn.to(device)


	def categoryFromOutput(output):
	    top_n, top_i = output.topk(1)
	    category_i = top_i[0].item()
	    return LABELS[category_i], category_i
	    
	    
	import random
	def randomTrainingExampleBatch(batch_size,flag,num=-1):
		if flag == 'train':
			X = tensor_X_train
			y = tensor_y_train
			data_size = n_data_size_train
		elif flag == 'test':
			X = tensor_X_test
			y = tensor_y_test
			data_size = n_data_size_test
		if num == -1:
			ran_num = random.randint(0,data_size-batch_size)
		else:
			ran_num = num
		pose_sequence_tensor = X[ran_num:(ran_num+batch_size)]
		pose_sequence_tensor = pose_sequence_tensor
		category_tensor = y[ran_num:ran_num+batch_size,:]
		return category_tensor.long(),pose_sequence_tensor
		    
	    
	import torch.optim as optim
	import time
	import math

	criterion = nn.CrossEntropyLoss()
	learning_rate = 0.0005
	optimizer = optim.SGD(rnn.parameters(),lr=learning_rate,momentum=0.9)
	#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

	n_iters = 100000
	#n_iters = 60000
	print_every = 10
	plot_every = 1000
	batch_size = 20

	# Keep track of losses for plotting
	current_loss = 0
	all_losses = []

	def timeSince(since):
	    now = time.time()
	    s = now - since
	    m = math.floor(s / 60)
	    s -= m * 60
	    return '%dm %ds' % (m, s)

	start = time.time()

	for iter in range(1, n_iters + 1):
	   
	    category_tensor, input_sequence = randomTrainingExampleBatch(batch_size,'train')
	    input_sequence = input_sequence.to(device)
	    category_tensor = category_tensor.to(device)
	    category_tensor = torch.squeeze(category_tensor)
	    
	    optimizer.zero_grad()
	    
	    output = rnn(input_sequence)
	    loss = criterion(output, category_tensor)
	    loss.backward()
	    optimizer.step() 
	    #scheduler.step()
	    

	    current_loss += loss.item()
	    
	    #print(loss.item())
	    
	    category = LABELS[int(category_tensor[0])]

	    # Print iter number, loss, name and guess
	    if iter % print_every == 0:
	    	    guess, guess_i = categoryFromOutput(output)
	    	    correct = '✓' if guess == category else '✗ (%s)' % category
	    	    print('%d %d%% (%s) %.4f  / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, guess, correct))
			
	    # Add current loss avg to list of losses
	    if iter % plot_every == 0:
	    	    all_losses.append(current_loss / plot_every)
	    	    current_loss = 0
	    
	    
def v1():
	import numpy as np

	## https://github.com/xieyulai/LSTM-for-Human-Activity-Recognition-using-2D-Pose_Pytorch/blob/master/lstm.ipynb

	# Useful Constants
	# Output classes to learn how to classify
	LABELS = [    
	    "JUMPING",
	    "JUMPING_JACKS",
	    "BOXING",
	    "WAVING_2HANDS",
	    "WAVING_1HAND",
	    "CLAPPING_HANDS"

	] 
	#DATASET_PATH = "data/HAR_pose_activities/database/"
	DATASET_PATH = "/home/popikeyshen/RNN-HAR-2D-Pose-database/"

	X_train_path = DATASET_PATH + "X_train.txt"
	X_test_path = DATASET_PATH + "X_test.txt"

	y_train_path = DATASET_PATH + "Y_train.txt"
	y_test_path = DATASET_PATH + "Y_test.txt"

	n_steps = 32 # 32 timesteps per series
	n_categories = len(LABELS)

	# Load the networks inputs

	def load_X(X_path):
	    file = open(X_path, 'r')
	    X_ = np.array(
		[elem for elem in [
		    row.split(',') for row in file
		]], 
		dtype=np.float32
	    )
	    file.close()
	    blocks = int(len(X_) / n_steps)
	    
	    X_ = np.array(np.split(X_,blocks))

	    return X_ 

	# Load the networks outputs
	def load_y(y_path):
	    file = open(y_path, 'r')
	    y_ = np.array(
		[elem for elem in [
		    row.replace('  ', ' ').strip().split(' ') for row in file
		]], 
		dtype=np.int32
	    )
	    file.close()
	    
	    # for 0-based indexing 
	    return y_ - 1

	X_train = load_X(X_train_path)
	X_test = load_X(X_test_path)


	# [ j0_x, j0_y, j1_x, j1_y , j2_x, j2_y, j3_x, j3_y, j4_x, j4_y, j5_x, j5_y, j6_x, j6_y, j7_x, j7_y, j8_x, j8_y, j9_x, j9_y, j10_x, j10_y, j11_x, j11_y, j12_x, j12_y, j13_x, j13_y, j14_x, j14_y, j15_x, j15_y, j16_x, j16_y, j17_x, j17_y ]

	
	from matplotlib import pyplot as plt
	plt.plot(X_train[100][2][:], X_train[100][3][:])
	plt.show()

	y_train = load_y(y_train_path)
	y_test = load_y(y_test_path)

	import torch
	
	X_train = np.array( [[[0,100,0] , [0,0,100]],  [[0,0,100] , [0,0,100]], [[0,100,0] , [0,0,100]],  [[0,0,100] , [0,0,100]]]  )
	y_train = np.array( [[1,0,1,0]])
	
	y_train = np.transpose( y_train, axes=(1, 0))
	#X_train = np.transpose( X_train, axes=(0, 1, 2))
	
	X_test = X_train
	y_test = y_train

	tensor_X_test = torch.from_numpy(X_test)
	print('test_data_size:',tensor_X_test.size())
	tensor_y_test = torch.from_numpy(y_test)
	print('test_label_size:',tensor_y_test.size())
	n_data_size_test = tensor_X_test.size()[0]
	print('n_data_size_test:',n_data_size_test)

	tensor_X_train = torch.from_numpy(X_train)
	print('train_data_size:',tensor_X_train.size())
	tensor_y_train = torch.from_numpy(y_train)
	print('train_label_size:',tensor_y_train.size())
	n_data_size_train = tensor_X_train.size()[0]
	print('n_data_size_train:',n_data_size_train)

	

	print(X_train.shape, y_train.shape)
	import torch.nn as nn

	#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	device = torch.device('cpu')



	class LSTM(nn.Module):
	    
		def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
			super(LSTM,self).__init__()
			self.hidden_dim = hidden_dim
			self.output_dim = output_dim
			self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
			self.fc = torch.nn.Linear(hidden_dim,output_dim)
			self.bn = nn.BatchNorm1d(2)  ## num of frames
			
		def forward(self,inputs):
			x = self.bn(inputs)
			lstm_out,(hn,cn) = self.lstm(x)
			out = self.fc(lstm_out[:,-1,:])
			return out


	n_hidden = 200   ## hidden layer can be big or small
	n_joints = 3  ## input shape x's y's                   ## or rgbn for other task
	n_categories = 6
	n_layer = 3
	rnn = LSTM(n_joints,n_hidden,n_categories,n_layer)
	rnn.to(device)


	def categoryFromOutput(output):
	    top_n, top_i = output.topk(1)
	    category_i = top_i[0].item()
	    return LABELS[category_i], category_i
	    
	    
	import random
	def randomTrainingExampleBatch(batch_size,flag,num=-1):
		if flag == 'train':
			X = tensor_X_train
			y = tensor_y_train
			data_size = n_data_size_train
		elif flag == 'test':
			X = tensor_X_test
			y = tensor_y_test
			data_size = n_data_size_test
		if num == -1:
			ran_num = random.randint(0,data_size-batch_size)
		else:
			ran_num = num
		pose_sequence_tensor = X[ran_num:(ran_num+batch_size)]
		pose_sequence_tensor = pose_sequence_tensor
		
		#print(y)
		#print(ran_num,batch_size)
		category_tensor = y[ran_num:ran_num+batch_size,:]
		return category_tensor.long(),pose_sequence_tensor
		    
	    
	import torch.optim as optim
	import time
	import math

	criterion = nn.CrossEntropyLoss()
	learning_rate = 0.005
	optimizer = optim.SGD(rnn.parameters(),lr=learning_rate,momentum=0.9)
	#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

	n_iters = 100000
	#n_iters = 60000
	print_every = 10
	plot_every = 1000
	batch_size = 2

	# Keep track of losses for plotting
	current_loss = 0
	all_losses = []

	def timeSince(since):
	    now = time.time()
	    s = now - since
	    m = math.floor(s / 60)
	    s -= m * 60
	    return '%dm %ds' % (m, s)

	start = time.time()

	for iter in range(1, n_iters + 1):
	   
	    category_tensor, input_sequence = randomTrainingExampleBatch(batch_size,'train')
	    input_sequence = input_sequence.to(device)
	    category_tensor = category_tensor.to(device)
	    category_tensor = torch.squeeze(category_tensor)
	    
	    optimizer.zero_grad()
	    
	    output = rnn(input_sequence.float() )
	    loss = criterion(output, category_tensor)
	    loss.backward()
	    optimizer.step() 
	    #scheduler.step()
	    

	    current_loss += loss.item()
	    
	    #print(loss.item())
	    
	    category = LABELS[int(category_tensor[0])]

	    # Print iter number, loss, name and guess
	    if iter % print_every == 0:
	    	    guess, guess_i = categoryFromOutput(output)
	    	    correct = '✓' if guess == category else '✗ (%s)' % category
	    	    print('%d %d%% (%s) %.4f  / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, guess, correct))
			
	    # Add current loss avg to list of losses
	    if iter % plot_every == 0:
	    	    all_losses.append(current_loss / plot_every)
	    	    current_loss = 0
	    	    
	    	    
v1()
