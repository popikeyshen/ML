import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

SEED=9876
torch.manual_seed(SEED)


train_csv = pd.read_csv("/data/fashion-mnist_train.csv")
test_csv = pd.read_csv("/data/fashion-mnist_test.csv")

device = torch.device("cpu")

train_csv.head()

y_train = train_csv['label'].values
X_train = train_csv.drop(['label'],axis=1).values

y_test = test_csv['label'].values
X_test = test_csv.drop(['label'],axis=1).values

plt.imshow(X_train[0].reshape(28, 28))
plt.imshow(X_train[1].reshape(28, 28))



BATCH_SIZE = 32

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)


### Base model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784,250)
        self.linear2 = nn.Linear(250,100)
        self.linear3 = nn.Linear(100,10)
    
    def forward(self,X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return X

mlp = MLP()
print(mlp)


def fit(model, train_loader, epoch_number=5):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epoch_number):
        correct = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            if batch_idx % 200 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data, float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))

def evaluate(model):
    correct = 0 
    for test_imgs, test_labels in test_loader:
        test_imgs = Variable(test_imgs).float()
        
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))

torch.manual_seed(SEED)
fit(mlp, train_loader)

evaluate(mlp)

def calc_weights(model):
    result = 0
    for layer in model.children():
        result += len(layer.weight.reshape(-1))
    return result

calc_weights(mlp)


### little model
class StudentMLP(nn.Module):
    def __init__(self):
        super(StudentMLP, self).__init__()
        self.linear1 = nn.Linear(784,16)
        self.linear2 = nn.Linear(16,10)
    
    def forward(self,X):
        X = F.relu(self.linear1(X))
        X = self.linear2(X)
        return X


smlp_simple = StudentMLP()
calc_weights(smlp_simple)

torch.manual_seed(SEED)
fit(smlp_simple, train_loader)

evaluate(smlp_simple)



def distill(teacher_model, student_model, train_loader, epoch_number=5, alpha=0.5, temperature=2):
    def error_and_output(var_X_batch, var_y_batch): # Задаем нашу особую функцию ошибки
        # Дивергенция Кульбака-Лейблера нужна, чтобы подсчитать кросс-энтропию между двумя распределениями
        # А именно между распределениями ответов модели-учителя и модели-ученика
        kldloss = nn.KLDivLoss()  
        # Для подсчета ошибки на данных воспользуемся уже готовой функцией для кросс-энтропии
        celoss = nn.CrossEntropyLoss()
        
        # Считаем выходы из сети-учителя
        teacher_logits = teacher_model(var_X_batch)
        # И выходы из сети-ученика
        student_logits = student_model(var_X_batch)
        
        # Рассчитываем распределение вероятностей ответов с помощью softmax с параметром T для сети-ученика
        soft_predictions = F.log_softmax( student_logits / temperature, dim=1 )
        # И для сети-учителя
        soft_labels = F.softmax( teacher_logits / temperature, dim=1 )
        # Считаем ошибку дистиляции - кросс-энтропию между распределениями ответов моделей
        distillation_loss = kldloss(soft_predictions, soft_labels)
        
        # Считаем ошибку на данных - кросс-энтропию между распределением ответов сети-ученика и правильным ответом
        student_loss = celoss(student_logits, var_y_batch)
        
        # Складываем с весами
        return distillation_loss * alpha + student_loss * (1 - alpha), student_logits
    
    optimizer = torch.optim.Adam(student_model.parameters())
    student_model.train()
    
    # Далее обучение проходит как обычно
    for epoch in range(epoch_number):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            loss, output = error_and_output(var_X_batch, var_y_batch)
            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            if batch_idx % 200 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data, float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))


torch.manual_seed(SEED)
smlp = StudentMLP()
distill(mlp, smlp, train_loader, temperature=10.0)


evaluate(smlp)


torch.manual_seed(SEED)
smlp = StudentMLP()
distill(mlp, smlp, train_loader, temperature=5.0)

evaluate(smlp)

torch.manual_seed(SEED)
smlp = StudentMLP()
distill(mlp, smlp, train_loader, temperature=2.0)


evaluate(smlp)



torch.manual_seed(SEED)
smlp = StudentMLP()
distill(mlp, smlp, train_loader, temperature=1.5)

evaluate(smlp)


torch.manual_seed(SEED)
smlp = StudentMLP()
distill(mlp, smlp, train_loader, temperature=5.0, alpha=0.2)

evaluate(smlp)


### https://www.coursera.org/learn/machine-learning-design/



