import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class AnnModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(AnnModel,self).__init__()

        self.fc1 = nn.Linear( input_dim,hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim,hidden_dim)
        self.act3 = nn.ELU()
        self.fc4 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):

        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        out = self.act3(out)
        out = self.fc4(out)

        return out


dataTrain = pd.read_csv("train.csv",dtype=np.float32)

targetsNumpy = dataTrain.label.values

featuresNumpy = dataTrain.loc[:,dataTrain.columns !="label"].values/255

xTrain,xTest,yTrain,yTest=train_test_split(featuresNumpy,targetsNumpy,test_size=0.2,random_state=42)

xTrainTensor = torch.from_numpy(xTrain)
yTrainTensor = torch.from_numpy(yTrain).type(torch.LongTensor)


xTestTensor = torch.from_numpy(xTest)
yTestTensor = torch.from_numpy(yTest).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(xTrainTensor,yTrainTensor)
test = torch.utils.data.TensorDataset(xTestTensor,yTestTensor)

trainLoder = DataLoader(train,batch_size=100,shuffle=False)
testLoder = DataLoader(test,batch_size=100,shuffle=False)

model = AnnModel((28*28),150,10)

error = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(),lr=0.02)

count = 0
lossList=[]
iterationList=[]
numOfEpoch=29

for epoch in range (numOfEpoch):

    for i, (images,labels) in enumerate(trainLoder):

        trainn = Variable(images.view(-1,28*28))
        labels = Variable(labels)

        optimizer.zero_grad()

        output = model(trainn)

        loss = error(output,labels)

        loss.backward()

        optimizer.step()

        count+=1



        if (count%50==0):

            correct=0
            total = 0

            for image,label in testLoder:

                testt = Variable(image.view(-1,28*28))

                outputs = model(testt)

                predicted = torch.max(outputs.data,1)[1]

                total+=len(label)

                correct += (predicted == label).sum()

            accuracy = 100*correct/float(total)

            lossList.append(loss.data)
            iterationList.append(count)

        if (count%500==0):

            print("Epoch: {} Iteration: {} Loss: {} Accuracy: {}%".format(epoch,count,loss.data,accuracy))

plt.plot(iterationList,lossList)
plt.xlabel("Number of Iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression")
plt.show()

