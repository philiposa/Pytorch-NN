import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self, in_features=4,h1=8,h2=9,out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.out(x)

        return x
    
torch.manual_seed(41)
model = Model()

df = pd.read_csv('iris.csv')
df['variety'] = df['variety'].replace('Setosa', 0.0)
df['variety'] = df['variety'].replace('Versicolor', 1.0)
df['variety'] = df['variety'].replace('Virginica', 2.0)

#Train data
x = df.drop('variety', axis=1)
y  = df['variety']

#convert to numpy
X = x.values
y = y.values

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=41)

#Convert x labels to tensor float
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#Convert y labels to tensor long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#create criterion to measure the error
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train model
epochs = 100
losses = []
for i in range(epochs):
    #go forward
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    #orint every 10 epoch
    if i % 10 == 0:
        print(f'Epoch: {i} and loss" {loss}')

    #back prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

