'''
Dataset
'''
from torch.utils.data import Dataset
import numpy as np

np.random.seed(4568)
x_np = np.random.random((5, 5))
y_np = np.random.random((5, 1))

class CustomDataset(Dataset): # torch.utils.data.Dataset 상속
    def __init__(self):
        self.x_data = x_np
        self.y_data = y_np

    def __len__(self): # 전체 데이터 수
        return len(self.x_data)

    def __getitem__(self, idx): # 어떤 인덱스 idx를 받았을 때, 이에 상응하는 입출력 데이터 변환
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()

'''
DataLoader
'''
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size = 2, # 각 minibatch의 크기 (한 번의 배치 안에 있는 샘플 사이즈), 통상적으로 2의 제곱수로 설정함 (예: 16, 32, 64...)
    shuffle = True # Epoch 마다 데이터셋을 섞어, 데이터가 학습되는 순서를 바꿈
)

'''
Model
'''
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(5,10),
    nn.Sigmoid(),
    nn.Linear(10,5),
    nn.Sigmoid(),
    nn.Linear(5, 2)
)

optimizer = optim.Adam(model.parameters(), lr = 1e-2)
criterion = nn.MSELoss()

'''
Train code
'''
nb_epochs = 20
for epoch in range(nb_epochs +1): 
    for batch_idx, samples in enumerate(dataloader): # enumerate(dataloader) : minibatch 인덱스와 데이터를 받음
        x_train, y_train = samples
        #H(x) 계산
        prediction = model(x_train)
    
        # cost 계산
        loss = criterion(prediction, y_train)
    
        # cost로 H(x) 개선
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        print('Epoch {:4d}/{} Batch {}/{} Cost : {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader), # 한 epoch 당 minibatch 개수
            loss.item()
            ))
