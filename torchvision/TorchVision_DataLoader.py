from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt

'''
네트워크 인풋
'''
# (10 x 3 x 100 x 100) = (batch_size, data_dimension, matrix_size)

'''
데이터 저장
'''
transfrom = transforms.Compose([transforms.Resize((100, 100)), # 사이즈 조정
                                transforms.ToTensor(), # 텐서로 변환
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


'''
커스텀 데이터 셋 구성
'''
# imfd 폴더에 아래 폴더안에 이미지가 들어가 있어야 불러올 수 있음
# imfd 폴더 아래에 있는 폴더들의 이름이 label로 지정됨
trainset = torchvision.datasets.ImageFolder(root = './imfd', transform = transfrom)

'''
데이터 로더
'''
# 배치사이즈가 커지면 이미지를 가지고 오는데 걸리는 시간이 걸리는데, 
# CPU가 이미를 가지고 올 때 CPU 프로세서를 얼마나 사용할 지 결정
train_loader = DataLoader(trainset, batch_size = 10, shuffle = True, num_workers = 0) 

'''
이미지 불러오는 함수 정의
'''
def imshow(img):
    import numpy as np
    img = img / 2 + 0.5 # unnormalized
    np_img = img.numpy() # 텐서를 넘파이로 저장
    # transpose : ToTensor를 통해 (Height x Width x Channel) 로 되어 있다가
    # Numpy로 변경하면서 (Channel x Height x Width)로 변경되었기 때문에 index 순으로 다시 변경
    plt.imshow(np.transpose(np_img, (1, 2, 0))) #넘파이로 저장된 이미지 출력
    print(np_img.shape)
    print((np.transpose(np_img, (1, 2, 0))).shape)


'''
데이터 출력 (batch_size x data_dimension x matrix_size) = (8 x 3 x 32 x 32)
'''
for batch_idx, (img, label) in enumerate(train_loader):
    print(batch_idx, img.shape, label.shape)

data_iter = iter(train_loader)
images, labels = data_iter.next()

print(images.shape)
imshow(torchvision.utils.make_grid(images, nrow = 5))
print(images.shape)
print((torchvision.utils.make_grid(images)).shape)
print(''.join('%5s ' % labels[j] for j in range(10)))