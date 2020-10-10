from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

'''
class my_network(nn.Module):
    def __init__(self):
        super(my_network, self).__init__() 
        # 사용할 함수들을 정의할 장소
    def forward(self, x):
        # 함수들을 사용하여 Network의 forward를 정의하는 장소
        return # 리턴 받을 값
'''

'''
네트워크 인풋
'''
# (10 x 3 x 100 x 100) = (batch_size, data_dimension, matrix_size)

'''
데이터 저장
'''
transfrom = transforms.Compose([transforms.Resize((32, 32)), # 사이즈 조정
                                transforms.ToTensor(), # 텐서로 변환
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
커스텀 데이터 셋 구성
'''
# imfd 폴더에 아래 폴더안에 이미지가 들어가 있어야 불러올 수 있음
trainset = torchvision.datasets.ImageFolder(root = './imfd', transform = transfrom)

'''
데이터 로더
'''
# 배치사이즈가 커지면 이미지를 가지고 오는데 걸리는 시간이 걸리는데, 
# CPU가 이미를 가지고 올 때 CPU 프로세서를 얼마나 사용할 지 결정
train_loader = DataLoader(trainset, batch_size = 10, shuffle = True, num_workers = 0) 

# class torch.nn.Conv2d(input_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True)
# input_channels : 사용할 인풋의 크기
# output_channels : 사용할 필터 수
# kernel_size : 사용할 커널 크기

'''
imgs = 0
for n, (img, labels) in enumerate(train_loader):
    print(n, img.shape, labels.shape)
    imgs = img
    break
# 기본적인 계산 과정
net = nn.Conv2d(3, 5, 5) # (channels, filter_size)
out1 = net(Variable(imgs))
print(out1.shape)
net2 = nn.Conv2d(5, 10, 5) # (filter_size, channels, filter_size)
out2 = net2(out1)
print(out2.shape)
'''


'''
네트워크 모델 정의
'''
class my_network(nn.Module):
    def __init__(self):
        super(my_network, self).__init__()
        self.net_1 = nn.Conv2d(3, 5, 5)
        self.net_2 = nn.Conv2d(5, 10, 5)

    def forward(self, x):
        out1 = self.net_1(x)
        out2 = self.net_2(out1)

        return out2

imgs = 0
for n, (img, labels) in enumerate(train_loader):
    print(n, img.shape, labels.shape)
    imgs = img
    break

model = my_network()
result = model(Variable(imgs))
print(result.shape)