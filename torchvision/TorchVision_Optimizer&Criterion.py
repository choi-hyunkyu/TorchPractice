'''
Loss는 어디서 생기는지,
input과 model을 가지고 output을 만들어내는 과정에서 label과 output이 정확이 같지 않다.
Type 유형
    회귀(Regression)
    이진분류(Binary Classification) -> output의 크기가 1, 출력이 한 개
    다중클래스분류(Multi Class Classification)
출력층에 쓰이는 활성화 함수
    항등사항(=나오는 대로 사용)
    Sigmoid Function
    Logistic Fuction
    ReLU Function
    Softmax Fucntion -> output의 개수가 10개면, output의 합이 1이 되게 만들어 줌
오차 함수(Error Function = Loss Function)
    제곱오차(Squared Error)
    BCELoss
    CrossEntropy
'''

'''
BackPropagation
    경사 하강법(Gradient Descent Method)
    Weight가 x일때 E(w)가 최소인 것을 찾는 것이 기본적인 원리
'''

'''
Optimizer(backpropagation 과정에서 사용되는 최적화 함수)
    Adadelta
    Adagrad
    Adam
    SparseAdam
    Adamax
    ASGD
    LBFGS
    RMSprop
    Rprop
    SGD
'''

'''
데이터 정의
'''
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

transfrom = transforms.Compose([transforms.Resize((100, 100)), # 사이즈 조정
                                transforms.ToTensor(), # 텐서로 변환
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


'''
커스텀 데이터 셋 구성
'''
# imfd 폴더에 아래 폴더안에 이미지가 들어가 있어야 불러올 수 있음
trainset = torchvision.datasets.ImageFolder(root = './imfd_train', transform = transfrom)
testset = torchvision.datasets.ImageFolder(root = './imfd_test', transform = transfrom)
'''
데이터 로더
'''
# 배치사이즈가 커지면 이미지를 가지고 오는데 걸리는 시간이 걸리는데, 
# CPU가 이미를 가지고 올 때 CPU 프로세서를 얼마나 사용할 지 결정
train_loader = DataLoader(trainset, batch_size = 10, shuffle = True, num_workers = 0)
test_loader = DataLoader(testset, batch_size = 10, shuffle = False, num_workers = 0)

'''
모델 정의
'''
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class my_network(nn.Module):
    def __init__(self):
        super(my_network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.fc1 = nn.Linear(10 * 22 * 22, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out1 = F.relu(self.conv1(x), inplace = True)
        out2 = F.max_pool2d(out1, (2, 2))
        out3 = F.relu(self.conv2(out2), inplace = True)
        out4 = F.max_pool2d(out3, (2, 2))
        # fully-connected layer에 입력될 때 한 개의 열의 형태로 입력해야 하므로,
        # 데이터셋의 batch_size와 matrix 형태 (30 x 5 x 5)를 곱한 형태로 입력한다.
        out5 = out4.view(10, 10 * 22 * 22)
        out6 = F.relu(self.fc1(out5), inplace = True)
        out = F.relu(self.fc2(out6), inplace = True)

        return out

model = my_network()

'''
옵티마이저, 손실함수 정의
'''
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
criterion = nn.CrossEntropyLoss()

'''
학습
'''
nb_epochs = 500
for epoch in range(nb_epochs):
    for batch_idx, samples in enumerate(train_loader):
        x_train, y_train = samples
        x_train, y_train = Variable(x_train), Variable(y_train)

        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = criterion(hypothesis, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print("Epoch: {}/{} | Batch: {}/{} | Loss: {:.6f}".format(epoch + 1, nb_epochs, batch_idx + 1, len(train_loader), loss))
print('finished')

# 모델 저장
torch.save(model, './data/' + 'torchvision_model.pt')

# 모델 불러오기
import torch
model =  torch.load('./data/' + 'torchvision_model.pt')

'''
평가
'''
total = 0.0
correct = 0.0
for samples in test_loader:
    x_test, y_test = samples
    result = model(Variable(x_test))
    _, predicted = torch.max(result.data, 1)
    total += y_test.size(0)
    correct += (predicted == y_test).sum()
print('Accurracy of the network in the 10 test images: {:.2f}'.format(100 * correct/total))