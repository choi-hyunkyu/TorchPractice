import torch

# 모델 저장
torch.save(model, PATH + 'model.pt')

# 모델 불러오기
model =  torch.load(PATH + 'model.pt')
