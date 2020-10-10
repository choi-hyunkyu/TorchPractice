'''
정확도
'''
def accuracy(out, y_train):
    hypothesis = torch.argmax(out, dim = 1)
    return (hypothesis == y_train).float().mean()