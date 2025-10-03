import torch
import torch.nn as nn
from sympy import Predicate
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
# 设定下载参数
transform=transforms.Compose([transforms.ToTensor()
,transforms.Normalize(0.1307,0.3081)])
# 下载训练集和测试集
train_Data=datasets.MNIST(
    root='D:\\pycharm\\Pytorch_start',
    train=True,
    download=False,
    transform=transform
)
test_Data=datasets.MNIST(
    root='D:\\pycharm\\Pytorch_start',
    train=False,
    download=False,
    transform=transform
)
train_loader=DataLoader(dataset=train_Data,shuffle=True,batch_size=64)#shuffle洗牌
test_loader=DataLoader(dataset=test_Data,shuffle=False,batch_size=64)
class DNN(nn.Module):
    def __init__(self):
        # 搭建神经网络各层
        super(DNN,self).__init__()
        self.net=nn.Sequential(
            nn.Flatten(),#先把图像铺平成一维
            nn.Linear(784,512),nn.ReLU(),
            nn.Linear(512,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,64) ,nn.ReLU(),
            nn.Linear(64,10)

        )
    def forward(self,x):
            '''前向传播'''
            y=self.net(x)
            return y
model=DNN().to('cuda:0')
print(model)
loss_fn=nn.CrossEntropyLoss()#自带softmax激活函数
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.5)
epochs=5
losses=[]
for epoch in range(epochs):
    for (x,y) in train_loader:
        x,y=x.to('cuda:0'),y.to('cuda:0')
        Pred = model(x)
        loss=loss_fn(Pred,y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
Fig=plt.figure()
plt.plot(range(len(losses)),losses)
plt.show()

correct=0
total=0
with torch.no_grad():
 for (x,y) in train_loader:  # 注意缩进
    x, y = x.to('cuda:0'), y.to('cuda:0')
    Pred=model(x)
    _,predicted=torch.max(Pred.data,dim=1)
    correct+=torch.sum((predicted==y))
    total+=y.size(0)
print(f'测试集精确度：{100*correct/total}%')