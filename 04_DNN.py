import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 小梯度批量下降
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
# x1=torch.rand(1000,1)
# x2=torch.rand(1000,1)
# x3=torch.rand(1000,1)
# y1=((x1+x2+x3)<1).float()
# y2=((1<(x1+x2+x3))&((x1+x2+x3)<2)).float()
# y3=((x1+x2+x3)>2).float()
# Data=torch.cat([x1,x2,x3,y1,y2,y3],axis=1)
# Data=Data.to('cuda:0')
# train_size=int(len(Data)*0.7)#训练集的样本数量
# test_size=len(Data)-train_size#测试集的样本数量
# Data=Data[torch.randperm(Data.size(0)),:]#打乱样本数据
# train_Data=Data[:train_size,:]#训练集样本
# test_Data=Data[train_size:,:]#测试集样本
# class DNN(nn.Module):
#     def __init__(self):
#         # 搭建神经网络各层
#         super(DNN,self).__init__()
#         self.net=nn.Sequential(
#             nn.Linear(3,5),nn.ReLU(),  #第1层：全连接层
#             nn.Linear(5,5),nn.ReLU(),  #第2层：全连接层
#             nn.Linear(5,5),nn.ReLU(),  #第4层：全连接层
#             nn.Linear(5,3)             #第4层：全连接层
#
#         )
#     def forward(self,x):
#             '''前向传播'''
#             y=self.net(x)
#             return y
#
# model=DNN().to('cuda:0')
# print(model)
#
# #损失函数的选择
# loss_fn=nn.MSELoss()
# #优化算法的选择
# learning_rate=0.01
# optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#训练网络
# epochs=1000
# losses=[]
# X=train_Data[:, :3]#前三行为输入特征
# Y=train_Data[:, -3:]#后三列为输出特征
# for epoch in range(epochs):
#    Pred=model(X)#一次前向传播
#    loss=loss_fn(Pred,Y)#计算损失函数
#    losses.append(loss.item())#记录损失函数的变化
#    optimizer.zero_grad()#清理上一轮滞留的梯度
#    loss.backward()#一次反向传播
#    optimizer.step()#优化内部参数
#

# Fig=plt.figure()
# plt.plot(range(epochs),losses)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()

#测试网络
# X=train_Data[:, :3]#前三行为输入特征
# Y=train_Data[:, -3:]#后三列为输出特征
#
# with torch.no_grad():
#     Pred=model(X)
#     Pred[:,torch.argmax(Pred,axis=1)]=1
#     Pred[Pred!=1]=0
#     correct=torch.sum((Pred==Y).all(1))#预测正确的样本
#     total=Y.size(0)
#     print(f'测试集精准度：{100*correct/total}%')

# 1. 方法一：添加安全全局（推荐，符合weights_only=True的安全设置）
# torch.serialization.add_safe_globals([DNN])  # 允许加载DNN类
# 2. 方法二：加载时关闭weights_only（简单但需信任模型来源）
# new_model = torch.load('model.pth', weights_only=False)
# # 保存网络
# torch.save(model,'model.pth')
# # 导入网络
# new_model = torch.load('model.pth', weights_only=False)

# 测试网络
#测试网络
# X=train_Data[:, :3]#前三行为输入特征
# Y=train_Data[:, -3:]#后三列为输出特征
#
# with torch.no_grad():
#     Pred=new_model(X)
#     Pred[:,torch.argmax(Pred,axis=1)]=1
#     Pred[Pred!=1]=0
#     correct=torch.sum((Pred==Y).all(1))#预测正确的样本
#     total=Y.size(0)
#     print(f'测试集精准度：{100*correct/total}%')

# 梯度下降
# df=pd.read_csv('Data.csv',index_col=0)
# arr=df.values
# arr=arr.astype(np.float32)
# ts=torch.tensor(arr)
# ts=ts.to('cuda:0')
# print(ts.shape)
# train_size=int(len(ts)*0.7)#训练集的样本数量
# test_size=len(ts)-train_size#测试集的样本数量
# ts=ts[torch.randperm(ts.size(0)),:]#打乱样本数据
# train_Data=ts[:train_size,:]#训练集样本
# test_Data=ts[train_size:,:]#测试集样本
#
# class DNN(nn.Module):
#     def __init__(self):
#         # 搭建神经网络各层
#         super(DNN,self).__init__()
#         self.net=nn.Sequential(
#             nn.Linear(8,32),nn.Sigmoid(),  #第1层：全连接层
#             nn.Linear(32,8),nn.Sigmoid(),  #第2层：全连接层
#             nn.Linear(8,4),nn.Sigmoid(),  #第4层：全连接层
#             nn.Linear(4,1) ,nn.Sigmoid()            #第4层：全连接层
#
#         )
#     def forward(self,x):
#             '''前向传播'''
#             y=self.net(x)
#             return y
# model=DNN().to('cuda:0')
# print(model)

# 训练网络
# loss_fn=nn.BCELoss(reduction='mean')
# learning_rate=0.005
# optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
# epochs=5000
# losses=[]
# X=train_Data[:,:-1]
# Y=train_Data[:,-1].reshape((-1,1))
# for epoch in range(epochs):
#     Pred=model(X)
#     loss=loss_fn(Pred,Y)
#     losses.append(loss.item())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# Fig=plt.figure()
# plt.plot(range(epochs),losses)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()

# 测试网络
# X=test_Data[:,:-1]
# Y=test_Data[:,-1].reshape((-1,1))
# with torch.no_grad():
#     Pred=model(X)
#     Pred[Pred>=0.5]=1
#     Pred[Pred<0.5]=0
#     correct=torch.sum((Pred==Y).all(1))
#     total=Y.size(0)
#     print(f'测试集精确度：{100*correct/total}%')
#





# 小批量梯度下降

#
#
# class MyData(Dataset):
#     def __init__(self, filepath):
#         df=pd.read_csv(filepath,index_col=0)
#         arr=df.values
#         arr=arr.astype(np.float32)
#         ts=torch.tensor(arr)
#         ts=ts.to('cuda:0')
#         self.X=ts[:,:-1]
#         self.Y = ts[:,-1].reshape((-1,1))
#         self.len=ts.shape[0]
#     def __getitem__(self,index):
#         return self.X[index],self.Y[index]
#     def __len__(self):
#         return self.len
#
# Data=MyData('Data.csv')
# train_size=int(len(Data)*0.7)
# test_size=len(Data)-train_size
# train_Data,test_Data=random_split(Data,[train_size,test_size])
#
# # 批次加载器
# train_loader=DataLoader(dataset=train_Data,shuffle=True,batch_size=128)#shuffle洗牌
# test_loader=DataLoader(dataset=test_Data,shuffle=False,batch_size=64)
# class DNN(nn.Module):
#     def __init__(self):
#         # 搭建神经网络各层
#         super(DNN,self).__init__()
#         self.net=nn.Sequential(
#             nn.Linear(8,32),nn.Sigmoid(),  #第1层：全连接层
#             nn.Linear(32,8),nn.Sigmoid(),  #第2层：全连接层
#             nn.Linear(8,4),nn.Sigmoid(),  #第4层：全连接层
#             nn.Linear(4,1) ,nn.Sigmoid()            #第4层：全连接层
#
#         )
#     def forward(self,x):
#             '''前向传播'''
#             y=self.net(x)
#             return y
# model=DNN().to('cuda:0')
# print(model)
# loss_fn=nn.BCELoss(reduction='mean')
# learning_rate=0.005
# optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
# epochs=500
# losses=[]
#
# for epoch in range(epochs):
#   for (x,y) in train_loader:#注意缩进
#     Pred=model(x)
#     loss=loss_fn(Pred,y)
#     losses.append(loss.item())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# Fig=plt.figure()
# plt.plot(range(len(losses)),losses)
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# plt.show()
#
#
# correct=0
# total=0
# with torch.no_grad():
#  for (X,Y) in train_loader:  # 注意缩进
#     Pred=model(X)
#     Pred[Pred>=0.5]=1
#     Pred[Pred<0.5]=0
#     correct+=torch.sum((Pred==Y).all(1))
#     total+=Y.size(0)
# print(f'测试集精确度：{100*correct/total}%')


