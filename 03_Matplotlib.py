import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
import numpy as np
# Matlab方式
# Fig1=plt.figure()
# x=[1,2,3,4,5]
# y1=[1,8,27,64,125]
# y2=[26,56,71,64,125]
# y3=[38,86,33,34,95]
# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,y3)
# 保存图像
# plt.savefig('figure.svg', format='svg')
# plt.show()


# 面向对象方式
# Fig2=plt.figure()
# ax2=plt.axes()
# ax2.plot(x,y)
# plt.show()

# 绘制多个子图
# Matlab
# Fig1=plt.figure()
# x=[1,2,3,4,5]
# y1=[1,8,27,64,125]
# y2=[26,56,71,64,125]
# y3=[78,86,88,89,95]
# plt.subplot(3,1,1),plt.plot(x,y1)
# plt.subplot(3,1,2),plt.plot(x,y2)
# plt.subplot(3,1,3),plt.plot(x,y3)
# plt.show()

# 面向对象
# Fig2,ax2=plt.subplots(3)
# ax2[0].plot(x,y1)
# ax2[1].plot(x,y2)
# ax2[2].plot(x,y3)
# plt.show()



# 二维图

# x=[1,2,3,4,5]
# y1=[0,1,2,3,4]
# y2=[1,2,3,4,5]
# y3=[2,3,4,5,6]
# y4=[3,4,5,6,7]
# y5=[4,5,6,7,8]
# y6=[5,6,7,8,9]
# Fig1=plt.figure()
#
# plt.plot(x,y1,color='#5B608D',linestyle='-',linewidth=0.8,marker='o',markersize=3)
#
# plt.plot(x,y2,color='#5B608D',linestyle='-.',linewidth=0.9,marker='^',markersize=4)
# plt.plot(x,y3,color='#F8F8FF',linestyle=':',linewidth=1,marker='s',markersize=5)
# # 隐藏
# plt.plot(x,y4,color='#FFDAB9',linestyle=' ',linewidth=1.4,marker='D',markersize=6)
#
# plt.plot(x,y5,color='#000080',linestyle='--',linewidth=1.6,marker='s',markersize=7)
# # 设置标记 . o s D ^
# plt.plot(x,y6,color='#000000',linestyle='-',linewidth=2.6,marker='o',markersize=8)
# plt.show()

#网格图
# x=np.linspace(0,10,1000)
# l=np.sin(x)*np.cos(x).reshape(-1,1)
# Fig1=plt.figure()
# plt.imshow(l)
# plt.colorbar()
# plt.show()


# 统计图
# data=np.random.randn(10000)
# Fig1=plt.figure()
# # bins 区间  alpha透明度  histtype图表类型
# plt.hist(data,bins=60,alpha=1,histtype='stepfilled',color='#FFE4E1',edgecolor='#708090')
# plt.show()

# 图窗属性
# 设置坐标轴上下限
# lim法
# x=[1,2,3,4,5]
# y=[1,8,27,64,125]
# Fig1=plt.figure()
# plt.plot(x,y)
# plt.title('This is the title')
# plt.xlabel('This is the xlabel')
# plt.ylabel('This is the ylabel')
# plt.xlim(1,5)
# plt.ylim(1,140)
# plt.show()

#图例

Fig1=plt.figure()
x=[1,2,3,4,5]
y1=[1,8,27,64,125]
y2=[26,56,71,99,111]
y3=[78,86,88,89,95]
plt.plot(x,y1,label='y1')
plt.plot(x,y2,label='y2')
plt.plot(x,y3,label='y3')
# upper center lower | left right center  frameon边框
plt.legend(loc='lower right',frameon=False)
# 网格
plt.grid()
plt.show()