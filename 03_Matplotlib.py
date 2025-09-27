import matplotlib.pyplot as plt
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
