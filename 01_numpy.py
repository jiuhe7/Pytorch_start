import numpy as np
from fontTools.feaLib.ast import simplify_name_attributes

# # 整数型数组
# arr1=np.array([1,2,3])
# print(arr1)
# # 浮点型数组
# arr2=np.array([1,3.1,4.7])
# print(arr2)
# # 类型与数组原类型一致
# # 同化定理
# arr1[0]=123.5
# print(arr1)
#
# arr2[0]=123
# print(arr2)
#
# # 共同改变定理
# arr3=arr1.astype(float)
# print(arr3)
# arr4=arr1.astype(int)
# print(arr4)
# 整数型数组与浮点数（组）做运算 or 遇到除法（即使是除以整数）


# 一维数组 x  (x,)
#二维数组 (x,y)
#三维数组 (x,y，z)

# arr5=np.ones(3)
# print(arr5)
#
# arr5=np.ones((3,4))
# print(arr5)
#
# arr5=np.ones(((3,4,5)))
# print(arr5)
#
# print(arr5.shape)


# arr1=np.arange(12)
#
# print(arr1)
# # -1可以自己计算  升维
# arr2=arr1.reshape((4,-1))
# print(arr2)
#
# # 降维
# arr2=arr2.reshape(-1)
# print(arr2)


# 一维向量  二维数组

# arr1=np.array([1,4,88])
# print(arr1)
#
#
# arr1=np.array([[1,4,88],[1,4,88]])
# print(arr1)

# # 递增数组
# arr1=np.arange(10)
# print(arr1)
#
#
# arr1=np.arange(19,30)
# print(arr1)
#
# arr1=np.arange(0,21,2)
# print(arr1)

# 同值数组,float

# arr1=np.zeros((2,4))
# print(arr1)
#
# arr1=np.ones(3)*7
# print(arr1)

# 随机数组
# 0-1均匀分布
# arr=np.random.random((3,4))
# print(arr)
#
# arr=np.random.random((3,4))*40+60
# print(arr)


# 整数型随机分布
# arr1=np.random.randint(10,100,(1,15))
# 10-100 的整数
# print(arr1)
#
# arr=(40*np.random.random((3,4))).astype(int)+60
# print(arr)

# 正态分布 (均值，标准差，形状)
# arr=np.random.normal(0,1,(3,5))
# print(arr)
# 标准正态
# arr=np.random.randn(2,3)
# print(arr)


# 访问向量

# arr=np.arange(1,10)
# print(arr)
#
# print(arr[3])
# print(arr[-2])
#
# # 修改数组
# arr[0]=100
# print(arr)

#
# arr=np.array([[2,4],[3,7],[6,1]])
# print(arr)
# print(arr[2,0])
#

# 花式索引
# arr=np.array([[2,4],[3,7],[6,1]])
# print(arr)
#
# print(arr[[1]])

# arr=np.arange(1,17).reshape(4,4)
# print(arr)
# # print(arr[[0,1,2],[0,1,2]])
#
# # 修改数组元素
# arr[[0,1,2,3],[0,1,2,3]]=100
# print(arr)
#

# 向量切片
# arr=np.arange(0,9)
# print(arr)
# print(arr[1:4])
#
# print(arr[1:])
#
# print(arr[:4])
# 间隔取样
# print(arr[::2])
# print(arr[::3])
# print(arr[1:-1:2])


# 矩阵切片
#
# arr=np.arange(1,21).reshape(4,5)
# print(arr)
# print(arr[1:3,1:-1])
# print(arr[::3,::2])
#
#提取矩阵的行
# arr=np.arange(1,21).reshape(4,5)
# print(arr)
# 第二行
# print(arr[2,:])
#1-2行
# print(arr[1:3,:])
# 行可以简写
# print(arr[2])

#提取矩阵的列
#一列输出的是向量
# cut=arr[:,2]
# cut=cut.reshape((-1,1))
# print(cut)

# 备份切片新变量
# arr=np.arange(10)
# print(arr)
# copy=arr[:3].copy()
#
# copy[0]=100
# print(copy)
# print(arr)

# 赋值不会创建新数组

# arr1=arr
# arr1[0]=100
# print(arr)

# 转置只对矩阵有效
# arr1=arr.reshape((2,-1))
# print(arr1.T)

# 矩阵的翻转
# 上下翻转np.flipud,左右翻转np.fliplr,向量只能上下翻转,因为向量是竖着的
# print(np.flipud(arr))
#
# arr=np.arange(1,21).reshape(4,5)
# # print(arr)
# print(np.flipud(arr))
# print("---------------------------")
# print(np.fliplr(arr))

# 向量拼接
# arr=np.arange(10)
# arr1=np.arange(10,21)
# arr2=np.concatenate([arr,arr1])
# print(arr2)
# 矩阵拼接
# arr=np.arange(10).reshape(2,-1)
# arr1=np.arange(10,20).reshape(2,-1)
# # 竖向拼接
# arr2=np.concatenate([arr,arr1])
# print(arr2)
# # 横向拼接
# arr2=np.concatenate([arr,arr1],axis=1)
# print(arr2)

# 向量分裂
# arr=np.arange(10,100,10)
# print(arr)
#
# arr1,arr2,arr3=np.split(arr,[2,8])
# print(arr1)
# print(arr2)
# print(arr3)

# arr=np.arange(1,9).reshape(2,4)
# print(arr,'\n')
# arr1,arr2=np.split(arr,[1])
# print(arr1,'\n')
# print(arr2)
# arr1,arr2,arr3=np.split(arr,[1,3],axis=1)
# print(arr1,'\n')
# print(arr2,'\n')
# print(arr3)

#数组的运算
# arr=np.arange(1,9).reshape(2,-1)
# print(arr)
#
# print(arr/1)

# arr=np.arange(-1,-9,-1).reshape(2,-1)
#
# print(arr*arr)


# 广播（适配）
#向量
# arr=np.array([1,0,10])
# print(arr)
# # 矩阵
# arr1=np.random.random((10,3))
# print(arr1)
# print(arr1*arr)

# arr=np.ones((3,5))
# print(arr)
# arr1=np.arange(3).reshape(3,1)
# print(arr1)
# print(arr*arr1)

#  矩阵乘积 混有向量的输出结果必为向量
# arr1=np.arange(5)
# arr2=np.arange(15).reshape(5,3)
# print(np.dot(arr1,arr2))



# 数学函数
# 绝对值

# arr=np.array([-10,0,10])
# print(arr)
# print(np.abs(arr))

# 三角函数

# theta=np.arange(3)*np.pi/2
# print(theta)
# sin_v=np.sin(theta)
# cos_v=np.cos(theta)
# tan_v=np.tan(theta)
#
# print(sin_v,'\n',cos_v,'\n',tan_v)

# 指数函数

x=np.arange(1,4)
print('e^x=',np.exp(x))
print('2^x=',2**x)
print('10^x=',10**x)