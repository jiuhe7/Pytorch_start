import numpy as np
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




arr=np.arange(1,21).reshape(4,5)
print(arr)












