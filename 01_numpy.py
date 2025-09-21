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


arr1=np.arange(12)

print(arr1)
# -1可以自己计算  升维
arr2=arr1.reshape((4,-1))
print(arr2)

# 降维
arr2=arr2.reshape(-1)
print(arr2)


# 一维向量  二维数组