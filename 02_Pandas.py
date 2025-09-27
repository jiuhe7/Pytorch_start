import pandas as pd
import numpy as np
from sympy.polys.polyoptions import Series

# 对象创建
# 字典创建法
# dict_v={'a':0,'b':0.25,'c':0.5,'d':0.75,'e':1}
# sr=pd.Series(dict_v)
# print(sr)

# 数组创建法
# v=[0,0.25,0.5,0.75,1]
# key=['a','b','c','d','e']
# sr=pd.Series(v,index=key)
#
# print(sr)
# sr=pd.Series(v)
# # Index 可以省略，省略后索引从零开始的顺序数字
# print(sr)

# # 一维对象的属性
# v=[53,24,45,75]
# k=['1号','2号','3号','4号' ]
# sr=pd.Series(v,index=k)
# print(sr)
# # values 数组
# print(sr.values)
# print(sr.index)

# 二维对象的创建
# 字典创建法
# v1=[53,24,45,75]
# k=['1号','2号','3号','4号' ]
# sr1=pd.Series(v1,index=k)
# print(sr1)
# v2=['男','女','女','女']
# i=['1号','2号','3号','4号' ]
# sr2=pd.Series(v2,index=i)
# print(sr2)
# df=pd.DataFrame({'年龄':sr1,'性别':sr2})
# print(df)

# 数组创建法
# v=np.array([[53,'男'],[24,'女'],[45,'女'],[75,'女']])
# i=['1号','2号','3号','4号' ]
# c=['年龄','性别']
# df=pd.DataFrame(v,index=i,columns=c)
# print(df)

# 对象的索引  .loc 显示索引 。iloc 隐式索引
# 一维
# v=[53,24,45,75]
# k=['1号','2号','3号','4号' ]
# sr=pd.Series(v,index=k)
# print(sr)
# print(sr.loc['3号'])
# # 花式索引
# print(sr.loc[['1号','3号']])
# # 修改元素
# sr.loc['3号']=100
# print(sr.loc['3号'])

# print(sr.iloc[[0,2]])
# # 修改元素
# sr.iloc[2]=100
# print(sr.iloc[2])

# 访问切片 显示索引覆盖最后元素 隐式缩阴跟之前一样
# print(sr.loc['1号':'3号'])
# print(sr.iloc[0:3])

# 二维
# 显式
# v=([[53,'男'],[24,'女'],[45,'女'],[75,'女']])
# i=['1号','2号','3号','4号' ]
# c=['年龄','性别']
# df=pd.DataFrame(v,index=i,columns=c)
# print(df)
# 显式
# print(df.loc['1号','年龄'])
# print(df.loc[['1号','3号'],['性别','年龄']])
# 隐式
# print(df.iloc[0,0])
# print(df.iloc[[0,2],[1,0]])

# 切片
# print(df.loc['1号':'3号','年龄'])
# print(df.loc['3号',:])
# print(df.loc[:,'年龄'])

# print(df.iloc[0:3,0])
# print(df.iloc[2,:])
# print(df.iloc[:,0])

# 对象的转置
# print(df.T)
# 翻转
# 左右翻转

# print(df.iloc[:,::-1])

# # 上下翻转
# print(df.iloc[::-1,:])

# 对象的重塑

# i=['1号','2号','3号','4号' ]
# v1=[10,20,30,40]
# v2=['女','男','男','女']
# v3=[1,2,3,4]
# sr1=pd.Series(v1,index=i)
# sr2=pd.Series(v2,index=i)
# sr3=pd.Series(v3,index=i)
# print(sr1)
# print(sr2)
# print(sr3)
#
# df=pd.DataFrame({"年龄":sr1,"性别":sr2})
# print(df)
# df['牌照']=sr3
#
# print(df)
# sr4=df['年龄']
# print(sr4)


# 拼接 key 可重复
# v1=[10,20,30,40]
# v2=[40,50,60]
# k1=['1号','2号','3号','4号']
# k2=['4号','5号','6号']
# sr1=pd.Series(v1,index=k1)
# sr2=pd.Series(v2,index=k2)
# print(sr1)
# print(sr2)
# sr3=pd.concat([sr1,sr2])
# print(sr3.index.is_unique)


# 一维与二维对象的合并
# v=([[53,'男'],[24,'女'],[45,'女'],[75,'女']])
# i=[1,2,3,4]
# c=[2,1]
# sr1=pd.DataFrame(v,index=i,columns=c)
# sr1=sr1.iloc[:,::-1]
# print(sr1)
# # 加列
# sr1['代码']=['a','b','c','d']
# print(sr1)
# # 加行
# sr1.loc['5']=['男','22','e']
# print(sr1)

# 二维与二维对象的合并

# v1=[[10,'女'],[20,'男'],[30,'男'],[40,'女']]
# v2=[[1,'是'],[2,'是'],[3,'是'],[4,'否']]
# v3=[[50,'男',5,'s'],[60,'女',6,'是']]
# i1=['1号','2号','3号','4号']
# i2=['1号','2号','3号','4号']
# i3=['5号','6号']
# c1=['年龄','性别']
# c2=['牌照','ikun']
# c3=['年龄','性别','牌照','ikun']
# df1=pd.DataFrame(v1,index=i1,columns=c1)
# df2=pd.DataFrame(v2,index=i2,columns=c2)
# df3=pd.DataFrame(v3,index=i3,columns=c3)
# print(df1)
# print(df2)
# print(df3)
# # 添加列属性
# df=pd.concat([df1,df2],axis=1)
# print(df)
# # 添加行个体
# df=pd.concat([df,df3])
# print(df)

# 对象与系数之间的运算
# sr=pd.Series([53,64,72],index=['1号','2号','3号'])
# print(sr)
# print(sr+10)
# v=[[53,'女'],[64,'男'],[72,'男']]
# df=pd.DataFrame(v,index=['1号','2号','3号'],columns=['年龄','性别'])
# print(df)
# df['年龄']+=10
# print(df)

# 对象与对象之间的运算

# v1=[10,20,30,40]
# k1=['1号','2号','3号','4号']
# sr1=pd.Series(v1,index=k1)
# print(sr1)
#
# v2=[1,2,3]
# k2=['1号','2号','3号']
# sr2=pd.Series(v2,index=k2)
# print(sr2)
#
# print(sr2+sr1)


# v1=[[10,'女'],[20,'男'],[30,'男'],[40,'女']]
# v2=[[1,'是'],[2,'是'],[3,'是'],[4,'否']]
# v3=[[50,'男',5,'s'],[60,'女',6,'是']]
# i1=['1号','2号','3号','4号']
# i2=['1号','2号','3号','4号']
# i3=['5号','6号']
# c1=['年龄','性别']
# c2=['牌照','ikun']
# c3=['年龄','性别','牌照','ikun']
# df1=pd.DataFrame(v1,index=i1,columns=c1)
# df2=pd.DataFrame(v2,index=i2,columns=c2)
# df3=pd.DataFrame(v3,index=i3,columns=c3)
# print(df1)
# print(df2)
#
# df1['加法']=df1['年龄']+df2['牌照']
# print(df1)

# 发现缺失值
# v1=[10,None,30,40]
# k1=['1号','2号','3号','4号']
# sr1=pd.Series(v1,index=k1)
# print(sr1.isnull())
# print(~sr1.isnull())

# v=([[53,'None'],[24,'女'],[None,'女'],[75,'女']])
# i=['1号','2号','3号','4号' ]
# c=['年龄','性别']
# df=pd.DataFrame(v,index=i,columns=c)
# print(df)
# print(df.isnull())
# print(~df.isnull())

# 剔除缺失值
# v1=[10,None,30,40]
# k1=['1号','2号','3号','4号']
# sr1=pd.Series(v1,index=k1)
# print(sr1)
# print(sr1.dropna())

# v=([[53,None],[24,'女'],[None,'女'],[75,'女'],[None,None]])
# i=['1号','2号','3号','4号','5号']
# c=['年龄','性别']
# df=pd.DataFrame(v,index=i,columns=c)
# print(df)
# print(df.dropna())
# # 不推荐剔除列属性
# # print(df.dropna(axis=1))
# # 一行全为空才剔除
# print(df.dropna(how='all'))


#填充缺失值

# v1=[10,None,30,40]
# k1=['1号','2号','3号','4号']
# sr1=pd.Series(v1,index=k1)
# print(sr1)
# # 常数（0）填充
# print(sr1.fillna(0))
# # 常数（均值）填充
# print(sr1.fillna(np.mean(sr1)))
# # 前值填充
# print(sr1.fillna(method='ffill'))
# # 后值填充
# print(sr1.fillna(method='bfill'))


#
# v=([[53,3534],[None,None],[24,56],[75,None],[56,366]])
# i=['1号','2号','3号','4号','5号']
# c=['年龄','财富']
# df=pd.DataFrame(v,index=i,columns=c)
# print(df)
# # 常数（0）填充
# print(df.fillna(0))
# # 常数（均值）填充
# print(df.fillna(np.mean(df)))
# # 前值填充
# print(df.fillna(method='ffill'))
# # 后值填充
# print(df.fillna(method='bfill'))

# 导入Excel信息
#
# df=pd.read_csv('行星数据.csv', index_col=0)

# 导入 CSV 文件并获取前 5 行数据，然后打印
# print(df.head(5))
# print(df.max())
# print(df.min())
# print(df.mean())
# print(df.std())
# print(df.sum())
# print('-----------------------------------------------')
# 描述方法
# print(df.describe())

# 数据透视
df=pd.read_csv('泰坦尼克.csv', index_col=0)
print(df.head())
# 一个特征：性别
# print(df.pivot_table('是否生还',index='性别'))
# 两个特征：性别、船舱等级
# print(df.pivot_table('是否生还',index='性别',columns='船舱等级'))

# 重置年龄列
age=pd.cut(df['年龄'],[0,25,120])
# print(age)
# 三个特征：性别、船舱等级、年龄
# print(df.pivot_table('是否生还',index=['性别',age],columns='船舱等级'))

# 三个特征：性别、船舱等级、年龄、费用
# 自动将费用分为两部分

fare=pd.qcut(df['费用'],3)
print(fare)
print(df.pivot_table('是否生还', index=['船舱等级',fare], columns=['性别',age]))