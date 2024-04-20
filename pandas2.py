#学习链接：https://blog.csdn.net/TeFuirnever/article/details/94331545?spm=1001.2014.3001.5506
#Series数据结构

import pandas as pd
import numpy as np


#一、Series创建
#1、pd.Series([list]，index=[list])
#参数list；index为可选参数，若不填写则默认index从0开始；若填写则index长度应该与value长度相等。
s=pd.Series([1,2,3,4,5],index=['a','b','c','f','e'])
print(s)
print('')

#2、pd.Series({dict})
#以一字典结构为参数
s=pd.Series({'a':1,'b':2,'c':3,'f':4,'e':5})
print(s)
print('')


#二、Series取值
#s[index] or s[[index的list]]
#取值操作类似数组，当取不连续的多个值时可以以list为参数
v = np.random.random_sample(50)
s = pd.Series(v)
s1 = s[[3, 13, 23, 33]]
s2 = s[3:13]
s3 = s[43]
print("s1", s1)
print('')
print("s2", s2)
print('')
print("s3", s3)
print('')

#Series取头和尾的值
#.head(n)；.tail(n)
#取出头n行或尾n行，n为可选参数，若不填默认5
print("s.head()", s.head())
print('')
print("s.head(3)", s.head(3))
print('')
print("s.tail()", s.tail())
print('')
print("s.head(3)", s.head(3))
print('')

#三、Series常用操作
v = [10, 3, 2, 2, np.nan]
v = pd.Series(v)
print("len():", len(v))  # Series长度,包括NaN
print("shape():", np.shape(v))  # 矩阵形状，（，）
print("count():", v.count())  # Series长度，不包括NaN
print("unique():", v.unique())  # 出现不重复values值
print("value_counts():\n", v.value_counts())  # 统计value值出现次数
print('')

#Series加法
sum = v[1:3] + v[1:3]
sum1 = v[1:4] + v[1:4]
sum2 = v[1:3] + v[1:4]
sum3 = v[:3] + v[1:]
print("sum", sum)
print("sum1", sum1)
print("sum2", sum2)
print("sum3", sum3)
print('')

#Series查找
#1、范围查找
s = {"ton": 20, "mary": 18, "jack": 19, "jim": 22, "lj": 24, "car": None}
sa = pd.Series(s,name="new")#给创建的序列添加名字new
print(sa[sa>19])
print('')

#2、中位数
print("sa.median()", sa.median())
print('')

#Series赋值
sa['ton'] = 99
print(sa)