#学习链接:https://blog.csdn.net/yiyele/article/details/80605909?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171232701116800226514823%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171232701116800226514823&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-80605909-null-null.142^v100^pc_search_result_base1&utm_term=pandas&spm=1018.2226.3001.4187

import numpy as np
import pandas as pd
#首先导入pandas库，一般都会用到numpy库

df = pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006],
#pd.DataFrame用法查询链接:https://blog.csdn.net/XDXDXDXDX111/article/details/133523012?spm=1001.2014.3001.5506
 "date":pd.date_range('20130102', periods=6),
#pd.date_range用法查询链接:https://blog.csdn.net/m0_46589710/article/details/105383077?spm=1001.2014.3001.5506
  "city":['Beijing ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai', 'BEIJING '],
 "age":[23,44,54,32,34,32],
 "category":['100-A','100-B','110-A','110-C','210-A','130-F'],
  "price":[1200,np.nan,2133,5433,np.nan,4432]},
  columns =['id','date','city','category','age','price'])
#用pandas创建数据表

#------------------------------------------------------------------------------------------------
#一、数据表信息查看

df.info()
#数据表基本信息（维度、列名称、数据格式、所占空间等）

print('')
print(df.shape)
#维度查看

print('')
print(df.dtypes)
#每一列数据的格式

print('')
print(df['id'].dtype)
#某一列格式

print('')
print(df.isnull())
#空值

print('')
print(df['id'].isnull())
#查看某一列空值

print('')
print(df['id'].unique())
#查看某一列的唯一值

print('')
print(df.values)
#查看数据表的值

print('')
print(df.columns)
#查看列名称

print('')
print(df.head(2))
print('')
print(df.tail(3))
#df.head(n)、df.tail(m):查看前n行后m行数据



#----------------------------------------------------------------------------
#二、数据表数据处理

print('')
df1=df.fillna(0)
#用0填充空值
print(df1.tail(3))
#df.fillna的用法查询链接：https://blog.csdn.net/Hudas/article/details/122923643?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171247167416800227419300%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171247167416800227419300&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-122923643-null-null.142^v100^pc_search_result_base1&utm_term=df.fillna&spm=1018.2226.3001.4187

print('')
df2=df['price'].fillna(df['price'].mean())
print(df2)
#使用列prince的均值对NA进行填充

print('')
df3=df['city']=df['city'].map(str.strip)
print(df3)
#清除city字段的字符空格
#str.strip用法查询链接：https://blog.csdn.net/lanhuazui10/article/details/119984549?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171247250716800186547593%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171247250716800186547593&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-119984549-null-null.142^v100^pc_search_result_base1&utm_term=str.strip&spm=1018.2226.3001.4187
#map方法查询链接：https://blog.csdn.net/aa150602/article/details/101141215?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171247311416800188561868%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171247311416800188561868&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-101141215-null-null.142^v100^pc_search_result_base1&utm_term=python%E4%B8%AD%E7%9A%84map%E6%96%B9%E6%B3%95&spm=1018.2226.3001.4187

print('')
df4=df['city']=df['city'].str.lower()
print(df4)
#大小写转换

print('')
df5=df2.astype('int')
print(df5)
#更改数据格式

print('')
df6=df.rename(columns={'category': 'category-size'})
print(df6)
#更改列名称

print('')
df7=df['city'].replace('sh', 'shanghai')
print(df7)
#数据替换
#replace用法查询链接：https://blog.csdn.net/wangyuxiang946/article/details/131509191?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171247385516800182176646%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171247385516800182176646&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-131509191-null-null.142^v100^pc_search_result_base1&utm_term=python%E4%B8%AD%E7%9A%84replace&spm=1018.2226.3001.4187

print('')
df8=df['city'].drop_duplicates()
#删除后出现的重复值
print(df8)
df9=df['city'].drop_duplicates(keep='last')
#删除先出现的重复值
print(df9)
#drop_duplicates()的用法查询链接：https://blog.csdn.net/qq_42453890/article/details/110916950?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171247393516800213035012%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171247393516800213035012&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-110916950-null-null.142^v100^pc_search_result_base1&utm_term=python%E4%B8%AD%E7%9A%84drop_duplicates%28%29&spm=1018.2226.3001.4187


#------------------------------------------------------------------------------
#三、对多个数据表操作

df0=pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006,1007,1008],
"gender":['male','female','male','female','male','female','male','female'],
"pay":['Y','N','Y','Y','N','Y','N','Y',],
"m-point":[10,12,20,40,40,40,30,20]})

df_inner=pd.merge(df,df0,how='inner')
df_left=pd.merge(df,df0,how='left')
df_right=pd.merge(df,df0,how='right')
df_outer=pd.merge(df,df0,how='outer')
print('')
print(df_inner)
print(df_left)
print(df_right)
print(df_outer)
#merge用法查询链接：https://blog.csdn.net/brucewong0516/article/details/82707492?ops_request_misc=&request_id=&biz_id=102&utm_term=pd.merge&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-82707492.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187

print()
result = df.append(df0)
print(result)
#append用法查询链接：https://blog.csdn.net/wangyuxiang946/article/details/122142534?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171247822916800182112870%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171247822916800182112870&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-122142534-null-null.142^v100^pc_search_result_base1&utm_term=append&spm=1018.2226.3001.4187

print()
dfn=df.set_index('id')
dfm=df0.set_index('id')
#set_index用法查询链接：https://blog.csdn.net/Ajdidfj/article/details/123178391?ops_request_misc=&request_id=&biz_id=102&utm_term=setindex&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-123178391.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
result1 = dfm.join(dfn,on='id')
print(result1)
#join用法查询链接:https://blog.csdn.net/bitcarmanlee/article/details/113311113?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171247902616800182132881%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171247902616800182132881&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-113311113-null-null.142^v100^pc_search_result_base1&utm_term=ValueError%3A%20columns%20overlap%20but%20no%20suffix%20specified%3A%20Index%28%5Bid%5D%2C%20dtype%3Dobject%29&spm=1018.2226.3001.4187

print()
res = pd.concat([df,df0], axis=1, join='inner')
print(res)
#concat用法查询链接：https://blog.csdn.net/Hudas/article/details/123009834?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171248106316800197012150%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171248106316800197012150&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123009834-null-null.142^v100^pc_search_result_base1&utm_term=pd.concat&spm=1018.2226.3001.4187

print()
dfs=df_outer.sort_values(by=['age'])
print(dfs)
#按照特定列的值排序

print()
dfs2=df_outer.sort_index()
print(dfs2)
#按照索引列排序

print()
df_outer['group'] = np.where(df_outer['price'] > 3000,'high','low')
#np.where用法查询链接：https://blog.csdn.net/Kingyanhui/article/details/121385646?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171247418916800188520395%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171247418916800188520395&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-121385646-null-null.142^v100^pc_search_result_base1&utm_term=np.where&spm=1018.2226.3001.4187
print(df_outer)
#如果price列的值>3000，group列显示high，否则显示low

print()
df_outer.loc[(df_outer['city'] == 'beijing') & (df_outer['price'] >= 4000), 'sign']=1
print(df_outer)
#对符合多个条件的数据进行分组标记
#df.loc用法查询：https://blog.csdn.net/weixin_47139649/article/details/126854365?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171248814016800185822738%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171248814016800185822738&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-126854365-null-null.142^v100^pc_search_result_base1&utm_term=df.loc&spm=1018.2226.3001.4187

print()
dfp=pd.DataFrame((x.split('-') for x in df_inner['category']),index=df_inner.index,columns=['category','size'])
print(dfp)
#对category字段的值依次进行分列，并创建数据表，索引值为df_inner的索引列，列名称为category和size
print()
dfp2=pd.DataFrame((x.split('l') for x in df_inner['gender']),index=df_inner.index,columns=['pre','out'])
print(dfp2)
#同上


#------------------------------------------------------------------------------
#四、对数据操作
print()
dft=df_inner.loc[0]
dft1=df_inner.loc[1]
print(dft)
print(dft1)
#按索引提取单行的数值

print()
dft2=df_inner.iloc[0:2]
print(dft2)
#按索引提取区域行数值

print()
dft3=df_inner.set_index('date')[:'2013-01-04']
print(dft3)
#设置日期为索引并且提取4日之前的所有数据
#写法与df_inner=df_inner.set_index('date')   dft3=df_inner[:'2013-01-04']两句一样的效果

print()
dfp3=df_inner.iloc[:3,:2] #冒号前后的数字不是索引的标签名称，而是数据所在的位置，从0开始，前三行，前两列。
dfp4=df_inner.iloc[[0,2,5],[4,5]] #提取第0、2、5行，4、5列
print(dfp3)
print(dfp4)
#iloc用法查询链接：https://blog.csdn.net/Fwuyi/article/details/123127754?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171248717816800215019425%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171248717816800215019425&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123127754-null-null.142^v100^pc_search_result_base1&utm_term=df.iloc&spm=1018.2226.3001.4187

print()
print(df_inner['city'].isin(['beijing']))
#判断city列的值是否为北京

print()
print(df_inner.loc[df_inner['city'].isin(['beijing','shanghai'])])
#判断city列里是否包含beijing和shanghai，然后将符合条件的数据提取出来

print()
print(pd.DataFrame(df_inner['category'].str[:4]))
#提取前三个字符，并生成数据表

print()
print(df_inner.loc[(df_inner['age'] > 25) & (df_inner['city'] == 'beijing'), ['id','city','age','category','gender']])
#使用“与”进行筛选
print(df_inner.loc[(df_inner['age'] > 25) | (df_inner['city'] == 'beijing'), ['id','city','age','category','gender']])
#使用“或”进行筛选
print(df_inner.loc[(df_inner['city'] != 'beijing'), ['id','city','age','category','gender']])
#使用“非”条件进行筛选
print(df_inner.loc[(df_inner['city'] != 'beijing'), ['id','city','age','category','gender']].count())
#对筛选后的数据每列进行计数
print(df_inner.loc[(df_inner['city'] != 'beijing'), ['id','city','age','category','gender']].age.count())
#对筛选后的数据按age列进行计数

print()
print(df_inner.query('city == ["beijing", "shanghai"]'))
#使用query函数进行筛选
print(df_inner.query('city == ["beijing", "shanghai"]').price.sum())
#对筛选后的结果按prince进行求和
#df.query用法查询链接：https://blog.csdn.net/alanguoo/article/details/88874742?ops_request_misc=&request_id=&biz_id=102&utm_term=df.query&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-88874742.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187

print()
print(df_inner.groupby('city').count())
#所有的列对城市进行计数汇总
print(df_inner.groupby('city')['id'].count())
#按城市对id字段进行计数
print(df_inner.groupby(['city','age'])['id'].count())
#对两个字段进行汇总计数
print(df_inner.groupby('city')['price'].agg([len,np.sum, np.mean]))
#对city字段进行汇总，并分别计算prince的合计和均值
#df.groupby用法查询链接：https://blog.csdn.net/mjm891116/article/details/124615642?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171249305516800184111541%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171249305516800184111541&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124615642-null-null.142^v100^pc_search_result_base1&utm_term=df.groupby&spm=1018.2226.3001.4187

print()
print(df_inner.sample(n=3))
#随机采样三行数据
weights = [0, 0, 0, 0, 0.5, 0.5]
print(df_inner.sample(n=2, weights=weights))
#设置采样权重，即采到的概率
print(df_inner.sample(n=6, replace=False))
#采样后不放回
print(df_inner.sample(n=6, replace=True))
#采样后放回,即可能取到重样的
#df.sample用法查询链接：https://blog.csdn.net/qq_40433737/article/details/107048681?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171249376716800184171696%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171249376716800184171696&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-107048681-null-null.142^v100^pc_search_result_base1&utm_term=df.sample&spm=1018.2226.3001.4187

print()
print(df_inner.describe().round(2).T) #round函数设置显示小数位，T表示转置
print(df_inner.describe().round(2))
#数据表描述性统计
print(df_inner['price'].std())
#计算列的标准差
print(df_inner['price'].cov(df_inner['age']))
#计算两个字段间的协方差
print(df_inner.cov())
#数据表中所有字段间的协方差
print(df_inner['price'].corr(df_inner['m-point'])) #相关系数在-1到1之间，接近1为正相关，接近-1为负相关，0为不相关
#两个字段的相关性分析
print(df_inner.corr())
#数据表的相关性分析

#------------------------------------------------------------------------
#五、数据输出
df_inner.to_excel('excel_to_python.xlsx', sheet_name='xyj')#写入Excel
df_inner.to_csv('excel_to_python.csv')#写入到CSV

#------------------------------------------------------------------------
#六、数据导入
print()
dfd= pd.DataFrame(pd.read_csv('excel_to_python.csv',header=0))
dfd2 = pd.DataFrame(pd.read_excel('excel_to_python.xlsx'))
#pd.read_csv使用方法查询链接：https://blog.csdn.net/weixin_47139649/article/details/126744842?spm=1001.2014.3001.5506
#pd.read_exxcel使用方法查询链接：https://blog.csdn.net/qq_46450354/article/details/129537002?ops_request_misc=&request_id=&biz_id=102&utm_term=pd.read_excel()%E5%8F%82%E6%95%B0%E8%AF%A6%E8%A7%A3&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-129537002.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
print(dfd)
print(dfd2)


















