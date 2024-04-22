#学习链接：https://blog.csdn.net/a373595475/article/details/79580734?spm=1001.2014.3001.5506

import numpy as np

#一、Ndarray
#基本的ndarray是使用NumPy中的数组函数创建的,
#numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)

a = np.array([1,2,3])#一维
print("a:",a)
b = np.array([[1,2],[3,4]])#二维
print("b:",b)
c= np.array([1,2,3,4,5], ndmin =  2) #最小维度
print("c:",c)
d = np.array([1,2,3], dtype = complex) #类型为复数
print("d:",d)
print('')



#二、数据类型
#numpy.dtype(object, align, copy)
#Object：被转换为数据类型的对象。Align：如果为true，则向字段添加间隔，使其类似 C 的结构体。
#Copy：生成dtype对象的新副本，如果为flase，结果是内建数据类型对象的引用。

dt = np.dtype(np.int32)# 使用数组标量类型
print(dt)
dt = np.dtype('i4') #int8，int16，int32，int64 可替换为等价的字符串 'i1'，'i2'，'i4'，以及其他。
print(dt)
print('')


#结构化数据类型。这里声明了字段名称和相应的标量数据类型。
dt = np.dtype([('age',np.float64)]) # 首先创建结构化数据类型。
a = np.array([(10,),(20,),(30,)], dtype = dt) # 将其应用于 ndarray 对象
print(a.dtype,a)
print('')


#定义名为student的结构化数据类型，其中包含字符串字段name，整数字段age和浮点字段marks。
# 此dtype应用于ndarray对象。
student = np.dtype([('name','S20'),  ('age',  'i1'),  ('marks',  'f4')])
a = np.array([('abc',  21,  50),('xyz',  18,  75)], dtype = student)
print(a)



#三、数组属性

#ndarray.shape这一数组属性返回一个包含数组维度的元组，它也可以用于调整数组大小。
a = np.array([[1,2,3],[4,5,6]])
print(a.shape)
print('')

a = np.array([[1,2,3],[4,5,6]])
a.shape = (3,2)#调整数组大小
print(a)
b = a.reshape(6,1)#或者用reshape
print (b)
print('')


#ndarray.ndim这一数组属性返回数组的维数。
a = np.arange(24)#等间隔数字的数组
print(a.ndim)
print('')
b = a.reshape(2,4,3)#b拥有三个维度,4行3列的两个二维数组相叠
print (b.ndim,b)
print('')


#numpy.itemsize这一数组属性返回数组中每个元素的字节单位长度。
x = np.array([1,2,3,4,5], dtype = np.int8)#数组的dtype为int8（一个字节）
print (x.itemsize)
x = np.array([1,2,3,4,5], dtype = np.float32)# 数组的 dtype 现在为 float32（四个字节）
print (x.itemsize)
print('')


#numpy.flags这个函数返回了ndarray对象属性的当前值。
x = np.array([1,2,3,4,5])
print (x.flags)
print('')


#四、数组创建例程
#新的ndarray对象可以通过任何下列数组创建例程或使用低级ndarray构造函数构造。

#numpy.empty它创建指定形状和dtype的未初始化数组。它使用以下构造函数：
#numpy.empty(shape, dtype = float, order = 'C')
x = np.empty([3,2], dtype =  int)
print(x)#数组元素为随机值，因为它们未初始化
print('')


#numpy.zeros返回特定大小，以0填充的新数组。numpy.zeros(shape, dtype = float, order = 'C')
x = np.zeros(5)#含有5个0的数组，默认类型为float
print(x)
print('')
x = np.zeros((2,3), dtype = np.int)
print(x)
print('')
x = np.zeros((4,2), dtype=[('x',  'i4'),('y',  'f4'),('z','S20')])
print(x)
print('')


#numpy.ones返回特定大小，以1填充的新数组。numpy.ones(shape, dtype = None, order = 'C')
x = np.ones(5)
print(x)# 含有 5 个 1 的数组，默认类型为 float
print('')
x = np.ones([2,2], dtype =  int)
print(x)
print('')


#numpy.asarray此函数类似于numpy.array，除了它有较少的参数。
#numpy.asarray(a, dtype = None, order = None)
x =  [1,2,3]
a = np.asarray(x)
print(a)# 将列表转换为 ndarray

a = np.asarray(x, dtype =  float)
print(a)# 设置了dtype
print('')

x =  (1,2,3)# 来自元组的ndarray
a = np.asarray(x)
print(a)
print('')

x = [[(1,2,3),(4,5),(1)],[(1)]]# 来自元组列表的 ndarray
a = np.asarray(x)
print(a.shape,a,x)
print('')


#numpy.frombuffer此函数将缓冲区解释为一维数组。暴露缓冲区接口的任何对象都用作参数来返回ndarray。
#numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
s = 'Hello World'
m=s.encode()
a = np.frombuffer(m, dtype='S1')
print(a)
print('')


#numpy.fromiter此函数从任何可迭代对象构建一个ndarray对象，返回一个新的一维数组。
#numpy.fromiter(iterable, dtype, count = -1)
list = range(5)#使用 range 函数创建列表对象
print(list)
it = iter(list)  # 使用迭代器创建 ndarray
print(it)
x = np.fromiter(it, dtype =  float)
print(x)
print('')


#numpy.arange这个函数返回ndarray对象，包含给定范围内的等间隔值
#numpy.arange(start, stop, step, dtype),start默认0，step默认1，dtype若不给出则默认stop的类型，stop必须给出
x = np.arange(5)#给出了stop，省略了其他
print(x)
x = np.arange(5, dtype =  float) #设置了dtype
print(x)
x = np.arange(10,20,2)# 设置了起始值和终止值参数
print(x)
print('')


#numpy.linspace此函数类似于arange()函数。在此函数中，指定了范围之间的均匀间隔数量，而不是步长
#numpy.linspace(start, stop, num, endpoint, retstep, dtype)num为等间隔的样例数，默认50,
# endpoint默认true,即stop包含在序列中
x = np.linspace(10,20,5)
print(x)
x = np.linspace(10,20,5,endpoint=False)#将endpoint设为false
print(x)
print('')


#numpy.logspace此函数返回一个ndarray对象，其中包含在对数刻度上均匀分布的数字。
#刻度的开始和结束端点是某个底数的幂，通常为10
#numpy.logscale(start, stop, num, endpoint, base, dtype)
#num默认50，base默认10
a = np.logspace(1.0,2.0,num=10)  # 默认底数是10
print(a)
a = np.logspace(1,10,num=10,base=2)
print(a)
print('')



#五、数组操作
#基本切片是Python中基本切片概念到n维的扩展。
#通过将start，stop和step参数提供给内置的slice函数来构造一个 Python slice对象。
#此slice对象被传递给数组来提取数组的一部分。
a = np.arange(10)#ndarray对象由arange()函数创建
s = slice(2,7,2)#分别用起始，终止和步长值2，7和2定义切片对象
print(a[s])#切片对象传递给ndarray时，会对它的一部分进行切片，从索引2到7，步长为2
b = a[2:7:2]#将由冒号分隔的切片参数（start:stop:step）直接提供给ndarray对象，也可以获得相同的结果
print(b)
b = a[5]#只输入一个参数，则将返回与索引对应的单个项目
print(b)
print(a[2:])#使用x:，则从该索引向后的所有项目将被提取
print(a[2:5])#如果使用两个参数（以:分隔），则对两个索引（不包括停止索引）之间的元素以默认步骤进行切片
print('')


#用于多维ndarray
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
print(a[1:])
print('')


#切片还可以包括省略号（...），来使选择元组的长度与数组的维度相同。
#如果在行位置使用省略号，它将返回包含行中元素的ndarray。
print(a[...,1]) #返回第二列元素的数组
print(a[1,...]) #第二行切片所有元素
print(a[...,1:])#第二列向后切片所有元素
print('')


#两种类型的高级索引：整数和布尔值。高级索引始终返回数据的副本。与此相反，切片只提供了一个视图。

#整数索引
#这种机制有助于基于N维索引来获取数组中任意元素。
#每个整数数组表示该维度的下标值。当索引的元素个数就是目标ndarray的维度时，会变得相当直接。
x = np.array([[1,  2],  [3,  4],  [5,  6]])
print(x)
y = x[[0,1,2],  [0,1,0]]#行索引包含所有行号，列索引指定要选择的元素
print(y)
print('')

x = np.array([[0,  1,  2],[3,  4,  5],[6,  7,  8],[9,  10,  11]])
rows = np.array([[0,0],[3,3]])
print(rows)
cols = np.array([[0,2],[0,2]])
print(cols)
y = x[rows,cols]#获取了4X3数组中的每个角处的元素。行索引是[0,0]和[3,3]，而列索引是[0,2]和[0,2]
print(y)
y = x[1:4,[1,2]]#对列使用高级索引,但高级索引会导致复制，并且可能有不同的内存布局
print(y)
print('')


#布尔索引
#当结果对象是布尔运算（例如比较运算符）的结果时，将使用此类型的高级索引
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
print(x[x >  5])#打印出大于5的元素
a = np.array([np.nan,  1,2,np.nan,3,4,5])
print(a[~np.isnan(a)])#使用了~（取补运算符）来过滤NaN
a = np.array([1,  2+6j,  5,  3.5+5j])
print(a[np.iscomplex(a)])#从数组中过滤掉非复数元素
print('')


#广播
#广播是指NumPy在算术运算期间处理不同形状的数组的能力。
#对数组的算术运算通常在相应的元素上进行。如果两个阵列具有完全相同的形状，则这些操作被无缝执行。
a = np.array([1,2,3,4])
b = np.array([10,20,30,40])
c = a * b
print(c)
print('')
#如果两个数组的维数不相同，则元素到元素的操作是不可能的。
#然而，在NumPy中仍然可以对形状不相似的数组进行操作，因为它拥有广播功能。
#较小的数组会广播到较大数组的大小，以便使它们的形状可兼容。
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]])
b = np.array([1.0,2.0,3.0])
print(a + b)
print('')


#迭代
#NumPy 包包含一个迭代器对象numpy.nditer。
#它是一个有效的多维迭代器对象，可以用于在数组上进行迭代。
#数组的每个元素可使用 Python 的标准Iterator接口来访问。
a = np.arange(0,60,5)
a = a.reshape(3,4)
for x in np.nditer(a):
    print(x)
print('')

#迭代的顺序匹配数组的内容布局，而不考虑特定的排序。这可以通过迭代上述数组的转置来看到
b=a.T
for x in np.nditer(b):
    print(x)
print('')

#nditer对象有另一个可选参数op_flags。 其默认值为只读，但可以设置为读写或只写模式。
# 这将允许使用此迭代器修改数组元素。
for x in np.nditer(a, op_flags=['readwrite']):
    x[...]=2*x
print(a)

#如果两个数组是可广播的，nditer组合对象能够同时迭代它们。
b = np.array([1,  2,  3,  4], dtype =  int)
for x,y in np.nditer([a,b]):
    print("%d:%d"  %  (x,y))
print('')


#numpy.ndarray.flat该函数返回数组上的一维迭代器，行为类似 Python 内建的迭代器。
a = np.arange(8).reshape(2, 4)
print(a.flat[5])# 返回展开数组中的下标的对应元素

#numpy.ndarray.flatten该函数返回折叠为一维的数组副本
a = np.arange(8).reshape(2, 4)
print(a.flatten())

#numpy.ravel这个函数返回展开的一维数组，并且按需生成副本。返回的数组和输入数组拥有相同数据类型。
a = np.arange(8).reshape(2, 4)
print(a.ravel())
print('')


#numpy.transpose这个函数翻转给定数组的维度。如果可能的话它会返回一个视图。
a = np.arange(12).reshape(3, 4)
print(np.transpose(a))

#numpy.ndarray.T,该函数属于ndarray类，行为类似于numpy.transpose
a = np.arange(12).reshape(3, 4)
print(a.T)
print('')


#数组的轴：https://blog.csdn.net/m0_66106755/article/details/128713022?spm=1001.2014.3001.5506
#数组的轴：https://blog.csdn.net/qq_41721660/article/details/128129236?spm=1001.2014.3001.5506
#numpy.rollaxis该函数向后滚动特定的轴，直到一个特定位置
#numpy.rollaxis(arr, axis, start)
#arr：输入数组, axis：要向后滚动的轴，其它轴的相对位置不会改变, start：默认为零，表示完整的滚动。会滚动到特定位置
a = np.arange(8).reshape(2, 2, 2)
print(np.rollaxis(a, 2))# 将轴 2 滚动到轴 0（宽度到深度）
print(np.rollaxis(a, 2, 1))# 将轴 2 滚动到轴 1：（宽度到高度）
print('')

#numpy.swapaxes该函数交换数组的两个轴
# numpy.swapaxes(arr, axis1, axis2)
a = np.arange(8).reshape(2, 2, 2)
# 现在交换轴 0（深度方向）到轴 2（宽度方向）
print(np.swapaxes(a, 2, 0))
print('')

#numpy.expand_dims通过在指定位置插入新的轴来扩展数组形状
# numpy.expand_dims(arr, axis),axis：新轴插入的位置
x = np.array(([1, 2], [3, 4]))
y = np.expand_dims(x, axis=0)
print(y)
y = np.expand_dims(x, axis=1)# 在位置 1 插入轴
print(y)
print('')

#numpy.squeeze函数从给定数组的形状中删除一维条目
# numpy.squeeze(arr, axis),axis：整数或整数元组，用于选择形状中单一维度条目的子集
x = np.arange(9).reshape(1, 3, 3)
y = np.squeeze(x)
print(y)
print('')


#broadcast返回一个对象，该对象封装了将一个数组广播到另一个数组的结果。该函数使用两个数组作为输入参数
x = np.array([[1], [2], [3]])
y = np.array([4, 5, 6])
b = np.broadcast(x, y)# 对 y 广播 x
print(b.shape)# shape 属性返回广播对象的形状
print('')


#numpy.concatenate数组的连接是指连接。 此函数用于沿指定轴连接相同形状的两个或多个数组。
# numpy.concatenate((a1, a2, ...), axis)
#a1, a2, ...：相同类型的数组序列，axis：沿着它连接数组的轴，默认为 0
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.concatenate((a, b)))#沿轴 0 连接两个数组
print(np.concatenate((a, b), axis=1))#沿轴 1 连接两个数组
print('')

#numpy.stack此函数沿新轴连接数组序列
# numpy.stack(arrays, axis),axis：返回数组中的轴，输入数组沿着它来堆叠
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.stack((a, b), 0))#沿轴 0 堆叠两个数组
print(np.stack((a, b), 1))#沿轴 1 堆叠两个数组
print('')

#numpy.hstack:numpy.stack函数的变体，通过堆叠来生成水平的单个数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.hstack((a, b))
print(c)
print('')

#numpy.vstack:numpy.stack函数的变体，通过堆叠来生成竖直的单个数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.vstack((a, b))
print(c)
print('')


#numpy.split该函数沿特定的轴将数组分割为子数组
# numpy.split(ary, indices_or_sections, axis)
#ary：被分割的输入数组,
# indices_or_sections：可以是整数，表明要从输入数组创建的等大小的子数组的数量。如果此参数是一维数组，则其元素表明要创建新子数组的点。
# axis：默认为 0
a = np.arange(9)
b = np.split(a, 3)#将数组分为三个大小相等的子数组
print(b)
b = np.split(a, [4, 7])#将数组在一维数组中表明的位置分割
print(b)
print('')

#numpy.hsplit是split()函数的特例，其中轴为 1 表示水平分割，无论输入数组的维度是什么
a = np.arange(16).reshape(4, 4)
b = np.hsplit(a, 2)
print(b)
print('')

#numpy.vsplit是split()函数的特例，其中轴为 0 表示竖直分割，无论输入数组的维度是什么
a = np.arange(16).reshape(4, 4)
b = np.vsplit(a, 2)
print(b)


#numpy.resize此函数返回指定大小的新数组。 如果新大小大于原始大小，则包含原始数组中的元素的重复副本
#numpy.resize(arr, shape)
#arr：要修改大小的输入数组
#shape：返回数组的新形状
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.resize(a, (2, 2))
print(b)
print('')
b = np.resize(a, (3, 3))#修改第二个数组的大小
print(b)#a 的第一行在 b 中重复出现，因为尺寸变大了
print('')


#numpy.append此函数在输入数组的末尾添加值。 附加操作不是原地的，而是分配新的数组。
# 此外，输入数组的维度必须匹配否则将生成ValueError
#numpy.append(arr, values, axis)
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.append(a, [7, 8, 9]))#向数组添加元素
print(np.append(a, [[7, 8, 9]], axis=0))#沿轴 0 添加元素
print(np.append(a, [[5, 5, 5], [7, 8, 9]], axis=1))#沿轴 1 添加元素
print('')


#numpy.insert此函数在给定索引之前，沿给定轴在输入数组中插入值。
#如果值的类型转换为要插入，则它与输入数组不同。 插入没有原地的，函数会返回一个新数组。
# 此外，如果未提供轴，则输入数组会被展开
# numpy.insert(arr, obj, values, axis)
a = np.array([[1,2],[3,4],[5,6]])
print(np.insert(a,3,[11,12]))#未传递 Axis 参数。 在插入之前输入数组会被展开
#传递了 Axis 参数。 会广播值数组来配输入数组
print(np.insert(a,1,[11],axis = 0))#沿轴 0 广播
print(np.insert(a,1,11,axis = 1))#沿轴 1 广播
print('')


#numpy.delete此函数返回从输入数组中删除指定子数组的新数组。
# 与insert()函数的情况一样，如果未提供轴参数，则输入数组将展开。
# Numpy.delete(arr, obj, axis)
a = np.arange(12).reshape(3,4)
print(np.delete(a,5))#未传递 Axis 参数。 在插入之前输入数组会被展开
print(np.delete(a,1,axis = 1))#删除第二列
a = np.array([1,2,3,4,5,6,7,8,9,10])
print(np.delete(a, np.s_[::2]))#包含从数组中删除的替代值的切片
print('')


#numpy.unique此函数返回输入数组中的去重元素数组。
# 该函数能够返回一个元组，包含去重数组和相关索引的数组。索引的性质取决于函数调用中返回参数的类型。
# numpy.unique(arr, return_index, return_inverse, return_counts)
#arr：输入数组，如果不是一维数组则会展开
#return_index：如果为true，返回输入数组中的元素下标
#return_inverse：如果为true，返回去重数组的下标，它可以用于重构输入数组
#return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数
a = np.array([5,2,6,2,7,5,6,8,2,9])
u = np.unique(a)
print(u)
u,indices1,indices2 = np.unique(a, return_index = True,return_counts = True)
print(u,indices1,indices2)
u,indices = np.unique(a,return_inverse = True)
print(u,indices)
print(u[indices])#使用下标重构原数组
print('')


#numpy.sort()：sort()函数返回输入数组的排序副本
# numpy.sort(a, axis, kind, order)
#axis 沿着它排序数组的轴，如果没有数组会被展开，沿着最后的轴排序
#kind 默认为'quicksort'快速排序，
#order 如果数组包含字段，则是要排序的字段
a = np.array([[3,7],[9,1]])
print(np.sort(a))
print(np.sort(a, axis =  0))  #沿轴 0 排序
# 在 sort 函数中排序字段
dt = np.dtype([('name',  'S10'),('age',  int)])
a = np.array([("raju",21),("anil",25),("ravi",  17),  ("amar",27)], dtype = dt)
print(np.sort(a, order =  'name')) #按 name 排序
print('')


#numpy.argsort()函数对输入数组沿给定轴执行间接排序，并使用指定排序类型返回数据的索引数组。
# 这个索引数组用于构造排序后的数组
x = np.array([3,  1,  2])
y = np.argsort(x)
print(y)
print(x[y])  #以排序后的顺序重构原数组
for i in y:
    print(x[i]) #使用循环重构原数组
print('')


#numpy.argmax() 和 numpy.argmin()分别沿给定轴返回最大和最小元素的索引
a = np.array([[30,40,70],[80,20,10],[50,90,60]])
print(np.argmax(a))
print (a.flatten())  #展开数组
maxindex = np.argmax(a, axis =  0)  #沿轴 0 的最大值索引
print(maxindex)
maxindex = np.argmax(a, axis =  1)  #沿轴 1 的最大值索引
print(maxindex)
minindex = np.argmin(a)
print(minindex)
print(a.flatten()[minindex])  #展开数组中的最小值
minindex = np.argmin(a, axis =  0)  #沿轴 0 的最小值索引
print(minindex)
minindex = np.argmin(a, axis =  1) #沿轴 1 的最小值索引
print(minindex)
print('')


#numpy.nonzero()函数返回输入数组中非零元素的索引
a = np.array([[30,40,0],[0,20,10],[50,0,60]])
print(np.nonzero (a))
print('')


#numpy.where()返回输入数组中满足给定条件的元素的索引
x = np.arange(9.).reshape(3,  3)
y = np.where(x >  3)
print(x[y])
print('')


#numpy.extract()返回满足任何条件的元素
x = np.arange(9.).reshape(3,  3)
condition = np.mod(x,2)  ==  0# 定义条件
print(condition)
print(np.extract(condition, x))
print('')


#在执行函数时，其中一些返回输入数组的副本，而另一些返回视图。
# 当内容物理存储在另一个位置时，称为副本。 另一方面，如果提供了相同内存内容的不同视图，我们将其称为视图
#无复制
#简单的赋值不会创建数组对象的副本。 相反，它使用原始数组的相同id()来访问它。
# id()返回 Python 对象的通用标识符，类似于 C 中的指针。此外，一个数组的任何变化都反映在另一个数组上。
a = np.arange(6)
print(id(a))
b = a#a 赋值给 b
print(id(b))#b 拥有相同 id()
b.shape = 3, 2 #修改 b 的形状
print(b)
print(a)#a 的形状也修改了
print('')

#视图或浅复制
#ndarray.view()方法，是一个新的数组对象，并可查看原始数组的相同数据。
# 与前一种情况不同，新数组的维数更改不会更改原始数据的维数
a = np.arange(6).reshape(3,2)
b = a.view()  #创建 a 的视图
#两个数组的 id() 不同
print(id(a))  #a 的 id()
print(id(b))  #b 的 id()
# 修改 b 的形状，并不会修改 a
b.shape =  2,3
print (b) #b 的形状
print(a)#a 的形状
print('')

#数组的切片也会创建视图
a = np.array([[10,10],  [2,3],  [4,5]])
s = a[:,  :2]  #创建切片
print(s)
print('')

#深复制
#ndarray.copy()函数创建一个深层副本。 它是数组及其数据的完整副本，不与原始数组共享
a = np.array([[10,10],  [2,3],  [4,5]])
b = a.copy()
# b 与 a 不共享任何内容
print(b is a)
b[0,0]  =  100  #修改 b 的内容
print(a)#a 保持不变
print('')



#六、字符串函数
#在字符数组类（numpy.char）中定义。numpy.char类中的函数在执行向量化字符串操作时非常有用。

#numpy.char.add()函数执行按元素的字符串连接。
print(np.char.add(['hello'],[' xyz']))
print(np.char.add(['hello', 'hi'],[' abc', ' xyz']))
print('')

#numpy.char.multiply()这个函数执行多重连接。
print(np.char.multiply('Hello ',3))
print('')

#numpy.char.center()此函数返回所需宽度的数组，以便输入字符串位于中心，并使用fillchar在左侧和右侧进行填充。
# np.char.center(arr, width,fillchar)
print(np.char.center('hello', 20,fillchar = '*'))
print('')

#numpy.char.capitalize()函数返回字符串的副本，其中第一个字母大写
print(np.char.capitalize('hello world'))
print('')

#numpy.char.title()返回输入字符串的按元素标题转换版本，其中每个单词的首字母都大写
print(np.char.title('hello how are you?'))
print('')

#numpy.char.lower()函数返回一个数组，其元素转换为小写。它对每个元素调用str.lower
print(np.char.lower(['HELLO','WORLD']))
print(np.char.lower('HELLO'))
print('')

#numpy.char.upper()函数返回一个数组，其元素转换为大写。它对每个元素调用str.upper
print(np.char.upper('hello'))
print(np.char.upper(['hello','world']))
print('')

#numpy.char.split()返回输入字符串中的单词列表。默认空格用作分隔符。否则指定的分隔符字符用于分割字符串
print(np.char.split ('hello how are you?'))
print(np.char.split ('TutorialsPoint,Hyderabad,Telangana', sep = ','))
print('')

#numpy.char.splitlines()函数返回数组中元素的单词列表，以换行符分割
print(np.char.splitlines('hello\nhow are you?'))
print(np.char.splitlines('hello\rhow are you?'))
#'\n'，'\r'，'\r\n'都会用作换行符
print('')

#numpy.char.strip()函数返回数组的副本，其中元素移除了开头或结尾处的特定字符
print(np.char.strip('ashok arora','a'))
print(np.char.strip(['arora','admin','java'],'a'))
print('')

#numpy.char.join()这个函数返回一个字符串，其中单个字符由特定的分隔符连接
print(np.char.join(':','dmy'))
print(np.char.join([':','-'],['dmy','ymd']))
print('')

#numpy.char.replace()这个函数返回字符串副本，其中所有字符序列的出现位置都被另一个给定的字符序列取代
print(np.char.replace ('He is a good boy', 'is', 'was'))
print('')



#七、位操作

#bitwise_and通过np.bitwise_and()函数对输入数组中的整数的二进制表示的相应位执行位与运算
a, b = 13, 17
print(bin(a), bin(b))#13 和 17 的二进制形式
print(np.bitwise_and(13, 17))
print('')

#bitwise_or通过np.bitwise_or()函数对输入数组中的整数的二进制表示的相应位执行位或运算
print(np.bitwise_or(13, 17))
print('')

#invert此函数计算输入数组中整数的位非结果。 对于有符号整数，返回补码
print(np.invert(np.array([13], dtype=np.uint8)))
# 比较 13 和 242 的二进制表示，我们发现了位的反转
print(np.binary_repr(13, width=8))#13 的二进制表示
print(np.binary_repr(242, width=8))#242 的二进制表示
print('')

#left_shift:numpy.left shift()函数将数组元素的二进制表示中的位向左移动到指定位置，右侧附加相等数量的 0
print(np.left_shift(10, 2))#将 10 左移两位
print(np.binary_repr(10, width=8))#10 的二进制表示
print(np.binary_repr(40, width=8))#40 的二进制表示
#  '00001010' 中的两位移动到了左边，并在右边添加了两个 0。
print('')

#right_shift:numpy.right_shift()函数将数组元素的二进制表示中的位向右移动到指定位置，左侧附加相等数量的0
print(np.right_shift(40, 2))#将 40 右移两位
print(np.binary_repr(40, width=8))#40 的二进制表示
print(np.binary_repr(10, width=8))#10 的二进制表示
#  '00001010' 中的两位移动到了右边，并在左边添加了两个 0
print('')



#八、算术运算

#三角函数NumPy 拥有标准的三角函数，它为弧度制单位的给定角度返回三角函数比值
a = np.array([0,30,45,60,90])  # 通过乘 pi/180 转化为弧度
print(np.sin(a*np.pi/180))
print(np.cos(a*np.pi/180))
print(np.tan(a*np.pi/180))
print('')

#numpy.degrees()函数通过将弧度制转换为角度制
print(np.degrees(np.pi/3))
print('')

#numpy.around()这个函数返回四舍五入到所需精度的值
#numpy.around(a,decimals),decimals 要舍入的小数位数。默认为0。如果为负，整数将四舍五入到小数点左侧的位置
a = np.array([1.0, 5.55, 123, 0.567, 25.532])
print(np.around(a, decimals =  1))
print(np.around(a, decimals =  -1))
print('')

#numpy.floor()此函数返回不大于输入参数的最大整数。
# 即标量x 的下限是最大的整数i ，使得i <= x。 注意在Python中，向下取整总是从 0 舍入
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
print(np.floor(a))
print('')

#numpy.ceil()返回输入值的上限，即，标量x的上限是最小的整数i ，使得i> = x
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
print(np.ceil(a))
print('')

#用于执行算术运算（如add()，subtract()，multiply()和divide()）的数组必须具有相同的形状或符合数组广播规则
a = np.arange(9, dtype = np.float_).reshape(3,3)
b = np.array([10,10,10])
print(np.add(a,b))
print(np.subtract(a,b))
print(np.multiply(a,b))
print(np.divide(a,b))
print('')

#numpy.reciprocal()返回参数逐元素的倒数。
# 由于Python处理整数除法的方式，对于绝对值大于1的整数元素，结果始终为0，对于整数0，则发出溢出警告。
a = np.array([0.25,  1.33,  1,  0,  100])
print(np.reciprocal(a))
b = np.array([100], dtype =  int)
print(np.reciprocal(b))
print('')

#numpy.power()此函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂
a = np.array([10,100,1000])
print(np.power(a,2))
b = np.array([1,2,3])
print(np.power(a,b))
print('')

#numpy.mod()返回输入数组中相应元素的除法余数。
# 函数numpy.remainder()也产生相同的结果
a = np.array([10,20,30])
b = np.array([3,5,7])
print(np.mod(a,b))
print(np.remainder(a,b))
print('')

#以下函数用于对含有复数的数组执行操作。
#numpy.real() 返回复数类型参数的实部。
#numpy.imag() 返回复数类型参数的虚部。
#numpy.conj() 返回通过改变虚部的符号而获得的共轭复数。
#numpy.angle() 返回复数参数的角度。函数的参数是degree。如果为true，返回的角度以角度制来表示，否则为以弧度制来表示
a = np.array([-5.6j,  0.2j,  11.  ,  1+1j])
print(np.real(a))
print(np.imag(a))
print(np.conj(a))
print(np.angle(a))
print(np.angle(a, deg =  True))
print('')

#numpy.amin() 和 numpy.amax()从给定数组中的元素沿指定轴返回最小值和最大值
a = np.array([[3,7,5],[8,4,3],[2,4,9]])
print (np.amin(a,1))
print (np.amin(a,0))
print(np.amin(a))
print (np.amax(a))
print (np.amax(a, axis =  0))
print('')

#numpy.ptp()函数返回沿轴的值的范围（最大值 - 最小值）
a = np.array([[3,7,5],[8,4,3],[2,4,9]])
print(np.ptp(a))
print(np.ptp(a, axis =  1))
print(np.ptp(a, axis =  0))
print('')

#numpy.percentile()百分位数是统计中使用的度量，表示小于这个值得观察值占某个百分比
# numpy.percentile(a, q, axis),a输入数组
# q要计算的百分位数，在 0 ~ 100 之间
# axis 沿着它计算百分位数的轴
a = np.array([[30,40,70],[80,20,10],[50,90,60]])
print(np.percentile(a,50))
print(np.percentile(a,50, axis =  1))
print(np.percentile(a,50, axis =  0))
print('')

#numpy.median()中值定义为将数据样本的上半部分与下半部分分开的值
a = np.array([[30,65,70],[80,95,10],[50,90,60]])
print(np.median(a))
print(np.median(a, axis =  0))
print(np.median(a, axis =  1))
print('')

#numpy.mean()算术平均值是沿轴的元素的总和除以元素的数量。
# numpy.mean()函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(np.mean(a))
print(np.mean(a, axis =  0))
print(np.mean(a, axis =  1))
print('')

#numpy.average()加权平均值是由每个分量乘以反映其重要性的因子得到的平均值。
# numpy.average()函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值。
# 该函数可以接受一个轴参数。 如果没有指定轴，则数组会被展开
a = np.array([1,2,3,4])
print(np.average(a)) # 不指定权重时相当于 mean 函数
wts = np.array([4,3,2,1])
print(np.average(a,weights=wts))
# 如果 returned 参数设为 true，则返回权重的和
print(np.average([1,2,3,4],weights=[4,3,2,1], returned=True))
print('')

#在多维数组中，可以指定用于计算的轴
a = np.arange(6).reshape(3,2)
wt = np.array([3,5])
print(np.average(a, axis =  1, weights = wt, returned =  True))
print('')

#标准差,是与均值的偏差的平方的平均值的平方根
# std = sqrt(mean((x - x.mean())**2))
print(np.std([1,2,3,4]))
print('')

#方差是偏差的平方的平均值，即mean((x - x.mean())** 2)
print(np.var([1,2,3,4]))
print('')



#九、矩阵库与线性代数
#NumPy 包包含一个 Matrix库numpy.matlib。此模块的函数返回矩阵而不是返回ndarray对象

#matlib.empty()函数返回一个新的矩阵，而不初始化元素
# numpy.matlib.empty(shape, dtype, order)
import numpy.matlib
print(np.matlib.empty((2,2)))#填充为随机数据
print('')

#numpy.matlib.zeros()此函数返回以零填充的矩阵
print(np.matlib.zeros((2,2)) )
print('')

#numpy.matlib.ones()此函数返回以一填充的矩阵
print(np.matlib.ones((2,2)))
print('')

#numpy.matlib.eye()这个函数返回一个矩阵，对角线元素为 1，其他位置为零
# numpy.matlib.eye(n, M,k, dtype), n返回矩阵的行数,  M返回矩阵的列数，默认为n, k对角线的索引
print(np.matlib.eye(n =  3, M =  4, k =  0, dtype =  float))
print('')

#numpy.matlib.identity()函数返回给定大小的单位矩阵。单位矩阵是主对角线元素都为 1 的方阵
print (np.matlib.identity(5, dtype =  float))
print('')

#numpy.matlib.rand()函数返回给定大小的填充随机值的矩阵
print(np.matlib.rand(3,3))
print('')

i = np.matrix('1,2;3,4')
print (i)
print('')


#NumPy 包包含numpy.linalg模块，提供线性代数所需的所有功能

#numpy.dot()返回两个数组的点积。 对于二维向量，其等效于矩阵乘法。
# 对于一维数组，它是向量的内积。 对于 N 维数组，它是a的最后一个轴上的和与b的倒数第二个轴的乘积
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.dot(a,b))
print('')

#numpy.vdot()返回两个向量的点积。 如果第一个参数是复数，那么它的共轭复数会用于计算。
# 如果参数id是多维数组，它会被展开
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.vdot(a,b))
print('')

#numpy.inner()返回一维数组的向量内积。 对于更高的维度，它返回最后一个轴上的和的乘积
print(np.inner(np.array([1,2,3]),np.array([0,1,0])))
print('')

a = np.array([[1, 2], [3, 4]])
b = np.array([[11, 12], [13, 14]])
print(np.inner(a, b))
print('')

#numpy.linalg.det()函数计算输入矩阵的行列式
a = np.array([[1,2], [3,4]])
print(np.linalg.det(a))
b = np.array([[6,1,1], [4, -2, 5], [2,8,7]])
print(np.linalg.det(b))
print('')

#numpy.linalg.inv()函数来计算矩阵的逆
x = np.array([[1,2],[3,4]])
y = np.linalg.inv(x)
print(y)


#numpy.linalg.solve()函数给出了矩阵形式的线性方程的解
#x + y + z = 6
#2y + 5z = -4
#2x + 5y - z = 27
#可以使用矩阵表示为：如果矩阵成为A、X和B，方程变为：AX = B  ,或X = A^(-1)B
a = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
ainv = np.linalg.inv(a)#a 的逆
b = np.array([[6], [-4], [27]])
x = np.linalg.solve(a, b)#计算：A^(-1)B
print(x)
print('')
# 这就是线性方程 x = 5, y = 3, z = -2 的解
#结果也可以使用下列函数获取
x = np.dot(ainv,b)
print(x)
print('')



#十、IO
#load()和save()函数处理 numPy 二进制文件（带npy扩展名）
#loadtxt()和savetxt()函数处理正常的文本文件

#numpy.save()文件将输入数组存储在具有npy扩展名的磁盘文件中,为了从outfile.npy重建数组，请使用load()函数
a = np.array([1,2,3,4,5])
np.save('outfile',a)
b = np.load('outfile.npy')
print(b)
print('')

#以简单文本文件格式存储和获取数组数据，是通过savetxt()和loadtx()函数完成的。
a = np.array([1,2,3,4,5])
np.savetxt('out.txt',a)
b = np.loadtxt('out.txt')
print(b)