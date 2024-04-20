#学习链接：https://blog.csdn.net/qq_34859482/article/details/80617391?spm=1001.2014.3001.5506
#总体流程：先创建画板和轴，然后中间绘图，最后保存和展示

import numpy as np
#一般画图都要用到数据处理
import os
#用于创建路径，保存生成的图片
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------------------------
#一、基础知识
fig = plt.figure()
#在任何绘图之前，我们需要一个Figure对象，可以理解成我们需要一张画板才能开始绘图
ax = fig.add_subplot(111)
#在画板的第1行第1列的第1个位置生成一个Axes对象来准备作画
#在拥有Figure对象之后，在作画前我们还需要轴，没有轴的话就没有绘图基准，所以需要添加Axes
ax.set(xlim=[0.5, 4.5], ylim=[-2, 8], title='An Example Axes',
       ylabel='Y-Axis', xlabel='X-Axis')
#plt.xlim(),plt.ylim()分别表示横纵轴的刻度范围
plt.savefig(os.path.join('fig01' , '001.png'))#保存的时候需要plt.show()在plt.savefig()之后，顺序颠倒会出现图片为空白
#plt.savefig用法查询链接:https://blog.csdn.net/qq_40481843/article/details/120443307?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171249892216800182755556%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171249892216800182755556&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-120443307-null-null.142^v100^pc_search_result_base1&utm_term=plt%E4%BF%9D%E5%AD%98%E5%9B%BE%E7%89%87%E5%88%B0%E6%9F%90%E6%96%87%E4%BB%B6%E5%A4%B9&spm=1018.2226.3001.4187
#plt.savefig用法查询链接:https://blog.csdn.net/weixin_47872288/article/details/128739356?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171249882616800226554640%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171249882616800226554640&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-128739356-null-null.142^v100^pc_search_result_base1&utm_term=plt%E4%BF%9D%E5%AD%98%E5%9B%BE%E7%89%87&spm=1018.2226.3001.4187
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(222)
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(221)
ax3 = fig.add_subplot(224)
plt.show()
#fig.add_subplot(2, 2, 1)的方式生成Axes，前面两个参数确定了面板的划分，
#例如 2， 2会将整个面板划分成 2 * 2 的方格，第三个参数取值范围是 [1, 2*2] 表示第几个Axes


fig, axes = plt.subplots(nrows=2, ncols=2)#fig和axes是subplots()返回的两个对象，名称可以自定义
#效果同fig = plt.figure()  ax = fig.add_subplot(111) 两段代码
#subplot()、subplots()在实际过程中，先创建了一个figure画窗，
#然后通过调用add_subplot()来向画窗中各个分块添加坐标区，
#其差别在于是分次添加(subplot())还是一次性添加(subplots())
axes[0,0].set(title='Upper Left')
axes[0,1].set(title='Upper Right')
axes[1,0].set(title='Lower Left')
axes[1,1].set(title='Lower Right')
fig.tight_layout() #自动调整布局，使标题之间不重叠
plt.savefig(os.path.join('fig01','002.png'))
plt.show()
#plt.subplots用法查询链接:https://blog.csdn.net/sunjintaoxxx/article/details/121098302?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171249872916800185883151%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171249872916800185883151&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-121098302-null-null.142^v100^pc_search_result_base1&utm_term=plt.subplots&spm=1018.2226.3001.4187


fig = plt.figure()
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])
#np.array用法查询链接：https://blog.csdn.net/u011699626/article/details/122194100?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171256272316800226513820%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171256272316800226513820&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-122194100-null-null.142^v100^pc_search_result_base1&utm_term=np.array&spm=1018.2226.3001.4187
#np.array用法查询链接：https://blog.csdn.net/weixin_47097527/article/details/127649907?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171256272316800226513820%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171256272316800226513820&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-127649907-null-null.142^v100^pc_search_result_base1&utm_term=np.array&spm=1018.2226.3001.4187
plt.subplot(1, 2, 1)
plt.plot(xpoints,ypoints)
plt.xticks([0,2,4,6],labels='abcd')#调整x的刻度并自定义
plt.yticks([0,50,100],labels='efg')#调整y的刻度并自定义
plt.title("pl1")
#将图像窗口分为1行2列，当前位置在1
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.grid(True)#增加网格线
#plt.plot用法查询链接：https://blog.csdn.net/qq_43186282/article/details/121513266?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.plot&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-121513266.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
plt.title("pl2")
#将图像窗口分为1行2列，当前位置在2
plt.suptitle("003")
plt.savefig(os.path.join('fig01','003.png'))
plt.show()
#若pl1建完后有plt.show()，则两张图分开显示，现在是两个图共用一个plt.show(),故合并显示
#plt.subplot用法查询链接：https://blog.csdn.net/mouselet3/article/details/127389508?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171250091616800182789142%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171250091616800182789142&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-127389508-null-null.142^v100^pc_search_result_base1&utm_term=plt.subplot&spm=1018.2226.3001.4187


fig = plt.figure()
x = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
plt.hist(x)
plt.title('fig01-004',fontsize='xx-large',fontweight='heavy',color='blue',fontstyle='italic'
          ,loc ='left',rotation=15,bbox=dict(facecolor='y', edgecolor='blue', alpha=0.65))
#plt.title用法查询链接:https://blog.csdn.net/TeFuirnever/article/details/88945563?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171258674116800182729451%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171258674116800182729451&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-88945563-null-null.142^v100^pc_search_result_base1&utm_term=plt.title&spm=1018.2226.3001.4187
plt.xlim(-10,110)#x轴的范围
plt.ylim(-1,4)#y轴的范围
plt.xticks([-10,10,20,30,40,50,60,70,80,90,100])#调整x的刻度
plt.yticks([-1,1,2,3,4])#调整y的刻度
plt.savefig(os.path.join('fig01','004.png'))
plt.show()
#plt.hist用法查询链接:https://blog.csdn.net/weixin_46707493/article/details/119832774?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.hist&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-119832774.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
#plt.hist用法查询链接:https://blog.csdn.net/weixin_45520028/article/details/113924866?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.hist&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-113924866.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187



#----------------------------------------------------------------------------------------
#二、创建数据画图基本知识
# 创建一些测试数据
x = np.linspace(0, 100, 4)
#np.linspace用法查询：https://blog.csdn.net/neweastsun/article/details/99676029
y = np.sin(x)
labels = ['A', 'B', 'C', 'D']
fig, ax = plt.subplots()
plt.xticks(x,labels,rotation = 30)
#plt.xticks用法查询链接：https://blog.csdn.net/weixin_50345615/article/details/126193271?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171256700916800188564032%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171256700916800188564032&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-126193271-null-null.142^v100^pc_search_result_base1&utm_term=plt.xticks&spm=1018.2226.3001.4187
ax.plot(x, y)
ax.set_title('fig02-001')
#ax.set_title用法查询链接:https://blog.csdn.net/weixin_43729592/article/details/117600365?spm=1001.2014.3001.5506
plt.savefig(os.path.join('fig02','001.png'))
plt.show()


fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharex=True)
#sharex、sharey：设置 x、y 轴是否共享属性，默认为 false，可设置为 ‘none’、‘all’、‘row’ 或 ‘col’。
# False 或 none 每个子图的 x 轴或 y 轴都是独立的，True 或 ‘all’：所有子图共享 x 轴或 y 轴，
# ‘row’ 设置每个子图行共享一个 x 轴或 y 轴，‘col’：设置每个子图列共享一个 x 轴或 y 轴
ax1.plot(x, y)
ax1.set_title('Sharing x axis')
ax4.scatter(x, y)#散点图
#plt.scatter用法查询链接：https://blog.csdn.net/qq_43186282/article/details/121513266?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.plot&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-121513266.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
plt.savefig(os.path.join('fig02','002.png'))
plt.show()
#四个子图中上两幅图并无x轴（与下子图共享），因为已设置sharex=True
# 若改为sharey=True，可观察到四副子图中右两幅无y轴（即与左子图共享）


fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
# 通过对subplot_kw传入参数，生成关于极坐标系的子图
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)
plt.savefig(os.path.join('fig02','003.png'))
plt.show()


fig=plt.figure()
ax=fig.add_subplot()
plt.rcParams['font.family'] = 'SimHei' # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
#plt.rcParams用法查询链接：https://blog.csdn.net/weixin_39010770/article/details/88200298?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171256669416800226511506%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171256669416800226511506&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-88200298-null-null.142^v100^pc_search_result_base1&utm_term=plt.rcParams&spm=1018.2226.3001.4187
b=np.arange(5)
plt.plot(b,b*1.0,'g.-',b,np.sin(b),'rx',b,b*2.0,'b')
plt.savefig(os.path.join('fig02','004.png'))
plt.show()
#plt.plot常用方法查询链接:https://blog.csdn.net/weixin_46707493/article/details/119722246?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.plot&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-119722246.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187


#散点图plt.scatter，加了参数s以后就是泡泡图
fig=plt.figure()
ax=fig.add_subplot()
np.random.seed(0)
x = np.random.rand(20)
y = np.random.rand(20)
#np.random.rand用法查询链接：https://blog.csdn.net/qq_40130759/article/details/79535575?ops_request_misc=&request_id=&biz_id=102&utm_term=np.random.rand&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-79535575.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
area = (50 * np.random.rand(20)) ** 2
plt.scatter(x, y, s=area, alpha=0.5)
plt.axis('off')#去除坐标轴
plt.savefig(os.path.join('fig02','005.png'))
plt.show()


fig=plt.figure()
ax=fig.add_subplot()
x = np.arange(0, 10, 1)
plt.plot(x, x, 'r--', x, np.cos(x), 'g--', marker='^')
plt.xlabel('\n'.join(('横向','的轴')),x=1,fontsize=10)#x=1表示放在x轴100%的位置（从左往右）
#'\n'.join(('横','轴'))表示将文字一行一个单引号里的内容，排成一列，标准语法是'\n'.join(list(x))
plt.ylabel('纵轴',rotation='horizontal',labelpad=12.5,y=0.5,fontsize=10)
#y=0.5表示放在y轴50%的位置（从下往上）,labelpad表示距离坐标轴的距离
#fontsize表示字体大小
#plt.ylabel用法查询链接:https://blog.csdn.net/dongfuguo/article/details/118706468
#plt.ylabel用法查询链接:https://blog.csdn.net/Okami_/article/details/108742440?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.ylabel%E5%AD%97%E5%8F%AA%E6%9C%89%E4%B8%80%E5%8D%8A&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-108742440.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
#plt.xlabel用法查询链接：https://blog.csdn.net/qq_43657442/article/details/115112870?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.xlabel&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-9-115112870.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
#plt.xlabel用法查询链接:https://blog.csdn.net/Caiqiudan/article/details/109679540?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.xlabel&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-8-109679540.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
plt.legend(["BJ", "SH"], loc='upper left')
#plt.legend用法查询链接：https://blog.csdn.net/qq_43186282/article/details/121513266?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.plot&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-121513266.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
plt.savefig(os.path.join('fig02','006.png'))
plt.show()


fig=plt.figure()
ax=fig.add_subplot()
x = np.linspace(0, 10, 30)
plt.plot(x, np.sin(x), c='#0ffef9', label='sin(x)')
ax = plt.gca()  # 获取当前子图 get current axes
ax.spines['right'].set_color('none')  # spines中文脊柱的意思，表示图边框四条脊柱线：top,bottom,left,right
ax.spines['top'].set_color('none')  # 使上轴线及右轴线颜色设置为none或‘white’
#去除上轴线及右边轴线
#plt.spines用法查询链接：https://blog.csdn.net/weixin_46707493/article/details/119722246?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.plot&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-119722246.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
plt.savefig(os.path.join('fig02','007.png'))
plt.show()


fig=plt.figure()
ax=fig.add_subplot()
x = np.linspace(0, 10, 30)
plt.plot(x, np.sin(x), c='#0ffef9', label='sin(x)')
ax = plt.gca()  # 获取当前子图 get current axes
ax.spines['right'].set_color('none')  # spines中文脊柱的意思，表示图边框四条脊柱线：top,bottom,left,right
ax.spines['top'].set_color('none')  # 使上轴线及右轴线颜色设置为none或‘white’
ax.spines['left'].set_position(('data', 0))  # 调整左轴线位置到横坐标0的位置
ax.spines['bottom'].set_position(('data', 0))  # 调整下轴线位置到纵坐标0的位置
#调整坐标轴为x,y直角坐标轴
plt.savefig(os.path.join('fig02','008.png'))
plt.show()


fig=plt.figure()
ax=fig.add_subplot()
x = np.linspace(0, 10, 30)
plt.plot(x, np.sin(x), c='#0ffef9', label='sin(x)')
# plt.text(x,y,s) x表示x轴坐标值,y表示y轴坐标值 通过x,y来控制标注的位置
plt.text(x=1, y=0.5, s='这是标注', fontsize=15, c='b', rotation=20)  # rotation控制文本字体角度
#文本标注：plt.text()：plt.text(x,y,s)，x表示x轴坐标值,y表示y轴坐标值 通过x,y来控制标注的位置
#用法查询链接：https://blog.csdn.net/weixin_46707493/article/details/119722246?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.plot&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-119722246.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
plt.savefig(os.path.join('fig02','009.png'))
plt.show()


fig=plt.figure()
ax=fig.add_subplot()
x = np.linspace(0, 10, 30)
plt.plot(x, np.sin(x), c='#0ffef9', label='sin(x)')
plt.annotate(s='底点', xy=(4.65, -1), xytext=(4.2, 0),
             arrowprops={'headwidth': 10, 'facecolor': 'r'}, fontsize=15)
#若未设置arrowprops参数则没有箭头显示，只有文本注释
#文本箭头注释:plt.annotate()查询链接：https://blog.csdn.net/weixin_46707493/article/details/119722246?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.plot&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-119722246.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
#plt.annotate(text,xy,xytext,arrowprops)，text表示注释文本，xy表示箭头端点位置，xytext表示文本注释的位置，arrowprops控制箭头，形式为字典
plt.savefig(os.path.join('fig02','010.png'))
plt.show()


fig=plt.figure()
ax=fig.add_subplot()
x = np.linspace(0, 10, 30)
y1 = x * 2
y2 = x ** 2 + 5
plt.figure(figsize=(8, 5))
plt.plot(x, y1, 'r', x, y2, 'g')  # 注：可以在一个plot函数内画出多条线
plt.fill_between(x, y1, y2, facecolor='k', alpha=0.2)
#plt.fill_between用法查询链接：https://blog.csdn.net/weixin_46707493/article/details/119722246?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.plot&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-119722246.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
# 第一个参数表示要覆盖的左右范围，第二个参数表示覆盖的下限，第三个参数表示覆盖的上限，在这里则表示填充y1曲线和y2曲线中间的区域
plt.savefig(os.path.join('fig02','011.png'))
plt.show()


#归纳练习
plt.rcParams['font.family'] = 'SimHei'  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2
# 创建画布并设置大小
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='linear line', marker='+')
# zorder控制绘图顺序，值越大画的越慢，防止先画从而被后画的图挡住，如下图防止红线被图例挡住从而不清晰
plt.plot(x, y2, color='red', lw=2.0, ls='--', label='square line', zorder=10)
plt.xlim(-4, 5)
plt.ylim(-5, 10)
plt.xticks(rotation=30, fontsize=12, c='k')  # rotation：旋转角度
plt.yticks(rotation=30, fontsize=12, c='k')
plt.grid(True)
plt.legend(loc='upper right', edgecolor='none', facecolor='g', fontsize=13)
# 注释
plt.text(-2, -4, '这是直线', fontsize=15, c='b')
plt.annotate('这是曲线', xy=(2, 2), xytext=(3, -1),
             arrowprops={'headwidth': 10, 'facecolor': 'g'}, fontsize=15, c='r')
# 填充
plt.fill_between(x, y1, y2, facecolor='g', alpha=0.2)
plt.savefig(os.path.join('fig02','012.png'))
plt.show()



#面向对象画图
#Figure：画布，顶层级，用来容纳所有绘图元素
#Axes：可以认为是figure这张画图上的子图，因为子图上一般都是坐标图，也可以愿意理解为轴域或者坐标系。
#Axis：axes的下属层级，用于处理所有和坐标轴有关的元素
#Tick：axis的下属层级，用来处理所有和刻度有关的元素
#综上可知，画出的图像是在画布figure上显示的，axes可理解成figure上的一个子图，axis可理解成子图axes的坐标轴，tick是坐标轴axis上的刻度
#我们来看看面对对象是如何画图的，其实很简单，只需要把plt改为面对自己创建的axes即可
plt.rcParams['font.family'] = 'SimHei'  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2
# 方法一
fig = plt.figure(figsize=(8, 5))  # 创建画布并设置大小
ax = fig.add_subplot(111)  # 111，表示在画布中创建一行，一列，编号为一的子图，即创建一个子图的意思
# 方法二
# fig,ax = plt.subplots(figsize =(8,5))
ax.plot(x, y1, label='linear line', marker='+')
ax.plot(x, y2, color='red', lw=2.0, ls='--', label='square line', zorder=10)
ax.set_xlim(-4, 5)
ax.set_ylim(-5, 10)
plt.xticks(rotation=30, fontsize=12, c='k')  # rotation：旋转角度
plt.yticks(rotation=30, fontsize=12, c='k')
ax.grid(True)
ax.legend(loc='upper right', edgecolor='none', facecolor='g', fontsize=13)
# 注释
ax.text(-2, -4, '这是直线', fontsize=15, c='b')
ax.annotate('这是曲线', xy=(2, 2), xytext=(3, -1),
            arrowprops={'headwidth': 10, 'facecolor': 'g'}, fontsize=15, c='r')
# 填充
ax.fill_between(x, y1, y2, facecolor='g', alpha=0.2)
plt.savefig(os.path.join('fig02','013.png'))
plt.show()



#-------------------------------------------------------------------------------------
#三、与pandas结合使用
#学习链接查询:https://blog.csdn.net/weixin_47989730/article/details/124356416?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171255967016800188528459%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171255967016800188528459&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124356416-null-null.142^v100^pc_search_result_base1&utm_term=plt.plot&spm=1018.2226.3001.4187
import pandas as pd

#x, y可传入(元组), [列表], np.array, pd.Series
fig=plt.figure()
ax=fig.add_subplot()
x = (3, 4, 5)  # (元组)
y1 = np.array([3, 4, 3])  # np.array
y2 = pd.Series([4, 5, 4])  # pd.Series
plt.plot(x, y1)
plt.plot(y2)  # x可省略,默认[0,1..,N-1]递增
plt.savefig(os.path.join('fig03','001.png'))
plt.show()  # plt.show()前可加多个plt.plot(),画在同一张图上


#可传入多组x, y
fig=plt.figure()
ax=fig.add_subplot()
x = (3, 4, 5)
y1 = np.array([3, 4, 3])
y2 = pd.Series([4, 5, 4])
plt.plot(x, y1, x, y2)  # 此时x不可省略
plt.savefig(os.path.join('fig03','002.png'))
plt.show()


#x, y可以不等长, x短，则后面的y循环使用x的序列
fig=plt.figure()
ax=fig.add_subplot()
dic1 = {'x列0': [0, 1, 2], 'x列1': [3, 4, 5]}
x = pd.DataFrame(dic1)
dic2 = {'y列0': [2, 3, 2], 'y列1': [3, 4, 3], 'y列2': [4, 5, 4], 'y列3': [5, 6, 5]}
y = pd.DataFrame(dic2)
print(x)
print(y)
plt.plot(x,y)
plt.savefig(os.path.join('fig03','003.png'))
plt.show()


#x, y可以不等长, x长，则后面的x循环使用y的序列
#"点型"
fig=plt.figure()
ax=fig.add_subplot()
marker=['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_','.',',']
dic1=[[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17]]
x=pd.DataFrame(dic1)
dic2=[[2,3,2.5],[3,4,3.5],[4,5,4.5],[5,6,5.5]]
y=pd.DataFrame(dic2)
print(x)
print(y)
for i in range(6):
    for j in range(4):
        plt.plot(x.loc[i],y.loc[j],"b"+marker[i*4+j]+":") # "b"蓝色,":"点线
plt.savefig(os.path.join('fig03','004.png'))
plt.show()


#x或y传入二维数组
#plt.plot(x, y, "格式控制字符串")，点和线的格式可以用"格式控制字符串"设置
#"格式控制字符串"最多可以包括三部分, "颜色", "点型", "线型"
fig=plt.figure()
ax=fig.add_subplot()
lst1 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
x = np.array(lst1)
lst2 = [[2, 3, 2], [3, 4, 3], [4, 5, 4]]
y = np.array(lst2)
print(x)
print(y)
plt.plot(x,y,"ob:") #"b"为蓝色, "o"为圆点, ":"为点线
#画图时，x的一列与对应的y的一列画出一个图
plt.savefig(os.path.join('fig03','005.png'))
plt.show()

color=['b','g','r','c','m','y','k','w']
linestyle=['-','--','-.',':']
dic1=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
x=pd.DataFrame(dic1)
dic2=[[2, 3, 2], [3, 4, 3], [4, 5, 4]]
y=pd.DataFrame(dic2)
for i in range(3):
    for j in range(3):
        plt.plot(x.loc[i],y.loc[j],color[i*2+j]+linestyle[j])
plt.savefig(os.path.join('fig03','006.png'))
plt.show()

#plt.plot(x, y, "格式控制字符串", 关键字=参数)
y=[2,3,2]
# 蓝色,线宽20,圆点,点尺寸50,点填充红色,点边缘宽度6,点边缘灰色
plt.plot(y,color="blue",linewidth=20,marker="o",markersize=50,
         markerfacecolor="red",markeredgewidth=6,markeredgecolor="grey")
plt.savefig(os.path.join('fig03','007.png'))
plt.show()


#-------------------------------------------------------------------------------------
#四、绘制基本的2D图

#线
x = np.linspace(0, 10, 200)
data_obj = {'x': x,
            'y1': 2 * x + 1,
            'y2': 3 * x + 1.2,
            'mean': 0.5 * x * np.cos(2*x) + 2.5 * x + 1.1}
fig, ax = plt.subplots()
#填充两条线之间的颜色
ax.fill_between('x', 'y1', 'y2', color='yellow', data=data_obj)
# 画中间的线
ax.plot('x', 'mean', color='black', data=data_obj)
#上面的作图，在数据部分只传入了字符串，这些字符串对一个这 data_obj 中的关键字，
# 当以这种方式作画时，将会在传入给 data 中寻找对应关键字的数据来绘图。
plt.savefig(os.path.join('fig04','001.png'))
plt.show()


#散点图
x = np.arange(10)
y = np.random.randn(10)
plt.scatter(x, y, color='red', marker='+')
plt.savefig(os.path.join('fig04','002.png'))
plt.show()


#饼图
#饼图自动根据数据的百分比画饼。labels是各个块的标签。
# autopct=%1.1f%%表示格式化百分比精确输出，explode，突出某些块，不同的值突出的效果不一样。
# pctdistance=1.12百分比距离圆心的距离，默认是0.6.
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, (ax1, ax2) = plt.subplots(2)
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
ax2.pie(sizes, autopct='%1.2f%%', shadow=True, startangle=90, explode=explode,pctdistance=1.12)
ax2.axis('equal')
ax2.legend(labels=labels, loc='upper right')
plt.savefig(os.path.join('fig04','003.png'))
plt.show()


#等高线（轮廓图）
#上面画了两个一样的轮廓图，contourf会填充轮廓线之间的颜色。
# 数据x, y, z通常是具有相同 shape 的二维矩阵。
# x, y 可以为一维向量，但是必需有 z.shape = (y.n, x.n)，这里y.n和x.n分别表示x、y的长度。
# Z通常表示的是距离X-Y平面的距离，传入X、Y则是控制了绘制等高线的范围。
fig, (ax1, ax2) = plt.subplots(2)
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
ax1.contourf(x, y, z)
ax2.contour(x, y, z)
plt.savefig(os.path.join('fig04','004.png'))
plt.show()


#绘制垂直柱状图
#matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
#x:柱子的横坐标，height：柱子的高度，width：柱子的宽度，default: 0.8，bottom：y轴的起始值，default: 0
#align：柱子与x轴坐标的对齐方式，{‘center’, ‘edge’}, default: ‘center’，
#lable：list[str] 将相应的横坐标替换成标签
import seaborn as sns
sns.set_style({'font.sans-serif': ['simhei', 'Arial']})
name_list = ['China', 'USA', 'India', 'Russia']
num_list = [14, 3.3, 7.8, 1.46]
plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list, bottom=1)
# plt.bar(range(len(num_list)), num_list, color=['r', 'g', 'b'], tick_label=name_list,bottom=1)
plt.ylabel("人口(亿)")
plt.savefig(os.path.join('fig04','005.png'))
plt.show()




#-------------------------------------------------------------------------------------
#五、布局、图例说明、边界

#区间上下限
#当绘画完成后，会发现X、Y轴的区间是会自动调整的，并不是跟我们传入的X、Y轴数据中的最值相同。
# 为了调整区间我们使用下面的方式：
#ax.set_xlim([xmin, xmax])   #设置X轴的区间
#ax.set_ylim([ymin, ymax])   #Y轴区间
#ax.axis([xmin, xmax, ymin, ymax])   #X、Y轴区间
#ax.set_ylim(bottom=-10)     #Y轴下限
#ax.set_xlim(right=25)       #X轴上限

x = np.linspace(0, 2*np.pi)
y = np.sin(x)
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x, y)
ax2.plot(x, y)
ax2.set_xlim([-1, 6])
ax2.set_ylim([-1, 3])
plt.savefig(os.path.join('fig05','001.png'))
plt.show()


#图例说明
#我们如果我们在一个Axes上做多次绘画，那么可能出现分不清哪条线或点所代表的意思。
# 添加图例说明，就可以解决这个问题了，见下例：
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label='Philadelphia')
ax.plot([1, 2, 3, 4], [30, 23, 13, 4], label='Boston')
ax.scatter([1, 2, 3, 4], [20, 10, 30, 15], label='Point')
ax.set(ylabel='Temperature (deg C)', xlabel='Time', title='A tale of two cities')
ax.legend()
#在绘图时传入 label 参数，并最后调用ax.legend()显示图例说明
plt.savefig(os.path.join('fig05','002.png'))
plt.show()


#多子图的布局
#当我们绘画多个子图时，就会有一些美观的问题存在，例如子图之间的间隔，子图与画板的外边间距以及子图的内边距
fig, axes = plt.subplots(2, 2, figsize=(9, 9))
fig.subplots_adjust(wspace=0.5, hspace=0.3,
                    left=0.125, right=0.9,
                    top=0.9,    bottom=0.1)
plt.show()
#通过fig.subplots_adjust()我们修改了子图水平之间的间隔wspace=0.5，
# 垂直方向上的间距hspace=0.3，左边距left=0.125 ，这里数值都是百分比的。
# 以 [0, 1] 为区间，选择left、right、bottom、top 注意 top 和 right 是 0.9 表示上、右边距为百分之10。
# 不确定如果调整的时候，fig.tight_layout()是一个很好的选择。
# 之前说到了内边距，内边距是子图的，也就是 Axes 对象，所以这样使用 ax.margins(x=0.1, y=0.1)，
# 当值传入一个值时，表示同时修改水平和垂直方向的内边距


#去边界，换轴
fig, ax = plt.subplots()
ax.plot([-2, 2, 3, 4], [-10, 20, 25, 5])
ax.spines['top'].set_visible(False)     #顶边界不可见
ax.xaxis.set_ticks_position('bottom')  # ticks 的位置为下方，分上下的。
ax.spines['right'].set_visible(False)   #右边界不可见
ax.yaxis.set_ticks_position('left')
# "outward"
# 移动左、下边界离 Axes 10 个距离
#ax.spines['bottom'].set_position(('outward', 10))
#ax.spines['left'].set_position(('outward', 10))
# "data"
# 移动左、下边界到 (0, 0) 处相交
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
# "axes"
# 移动边界，按 Axes 的百分比位置
#ax.spines['bottom'].set_position(('axes', 0.75))
#ax.spines['left'].set_position(('axes', 0.3))
plt.savefig(os.path.join('fig05','003.png'))
plt.show()


#更多补充内容可以参考两篇一样优秀的博文：
# https://blog.csdn.net/weixin_41558411/article/details/115582012?spm=1001.2014.3001.5506
# https://blog.csdn.net/qq_62592360/article/details/132139856?spm=1001.2014.3001.5506