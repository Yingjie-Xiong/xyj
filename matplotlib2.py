#画3D图
#当使用Matplotlib绘制三维图表时，你可以使用mpl_toolkits.mplot3d模块来创建三维坐标轴，
# 并使用相应的函数来绘制三维散点图、曲面图、线框图等。
#格式：from mpl_toolkits.mplot3d import axes3d
#ax3d = plt.gca(projection='3d')
#ax3d.plot_wireframe()	# 绘制3d线框图
#ax3d.plot_surface()    # 绘制3d曲面图
#ax3d.scatter()			# 绘制3d散点图


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#绘制三维散点图
#ax3d.scatter(x, y, z, marker='',s = 60,edgecolor='',facecolor='',zorder=3, c=d,cmap='jet')
# x,y,z确定一组散点坐标,marker是点型,s是点的大小,edgecolor是边缘色,facecolor是填充色,zorder绘制图层编号,
#c是设置过渡性颜色，cmap是颜色映射


n = 500
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
z = np.random.normal(0, 1, n)
d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
plt.figure('3D Scatter')
ax3d = plt.gca(projection='3d')
ax3d.set_xlabel('X', fontsize=14)
ax3d.set_ylabel('Y', fontsize=14)
ax3d.set_zlabel('Z', fontsize=14)
ax3d.scatter(x, y, z, s=60, alpha=0.6,c=d, cmap='jet')
plt.savefig(os.path.join('fig06', '001.png'))
plt.show()


# 准备数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
z = [3, 6, 9, 12, 15]
# 创建三维坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制散点图
ax.scatter(x, y, z)
# 设置坐标轴标签
plt.rcParams['font.family'] = 'SimHei' # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig(os.path.join('fig06', '002.png'))
plt.show()


#绘制三维曲面图
#ax3d.plot_surface(x, y,z,rstride=30,cstride=30,cmap='jet')
# x,y网格点坐标矩阵,z为每个坐标点的值,rstride是行跨距,cstride是列跨距


n = 1000
x, y = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))
# 根据x,y 计算当前坐标下的z高度值
z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
plt.figure('Surface', facecolor='lightgray')
ax3d = plt.gca(projection='3d')
ax3d.set_xlabel('X', fontsize=14)
ax3d.set_ylabel('Y', fontsize=14)
ax3d.set_zlabel('Z', fontsize=14)
ax3d.plot_surface(x, y, z, rstride=50,cstride=50, cmap='jet')
plt.savefig(os.path.join('fig06', '003.png'))
plt.show()



# 生成网格数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
#使用meshgrid()函数生成二维的X和Y坐标网格，根据这些坐标计算Z值，并使用plot_surface()函数绘制曲面图。
Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
# 创建三维坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制曲面图
ax.plot_surface(X, Y, Z, cmap='viridis')
# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig(os.path.join('fig06' , '004.png'))
plt.show()


#绘制三维线框图
#ax3d.plot_wireframe(x, y,z,rstride=30,cstride=30,linewidth=1,color='')
# x,y网格点坐标矩阵,z为每个坐标点的值,rstride是行跨距,cstride是列跨距,


# 生成网格点坐标矩阵
n = 1000
x, y = np.meshgrid(np.linspace(-3, 3, n),np.linspace(-3, 3, n))
# 根据x,y 计算当前坐标下的z高度值
z = (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
plt.figure('Wireframe', facecolor='lightgray')
ax3d = plt.gca(projection='3d')
ax3d.set_xlabel('X', fontsize=14)
ax3d.set_ylabel('Y', fontsize=14)
ax3d.set_zlabel('Z', fontsize=14)
ax3d.plot_wireframe(x, y, z, rstride=10,cstride=10, color='dodgerblue')
plt.savefig(os.path.join('fig06' , '005.png'))
plt.show()

