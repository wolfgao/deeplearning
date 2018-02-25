# -*- coding: UTF-8 -*-
from math import *
from decimal import Decimal, getcontext
from copy import deepcopy
import numpy as np
from vector import Vector
from plane import Plane
from fractions import Fraction
# 不要修改这里！
from helper import *
seed = 114406

def shape(M):
	#假定所有传入的数组都是二维数组
	rows = len(M)
	cols = len(M[0])
	for i, row in enumerate(M):
		#assume all rows have same number of columns
		assert cols == len(row), "第 %d 行的列数和其他行不同，%d列" %(i, len(row))
	return rows, cols

def matxRound(M, decPts=4):
	rows = shape(M)[0]
	cols = shape(M)[1]
	for i in range(rows):
		for j in range(cols):
			M[i][j] = round(M[i][j], decPts)
	return

#计算矩阵的转置
#设A为m×n阶矩阵（即m行n列），第i 行j 列的元素是a(i,j)，即：A=a(i,j)
#定义A的转置为这样一个n×m阶矩阵B，满足B=a(j,i)，即 b (i,j)=a (j,i)（B的第i行第j列元素是A的第j行第i列元素），记A'=B。
#(有些书记为  ，这里T为A的上标）直观来看，将A的所有元素绕着一条从第1行第1列元素出发的右下方45度的射线作镜面反转，即得到A的转置。
def transpose(M):
	rows = shape(M)[0]
	cols = shape(M)[1]
	'''
	#经验证，发现问题出在“*”上，*应该是拷贝了5个引用（想象一下指针），每个对象指向同一块内容。
	#所以只要改变其中任何一个内容，其它几个内容都会改变
	new_rows = [0]*rows
	M_T = [new_rows]*cols
	#因此N维数组的建立和赋值应该是两个for循环完成
	'''
	M_T = [[0 for i in range(rows)] for j in range(cols)]
	for i in range(cols):
		for j in range(rows):
			M_T[i][j] = M[j][i]

	return M_T

# TODO 计算矩阵乘法 AB，如果无法相乘则raise ValueError
# 1)当矩阵A的列数等于矩阵B的行数时，A与B可以相乘。
# 2)矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
# 3) 乘积C的第m行第n列的元素等于矩阵A的第m行的元素与矩阵B的第n列对应元素乘积之和。
def matxMultiply(A, B):
	cols_A = shape(A)[1]
	rows_A = shape(A)[0]
	rows_B = shape(B)[0]
	cols_B = shape(B)[1]

	C = [[0 for i in range(cols_B)] for j in range(rows_A)]
	try:
		if cols_A != rows_B:
			raise ValueError

		#矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
		for i in range(rows_A):
			for j in range(cols_B):
				p =0
				while p< cols_A:
					C[i][j] += A[i][p]*B[p][j]
					p+=1

	except ValueError:
            raise ValueError("The columns of the Matrix A aren't equal to the rows of the Matrix B, so they can't be multiplied" )

	return C

# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
	cols_A = shape(A)[1]
	rows_A = shape(A)[0]
	rows_B = shape(b)[0]
	cols_B = shape(b)[1]
	new_A = [[0 for i in range(cols_A+cols_B)] for j in range(rows_A)]
	#new_A = deepcopy(A)

	try:
		if rows_A != rows_B:
			raise ValueError

		for i in range(rows_A):
			for j in range(cols_A+cols_B):
				if j<cols_A:
					new_A[i][j] = A[i][j]
				else:
					#print b[i][j-cols_A]
					new_A[i][j] = b[i][j-cols_A]
	except ValueError:
		raise ValueError("The rows of the matrix A must be equal to the rows of b")
	return new_A


'''
2.2 初等行变换
交换两行
把某行乘以一个非零常数
把某行加上另一行的若干倍：
'''
# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
	#获得matrix 的行数，传入的参数行号应该小于行数
    rows = shape(M)[0]
    assert r1 < rows
    assert r2 < rows

    array = M[r2]
    M[r2] = M[r1]
    M[r1] = array

# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
	if scale ==0:
		raise ValueError ("scale 不应该为0")
	rows = shape(M)[0]
	assert r<rows, "输入的行号应该小于行数"

	M[r] = [x*scale for x in M[r]]

# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale, tolerance=1e-10):
    rows = shape(M)[0]
    assert r1 <rows, "输入的行号应该小于行数"
    assert r2 <rows, "输入的行号应该小于行数"
    if abs(scale) <tolerance:
		raise ValueError ("scale 不应该为0")

    M[r1] = [x+y*scale for x,y in zip(M[r1], M[r2])]

'''
2.3.1 算法
步骤1 检查A，b是否行数相同
步骤2 构造增广矩阵Ab
步骤3 逐列转换Ab为化简行阶梯形矩阵 中文维基链接
对于Ab的每一列（最后一列除外）
    当前列为列c
    寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
    如果绝对值最大值为0
        那么A为奇异矩阵，返回None (你可以在选做问题2.4中证明为什么这里A一定是奇异矩阵)
    否则
        使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c）
        使用第二个行变换，将列c的对角线元素缩放为1
        多次使用第三个行变换，将列c的其他元素消为0
步骤4 返回Ab的最后一列
注： 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。
'''
""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16

    返回列向量 x 使得 Ax = b
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""
def get_col_w_max_value(A, row, col):
    rows = shape(A)[0]
    array = []
    for i in range(row, rows):
        array.append( abs(A[i][col]))
    print array
    #因为从row行开始计算，因此返回时要加上row
    row += array.index(max(array))
    return row, max(array)

def clear_coefficients_below(A, row, col, eps = 1.0e-16):
    rows = len(A)
    #起点要从row+1开始，就是row的下一行开始
    for k in range(row+1, rows):
        if not (abs(A[k][col]) < eps):
            alpha = -(A[k][col]/A[row][col])
            if not (abs(alpha) < eps):
                addScaledRow(A, k, row, alpha)

def clear_coefficients_above(A, col, eps = 1.0e-16):
    row = col
    beta = A[row][col]
    #起点要从row-1开始，就是row的下一行开始
    for k in range(row-1, -1, -1):
        if not (abs(beta) < eps):
            alpha = -(A[k][col]/beta)
            #print col, row, k, alpha
            if not (abs(alpha) < eps):
                addScaledRow(A, k, row, alpha)

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    rows_A = shape(A)[0]
    cols_A = shape(A)[1]
    rows_B = shape(b)[0]

    # 首先检查A矩阵是不是方阵（即行数和列数相等的矩阵。若行数和列数不相等，那就谈不上奇异矩阵和非奇异矩阵）
    if rows_A != cols_A:
    	print "This matrix is not a square matrix, columns is not equal to rows"
        return None

    #第二要检查A矩阵的轶是否和行数或者列数相等
    #这部分因为要求不能用numpy暂时跳过，等消元后再检查是否该列的|A|为0
    for row in range(rows_A):
    	clear_coefficients_below(A, row, row, epsilon)
    	clear_coefficients_above(A, row, epsilon)
    print A

    #通过两个检查后，step 1 已经在augmentMatrix方法中包含，因此直接step 2
    Ab = augmentMatrix(A,b)
    #开始消元
    for col in range(cols_A):
        row = col
        #寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
        row_max = get_col_w_max_value(Ab, row, col)[0]
        max_val = get_col_w_max_value(Ab, row, col)[1]

        #  如果绝对值最大值为0
        # 那么A为奇异矩阵，返回None (你可以在选做问题2.4中证明为什么这里A一定是奇异矩阵)
        # 若不等于0，称矩阵A为非奇异矩阵。
        if max_val == 0:
        	print "This matrix is a singular matrix"
        	return None
        else:
            #非奇异矩阵：
            if(row_max != row ):
                swapRows(Ab, row, row_max)


        #多次使用第三个行变换，将列c的其他元素消为0
        clear_coefficients_below(Ab, row, col, epsilon)
        #使用第二个行变换，将列c的对角线元素缩放为1
        alpha = 1.0/Ab[row][col]
        if not (abs(Ab[row][col] - 1) <epsilon):
            scaleRow(Ab, row, alpha)
    #把矩阵精细到小数点4位
    matxRound(Ab, decPts=4)
    print "after clear below efficiency, the Ab is ..."
    print Ab
    j = cols_A
    while j>0:
        clear_coefficients_above(Ab, j-1, epsilon)
        j -= 1

    #把矩阵精细到小数点4位
    matxRound(Ab, decPts=4)
    #消元后再次检查是否为奇异矩阵
    print "Ab is...."
    print Ab
    for col in range(cols_A):
    	max_val = get_col_w_max_value(Ab, col, col)[1]
    	if max_val == 0:
            return None

    new_b = []
    #第四步，返回最后一列
    for i in range(rows_B):
        #print Ab[i][cols_A]
        new_b.append([Fraction(Ab[i][cols_A]).limit_denominator()])
    return new_b

class MyFraction(Fraction):
    def is_near_zero(self, eps=1e-16):
        return abs(self.numerator) < eps

# 不要修改这里！
# 运行一次就够了！

from helper import *
from matplotlib import pyplot as plt
import random

X,Y = generatePoints(seed,num=100)

## 可视化

plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')
#plt.show()

m1 = 0
b1 = 0

# 不要修改这里！
plt.xlim((-5,5))
x_vals = plt.axes().get_xlim()
y_vals = [m1*x+b1 for x in x_vals]
plt.plot(x_vals, y_vals, '-', color='r')

plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')




#任意找两个点，连城一条线
index_a = int(random.random()*100)
index_b = int(random.random()*100)
while index_b == index_a:
	index_b = int(random.random()*100)

point_a = Vector([X[index_a],Y[index_a]])
point_b = Vector([X[index_b],Y[index_b]])

direction_vector = Vector([X[index_a],Y[index_a]]).minus(Vector([X[index_b], Y[index_b]]))
#因为normal vector = [A,B], Direction vector = [B, -A]
#y=mx+b 的normal vector是(m, -1)因此，随便用两个点的normal vector已经获得，我们因此可以获得一条线
#在这条线上，任何一点都可以base point因此K的求值很简单了。

normal_vector = Vector([direction_vector.coordinates[1], -direction_vector.coordinates[0]])
A = normal_vector.coordinates[0]
B = normal_vector.coordinates[1]
K1 = [Decimal(x)*Decimal(y) for x,y in zip(point_a.coordinates, normal_vector.coordinates)]
K = sum(K1)

#按照y=mx+b来计算
m = Decimal(-A/B).quantize(Decimal('0.0000'))
b = Decimal(K/B).quantize(Decimal('0.0000'))

#print m, b

#画出这条线来
plt.xlim((-5,5))
x_vals = plt.axes().get_xlim()
y_vals = [m*Decimal(x)+Decimal(b) for x in x_vals]
plt.plot(x_vals, y_vals, '-', color='r')

#plt.show()


# TODO 实现以下函数并输出所选直线的MSE

def calculateMSE(X,Y,m,b):
    MSE = 0
    n = len(X)
    assert n == len(Y)
    for i in range(n):
    	MSE += (Y[i]-m*X[i]-b)**2
    return MSE/n

#print(calculateMSE(X,Y,m1,b1))


# TODO 实现线性回归
'''
参数：X, Y 存储着一一对应的横坐标与纵坐标的两个一维数组
返回：m，b 浮点数
'''

def linearRegression(X,Y):
	#Step 1:本项目所有矩阵都为二维矩阵，因此首先把X,Y都变成二维的
	c = [[1] for i in range(len(X))]
	new_X = [[X[i]] for i in range(len(X))]
	new_Y = [[Y[i]] for i in range(len(Y))]

	#Step2: 合并X和常量,计算XTY, 和 XTX
	new_X = augmentMatrix(new_X,c)
	b =  matxMultiply(transpose(new_X), new_Y)
	A = matxMultiply(transpose(new_X),new_X)

	#Step3: 用高斯消元法去解 Ax=b, 实际上是求解  XTXh=XTY
	h = gj_Solve(A,b)
	m = float(h[0][0].numerator/h[0][0].denominator)
	b = float(h[1][0].numerator/h[1][0].denominator)

	return m, b


#m2,b2 = linearRegression(X,Y)
#assert isinstance(m2,float),"m is not a float"
#assert isinstance(b2,float),"b is not a float"
#print(m2,b2)


#A = generateMatrix(3,seed,singular=False)
#b = np.ones(shape=(3,1),dtype=int) # it doesn't matter

A = [[ 6, -4,  8],[-6,  6,  6],[2, -2, -2]]
b = [[0],[1],[2]]
#print np.linalg.matrix_rank(A)
#Ab = augmentMatrix(A,b) # 请确保你的增广矩阵已经写好了
#printInMatrixFormat(Ab,padding=3,truncating=0)

print gj_Solve(A,b)

