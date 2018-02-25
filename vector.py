# -*- coding: UTF-8 -*-

import math   # 导入 math 模块
#不得不导入这个模块，因为很多地方由于精度问题报出了python ValueError: math domain error
from decimal import *
#可以设定有效数字

class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            for coordinate in coordinates:
                coordinate = Decimal(str(coordinate)).quantize(Decimal('0.000'))
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')


    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def plus(self, v):
        '''
        zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
        '''
        new_coordinates = [Decimal(x)+Decimal(y) for x,y in zip(self.coordinates, v.coordinates)]

        return Vector(new_coordinates)

    def minus(self, v):
        new_coordinates = [Decimal(x)-Decimal(y) for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def multiply(self, scale):
        new_coordinates = [Decimal(scale)*Decimal(x) for x in self.coordinates]
        return Vector(new_coordinates)

    '''
    如果一个向量是另一个向量的倍数，那么每一个coordinate都应该是同样的倍数
    这里定义0为错误的情况，就是不存在倍数相同的情况
    '''
    def getScale(self,v,tolerance=1e-10):
        #排除不在同一维的向量，必须维数相同
        if self.dimension != v.dimension:
            raise ValueError('The two vectors must have same dimensions')
        new_coordinates =[Decimal(str(x/y)).quantize(Decimal('0.000')) for x,y in zip(self.coordinates, v.coordinates)]
        #至少是1微的
        old_scale = new_coordinates[0]
        for scale in new_coordinates:
            if (old_scale - scale >tolerance):
                return 0
        return old_scale


    '''
    计算向量的Magnitude
    数学公式：sqrt(x^2+y^2+...n^2)
    '''
    def magnitude(self):
        coordinates_squared = [(Decimal(x)**2) for x in self.coordinates]
        magnitude =  math.sqrt(sum(coordinates_squared))
        return Decimal(str(magnitude)).quantize(Decimal('0.000'))

    '''
    计算向量的归1值，normalize值
    公式：(1/Magnitude)*coordinates
    注意⚠️：magnitude可能是0
    '''
    def normalize(self):
        try:
            normal_coordinates = [(1/self.magnitude())*x for x in self.coordinates]

        except ZeroDivisionError:
            return Exception('Can\'t normalize the zero vector')

        return Vector(normal_coordinates)

    '''
    计算两个向量的乘积，这里看内乘积，就是Inner Products 或者（Dot Products)
    公式：sum(V1.coordinates*V2.coordinates)
     or: v1*v2 = v1.magnitude*v2.magnitude*cosɵƟ
    '''
    def dotProduct(self, v):
        new_coordinates = [Decimal(x)*Decimal(y) for x,y in zip(self.coordinates, v.coordinates)]
        return Decimal(str(sum(new_coordinates))).quantize(Decimal('0.000'))

    '''
    计算两个vector的angle
    公式：arcos(v1*v2/v1.magnitude*v2.magnitude)
    注意：某个vector的magnitude为0
    当前的情况下，在angle等于0的情况下，由于计算精度的情况会出现下面错误
    python ValueError: math domain error
    '''
    def dotAngle(self, v, in_degrees=False):
        try:
            thet = self.dotProduct(v)/(self.magnitude()*v.magnitude())
            #print "thet is %f" %thet
            thet = Decimal(str(thet)).quantize(Decimal('0.000'))
            #print "After decimal, thet is %f" %Decimal(thet)
            angle = math.acos(Decimal(thet))
            #print "angle is %f" %angle
            angle = Decimal(str(angle)).quantize(Decimal('0.000'))
        except ZeroDivisionError:
            return Exception('Can\'t normalize the zero vector')
        if(in_degrees):
            return math.degrees(angle)
        else:
            return angle


    '''
    判断两个向量是否平行 parallel还是正交 orthogonal需要先判断他们的angle
    注意如果一个0向量比较的，可以同时是parallel和orthogonal
    tolerance是误差范围
    '''
    def is_orthogonal_to(self, v, tolerance=1e-10):
        return ( self.is_Zero()
            or v.is_Zero()
            or abs(self.dotProduct(v))<= tolerance )

    '''
    判断两个向量是否平行 parallel还是正交 orthogonal需要先判断他们的angle
    注意如果一个0向量比较的，可以同时是parallel和orthogonal
    tolerance是误差范围
    '''
    def is_parallel_to(self,v):
        return ( self.is_Zero()
            or v.is_Zero()
            or self.dotAngle(v) == 0
            or self.dotAngle(v) == Decimal(str(math.pi)).quantize(Decimal('0.000')) )

    '''
    判断此向量是否为0
    '''
    def is_Zero(self, tolerance=1e-10):
        return self.magnitude() < Decimal(tolerance)

    '''
    这里计算一下向量在另外一个变量的平行投影
    公式：v.parallel = (v*b.normal)*b.normal
    '''
    def parallelValues(self, b):
        return b.normalize().multiply(self.dotProduct(b.normalize()))

    '''
    一个向量在另外一个向量的投影 Vector Projections
    projb(v) = v的parallel/b，
    '''
    def project(self,b):
        return self.parallelValues(b)/b

    '''
    这里计算一下向量在另外一个垂直变量:实际上就是这个变量减去他的平行变量
    公式：v-v.parallelValues(self, b)
    '''
    def orthogonalValues(self, b):
        return self.minus(self.parallelValues(b))


    '''
    Cross Products,不同于inner products
    公式：
        v*w = [v.y*w.z-w.y*v.z, -(v.x*w.z-w.x*v.z), v.x*w.y-w.x*v.y]
    '''
    def crossProducts(self,v):
        '''
        x1 = self.coordinates[0]
        y1=self.coordinates[1]
        z1=self.coordinates[2]

        x2= v.coordinates[0]
        y2= v.coordinates[1]
        z2= v.coordinates[2]
        '''
        try:
            x1, y1, z1 = self.coordinates
            x2, y2, z2 = v.coordinates
            new_coordinates = [y1*z2-y2*z1,
                                        -(x1*z2-x2*z1),
                                        x1*y2-x2*y1]

            return Vector(new_coordinates)

        except ValueError as e:
            msg = str(e)
            # 可能这个vector 只有2个变量，二维的，我们就补充一维
            if msg == "need more than 2 values to unpack":
                self_embedded_in_R3 = Vector(self.coordinates + (0,))
                v_embedded_in_R3 = Vector(v.coordinates + (0,))
                return self_embedded_in_R3.crossProducts(v_embedded_in_R3)
            #可能这个vector是一维的，或者4微以上
            elif (msg == "too many values to unpack" or
                msg == "need more than 1 value to unpack"):
                raise Exception(e)
            else:
                raise e

    '''
    Area of Cross prducts
    '''
    def area_crossProducts(self,v):
        return self.crossProducts(v).magnitude()


    '''
    Area of Cross prducts of this triangle
    '''
    def area_crossProducts(self,v):
        return 0.5*self.crossProducts(v).magnitude()



#test...
'''
print Vector((-0.221,7.437)).magnitude()
print Vector((-0.221,7.437)).normalize()
print Vector((5.581,-2.136)).normalize()
print Vector((8.813,-1.331,-6.247)).magnitude()
print Vector((1.996,3.108,-4.554)).normalize()


print math.degrees(Vector([-7.579,-7.88]).dotAngle(Vector([22.737,23.64])))
print math.degrees(Vector([-2.029,9.97,4.172]).dotAngle(Vector([-9.231,-6.639,-7.245])))
print math.degrees(Vector([-2.328,-7.284,-1.214]).dotAngle(Vector([-1.821,1.072,-2.94])))

print Vector([-2.118,4.827]).is_orthogonal_to(Vector([0]))
print Vector([-2.118,4.827]).is_parallel_to(Vector([0]))
print Vector([0]).is_Zero()


print Vector([3.039,1.879]).parallelValues(Vector([0.825,2.036]))
print Vector([-9.88,-3.264,-8.159]).orthogonalValues(Vector([-2.155,-9.353,-9.473]))
print Vector([3.009,-6.172,3.692,-2.51]).parallelValues(Vector([6.404,-9.144,2.759,8.718]))
print Vector([3.009,-6.172,3.692,-2.51]).orthogonalValues(Vector([6.404,-9.144,2.759,8.718]))


print Vector([8.462,7.893]).crossProducts(Vector([6.984,-5.975]))
print
print Vector([-8.987]).area_crossProducts(Vector([-4.268]))
print
print Vector([1.5,9.547,3.691,3.454]).area_crossProducts(Vector([-6.007,0.124,5.772,5.410]))
'''





