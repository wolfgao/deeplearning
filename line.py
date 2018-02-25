# -*- coding: UTF-8 -*-
'''
In this class file, methods:__init__,set_basepoint,__str__,first_nonzero_index are defined by Udacity
The other methods "is_parallel, is_same_line,get_intersection" are defined by me
'''


from decimal import *

from vector import Vector

getcontext().prec = 30


class Line(object):

    NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 2
        ''' Normal vector
        就是比如Ax+By=K，那么它的normal vector就是[B,A]
        如果两条line的normal vector是倍数关系，他们是平行的，否则是有intersection的
        '''
        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

        if not constant_term:
            constant_term = Decimal('0')
        self.constant_term = Decimal(constant_term)

        self.set_basepoint()


    '''
    最初当Vector(xt) = Vector(x0) +t*Vector(x)
    我们把t用几何的一条线来表示，任何上面的点都可以成为base point。
    如果我们把t按照一条线的话，我们可以用这样的方程式来表达：y = mx + b, (y,x,b)都是不同的coordinates
    也可以通过Ax+By=K来表达。
    因此basepoint可以这么来取：当x=0, y= k/B (B不等于0),base point 就是（0, K/B)
    当y = 0, x = K/A (A 不等于0)，base point就是（K/A, 0)
    [A,B]*[x,y] = K，假如K=0, 不影响线的方向，那么=>[A,B]*[x,y] = 0，因此normal vector = [A,B]
    Direction vector = [B, -A]
    '''
    def set_basepoint(self):
        try:
            n = self.normal_vector
            c = self.constant_term
            basepoint_coords = ['0']*self.dimension
            '''
            The previous is "initial_index = Line.first_nonzero_index(n.coordinates)"
            That will pop up "TypeError: 'Vector' object is not iterable"
            so change n to n.coordinates, that actually is a tuple, can be iterable.
            But with that change, it will pop up "TypeError: 'Vector' object does not support indexing"
            so have to change Decimal(n[initial_index]) to Decimal(n.coordinates[initial_index])
            '''
            initial_index = Line.first_nonzero_index(n.coordinates)
            #print initial_index
            initial_coefficient = Decimal(n.coordinates[initial_index])
            #print initial_coefficient
            basepoint_coords[initial_index] = c/initial_coefficient
            self.basepoint = Vector(basepoint_coords)

        except Exception as e:
            if str(e) == Line.NO_NONZERO_ELTS_FOUND_MSG:
                self.basepoint = None
            else:
                raise e


    def __str__(self):

        num_decimal_places = 3

        def write_coefficient(coefficient, is_initial_term=False):
            coefficient = round(coefficient, num_decimal_places)
            if coefficient % 1 == 0:
                coefficient = int(coefficient)

            output = ''

            if coefficient < 0:
                output += '-'
            if coefficient > 0 and not is_initial_term:
                output += '+'

            if not is_initial_term:
                output += ' '

            if abs(coefficient) != 1:
                output += '{}'.format(abs(coefficient))

            return output

        n = self.normal_vector

        try:
            initial_index = Line.first_nonzero_index(n.coordinates)
            terms = [write_coefficient(n.coordinates[i], is_initial_term=(i==initial_index)) + 'x_{}'.format(i+1)
                     for i in range(self.dimension) if round(n.coordinates[i], num_decimal_places) != 0]
            output = ' '.join(terms)

        except Exception as e:
            if str(e) == self.NO_NONZERO_ELTS_FOUND_MSG:
                output = '0'
            else:
                raise e

        constant = round(self.constant_term, num_decimal_places)
        if constant % 1 == 0:
            constant = int(constant)
        output += ' = {}'.format(constant)

        return output


    @staticmethod
    def first_nonzero_index(iterable):
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(Line.NO_NONZERO_ELTS_FOUND_MSG)

############################# coding by wolfgao ###########################
    def get_normal_vector(self):
        return self.normal_vector

    def get_constant_term(self):
        return self.constant_term
    '''
    如果一条线的normal vector和另外一条线的normal vector平行的话，说明这两个线至少是平行的。

    '''
    def is_parallel_with(self, line):
        return self.normal_vector.is_parallel_to(line.normal_vector)

    '''
    是不是同一条线，第一他们的normal vector 是平行的；
    其次，如果在这两条线上找一个点，比如base point和另外一条线的base point连线，
    如果和normal vector是orthogonal，那么这两条线是same
    要考虑如果你的normal vector is zero的情况，直线定义的左侧为0，右侧就是常量K了
    还有另外一条线的normal vector is zero的情况
    '''
    def __eq__(self, line):
        if self.normal_vector.is_Zero():
            if not line.normal_vector.is_Zero():
                return False
            else:
                diff =  self.constant_term - line.constant_term
                return MyDecimal(diff).is_near_zero()
        elif line.normal_vector.is_Zero():
            return False

        if (self.is_parallel_with(line)):
            new_vector = self.basepoint.minus(line.basepoint)
            return new_vector.is_orthogonal_to(self.normal_vector)
        else:
            return False

    '''
    如果一条线不和另外一条线平行，在2D的界面上，两条线必然要相交
    其实就是在解一个二元1次方程:
    A1x+B1y=K1 and A2x+B2y=K2
    因此：x= (K1B2-K2B1)/(A1B2-A2B1)
        y = (K1A2-K2A1)/(B1A2-B2A1)
    '''
    def get_intersec_p(self, line):
        A1 = Decimal(self.normal_vector.coordinates[0])
        B1 = Decimal(self.normal_vector.coordinates[1])
        K1 = Decimal(self.constant_term)

        A2 = Decimal(line.normal_vector.coordinates[0])
        B2 = Decimal(line.normal_vector.coordinates[1])
        K2 = Decimal(line.constant_term)

        if(self.is_parallel_with(line)):
           return None
        else:
            try:
                x = (K1*B2-K2*B1)/(A1*B2-A2*B1)
                y = (K1*A2-K2*A1)/(B1*A2-B2*A1)
                x = Decimal(str(x)).quantize(Decimal('0.000'), ROUND_CEILING)
                y = Decimal(str(y)).quantize(Decimal('0.000'), ROUND_CEILING)
            except ZeroDivisionError:
                return Exception('Can\'t devided by zero because the 2 lines are the same lines, no intersection')
            return Vector([x,y])


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


print Line(Vector([4.046,2.836]),1.21).is_parallel_with(Line(Vector([10.115,7.09]),3.025))
print Line(Vector([7.204,3.182]),8.68).is_parallel_with(Line(Vector([8.172,4.114]),9.883))
print Line(Vector([1.182,5.562]),6.744).is_parallel_with(Line(Vector([1.773,8.343]),9.525))

print "-----------------"
print Line(Vector([4.046,2.836]),1.21) == (Line(Vector([10.115,7.09]),3.025))
print Line(Vector([7.204,3.182]),8.68) == (Line(Vector([8.172,4.114]),9.883))
print Line(Vector([1.182,5.562]),6.744) == (Line(Vector([1.773,8.343]),9.525))
print "-----------------"
print Line(Vector([4.046,2.836]),1.21).get_intersec_p(Line(Vector([10.115,7.09]),3.025))
print Line(Vector([7.204,3.182]),8.68).get_intersec_p(Line(Vector([8.172,4.114]),9.883))
print Line(Vector([1.182,5.562]),6.744).get_intersec_p(Line(Vector([1.773,8.343]),9.525))
print "-----------------"