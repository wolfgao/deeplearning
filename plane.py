# -*- coding: UTF-8 -*-

from decimal import *

from vector import Vector

getcontext().prec = 30


class Plane(object):

    NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 3

        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

        if not constant_term:
            constant_term = Decimal('0')
        self.constant_term = Decimal(constant_term)

        self.set_basepoint()


    def set_basepoint(self):
        try:
            n = self.normal_vector
            c = self.constant_term
            basepoint_coords = ['0']*self.dimension

            initial_index = Plane.first_nonzero_index(n.coordinates)
            initial_coefficient = Decimal(n.coordinates[initial_index])

            basepoint_coords[initial_index] = c/initial_coefficient
            self.basepoint = Vector(basepoint_coords)

        except Exception as e:
            if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
                self.basepoint = None
            else:
                raise e


    def __str__(self):

        #num_decimal_places = 3
        #其实就是小数点保留3位

        def write_coefficient(coefficient, is_initial_term=False):
            coefficient = Decimal(coefficient).quantize(Decimal('0.000'))
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
            initial_index = Plane.first_nonzero_index(n.coordinates)
            terms = [write_coefficient(n.coordinates[i], is_initial_term=(i==initial_index)) + 'x_{}'.format(i+1)
                     for i in range(self.dimension) if Decimal(n.coordinates[i]).quantize(Decimal('0.000')) != 0]
            output = ' '.join(terms)

        except Exception as e:
            if str(e) == self.NO_NONZERO_ELTS_FOUND_MSG:
                output = '0'
            else:
                raise e

        constant = Decimal(self.constant_term).quantize(Decimal('0.000'))
        if constant % 1 == 0:
            constant = int(constant)
        output += ' = {}'.format(constant)

        return output


    @staticmethod
    def first_nonzero_index(iterable):
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(Plane.NO_NONZERO_ELTS_FOUND_MSG)

#######################################
    '''
    如果一个平面的normal vector和另外一平面的normal vector平行的话，说明这两个平面至少是平行的。

    '''
    def is_parallel_with(self, plane):
        return self.normal_vector.is_parallel_to(plane.normal_vector)

    '''
    是不是同一平面，第一他们的normal vector 是平行的；
    其次，如果在这两平面上找一个点，比如base point和另外一平面的base point连线，
    如果和其中的normal vector是orthogonal，那么这两条平面是same
    要考虑如果其中一个平面的normal vector is zero的情况，左侧为0，右侧就是常量K了
    如果同时为0，那么比较一下他们的常量K的差别
    '''
    def __eq__(self, plane):
        if self.normal_vector.is_Zero():
            if not plane.normal_vector.is_Zero():
                return False
            else:
                diff =  self.constant_term - plane.constant_term
                return MyDecimal(diff).is_near_zero()
        elif plane.normal_vector.is_Zero():
            return False

        if (self.is_parallel_with(plane)):
            new_vector = self.basepoint.minus(plane.basepoint)
            return new_vector.is_orthogonal_to(self.normal_vector)
        else:
            return False



class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


#print Plane(Vector([-0.412,3.806,0.728]),-3.46) == Plane(Vector([1.03,-9.515,-1.82]),8.65)
#print Plane(Vector([-0.412,3.806,0.728]),-3.46).is_parallel_with(Plane(Vector([1.03,-9.515,-1.82]),8.65))

#print Plane(Vector([2.611,5.528,0.283]),4.6).is_parallel_with(Plane(Vector([7.715,8.306,5.342]),3.76))
#print Plane(Vector([2.611,5.528,0.283]),4.6) == Plane(Vector([7.715,8.306,5.342]),3.76)

#print Plane(Vector([-7.926,8.625,-7.212]),-7.952).is_parallel_with(Plane(Vector([-2.642,2.875,-2.404]),-2.443))

#print Plane(Vector([-7.926,8.625,-7.212]),-7.952) == Plane(Vector([-2.642,2.875,-2.404]),-2.443)


