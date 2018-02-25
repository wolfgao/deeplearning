# -*- coding: UTF-8 -*-

from decimal import Decimal, getcontext
from copy import deepcopy

from vector import Vector
from plane import Plane

getcontext().prec = 30


class LinearSystem(object):

    ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
    NO_SOLUTIONS_MSG = 'No solutions'
    INF_SOLUTIONS_MSG = 'Infinitely many solutions'

    def __init__(self, planes):
        try:
            d = planes[0].dimension
            for p in planes:
                assert p.dimension == d

            self.planes = planes
            self.dimension = d
            self.len = self.__len__()

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    '''
    下面的三个方程，row都是从0起始，那么这里就指行号，比如1指第一个方程
    '''
    def swap_rows(self, row1, row2):
        #行号不能超过方程的总数，否则就会 index 溢出
        assert row1 < self.len
        assert row2 < self.len
        self[row1], self[row2] = self[row2], self[row1]


    def multiply_coefficient_and_row(self, coefficient, row):
        #行号不能超过方程的总数，否则就会 index 溢出
        assert row < self.len
        new_normal_vector = self[row].normal_vector.multiply(coefficient)
        new_constant_term = Decimal(self[row].constant_term)*Decimal(coefficient)
        self[row] = Plane(new_normal_vector,new_constant_term)
        return Plane(new_normal_vector,new_constant_term)


    def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):
        assert row_to_add < self.len
        assert row_to_be_added_to <self.len

        #调用multiply_coefficient_and_row之后要还原row_to_add，因为在那个方法里面我们把row_to_add换掉了。
        p_row_to_add = self[row_to_add]
        new_plane = self.multiply_coefficient_and_row(coefficient, row_to_add)
        self[row_to_add] = p_row_to_add
        new_normal_vector = new_plane.normal_vector.plus(self[row_to_be_added_to].normal_vector)
        new_constant_term = new_plane.constant_term + self[row_to_be_added_to].constant_term
        #改变了被加的那行的方程
        self[row_to_be_added_to] = Plane(new_normal_vector,new_constant_term)
        return Plane(new_normal_vector,new_constant_term)

    def compute_triangular_form(self):
        new_quatations = deepcopy(self)
        len = new_quatations.len
        num_variables = new_quatations.dimension
        j=0
        #遍历copy后的方程组，对每个变量前的系数进行判断
        for i in range(len):
            while j<num_variables:

                c = MyDecimal(new_quatations[i].normal_vector.coordinates[j])
                #print i, j,c
                if c.is_near_zero():
                    swap_succeeded = new_quatations.swap_with_row_below_for_nonzero(i,j)
                    if not swap_succeeded:
                        j +=1
                        continue

                new_quatations.clear_coefficients_below(i,j)
                j +=1
                break
        return new_quatations
        '''
        #这个是我之前的算法，验证也是work的，但是扩展性不够，很多代码是写死的，需要进一步优化
        #上面的算法是老师提供的，我自己调试修改了部分代码，总体要优于我下面的算法
        for i, p in enumerate(new_quatations):
            if i == 0:
                if(p.first_nonzero_index(p.normal_vector.coordinates) !=0):
                    for k, plane in enumerate(new_quatations):
                        if(0== plane.first_nonzero_index(plane.normal_vector.coordinates)):
                            #print k
                            new_quatations.swap_rows(0, k)
                            break
                else:
                    pass
            elif i == 1:
                #如果第一维的系数不等于0，那么需要消元
                if p.first_nonzero_index(p.normal_vector.coordinates) == 0:
                    #need 消元
                    coefficient = -(Decimal(p.normal_vector.coordinates[0])/Decimal(new_quatations[0].normal_vector.coordinates[0]))
                    new_quatations.add_multiple_times_row_to_row(coefficient,0,1)
                elif p.first_nonzero_index(p.normal_vector.coordinates) == 0:
                    pass
            elif i == 2:
                if p.first_nonzero_index(p.normal_vector.coordinates) == 0:
                    #如果第一维的系数不等于0，那么需要消元，先消第一维的元
                    coefficient = -(Decimal(p.normal_vector.coordinates[0])/Decimal(new_quatations[0].normal_vector.coordinates[0]))
                    new_quatations.add_multiple_times_row_to_row(coefficient,0,2)
                    #检查一下新的方程组中第3个方程的第二个维系数，如果还不为0，需要继续消元
                    if new_quatations[i].first_nonzero_index(new_quatations[i].normal_vector.coordinates) ==1:
                         #第二个维度的常量还是1，因此还需要再消元，和第二个方程直接消元
                        coefficient = -(Decimal(new_quatations[i].normal_vector.coordinates[1])/Decimal(new_quatations[1].normal_vector.coordinates[1]))
                        new_quatations.add_multiple_times_row_to_row(coefficient,1,2)
                if p.first_nonzero_index(p.normal_vector.coordinates) == 1:
                    #如果第二维的系数不等于0，那么需要消第二维的元
                    coefficient = -(Decimal(p.normal_vector.coordinates[1])/Decimal(new_quatations[1].normal_vector.coordinates[1]))
                    new_quatations.add_multiple_times_row_to_row(coefficient,1,2)
            if i>=3 and i<len:
                new_quatations[i] = Plane()
        return new_quatations
        '''

    def swap_with_row_below_for_nonzero(self, row, col):
        num_equations = len(self)
        #起点要从row+1开始，就是row的下一行开始
        for k in range(row+1,num_equations):
            coefficient = MyDecimal(self[k].normal_vector.coordinates[col])
            #print self[k+1].normal_vector
            if not coefficient.is_near_zero():
                self.swap_rows(row, k)
                return True
        return False

    def clear_coefficients_below(self, row, col):
        num_equations = len(self)
        beta = MyDecimal(self[row].normal_vector.coordinates[col])
        #起点要从row+1开始，就是row的下一行开始
        for k in range(row+1, num_equations):
            n = self[k].normal_vector
            gamma = n.coordinates[col]
            alpha = -Decimal(gamma)/Decimal(beta)
            #print row, k, alpha
            self.add_multiple_times_row_to_row(alpha, row, k)


    def indices_of_first_nonzero_terms_in_each_row(self):
        num_equations = len(self)
        num_variables = self.dimension

        indices = [-1] * num_equations

        for i,p in enumerate(self.planes):
            try:
                indices[i] = p.first_nonzero_index(p.normal_vector.coordinates)
            except Exception as e:
                if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
                    continue
                else:
                    raise e

        return indices

    #Here 我们通过这个方法来获得更简单的三角形方程组，确定每个方程leading变量的系数是1
    # Each pivot variable had coefficient 1
    # Each pivot variable is in own column
    # 0=k,0=0 must be put the end row of this quatations.
    def compute_rref(self):
        #首先获得三角形方程组
        tf = self.compute_triangular_form()
        #然后我们检查每个方程组的系数
        len = tf.len
        num_variables = tf.dimension
        pivot_indices = tf.indices_of_first_nonzero_terms_in_each_row()
        j = num_variables-1

        for i in range(len-1,-1,-1):
            j = pivot_indices[i]

            if j<0:
                continue

            d = MyDecimal(1 - j)
            if not d.is_near_zero() and j!=0 :
                alpha = Decimal('1')/j
                tf.multiply_coefficient_and_row(alpha,i)
            tf.clear_coefficients_above(i, j)

        return tf

    def clear_coefficients_above(self, row, col):
        num_equations = len(self)
        beta = MyDecimal(self[row].normal_vector.coordinates[col])
        #起点要从row-1开始，就是row的下一行开始
        for k in range(0, row, 1):
            gamma = self[k].normal_vector.coordinates[col]
            alpha = -Decimal(gamma)/Decimal(beta)
            #print row, k, alpha
            self.add_multiple_times_row_to_row(alpha, row, k)

    def extract_direction_vectors_for_paramaterization(self):
        num_equations = self.len
        num_variables = self.dimension

        pivot_indices = self.indices_of_first_nonzero_terms_in_each_row()
        free_variable_indices = set(range(num_variables)) - set(pivot_indices)
        print free_variable_indices

        direction_vectors = []

        for free_var in free_variable_indices:
            vector_coords = [0] * num_variables
            vector_coords[free_var] = 1
            for i, p in enumerate(self.planes):
                pivot_var = pivot_indices[i]
                if pivot_var <0:
                    break
                vector_coords[pivot_var] = -p.normal_vector.coordinates[free_var]
            direction_vectors.append(Vector(vector_coords))
        return direction_vectors


    def __len__(self):
        return len(self.planes)


    def __getitem__(self, i):
        return self.planes[i]


    def __setitem__(self, i, x):
        try:
            assert x.dimension == self.dimension
            self.planes[i] = x

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
        ret += '\n'.join(temp)
        return ret

class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


class Parameterization(object):
    """docstring for Parameterization"""
    def __init__(self, base_point, direction_vector):
        super(Parameterization, self).__init__()
        self.base_point = base_point
        self.direction_vector = direction_vector

    def __str__(self):
        j = self.indices_of_first_nonzero_terms_in_each_row()
        print j

p1 = Plane(normal_vector=Vector(['0.786','0.786','0.588']), constant_term='-0.714')
p2 = Plane(normal_vector=Vector(['-0.138','-0.138','0.244']), constant_term='0.319')
s = LinearSystem([p1,p2])
r = s.compute_rref()
print r

print r.extract_direction_vectors_for_paramaterization()


'''
p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
r = s.compute_rref()
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='-1') and
        r[1] == p2):
    print 'test case 1 failed'

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
r = s.compute_rref()
if not (r[0] == p1 and
        r[1] == Plane(constant_term='1')):
    print 'test case 2 failed'

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
s = LinearSystem([p1,p2,p3,p4])
r = s.compute_rref()
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='0') and
        r[1] == p2 and
        r[2] == Plane(normal_vector=Vector(['0','0','-2']), constant_term='2') and
        r[3] == Plane()):
    print 'test case 3 failed'

p1 = Plane(normal_vector=Vector(['0','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','-1','1']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','2','-5']), constant_term='3')
s = LinearSystem([p1,p2,p3])
r = s.compute_rref()
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term=Decimal('23')/Decimal('9')) and
        r[1] == Plane(normal_vector=Vector(['0','1','0']), constant_term=Decimal('7')/Decimal('9')) and
        r[2] == Plane(normal_vector=Vector(['0','0','1']), constant_term=Decimal('2')/Decimal('9'))):
    print 'test case 4 failed'

'''
'''
p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
t = s.compute_triangular_form()
if not (t[0] == p1 and
        t[1] == p2):
    print 'test case 1 failed'

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
t = s.compute_triangular_form()
#print t
if not (t[0] == p1 and
        t[1] == Plane(constant_term='1')):
    print 'test case 2 failed'

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
s = LinearSystem([p1,p2,p3,p4])
t = s.compute_triangular_form()
if not (t[0] == p1 and
        t[1] == p2 and
        t[2] == Plane(normal_vector=Vector(['0','0','-2']), constant_term='2') and
        t[3] == Plane()):
    print 'test case 3 failed'

p1 = Plane(normal_vector=Vector(['0','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','-1','1']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','2','-5']), constant_term='3')
s = LinearSystem([p1,p2,p3])
t = s.compute_triangular_form()

if not (t[0] == Plane(normal_vector=Vector(['1','-1','1']), constant_term='2') and
        t[1] == Plane(normal_vector=Vector(['0','1','1']), constant_term='1') and
        t[2] == Plane(normal_vector=Vector(['0','0','-9']), constant_term='-2')):
    print 'test case 4 failed'


'''
'''
p0 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p1 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p2 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p3 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')

s = LinearSystem([p0,p1,p2,p3])
print s
print s.swap_rows(2,3)
print s

print s.add_multiple_times_row_to_row(2,2,4)

p0 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p1 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p2 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p3 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')

s = LinearSystem([p0,p1,p2,p3])
print "Before test, the s is "
print s

s.swap_rows(0,1)
if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
    print 'test case 1 failed'

s.swap_rows(1,3)
if not (s[0] == p1 and s[1] == p3 and s[2] == p2 and s[3] == p0):
    print 'test case 2 failed'

s.swap_rows(3,1)
if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
    print 'test case 3 failed'

s.multiply_coefficient_and_row(1,0)
if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
    print 'test case 4 failed'

s.multiply_coefficient_and_row(-1,2)
if not (s[0] == p1 and
        s[1] == p0 and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print 'test case 5 failed'

s.multiply_coefficient_and_row(10,1)
if not (s[0] == p1 and
        s[1] == Plane(normal_vector=Vector(['10','10','10']), constant_term='10') and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print 'test case 6 failed'

s.add_multiple_times_row_to_row(0,0,1)
if not (s[0] == p1 and
        s[1] == Plane(normal_vector=Vector(['10','10','10']), constant_term='10') and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print 'test case 7 failed'

s.add_multiple_times_row_to_row(1,0,1)
if not (s[0] == p1 and
        s[1] == Plane(normal_vector=Vector(['10','11','10']), constant_term='12') and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print 'test case 8 failed'

s.add_multiple_times_row_to_row(-1,1,0)
if not (s[0] == Plane(normal_vector=Vector(['-10','-10','-10']), constant_term='-10') and
        s[1] == Plane(normal_vector=Vector(['10','11','10']), constant_term='12') and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print 'test case 9 failed'

print "After test case 9, s = "
print s

#print s.indices_of_first_nonzero_terms_in_each_row()
#print '{},{},{},{}'.format(s[0],s[1],s[2],s[3])
#print len(s)

#s[0] = p1
#print s

#print MyDecimal('1e-9').is_near_zero()
#print MyDecimal('1e-11').is_near_zero()
'''
