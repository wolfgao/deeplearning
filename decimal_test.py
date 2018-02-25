# -*- coding: UTF-8 -*-
# 目前看来默认的选项ROUND_HALF_EVEN是最好的选择，否则你可以看看下面的结果。
# 之前选择了ROUND_CEILING，结果-0.999856四舍五入变成了-0.999
# 官方文档：
# https://docs.python.org/2/whatsnew/2.4.html#pep-327-decimal-data-type
#
from decimal import *
import math


x = Decimal('-0.999856')
y = Decimal('0.999856')

print "The origin value is %f" %x
print "After decimal, it is ..."
print x.quantize(Decimal('0.000'), ROUND_HALF_EVEN)
print x.quantize(Decimal('0.000'), ROUND_HALF_DOWN)
print x.quantize(Decimal('0.000'), ROUND_CEILING)
print x.quantize(Decimal('0.000'), ROUND_FLOOR)
print x.quantize(Decimal('0.000'), ROUND_UP)
print x.quantize(Decimal('0.000'), ROUND_DOWN)

print "The origin value is %f" %y
print "After decimal, it is ..."
print y.quantize(Decimal('0.000'), ROUND_HALF_EVEN)
print y.quantize(Decimal('0.000'), ROUND_HALF_DOWN)
print y.quantize(Decimal('0.000'), ROUND_CEILING)
print y.quantize(Decimal('0.000'), ROUND_FLOOR)
print y.quantize(Decimal('0.000'), ROUND_UP)
print y.quantize(Decimal('0.000'), ROUND_DOWN)

print math.acos(Decimal(str('-1.0')))

#print "Before trap the error, we try to get 1/0 = %f" %(Decimal(1)/Decimal(0))
'''
traps is a dictionary specifying what happens on encountering certain error conditions:
either an exception is raised or a value is returned.
Some examples of error conditions are division by zero, loss of precision, and overflow.
'''
# turn off the traps
getcontext().traps[DivisionByZero] = False
print "After trap the error, we try to get 1/0 = %s" %(Decimal(1)/Decimal(0))
print Decimal(1)/Decimal(0)