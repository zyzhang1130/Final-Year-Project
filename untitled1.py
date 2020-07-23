from theano import tensor as T
from theano import function
import numpy

X = T.shared(numpy.array([0,1,2,3,4]))
Y = T.vector()
f = function([Y], updates=[(X[2:4], Y)] 