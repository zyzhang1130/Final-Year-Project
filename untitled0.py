from theano import tensor as T
from theano import function, shared
import numpy

X = shared(numpy.array([0,1,2,3,4]))
Y = T.vector()
X_update = (X, T.set_subtensor(X[2:4], Y))
f = function([Y], updates=[X_update])
f([100,10])
print (X.get_value()) # [0 1 100 10 4]