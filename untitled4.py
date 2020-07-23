import theano
from theano import tensor as T
import numpy as np
import math 
import cmath 

r = T.ivector()
new_r = T.set_subtensor(r[10:], 5)