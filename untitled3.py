import theano
from theano import tensor as T
import numpy as np
import math 
import cmath 
global W
l=1

def inputvector(theta,phi,zeta):
    global xreal,ximag
    xreal=np.asarray([math.cos(2*theta-phi)*math.cos(phi), math.cos(2*theta-phi)*math.sin(phi)*math.cos(zeta), math.cos(2*theta-phi)*math.sin(phi)*math.sin(zeta)])
    ximag=np.asarray([-math.sin(2*theta-phi)*math.sin(phi), math.sin(2*theta-phi)*math.cos(phi)*math.cos(zeta), math.sin(2*theta-phi)*math.cos(phi)*math.sin(zeta)])
    return xreal,ximag


def weights(w11r,w12r,w13r,w21r,w22r,w23r,w31r,w32r,w33r,w11i,w12i,w13i,w21i,w22i,w23i,w31i,w32i,w33i):
    global W
    W = theano.shared(np.asarray([w11r, w12r, w13r,w21r,w22r,w23r,w31r,w32r,w33r,w11i,w12i,w13i,w21i,w22i,w23i,w31i,w32i,w33i]), 'W')
    return W

xreal=T.dvector('xreal')
ximag=T.dvector('ximag')
#W1real=T.dvector('W1real')
#W2real=T.dvector('W2real')
#W3real=T.dvector('W3real')
#W1imag=T.dvector('W3imag')
#W2imag=T.dvector('W3imag')
#W3imag=T.dvector('W3imag')
#W=T.dvector('W')
#dot = T.dot(x,W)
#func=theano.function(inputs=[W,x],outputs=dot)
weights(np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1))

#W1real=T.set_subtensor(W[0:3],W[0:3])
#W2real=T.set_subtensor(W[3:6],W[3:6])
#W3real=T.set_subtensor(W[6:9],W[6:9])
#W1imag=T.set_subtensor(W[9:12],W[9:12])
#W2imag=T.set_subtensor(W[12:15],W[12:15])
#W3imag=T.set_subtensor(W[15:18],W[15:18])

cost=T.mean(T.sqr(T.sum(T.transpose(W[0:3])*xreal)-T.sum(T.transpose(W[9:12])*ximag))+T.sqr(T.sum(T.transpose(W[0:3])*ximag)+T.sum(T.transpose(W[9:12])*xreal))+l*(T.sqr(T.sum(T.transpose(W[3:6])*xreal)-T.sum(T.transpose(W[12:15])*ximag))+T.sqr(T.transpose(T.sum(W[3:6])*ximag)+T.sum(T.transpose(W[12:15])*xreal))+T.sqr(T.sum(T.transpose(W[6:9])*xreal)-T.sum(T.transpose(W[15:18])*ximag))+T.sqr(T.sum(T.transpose(W[6:9])*ximag)+T.sum(T.transpose(W[15:18])*xreal))-1))



gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.1 * gradients[0])
updates = [(W, W_updated)]
#updates1=[updates[0][0]]
#updates2=[updates[1][0]]
#updates3=[updates1,updates2]
train=theano.function(inputs=[xreal,ximag],outputs=cost,updates=updates,allow_input_downcast=True)
for i in range(100):
    inputvector(np.random.rand(1),np.random.rand(1),np.random.rand(1))
    train(xreal,ximag)
    print (W.get_value())



    
#a=[[1, 2,3], [4, 5,6],[7,8,9]]
#b=[[0.1, 0.2, 0.3]]
#print(func(a,b))


