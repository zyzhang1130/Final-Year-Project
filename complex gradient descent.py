import theano
from theano import tensor as T
import numpy as np
import math 
import cmath 
global W1,W2,W1real,W2real,W3real,W1imag,W2imag,W3imag,xreal,ximag
l=1

def inputvector(theta,phi,zeta):
    global xreal,ximag
    xreal=[math.cos(2*theta-phi)*math.cos(phi), math.cos(2*theta-phi)*math.sin(phi)*math.cos(zeta), math.cos(2*theta-phi)*math.sin(phi)*math.sin(zeta)]
    ximag=[-math.sin(2*theta-phi)*math.sin(phi), math.sin(2*theta-phi)*math.cos(phi)*math.cos(zeta), math.sin(2*theta-phi)*math.cos(phi)*math.sin(zeta)]   
    return xreal,ximag


def weights(w11r,w12r,w13r,w21r,w22r,w23r,w31r,w32r,w33r,w11i,w12i,w13i,w21i,w22i,w23i,w31i,w32i,w33i):
    global W1,W2,W1real,W2real,W3real,W1imag,W2imag,W3imag
    W1real = theano.shared(np.asarray([w11r, w12r, w13r]), 'W1real')
    W2real = theano.shared(np.asarray([w21r, w22r, w23r]), 'W2real')
    W3real = theano.shared(np.asarray([w31r, w32r, w33r]), 'W3real')
    
    W1imag = theano.shared(np.asarray([w11i, w12i, w13i]), 'W1imag')
    W2imag = theano.shared(np.asarray([w21i, w22i, w23i]), 'W2imag')
    W3imag = theano.shared(np.asarray([w31i, w32i, w33i]), 'W3imag')
    W1=[W1real,W1imag]
    W2=[W2real,W2imag]
    W3=[W3real,W3imag]
    W=[W1,W2,W3]
    return W1,W2,W1real,W2real,W3real,W1imag,W2imag,W3imag

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
W=[W1real,W2real,W3real,W1imag,W2imag,W3imag]
cost=T.mean(T.sqr(W1real*xreal-W1imag*ximag)+T.sqr(W1real*ximag+W1imag*xreal)+l*(T.sqr(W2real*xreal-W2imag*ximag)+T.sqr(W2real*ximag+W2imag*xreal)+T.sqr(W3real*xreal-W3imag*ximag)+T.sqr(W3real*ximag+W3imag*xreal)-1))
gradient=T.grad(cost,W)
W_updated=[0,0,0,0,0,0]
for i in range(6):
    W_updated[i] = W[i] - (0.1 * gradient[i])
updates=[(W,W_updated)]


#updates1=[updates[0][0]]
#updates2=[updates[1][0]]
#updates3=[updates1,updates2]
train=theano.function(inputs=[xreal,ximag],outputs=cost,updates=updates,allow_input_downcast=True)
for i in range(100):
    inputvector(np.random.rand(1),np.random.rand(1),np.random.rand(1))
    train(xreal,ximag)



    
#a=[[1, 2,3], [4, 5,6],[7,8,9]]
#b=[[0.1, 0.2, 0.3]]
#print(func(a,b))


