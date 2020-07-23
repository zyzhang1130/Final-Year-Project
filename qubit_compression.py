
import theano
from theano import tensor as T
from theano import printing
import numpy as np
import math 
import cmath 
global W
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import statistics
from sklearn.preprocessing import normalize

l=0.8

def weights(w11r,w12r,w21r,w22r,w11i,w12i,w21i,w22i):
    global W
    W = theano.shared(np.asarray([w11r, w12r,w21r,w22r,w11i,w12i,w21i,w22i]), 'W')
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
weights(np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1))

#W1real=T.set_subtensor(W[0:3],W[0:3])
#W2real=T.set_subtensor(W[3:6],W[3:6])
#W3real=T.set_subtensor(W[6:9],W[6:9])
#W1imag=T.set_subtensor(W[9:12],W[9:12])
#W2imag=T.set_subtensor(W[12:15],W[12:15])
#W3imag=T.set_subtensor(W[15:18],W[15:18])

cost=T.sqr(T.sum(T.transpose(W[0:2])*xreal)-T.sum(T.transpose(W[4:6])*ximag))+T.sqr(T.sum(T.transpose(W[0:2])*ximag)+T.sum(T.transpose(W[4:6])*xreal))+l*T.sqr(T.sqr(T.sum(T.transpose(W[2:4])*xreal)-T.sum(T.transpose(W[6:8])*ximag))+T.sqr(T.sum(T.transpose(W[2:4])*ximag)+T.sum(T.transpose(W[6:8])*xreal))-1)
loss=[]
gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.003* gradients[0])
updates = [(W, W_updated)]
#updates1=[updates[0][0]]
#updates2=[updates[1][0]]
#updates3=[updates1,updates2]
train=theano.function(inputs=[xreal,ximag],outputs=cost,updates=updates,allow_input_downcast=True)
for i in range(1500):    
    x=normalize(np.asarray([np.random.rand(4)]))
    ximag=x[0][0:2]
    xreal=x[0][2:4]
    loss.append(train(xreal,ximag))

epoch=list(range(len(loss)))
plt.figure()
plt.plot(epoch,loss)
plt.show()

a=W.get_value()
U = np.zeros((2,2), dtype=complex)
#Uinverse = np.linalg.inv(U) 
for i in range(2):
    for j in range(2):
        U[i,j]=complex(a[j+2*i],a[4+j+2*i])
        
Uinverse = np.linalg.inv(U) 
result=[]
norm=[]
initial=[]
ss=[]
for j in range(1000):       
        x=normalize(np.asarray([np.random.rand(4)]))
        ss.append(x[0][0])
        initial.append(x[0][0]*x[0][0]/(x[0][0]*x[0][0]+x[0][2]*x[0][2]))
        ximag=x[0][0:2]
        xreal=x[0][2:4]
        X = np.zeros((2,1), dtype=complex)
        for i in range(2):
            X[i,0]=complex(xreal[i],ximag[i])
        result.append(abs(np.dot(U,X))[0])
        norm.append(sum(abs(np.dot(U,X))*abs(np.dot(U,X))))
        

varnorm=np.var(norm)
meannorm=np.mean(norm)
print(varnorm)
print(meannorm)
plt.figure()
n, bins, patches = plt.hist(initial, 20,facecolor='blue', alpha=0.5)
plt.show()
ss=np.asarray(ss)
plt.figure()
n, bins, patches = plt.hist(ss, 20,facecolor='blue', alpha=0.5)
plt.show()
plt.figure()
result=np.asarray(result)
n, bins, patches = plt.hist(result, 20,facecolor='blue', alpha=0.5)
plt.show()
#a=[[1, 2,3], [4, 5,6],[7,8,9]]
#b=[[0.1, 0.2, 0.3]]
#print(func(a,b))


