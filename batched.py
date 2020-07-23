import theano
from theano import tensor as T
from theano import printing
import numpy as np
import math 
import cmath 
global W
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

l=1

def inputvector(theta1,phi1,zeta1,theta2,phi2,zeta2,theta3,phi3,zeta3):
    global xreal,ximag
    xreal=[np.asarray([math.cos(2*theta1-phi1)*math.cos(phi1), math.cos(2*theta1-phi1)*math.sin(phi1)*math.cos(zeta1), math.cos(2*theta1-phi1)*math.sin(phi1)*math.sin(zeta1)]),np.asarray([math.cos(2*theta2-phi2)*math.cos(phi2), math.cos(2*theta2-phi2)*math.sin(phi2)*math.cos(zeta2), math.cos(2*theta2-phi2)*math.sin(phi2)*math.sin(zeta2)]),np.asarray([math.cos(2*theta3-phi3)*math.cos(phi3), math.cos(2*theta3-phi3)*math.sin(phi3)*math.cos(zeta3), math.cos(2*theta3-phi3)*math.sin(phi3)*math.sin(zeta3)])]
    ximag=[np.asarray([-math.sin(2*theta1-phi1)*math.sin(phi1), math.sin(2*theta1-phi1)*math.cos(phi1)*math.cos(zeta1), math.sin(2*theta1-phi1)*math.cos(phi1)*math.sin(zeta1)]),np.asarray([-math.sin(2*theta2-phi2)*math.sin(phi2), math.sin(2*theta2-phi2)*math.cos(phi2)*math.cos(zeta2), math.sin(2*theta2-phi2)*math.cos(phi2)*math.sin(zeta2)]),np.asarray([-math.sin(2*theta3-phi3)*math.sin(phi3), math.sin(2*theta3-phi3)*math.cos(phi3)*math.cos(zeta3), math.sin(2*theta3-phi3)*math.cos(phi3)*math.sin(zeta3)])]
    return xreal,ximag

def testvector(theta,phi,zeta):
    global xreal,ximag
    xreal=np.asarray([math.cos(2*theta-phi)*math.cos(phi), math.cos(2*theta-phi)*math.sin(phi)*math.cos(zeta), math.cos(2*theta-phi)*math.sin(phi)*math.sin(zeta)])
    ximag=np.asarray([-math.sin(2*theta-phi)*math.sin(phi), math.sin(2*theta-phi)*math.cos(phi)*math.cos(zeta), math.sin(2*theta-phi)*math.cos(phi)*math.sin(zeta)])
    return xreal,ximag


def weights(w11r,w12r,w13r,w21r,w22r,w23r,w31r,w32r,w33r,w11i,w12i,w13i,w21i,w22i,w23i,w31i,w32i,w33i):
    global W
    W = theano.shared(np.asarray([w11r, w12r, w13r,w21r,w22r,w23r,w31r,w32r,w33r,w11i,w12i,w13i,w21i,w22i,w23i,w31i,w32i,w33i]), 'W')
    return W

xreal=T.dmatrix('xreal')
ximag=T.dmatrix('ximag')
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

#cost=T.mean(T.sqr(T.sum(T.dot(xreal,W[0:3]))-T.sum(T.dot(ximag,W[9:12])))+T.sqr(T.sum(T.dot(ximag,W[0:3]))+T.sum(T.dot(xreal,W[9:12])))+l*T.sqr(T.sqr(T.sum(T.dot(xreal,W[3:6]))-T.sum(T.dot(ximag,W[12:15])))+T.sqr(T.sum(T.dot(ximag,W[3:6]))+T.sum(T.dot(xreal,W[12:15])))+T.sqr(T.sum(T.dot(xreal,W[6:9]))-T.sum(T.dot(ximag,W[15:18])))+T.sqr(T.sum(T.dot(ximag,W[6:9]))+T.sum(T.dot(xreal,W[15:18])))-1))
#cost=T.mean(T.sqr(T.sum(T.dot(T.transpose(W[0:3]),xreal))-T.sum(T.dot(T.transpose(W[9:12]),ximag))))
cost=T.mean(T.sqr(T.sum(T.dot(T.transpose(W[0:3]),xreal))-T.sum(T.dot(T.transpose(W[9:12]),ximag)))+T.sqr(T.sum(T.dot(T.transpose(W[0:3]),ximag))+T.sum(T.dot(T.transpose(W[9:12]),xreal)))+l*T.sqr(T.sqr(T.sum(T.dot(T.transpose(W[3:6]),xreal))-T.sum(T.dot(T.transpose(W[12:15]),ximag)))+T.sqr(T.sum(T.dot(T.transpose(W[3:6]),ximag))+T.sum(T.dot(T.transpose(W[12:15]),xreal)))+T.sqr(T.sum(T.dot(T.transpose(W[6:9]),xreal))-T.sum(T.dot(T.transpose(W[15:18]),ximag)))+T.sqr(T.sum(T.dot(T.transpose(W[6:9]),ximag))+T.sum(T.dot(T.transpose(W[15:18]),xreal)))-1))


loss=[]
gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.006 * gradients[0])
updates = [(W, W_updated)]
#updates1=[updates[0][0]]
#updates2=[updates[1][0]]
#updates3=[updates1,updates2]
train=theano.function(inputs=[xreal,ximag],outputs=cost,updates=updates,allow_input_downcast=True)
for i in range(600):
    inputvector(np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1))
    loss.append(train(xreal,ximag))
    a=W.get_value()
    print(a)

epoch=list(range(len(loss)))
plt.figure()
plt.plot(epoch,loss)
plt.show()

a=W.get_value()
U = np.zeros((3,3), dtype=complex)

for i in range(3):
    for j in range(3):
        U[i,j]=complex(a[j+3*i],a[9+j+3*i])
        
Uinverse = np.linalg.inv(U) 
result=[]
norm=[]
for j in range(100):       
        testvector(np.random.rand(1),np.random.rand(1),np.random.rand(1))
        X = np.zeros((3,1), dtype=complex)
        for i in range(3):
            X[i,0]=complex(xreal[i],ximag[i])
        result.append(abs(np.dot(U,X))[0])
        norm.append(sum(abs(np.dot(U,X))*abs(np.dot(U,X))))
num_bins = 5
plt.figure()
n, bins, patches = plt.hist(result, num_bins, facecolor='blue', alpha=0.5)
plt.show()



#a=[[1, 2,3], [4, 5,6],[7,8,9]]
#b=[[0.1, 0.2, 0.3]]
#print(func(a,b))


