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

l=1
l2=1
l3=0

def inputvector(theta,phi,zeta):
    global xreal,ximag
    xreal=np.asarray([math.cos(2*theta-phi)*math.cos(phi), math.cos(2*theta-phi)*math.sin(phi)*math.cos(zeta), math.cos(2*theta-phi)*math.sin(phi)*math.sin(zeta)])
    ximag=np.asarray([-math.sin(2*theta-phi)*math.sin(phi), math.sin(2*theta-phi)*math.cos(phi)*math.cos(zeta), math.sin(2*theta-phi)*math.cos(phi)*math.sin(zeta)])
    return xreal,ximag

def inputvector3(theta):
    global xreal,ximag
    xreal=np.asarray([math.sin(theta), math.cos(theta)])
    ximag=np.asarray([0, 0])
    return xreal,ximag


#def weights(w11r,w12r,w21r,w22r,w11i,w12i,w21i,w22i):
#    global W
#    W = theano.shared(np.asarray([w11r, w21r,w12r,w22r,w11i,w21i,w12i,w22i]), 'W')
#    return W


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

#cost=T.sqr(T.sum(W[0]*W[1])+T.sum(W[4]*W[5])+T.sum(W[2]*W[3])+T.sum(W[6]*W[7]))+T.sqr(T.sum(W[0]*W[5])-T.sum(W[4]*W[1])+T.sum(W[2]*W[7])-T.sum(W[3]*W[6]))+l2*(T.sqr(T.sum(T.sqr(W[0])+T.sqr(W[2])+T.sqr(W[4])+T.sqr(W[6]))-1)+T.sqr(T.sum(T.sqr(W[1])+T.sqr(W[3])+T.sqr(W[7])+T.sqr(W[5]))-1))
cost=l*(T.sqr(T.sum(T.transpose(W[0:2])*xreal)-T.sum(T.transpose(W[4:6])*ximag))+T.sqr(T.sum(T.transpose(W[0:2])*ximag)+T.sum(T.transpose(W[4:6])*xreal)))+l2*(T.sqr(T.sum(W[0]*W[1])+T.sum(W[4]*W[5])+T.sum(W[2]*W[3])+T.sum(W[6]*W[7]))+T.sqr(T.sum(W[0]*W[5])-T.sum(W[4]*W[1])+T.sum(W[2]*W[7])-T.sum(W[3]*W[6])))+l3*(T.sqr(T.sum(T.sqr(W[0])+T.sqr(W[2])+T.sqr(W[4])+T.sqr(W[6]))-1)+T.sqr(T.sum(T.sqr(W[1])+T.sqr(W[3])+T.sqr(W[7])+T.sqr(W[5]))-1))
#cost=l*(T.sqr(T.sum(T.transpose(W[0:2])*xreal)-T.sum(T.transpose(W[4:6])*ximag))+T.sqr(T.sum(T.transpose(W[0:2])*ximag)+T.sum(T.transpose(W[4:6])*xreal)))+T.sqr(T.sum(W[0]*W[1])+T.sum(W[4]*W[5])+T.sum(W[2]*W[3])+T.sum(W[6]*W[7]))+T.sqr(T.sum(W[0]*W[5])-T.sum(W[4]*W[1])+T.sum(W[2]*W[7])-T.sum(W[3]*W[6]))+l2*(T.sqr(T.sum(T.sqr(W[0])+T.sqr(W[2])+T.sqr(W[4])+T.sqr(W[6]))-1)+T.sqr(T.sum(T.sqr(W[1])+T.sqr(W[3])+T.sqr(W[7])+T.sqr(W[5]))-1))
#+T.sqr(T.sum(T.sum(W[0]*W[1]),T.sum(W[3]*W[4]),T.sum(W[6]*W[7]),T.sum(-W[9]*W[10]),T.sum(-W[12]*W[13]),T.sum(-W[15]*W[16])))#minimize the inner product of first and second column real part


#cost=10*T.sqr(T.sum(W[0]*xreal[0])+T.sum(W[3]*xreal[1])+T.sum(W[6]*xreal[2])-T.sum(W[9]*ximag[0])-T.sum(W[12]*ximag[1])-T.sum(W[15]*ximag[2]))+T.sqr(T.sum(W[0]*ximag[0])+T.sum(W[3]*ximag[1])+T.sum(W[6]*ximag[2])+T.sum(W[9]*xreal[0])+T.sum(W[12]*xreal[1])+T.sum(W[15]*xreal[2]))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[3:6]))+T.sum(T.transpose(W[9:12])*T.transpose(W[12:15])))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[12:15]))-T.sum(T.transpose(W[3:6])*T.transpose(W[9:12])))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[6:9]))+T.sum(T.transpose(W[9:12])*T.transpose(W[15:18])))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[15:18]))-T.sum(T.transpose(W[6:9])*T.transpose(W[9:12])))+T.sqr(T.sum(T.transpose(W[3:6])*T.transpose(W[6:9]))+T.sum(T.transpose(W[12:15])*T.transpose(W[15:18])))+T.sqr(T.sum(T.transpose(W[3:6])*T.transpose(W[15:18]))-T.sum(T.transpose(W[6:9])*T.transpose(W[12:15])))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[0:3]))+T.sum(T.transpose(W[9:12])*T.transpose(W[9:12]))-1)+T.sqr(T.sum(T.transpose(W[3:6])*T.transpose(W[3:6]))+T.sum(T.transpose(W[12:15])*T.transpose(W[12:15]))-1)+T.sqr(T.sum(T.transpose(W[6:9])*T.transpose(W[6:9]))+T.sum(T.transpose(W[15:18])*T.transpose(W[15:18]))-1)

loss=[]
gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.05 * gradients[0])
updates = [(W, W_updated)]
#updates1=[updates[0][0]]
#updates2=[updates[1][0]]
#updates3=[updates1,updates2]
train=theano.function(inputs=[xreal,ximag],outputs=cost,updates=updates,allow_input_downcast=True,on_unused_input='warn')
ip=[]
for i in range(1000):
#    inputvector(10*np.random.rand(1),10*np.random.rand(1),10*np.random.rand(1))
    inputvector3(10*np.random.rand(1))
#    inputvector2(10*np.random.rand(1),10*np.random.rand(1))
#    ip.append(np.sqrt(xreal*xreal+ximag*ximag))
    ip.append(np.array([complex(xreal[0],ximag[0]),complex(xreal[1],ximag[1])]))
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
finalvector=[]
abso=[]
ip2=[]
for j in range(100):       
#        inputvector(10*np.random.rand(1),10*np.random.rand(1),10*np.random.rand(1))
        inputvector3(10*np.random.rand(1))
#        ip2.append(np.sqrt(xreal*xreal+ximag*ximag))
        ip2.append(np.array([complex(xreal[0],ximag[0]),complex(xreal[1],ximag[1])]))
        
        X = np.zeros((2,1), dtype=complex)
        for i in range(2):
            X[i,0]=complex(xreal[i],ximag[i])
        result.append(abs(np.dot(U,X))[0]) 
        finalvector.append(np.dot(U,X))
        abso.append(abs(np.dot(U,X)).T[0])
        norm.append(sum(abs(np.dot(U,X))*abs(np.dot(U,X))))
        

result=np.asarray(result)
varnorm=np.var(norm)
meannorm=np.mean(norm)
print(varnorm)
print(meannorm)
num_bins = 20
plt.figure()
n, bins, patches = plt.hist(result, num_bins, facecolor='blue', alpha=0.5)
plt.show()

ip=np.asarray(ip)
plt.figure()
colors = ['red', 'tan']
plt.hist(ip, num_bins, histtype='bar', color=colors, label=['v1', 'v2'])
plt.legend(prop={'size': 10})
plt.show()
#a=[[1, 2,3], [4, 5,6],[7,8,9]]
#b=[[0.1, 0.2, 0.3]]
#print(func(a,b))
ip2=np.asarray(ip2)
plt.figure()
colors = ['red', 'tan']
plt.hist(ip2, num_bins, histtype='bar', color=colors, label=['v1', 'v2'])
plt.legend(prop={'size': 10})
plt.show()

abso=np.asarray(abso)
plt.figure()
colors = ['red', 'tan']
plt.hist(abso, num_bins, histtype='bar', color=colors, label=['v1', 'v2'])
plt.legend(prop={'size': 10})
plt.show()
U=np.matrix(U)

#(U[0,0].real*U[0,1].real+U[0,0].imag*U[0,1].imag+U[1,0].real*U[1,1].real+U[1,0].imag*U[1,1].imag)**2