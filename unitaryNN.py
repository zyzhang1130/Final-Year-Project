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

l=2
l2=1

def inputvector(theta,phi,zeta):
    global xreal,ximag
    xreal=np.asarray([math.cos(2*theta-phi)*math.cos(phi), math.cos(2*theta-phi)*math.sin(phi)*math.cos(zeta), math.cos(2*theta-phi)*math.sin(phi)*math.sin(zeta)])
    ximag=np.asarray([-math.sin(2*theta-phi)*math.sin(phi), math.sin(2*theta-phi)*math.cos(phi)*math.cos(zeta), math.sin(2*theta-phi)*math.cos(phi)*math.sin(zeta)])
    return xreal,ximag

def inputvector3(theta,phi):
    global xreal,ximag
    xreal=np.asarray([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi),math.cos(theta)])
    ximag=np.asarray([0, 0, 0])
    return xreal,ximag


def weights(w11r,w12r,w13r,w21r,w22r,w23r,w31r,w32r,w33r,w11i,w12i,w13i,w21i,w22i,w23i,w31i,w32i,w33i):
    global W
    W = theano.shared(np.asarray([w11r, w21r, w31r,w12r,w22r,w32r,w13r,w23r,w33r,w11i,w21i,w31i,w12i,w22i,w32i,w13i,w23i,w33i]), 'W')
    return W


#def weights(w11r,w12r,w13r,w21r,w22r,w23r,w31r,w32r,w33r,w11i,w12i,w13i,w21i,w22i,w23i,w31i,w32i,w33i):
#    global W
#    W = theano.shared(np.asarray([w11r, w12r, w13r,w21r,w22r,w23r,w31r,w32r,w33r,w11i,w12i,w13i,w21i,w22i,w23i,w31i,w32i,w33i]), 'W')
#    return W

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

#cost=l*T.sqr(T.sum(T.transpose(W[0:3])*xreal)-T.sum(T.transpose(W[9:12])*ximag))+T.sqr(T.sum(T.transpose(W[0:3])*ximag)+T.sum(T.transpose(W[9:12])*xreal))+T.sqr(T.sum(W[0]*W[1])+T.sum(W[3]*W[4])+T.sum(W[6]*W[7])+T.sum(W[9]*W[10])+T.sum(W[12]*W[13])+T.sum(W[15]*W[16]))+T.sqr(T.sum(W[0]*W[10])-T.sum(W[9]*W[1])+T.sum(W[3]*W[13])-T.sum(W[4]*W[12])+T.sum(W[6]*W[16])-T.sum(W[15]*W[7]))+T.sqr(T.sum(W[0]*W[2])+T.sum(W[3]*W[5])+T.sum(W[6]*W[8])+T.sum(W[9]*W[11])+T.sum(W[12]*W[14])+T.sum(W[15]*W[17]))+T.sqr(T.sum(W[0]*W[11])-T.sum(W[9]*W[2])+T.sum(W[3]*W[14])-T.sum(W[5]*W[12])+T.sum(W[6]*W[17])-T.sum(W[15]*W[8]))+T.sqr(T.sum(W[1]*W[2])+T.sum(W[4]*W[5])+T.sum(W[7]*W[8])+T.sum(W[10]*W[11])+T.sum(W[13]*W[14])+T.sum(W[16]*W[17]))+T.sqr(T.sum(W[1]*W[11])-T.sum(W[10]*W[2])+T.sum(W[4]*W[14])-T.sum(W[5]*W[13])+T.sum(W[7]*W[17])-T.sum(W[16]*W[8]))+l2*(T.sqr(T.sum(T.sqr(W[0])+T.sqr(W[3])+T.sqr(W[6])+T.sqr(W[9])+T.sqr(W[12])+T.sqr(W[15]))-1)+T.sqr(T.sum(T.sqr(W[1])+T.sqr(W[4])+T.sqr(W[7])+T.sqr(W[10])+T.sqr(W[13])+T.sqr(W[16]))-1)+T.sqr(T.sum(T.sqr(W[2])+T.sqr(W[5])+T.sqr(W[8])+T.sqr(W[11])+T.sqr(W[14])+T.sqr(W[17]))-1))#normalize second column

#+T.sqr(T.sum(T.sum(W[0]*W[1]),T.sum(W[3]*W[4]),T.sum(W[6]*W[7]),T.sum(-W[9]*W[10]),T.sum(-W[12]*W[13]),T.sum(-W[15]*W[16])))#minimize the inner product of first and second column real part


cost=10*T.sqr(T.sum(W[0]*xreal[0])+T.sum(W[3]*xreal[1])+T.sum(W[6]*xreal[2])-T.sum(W[9]*ximag[0])-T.sum(W[12]*ximag[1])-T.sum(W[15]*ximag[2]))+T.sqr(T.sum(W[0]*ximag[0])+T.sum(W[3]*ximag[1])+T.sum(W[6]*ximag[2])+T.sum(W[9]*xreal[0])+T.sum(W[12]*xreal[1])+T.sum(W[15]*xreal[2]))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[3:6]))+T.sum(T.transpose(W[9:12])*T.transpose(W[12:15])))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[12:15]))-T.sum(T.transpose(W[3:6])*T.transpose(W[9:12])))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[6:9]))+T.sum(T.transpose(W[9:12])*T.transpose(W[15:18])))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[15:18]))-T.sum(T.transpose(W[6:9])*T.transpose(W[9:12])))+T.sqr(T.sum(T.transpose(W[3:6])*T.transpose(W[6:9]))+T.sum(T.transpose(W[12:15])*T.transpose(W[15:18])))+T.sqr(T.sum(T.transpose(W[3:6])*T.transpose(W[15:18]))-T.sum(T.transpose(W[6:9])*T.transpose(W[12:15])))+T.sqr(T.sum(T.transpose(W[0:3])*T.transpose(W[0:3]))+T.sum(T.transpose(W[9:12])*T.transpose(W[9:12]))-1)+T.sqr(T.sum(T.transpose(W[3:6])*T.transpose(W[3:6]))+T.sum(T.transpose(W[12:15])*T.transpose(W[12:15]))-1)+T.sqr(T.sum(T.transpose(W[6:9])*T.transpose(W[6:9]))+T.sum(T.transpose(W[15:18])*T.transpose(W[15:18]))-1)

loss=[]
gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.05 * gradients[0])
updates = [(W, W_updated)]
#updates1=[updates[0][0]]
#updates2=[updates[1][0]]
#updates3=[updates1,updates2]
train=theano.function(inputs=[xreal,ximag],outputs=cost,updates=updates,allow_input_downcast=True)
ip=[]
for i in range(1000):
    inputvector(10*np.random.rand(1),10*np.random.rand(1),10*np.random.rand(1))
#    inputvector3(10*np.random.rand(1),10*np.random.rand(1))
#    inputvector2(10*np.random.rand(1),10*np.random.rand(1))
    ip.append(np.sqrt(xreal*xreal+ximag*ximag))
    loss.append(train(xreal,ximag))

epoch=list(range(len(loss)))
plt.figure()
plt.plot(epoch,loss)
plt.show()

a=W.get_value()
U = np.zeros((3,3), dtype=complex)
#Uinverse = np.linalg.inv(U) 
for i in range(3):
    for j in range(3):
        U[i,j]=complex(a[j+3*i],a[9+j+3*i])
        
Uinverse = np.linalg.inv(U) 
result=[]
norm=[]
finalvector=[]
abso=[]
ip2=[]
for j in range(100):       
        inputvector(10*np.random.rand(1),10*np.random.rand(1),10*np.random.rand(1))
#        inputvector3(10*np.random.rand(1),10*np.random.rand(1))
        ip2.append(np.sqrt(xreal*xreal+ximag*ximag))
        X = np.zeros((3,1), dtype=complex)
        for i in range(3):
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
colors = ['red', 'tan', 'lime']
plt.hist(ip, num_bins, histtype='bar', color=colors, label=['v1', 'v2', 'v3'])
plt.legend(prop={'size': 10})
plt.show()
#a=[[1, 2,3], [4, 5,6],[7,8,9]]
#b=[[0.1, 0.2, 0.3]]
#print(func(a,b))
ip2=np.asarray(ip2)
plt.figure()
colors = ['red', 'tan', 'lime']
plt.hist(ip2, num_bins, histtype='bar', color=colors, label=['v1', 'v2', 'v3'])
plt.legend(prop={'size': 10})
plt.show()

abso=np.asarray(abso)
plt.figure()
colors = ['red', 'tan', 'lime']
plt.hist(abso, num_bins, histtype='bar', color=colors, label=['v1', 'v2', 'v3'])
plt.legend(prop={'size': 10})
plt.show()
