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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

l=1
l2=1
l3=1
l4=1

#def inputvector(theta,phi,zeta):
#    global xreal,ximag
#    xreal=np.asarray([math.cos(2*theta-phi)*math.cos(phi), math.cos(2*theta-phi)*math.sin(phi)*math.cos(zeta), math.cos(2*theta-phi)*math.sin(phi)*math.sin(zeta)])
#    ximag=np.asarray([-math.sin(2*theta-phi)*math.sin(phi), math.sin(2*theta-phi)*math.cos(phi)*math.cos(zeta), math.sin(2*theta-phi)*math.cos(phi)*math.sin(zeta)])
#    return xreal,ximag

def inputvector(theta,phi):
    global xreal
    xreal=np.asarray([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi),math.cos(theta)])
    return xreal

def inputvector2(theta,theta1,theta2,theta3):
    global xreal
    xreal=np.asarray([ [math.cos(theta)], [math.sin(theta)],[0]])
    Rx = [[1, 0, 0], [0, math.cos(theta1), -math.sin(theta1)],[0,math.sin(theta1),math.cos(theta1)]]
    Ry = [[math.cos(theta2), 0, math.sin(theta2)],[0, 1, 0],[-math.sin(theta2),0,math.cos(theta2)]]
    Rz = [[math.cos(theta3), -math.sin(theta3),0],[math.sin(theta3),math.cos(theta3),0],[ 0, 0,1]]
    xreal=np.matmul(Rx,xreal)
    xreal=np.matmul(Ry,xreal)
    xreal=np.matmul(Rz,xreal)
    xreal=np.squeeze(xreal)
    return xreal



def weights(w11r,w12r,w13r,w21r,w22r,w23r,w31r,w32r,w33r):
    global W
    W = theano.shared(np.asarray([w11r, w12r, w13r,w21r,w22r,w23r,w31r,w32r,w33r]), 'W')
    return W

xreal=T.dvector('xreal')
#W1real=T.dvector('W1real')
#W2real=T.dvector('W2real')
#W3real=T.dvector('W3real')
#W1imag=T.dvector('W3imag')
#W2imag=T.dvector('W3imag')
#W3imag=T.dvector('W3imag')
#W=T.dvector('W')
#dot = T.dot(x,W)
#func=theano.function(inputs=[W,x],outputs=dot)
weights(np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1))

#W1real=T.set_subtensor(W[0:3],W[0:3])
#W2real=T.set_subtensor(W[3:6],W[3:6])
#W3real=T.set_subtensor(W[6:9],W[6:9])
#W1imag=T.set_subtensor(W[9:12],W[9:12])
#W2imag=T.set_subtensor(W[12:15],W[12:15])
#W3imag=T.set_subtensor(W[15:18],W[15:18])

#cost=T.sqr(T.sum(T.transpose(W[0:3])*xreal))+T.sqr(T.sum(W[0]*W[1])+T.sum(W[3]*W[4])+T.sum(W[6]*W[7]))+l2*T.sqr(T.sum(W[0]*W[2])+T.sum(W[3]*W[5])+T.sum(W[6]*W[8]))+l3*T.sqr(T.sum(W[2]*W[1])+T.sum(W[5]*W[4])+T.sum(W[8]*W[7]))+l4*T.sqr(T.sum(T.sqr(W[2])+T.sqr(W[5])+T.sqr(W[8]))-1)+T.sqr(T.sum(T.sqr(W[0])+T.sqr(W[3])+T.sqr(W[6]))-1)+T.sqr(T.sum(T.sqr(W[1])+T.sqr(W[4])+T.sqr(W[7]))-1)
cost=T.sqr(T.sum(T.transpose(W[0:3])*xreal))+l4*T.sqr(T.sum(T.sqr(W[2])+T.sqr(W[5])+T.sqr(W[8]))-1)+T.sqr(T.sum(T.sqr(W[0])+T.sqr(W[3])+T.sqr(W[6]))-1)+T.sqr(T.sum(T.sqr(W[1])+T.sqr(W[4])+T.sqr(W[7]))-1)

normip=[]
loss=[]
gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.01 * gradients[0])
updates = [(W, W_updated)]
#updates1=[updates[0][0]]
#updates2=[updates[1][0]]
#updates3=[updates1,updates2]
train=theano.function(inputs=[xreal],outputs=cost,updates=updates,allow_input_downcast=True)
ip=[]
for i in range(2000):
    #inputvector(10*np.random.rand(1),0.7)
#    inputvector(10*np.random.rand(1),0.7)
    inputvector2(10*np.random.rand(1),0.1,0.2,0.3)
    normip.append(sum(abs(xreal)*abs(xreal)))
    ip.append(xreal)
    loss.append(train(xreal))

epoch=list(range(len(loss)))
plt.figure()
plt.plot(epoch,loss)
plt.show()

a=W.get_value()
U = np.zeros((3,3))
#Uinverse = np.linalg.inv(U) 
for i in range(3):
    for j in range(3):
        U[i,j]=a[j+3*i]
        
Uinverse = np.linalg.inv(U) 
result=[]
norm=[]
finalvector=[]
testip=[]
normtest=[]
for j in range(100):       
#        inputvector(3,10*np.random.rand(1))
        inputvector2(10*np.random.rand(1),0.1,0.2,0.3)
        X = np.zeros((3,1))
        testip.append(xreal)
        normtest.append(sum(abs(xreal)*abs(xreal)))
        for i in range(3):
            X[i,0]=xreal[i]
        result.append(abs(np.matmul(U,X))[0])
        finalvector.append(np.matmul(U,X))
        norm.append(sum(abs(np.matmul(U,X))*abs(np.dot(U,X))))
        

varnorm=np.var(norm)
meannorm=np.mean(norm)
print('norm variance:',varnorm)
print('norm mean:',meannorm)
result=np.asarray(result)
num_bins = 20
plt.figure()
n, bins, patches = plt.hist(result, num_bins, facecolor='blue', alpha=0.5)
plt.show()


ip=np.asarray(ip)
testip=np.asarray(testip)
plt.figure()
colors = ['red', 'tan', 'lime']
plt.hist(ip, num_bins, histtype='bar', color=colors, label=['v1', 'v2', 'v3'])
plt.legend(prop={'size': 10})
plt.show()
#a=[[1, 2,3], [4, 5,6],[7,8,9]]
#b=[[0.1, 0.2, 0.3]]
#print(func(a,b))
x=testip[:,0]
y=testip[:,1]
z=testip[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o', )

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
ax.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True)
ax.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True)
ax.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True)
plt.show()

finalvector=np.squeeze(finalvector)
finalvector=np.asarray(finalvector)
x=finalvector[:,0]
y=finalvector[:,1]
z=finalvector[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
ax.quiver(0, 0, 0, U[0,0], U[1,0], U[2,0], length=sum(U[:,0]**2), normalize=True)
ax.quiver(0, 0, 0, U[0,1], U[1,1], U[2,1], length=sum(U[:,1]**2), normalize=True)
ax.quiver(0, 0, 0, U[0,2], U[1,2], U[2,2], length=sum(U[:,2]**2), normalize=True)


plt.show()
