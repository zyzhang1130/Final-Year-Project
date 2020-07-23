
import numpy as np
from matplotlib import pyplot as plt
import math
import time


import numpy as np
import argparse


import shutil
import os
import time
import scipy.linalg as la

from scipy.sparse.linalg import eigsh

from copy import copy, deepcopy

def EVD(X):
    # s, U = np.linalg.eig(X)
    s, U = eigsh(X,k=100)
    idx = s.argsort()[::-1] # decreasing order
    return s[idx], U[:,idx]

X = deepcopy(testip)
#X -= X.mean(axis=0)
# X /= np.std(X,axis=0)
Sigma = X.T.dot(X) / (X.shape[0]-1)
t = time.time()
s, U = EVD(Sigma)
elapsed = time.time() - t
s, U = np.real(s), np.real(U)

plt.figure()
plt.plot(s[:16]/np.sum(s))
plt.title('Percentage of data variances captured by the first 15 principal directions')

k = 0
var = 0
tot_var = np.sum(s)
while var < 0.95:
    k += 1
    var = np.sum(s[:k])/ tot_var

print('k=',k)
print('captured variance=',var)
    

X=X.dot(U[:,0:k])

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

ax.quiver(0, 0, 0, U[0,0], U[1,0], U[2,0], length=s[0]*1, normalize=True)
ax.quiver(0, 0, 0, U[0,1], U[1,1], U[2,1], length=s[1]*1, normalize=True)

plt.show()


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

ax.quiver(0, 0, 0, U[0,0], U[1,0], U[2,0], length=s[0]*1, normalize=True)
ax.quiver(0, 0, 0, U[0,1], U[1,1], U[2,1], length=s[1]*1, normalize=True)

plt.show()