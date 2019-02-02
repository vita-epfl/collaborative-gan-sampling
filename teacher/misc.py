from utils import random_n_sphere

# from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

######### loss function ##########

# y = np.arange(1e-3,1.0-1e-3,1e-3)

# loss_ns = -np.log(y)
# loss_mm = np.log(1-y)

# der_ns = -1/y
# der_mm = -1/(1-y)

# plt.semilogy(1-y,abs(der_mm),'r-.',label='norm minimax')
# plt.semilogy(1-y,abs(der_ns),'b-.',label='norm non-saturating')
# plt.xlabel('$loss\' = 1- y = 1- D(G(z))$')
# plt.ylabel('$\Delta L_G$')
# plt.grid()
# plt.legend()
# plt.show()

# plt.plot(y,loss_mm,'r',label='loss minimax')
# plt.plot(y,loss_ns,'b',label='loss non-saturating')

# plt.plot(y,der_mm,'r-.',label='grad minimax')
# plt.plot(y,der_ns,'b-.',label='grad non-saturating')

# plt.xlabel('$y = D(x) = D(G(z))$')
# plt.ylabel('$L_G$')

# plt.xlim([0,1])
# plt.ylim([-10,10])
# plt.grid()
# plt.legend()
# plt.show()

######### n sphere ##########

points = random_n_sphere(1000,2)
plt.plot(points[:,0],points[:,1],'o')
plt.grid()
plt.axis('equal')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()
print(points.shape)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# points = random_n_sphere(10000,3)
# ax.scatter(points[:,0], points[:,1], points[:,2])
# ax.set_aspect('equal')
# plt.show()