# John Eargle
# 2017


import psiko as pk

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,1,500)
omega = 2
y = np.array([pk.square_comp(x, omega, k) for k in range(1,6)])
for i in range(5):
    plt.plot(x,[y[:i,j].sum() for j in range(500)])
plt.show()


omega=2
x = np.linspace(0,1,500)
ns = [1, 2, 8, 32, 128, 512]
y = np.array([square(x, omega, i) for i in ns])
print('np.shape(y):', np.shape(y))
y2 = np.array([square2(x, omega, i) for i in ns])
print('np.shape(y2):', np.shape(y2))
print('np.abs(y-y2).max():', np.abs(y-y2).max())
print('np.abs(y-y2).sum():', np.abs(y-y2).sum())
for i in y:
    plt.plot(x,i)
plt.show()
