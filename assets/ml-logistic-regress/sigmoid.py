

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

x1 = x2 = np.linspace(-2, 2, 100)
xx1, xx2 = np.meshgrid(x1, x2)
Z = 1 / (1 + np.exp(-(xx1 + xx2)))

surf = ax.plot_surface(xx1, xx2, Z, cmap=cm.coolwarm, antialiased=False)
surf.set_clim([0, 1]) # <- this is the important line

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('P(C=red)')
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_zticks(np.arange(0, 1.2, .2))
cbar = fig.colorbar(surf, ticks=np.arange(0, 1.2, .2))
cbar.set_ticks(np.arange(0, 1.2, .2))
