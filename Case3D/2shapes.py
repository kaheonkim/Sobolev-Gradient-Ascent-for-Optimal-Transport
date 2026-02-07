import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from functions import *
from time import time
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (13, 8)
plt.rcParams['image.cmap'] = 'viridis'

# Grid size
# Grid dimensions
n = 200
n1, n2, n3 = n, n, n
lr = 5e-3
num_iter = 2000
option = 1

x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2),
                    np.linspace(0.5/n3, 1-0.5/n3, n3))

nu1, nu2, nu3 = np.zeros((n1,n2,n3)), np.zeros((n1,n2,n3)), np.zeros((n1,n2,n3))

nu1[(x-0.5)**2+(y-0.5)**2+(z-0.5)**2< 0.4**2] = 1
nu2[(abs(x-0.5)<0.2) & (abs(y-0.5)<0.2) & (abs(z-0.5)<0.2)] = 1
# Convert boolean masks to numerical arrays

nu1 *= n1 * n2 * n3 / np.sum(nu1)
nu2 *= n1 * n2 * n3 / np.sum(nu2)

train_option = 'sqrt'

output_dir = 'image/2shapes/option'+str(option)+'/' + train_option +'/' + str(lr) + '_' + str(num_iter) +'/' 
os.makedirs(output_dir, exist_ok=True) 

nu = [nu1,nu2]
if option == 1:
    weight = [1/2,1/2]
elif option == 2:
    weight = [1/3,2/3]
elif option == 3:
    weight = [2/3,1/3]

if train_option == 'sqrt':
    mu_sga = sga3D_sqrt(nu, weight, num_iter,  lr=lr, output_dir = output_dir, record_option = True, return_option = True)
elif train_option == 'wdha':
    mu_sga = frechet_mean_weighted(nu, weight, num_iter,output_dir = output_dir, init_lr = lr, plot_option = True,save_option = True, return_option = True)

