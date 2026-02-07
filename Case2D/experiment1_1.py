import numpy as np
import matplotlib.pyplot as plt

from time import time
from functions import *
from w2 import BFM
from metric import *
from scipy.ndimage import gaussian_filter

plt.rcParams['figure.figsize'] = (13, 8)
plt.rcParams['image.cmap'] = 'viridis'


n1, n2 = 1024, 1024
x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2))
func1 = 1/2 * (x**2 + y**2)

mu1,mu2,mu3,mu4 = np.zeros((n2,n1)), np.zeros((n2,n1)), np.zeros((n2,n1)), np.zeros((n2,n1))
mask1 = (((x-0.3)**2+(y-0.3)**2)<0.15**2) | (((x-0.7)**2+(y-0.7)**2)<0.15**2)
mu1[mask1] = 1
mu1 = gaussian_filter(mu1, sigma=50)
mu1[~mask1] = 0
mu1 *= n1*n2/np.sum(mu1)

mask2 = (1*(x-0.5)**2-(0.4-(y-0.5))**3*(0.4+(y-0.5)) < 0)
mu2[mask2] = 1
mu2 = gaussian_filter(mu2, sigma=50)
mu2[~mask2] = 0
mu2 *= n1*n2/np.sum(mu2)

mask3 = ((x-0.5)**2+(y-0.5)**2>0.2**2)&((x-0.5)**2+(y-0.5)**2<0.4**2)
mu3[mask3] = 1
mu3 = gaussian_filter(mu3, sigma=50)
mu3[~mask3] = 0
mu3 *= n1*n2/np.sum(mu3)

r,p = 0.35, 0.4                    
theta = np.arctan2(y - 0.5, x - 0.5)
r_grid = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
mu4[r_grid < r * np.abs(np.cos(2 * (theta - np.pi / 4)))**p] = 1
mu4 = gaussian_filter(mu4, sigma=50)
mu4[~(r_grid < r * np.abs(np.cos(2 * (theta - np.pi / 4)))**p)] = 0
mu4 *= n1*n2/np.sum(mu4)

mu = [mu1,mu2,mu3,mu4]

weight = [1/4,1/4,1/4,1/4]


sga_sqrt(mu, weight, 300, 0.5, 'experiment1/1024/sga05/weight_equal', scheme = 'parallel', record_option=True, save_option = True, return_option = False)
sga_sqrt(mu, weight, 300, 0.5, 'experiment1/1024/sga05/weight_equal', scheme = 'sequential', record_option=True, save_option = True, return_option = False)
sga_sqrt(mu, weight, 300, 0.5, 'experiment1/1024/sga05/weight_equal', scheme = 'random', record_option=True,save_option = True, return_option = False)



