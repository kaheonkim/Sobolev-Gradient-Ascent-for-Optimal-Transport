import numpy as np
import numpy.ma as ma
import os
import ot
import time
import matplotlib.pyplot as plt

from w2 import BFM
from matplotlib.colors import LinearSegmentedColormap
from metric import *
from scipy.fftpack import dctn, idctn

plt.rcParams['figure.figsize'] = (13, 8) 
plt.rcParams['image.cmap'] = 'viridis'

def plotting(dist,directory,origin = 'lower',save_option = False):

  colors = [
    (1.0, 1.0, 1.0),  # white
    (0.0, 0.0, 0.4),  # bright cyan
    (0.0, 0.0, 0.2),  # rich mid-blue
    (0.0, 0.0, 0.1),  # deep blue
    (0.0, 0.0, 0.0)   # black
    ]
  custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
  vmin, vmax = 0, 100
  plt.imshow(dist, cmap=custom_cmap, origin = origin, vmin = vmin, vmax = vmax)

  plt.xticks([0, plt.gca().get_xlim()[1]], ['0', '1'])  # Custom x-axis labels
  plt.yticks([0, plt.gca().get_ylim()[1]], ['0', '1'])  # Custom y-axis labels
  if save_option:
    plt.savefig(directory+'/image.jpg')
  plt.show()


# Initialize Fourier kernel
def initialize_kernel(n1, n2):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel[0,0] = 1   
    return kernel

def dct2(a):
    return dctn(a, norm='ortho')

def idct2(a):
    return idctn(a, norm='ortho')
def update_potential(phi, rho, nu, weight, kernel, sigma):
    n1, n2 = nu.shape

    rho -= nu
    workspace = dct2(rho) / kernel
    workspace[0,0] = 0
    workspace = idct2(workspace)
    workspace *= weight
    phi += sigma * workspace
    h1 = np.sum(workspace * rho) / (n1*n2)

    return h1

def grad_norm(rho):
    n2, n1 = np.shape(rho)
    kernel = initialize_kernel(n1,n2)
    workspace = dct2(rho) / kernel
    workspace[0,0] = 0
    workspace = idct2(workspace)

    return np.sum(workspace * rho) / (n1*n2)

def compute_w2(phi, psi, mu, nu):
  n1, n2 = mu.shape
  x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1), np.linspace(0.5/n2,1-0.5/n2,n1))
  return np.sum(0.5 * (x*x+y*y) * (mu + nu) - nu*phi - mu*psi)/(n1*n2)

# Parameters for Armijo-Goldstein
scaleDown = 0.95
scaleUp   = 1/scaleDown
upper = 0.75
lower = 0.25



def compute_ot(phi, psi, bf,mu, nu, weight, sigma):
    n2, n1 = np.shape(phi)
    kernel = initialize_kernel(n1, n2)
    rho = np.copy(mu)

    x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2))

    rho = np.zeros((n2,n1))
    bf.pushforward(rho, phi, nu)
    gradSq = update_potential(psi, rho, mu, weight, kernel, sigma)

    bf.ctransform(phi, psi)
    bf.ctransform(psi, phi)

    bf.ctransform(psi, phi)
    bf.ctransform(phi, psi)

    new_w2 = compute_w2(phi, psi, mu, nu)

    return new_w2

def objective(dists, weights, potentials, bf):
  n2, n1 = np.shape(dists[0])
  m = len(weights)
  x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),np.linspace(0.5/n2,1-0.5/n2,n2))
  id = 1/2 * (x**2 + y**2)
  convexm, conjugatem = id - np.tensordot(-weights[:-1]/weights[-1], potentials, axes=([0], [0])), np.zeros((n2,n1))
  bf.ctransform(conjugatem, convexm)
  ctransformm = id - conjugatem
  value = weights[-1] * np.sum(dists[-1] * ctransformm) / (n1*n2)
  for j in range(m-1):
    convexj, conjugatej = id - potentials[j], np.zeros((n2,n1))
    bf.ctransform(conjugatej, convexj)
    ctransformj = id - conjugatej
    value += weights[j] * np.sum(dists[j] * ctransformj) / (n1*n2)
    
  return value



import time
def sga(dists, weights, n_iter,sigma, name, scheme='parallel', record_option = False, plot_option = True, origin = 'lower', save_option = True, return_option = False):
   n2, n1 = np.shape(dists[0])
   x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),np.linspace(0.5/n2,1-0.5/n2,n2))
   kernel, id = initialize_kernel(n1, n2), 1/2 * (x**2 + y**2)
   bf = BFM(n1, n2, np.ones((n2,n1)))
   weights, dists = np.array(weights), np.array(dists)
    
   weights, dists = weights[weights > 0], dists[weights > 0]
   cconcaves, distm = np.zeros((len(dists) - 1,n2,n1)), dists[-1]
   if record_option:
    error = [objective(dists, weights, cconcaves, bf)]
   time1 = time.time()
   record_time = 0
   if scheme == 'parallel':
    old_cconcaves = np.copy(cconcaves)
    for i in range(n_iter):
      new_cconcaves = []
      for j in range(len(dists)-1):
        rhom, rhoj, distj, old_cconcaves_ = np.zeros((n2,n1)), np.zeros((n2,n1)),dists[j], np.copy(old_cconcaves)
        
        cconcavem, cconcavej = np.tensordot(-weights[:-1]/weights[-1], old_cconcaves_, axes=([0], [0])), old_cconcaves_[j]
        ## Change to convex function
        convexm, convexj = id - cconcavem, id - cconcavej

        # Gradient Ascent
        conjugatem, conjugatej = np.zeros((n2,n1)), np.zeros((n2,n1))
        bf.ctransform(conjugatem, convexm)
        bf.ctransform(conjugatej, convexj)
        
        bf.pushforward(rhom, conjugatem, distm)
        bf.pushforward(rhoj, conjugatej, distj)

        workspace = dct2((rhom-rhoj)) / kernel
        workspace[0,0] = 0
        workspace = idct2(workspace)
        workspace *= weights[j]
        cconcavej += sigma  * workspace
        new_cconcaves.append(cconcavej)
        if record_option:
          time1_record = time.time()
          error.append(objective(dists, weights, old_cconcaves, bf))
          time2_record = time.time()
          record_time += (time2_record - time1_record)
      old_cconcaves = np.array(new_cconcaves)
    cconcaves = old_cconcaves
        

   elif scheme == 'sequential':
    for i in range(n_iter):
        for j in range(len(dists)-1):
          rhom, rhoj, distj = np.zeros((n2,n1)), np.zeros((n2,n1)),dists[j]

          # Two potentials
          cconcavem, cconcavej = np.tensordot(-weights[:-1]/weights[-1], cconcaves, axes=([0], [0])), cconcaves[j]
          ## Change to convex function
          convexm, convexj = id - cconcavem, id - cconcavej

          # Gradient Ascent
          conjugatem, conjugatej = np.zeros((n2,n1)), np.zeros((n2,n1))
          bf.ctransform(conjugatem, convexm)
          bf.ctransform(conjugatej, convexj)
          
          bf.pushforward(rhom, conjugatem, distm)
          bf.pushforward(rhoj, conjugatej, distj)

          workspace = dct2((rhom-rhoj)) / kernel
          workspace[0,0] = 0
          workspace = idct2(workspace)
          workspace *= weights[j+1]
          cconcavej += sigma  * workspace

          cconcaves[j] = cconcavej
          if record_option:
            time1_record = time.time()
            error.append(objective(dists, weights, cconcaves, bf))
            time2_record = time.time()
            record_time += (time2_record - time1_record)

   elif scheme == 'random':
     for i in range(n_iter):
        j = np.random.choice(range(len(weights)-1))
        rhom, rhoj, distj = np.zeros((n2,n1)), np.zeros((n2,n1)),dists[j]

        # Two potentials
        cconcavem, cconcavej = np.tensordot(-weights[:-1]/weights[-1], cconcaves, axes=([0], [0])), cconcaves[j]
        ## Change to convex function
        convexm, convexj = id - cconcavem, id - cconcavej

        # Gradient Ascent
        conjugatem, conjugatej = np.zeros((n2,n1)), np.zeros((n2,n1))
        bf.ctransform(conjugatem, convexm)
        bf.ctransform(conjugatej, convexj)
        
        bf.pushforward(rhom, conjugatem, distm)
        bf.pushforward(rhoj, conjugatej, distj)

        workspace = dct2((rhom-rhoj)) / kernel
        workspace[0,0] = 0
        workspace = idct2(workspace)
        workspace *= weights[j+1]
        cconcavej += sigma  * workspace

        cconcaves[j] = cconcavej
        if record_option:
          time1_record = time.time()
          error.append(objective(dists, weights, cconcaves, bf))
          time2_record = time.time()
          record_time += (time2_record - time1_record)
   barycenter = np.zeros((n2,n1))
   convex_potential = id - cconcaves[0]
   bf.ctransform(convex_potential, convex_potential)
   bf.pushforward(barycenter, convex_potential, dists[0])
   output_dir = '/youraddress/SGA/Case2D/image/'+ name + '/' + scheme
   os.makedirs(output_dir, exist_ok=True) 
   if save_option:
    time2 = time.time()
    elapsed = int(time2 - time1 - record_time)
    param_file = os.path.join(output_dir, "config_.txt")
    with open(param_file, "w") as f:
      f.write(f"Number of Grids : {n1} x {n2}\n")
      f.write(f"Elapsed Time : {elapsed}\n")
      f.write(f"Initial lr : {sigma}\n")
      for l in range(len(weights)):
        f.write(f"Weight{l}: {weights[l]}\n")
      f.write(f"Final objective Value : {objective(dists, weights, cconcaves, bf)}\n")
      # f.write(f"W2 value : {avgw2(dists,barycenter,weights)}")
    np.save(output_dir + '/result.npy',barycenter)
   if record_option:
    np.save(output_dir + '/error.npy',error)
    
   plotting(barycenter,output_dir,origin = origin, save_option = save_option)
  
   if return_option == True:
    return barycenter

def sga_sqrt(dists, weights, n_iter,sigma, name, scheme='parallel', record_option = False, plot_option = True, origin = 'lower', save_option = True, return_option = False):
   n2, n1 = np.shape(dists[0])
   x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),np.linspace(0.5/n2,1-0.5/n2,n2))
   kernel, id = initialize_kernel(n1, n2), 1/2 * (x**2 + y**2)
   bf = BFM(n1, n2, np.ones((n2,n1)))
   weights, dists = np.array(weights), np.array(dists)
    
   weights, dists = weights[weights > 0], dists[weights > 0]
   cconcaves, distm = np.zeros((len(dists) - 1,n2,n1)), dists[-1]
   if record_option:
    error = [objective(dists, weights, cconcaves, bf)]
   time1 = time.time()
   record_time = 0
   if scheme == 'parallel':
    old_cconcaves = np.copy(cconcaves)
    for i in range(n_iter):
      new_cconcaves = []
      for j in range(len(dists)-1):
        rhom, rhoj, distj, old_cconcaves_ = np.zeros((n2,n1)), np.zeros((n2,n1)),dists[j], np.copy(old_cconcaves)
        
        cconcavem, cconcavej = np.tensordot(-weights[:-1]/weights[-1], old_cconcaves_, axes=([0], [0])), old_cconcaves_[j]
        ## Change to convex function
        convexm, convexj = id - cconcavem, id - cconcavej

        # Gradient Ascent
        conjugatem, conjugatej = np.zeros((n2,n1)), np.zeros((n2,n1))
        bf.ctransform(conjugatem, convexm)
        bf.ctransform(conjugatej, convexj)
        
        bf.pushforward(rhom, conjugatem, distm)
        bf.pushforward(rhoj, conjugatej, distj)

        workspace = dct2((rhom-rhoj)) / kernel
        workspace[0,0] = 0
        workspace = idct2(workspace)
        workspace *= weights[j]
        cconcavej += sigma/np.sqrt(i+1)  * workspace
        new_cconcaves.append(cconcavej)
        if record_option:
          time1_record = time.time()
          error.append(objective(dists, weights, old_cconcaves, bf))
          time2_record = time.time()
          record_time += (time2_record - time1_record)
      old_cconcaves = np.array(new_cconcaves)
    cconcaves = old_cconcaves
        

   elif scheme == 'sequential':
    for i in range(n_iter):
        for j in range(len(dists)-1):
          rhom, rhoj, distj = np.zeros((n2,n1)), np.zeros((n2,n1)),dists[j]

          # Two potentials
          cconcavem, cconcavej = np.tensordot(-weights[:-1]/weights[-1], cconcaves, axes=([0], [0])), cconcaves[j]
          ## Change to convex function
          convexm, convexj = id - cconcavem, id - cconcavej

          # Gradient Ascent
          conjugatem, conjugatej = np.zeros((n2,n1)), np.zeros((n2,n1))
          bf.ctransform(conjugatem, convexm)
          bf.ctransform(conjugatej, convexj)
          
          bf.pushforward(rhom, conjugatem, distm)
          bf.pushforward(rhoj, conjugatej, distj)

          workspace = dct2((rhom-rhoj)) / kernel
          workspace[0,0] = 0
          workspace = idct2(workspace)
          workspace *= weights[j+1]
          cconcavej += sigma/np.sqrt(i+1)  * workspace

          cconcaves[j] = cconcavej
          if record_option:
            time1_record = time.time()
            error.append(objective(dists, weights, cconcaves, bf))
            time2_record = time.time()
            record_time += (time2_record - time1_record)

   elif scheme == 'random':
     for i in range(n_iter):
        j = np.random.choice(range(len(weights)-1))
        rhom, rhoj, distj = np.zeros((n2,n1)), np.zeros((n2,n1)),dists[j]

        # Two potentials
        cconcavem, cconcavej = np.tensordot(-weights[:-1]/weights[-1], cconcaves, axes=([0], [0])), cconcaves[j]
        ## Change to convex function
        convexm, convexj = id - cconcavem, id - cconcavej

        # Gradient Ascent
        conjugatem, conjugatej = np.zeros((n2,n1)), np.zeros((n2,n1))
        bf.ctransform(conjugatem, convexm)
        bf.ctransform(conjugatej, convexj)
        
        bf.pushforward(rhom, conjugatem, distm)
        bf.pushforward(rhoj, conjugatej, distj)

        workspace = dct2((rhom-rhoj)) / kernel
        workspace[0,0] = 0
        workspace = idct2(workspace)
        workspace *= weights[j+1]
        cconcavej += sigma/np.sqrt(i+1)  * workspace

        cconcaves[j] = cconcavej
        if record_option:
          time1_record = time.time()
          error.append(objective(dists, weights, cconcaves, bf))
          time2_record = time.time()
          record_time += (time2_record - time1_record)
   barycenter = np.zeros((n2,n1))
   convex_potential = id - cconcaves[0]
   bf.ctransform(convex_potential, convex_potential)
   bf.pushforward(barycenter, convex_potential, dists[0])
   output_dir = '/youraddress/SGA/Case2D/image/'+ name + '/' + scheme
   os.makedirs(output_dir, exist_ok=True) 
   if save_option:
    time2 = time.time()
    elapsed = int(time2 - time1 - record_time)
    param_file = os.path.join(output_dir, "config_.txt")
    with open(param_file, "w") as f:
      f.write(f"Number of Grids : {n1} x {n2}\n")
      f.write(f"Elapsed Time : {elapsed}\n")
      f.write(f"Initial lr : {sigma}\n")
      for l in range(len(weights)):
        f.write(f"Weight{l}: {weights[l]}\n")
      f.write(f"Final objective Value : {objective(dists, weights, cconcaves, bf)}\n")
      # f.write(f"W2 value : {avgw2(dists,barycenter,weights)}")
    np.save(output_dir + '/result.npy',barycenter)
   if record_option:
    np.save(output_dir + '/error.npy',error)
    
   plotting(barycenter,output_dir,origin = origin, save_option = save_option)
  
   if return_option == True:
    return barycenter

def wdha(dists,weights, n_iter, init_lr, name, plot_option = True,origin = 'lower',save_option = True, return_option = False):
  weights, dists = np.array(weights), np.array(dists)
  n2, n1 = np.shape(dists[0])
  x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2))
  
  dists, weights = dists[weights>0], weights[weights>0]
  n_dist, id, rd = len(dists), 1/2*(x**2 + y**2), dists[0]
  id -= np.mean(id)
  w2_list = np.zeros(n_dist)
  phi, psi = np.array([id] * n_dist), np.array([id] * n_dist)
  bf = BFM(n1, n2, rd)
  time1 = time.time()
  for i in range(n_iter):
    lr = init_lr/np.sqrt(i+1)
    prev_psi = psi
    w2dist = []
    for j in range(n_dist):
      new_w2 = compute_ot(phi[j], psi[j], bf, rd, dists[j], weights[j], lr)
      w2dist.append(new_w2)

    rho = np.ones_like(rd)
    bf.pushforward(rho, id+ lr*(np.average(prev_psi,axis=0,weights=weights)-id), rd)
    rd = rho

  output_dir = '/youraddress/SGA/Case2D/image/'+ name
  if save_option:
    time2 = time.time()
    elapsed = int(time2 - time1)
    os.makedirs(output_dir, exist_ok=True) 
    param_file = os.path.join(output_dir, "config.txt")
    with open(param_file, "w") as f:
      f.write(f"Number of Grids : {n1} x {n2}\n")
      f.write(f"Elapsed Time : {elapsed}\n")
      f.write(f"init_lr : {init_lr}\n")
      f.write(f"Average functional value : {np.average(w2dist, weights= weights)}")
      for l in range(len(weights)):
        f.write(f"Weight{l}: {weights[l]}\n")
      # f.write(f"Average W2 : {avgw2(dists,rd,weights)}")
    np.save(output_dir + '/result.npy',rd)
    
  plotting(rd,output_dir,origin = origin, save_option = save_option)
  if return_option == True:
    return rd




def wdha_constant(dists,weights, n_iter, lr, name, plot_option = True,origin = 'lower',save_option = True, return_option = False):
  #In this example, we kept constant learning rate
  weights, dists = np.array(weights), np.array(dists)
  n2, n1 = np.shape(dists[0])
  x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2))
  
  dists, weights = dists[weights>0], weights[weights>0]
  n_dist, id, rd = len(dists), 1/2*(x**2 + y**2), dists[0]
  id -= np.mean(id)
  phi, psi = np.array([id] * n_dist), np.array([id] * n_dist)
  bf = BFM(n1, n2, rd)
  time1 = time.time()
  for i in range(n_iter):
    prev_psi = psi
    w2dist = []
    for j in range(n_dist):
      new_w2 = compute_ot(phi[j], psi[j], bf, rd, dists[j], weights[j], lr)
      w2dist.append(new_w2)
    rho = np.ones_like(rd)
    bf.pushforward(rho, id+ lr*(np.average(prev_psi,axis=0,weights=weights)-id), rd)
    rd = rho


  output_dir = '/youraddress/SGA/Case2D/image/'+ name
  os.makedirs(output_dir, exist_ok=True) 
    
  if save_option:
    time2 = time.time()
    elapsed = int(time2 - time1)
    param_file = os.path.join(output_dir, "config.txt")
    with open(param_file, "w") as f:
      f.write(f"Number of Grids : {n1} x {n2}\n")
      f.write(f"Elapsed Time : {elapsed}\n")
      f.write(f"lr : {lr}\n")
      f.write(f"Average functional value : {np.average(w2dist, weights= weights)}")
      for l in range(len(weights)):
        f.write(f"Weight{l}: {weights[l]}\n")
      # f.write(f"Average W2 : {avgw2(dists,rd,weights)}")
    np.save(output_dir + '/result.npy',rd)
    
  plotting(rd,output_dir,origin = origin, save_option = save_option)
  if return_option == True:
    return rd



def frechet_mean_pot(dists, weights, n_iter, reg,name,plot_option = True,origin = 'lower', save_option=True, return_option = False):
    weights, dists = np.array(weights), np.array(dists)
    dists, weights = dists[weights>0], weights[weights>0]
    n2,n1 = np.shape(dists[0])
    tic = time.time()
    rd = ot.bregman.convolutional_barycenter2d(dists, reg, weights, numItermax=n_iter,stopThr=0.0)
    toc = time.time()

    output_dir = '/youraddress/SGA/Case2D/image/'+ name
    
    if save_option:
      elapsed = int(toc - tic)
      os.makedirs(output_dir, exist_ok=True) 
      param_file = os.path.join(output_dir, "config.txt")
      with open(param_file, "w") as f:
        f.write(f"Number of Grids : {n1} x {n2}\n")
        f.write(f"Elapsed Time : {elapsed}\n")
        f.write(f"Initial reg : {reg}\n")
        for l in range(len(weights)):
          f.write(f"Weight{l}: {weights[l]}\n")
        f.write(f"Average W2 : {avgw2(dists,rd,weights)}")
      np.save(output_dir + '/result.npy',rd)
      
    plotting(rd,output_dir, origin = origin,save_option = save_option)
    if return_option == True:
      return rd


def frechet_mean_pot_debiased(dists, weights, n_iter, reg,name,plot_option = True,origin = 'lower', save_option=True, return_option = False):
    weights, dists = np.array(weights), np.array(dists)
    dists, weights = dists[weights>0], weights[weights>0]
    n2,n1 = np.shape(dists[0])
    dists, weights = np.array(dists), np.array(weights)
    tic = time.time()
    rd = ot.bregman.convolutional_barycenter2d_debiased(dists, reg, weights, numItermax=n_iter,stopThr=0.0)
    toc = time.time()

    output_dir = '/youraddress/SGA/Case2D/image/'+ name
    if save_option:
      elapsed = int(toc - tic)
      os.makedirs(output_dir, exist_ok=True) 
      param_file = os.path.join(output_dir, "config.txt")
      with open(param_file, "w") as f:
        f.write(f"Number of Grids : {n1} x {n2}\n")
        f.write(f"Elapsed Time : {elapsed}\n")
        f.write(f"Initial reg : {reg}\n")
        for l in range(len(weights)):
          f.write(f"Weight{l}: {weights[l]}\n")
        f.write(f"Average W2 : {avgw2(dists,rd,weights)}")
      np.save(output_dir + '/result.npy',rd)
    plotting(rd,output_dir, origin = origin, save_option = save_option)
    if return_option == True:
      return rd
      