import os
import numpy as np
import matplotlib.pyplot as plt

from w23d_ import BFM3D2
from time import time
from scipy.fftpack import dctn, idctn
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

plt.rcParams['figure.figsize'] = (13, 8)
plt.rcParams['image.cmap'] = 'viridis'


def plot_isosurface(ax, data, title, color, n, iso_level=0.8):
    verts, faces, _, _ = measure.marching_cubes(data, level=iso_level)
    mesh = Poly3DCollection(verts[faces], alpha=0.05, facecolor=color, edgecolor='k')
    ax.add_collection3d(mesh)
    n1,n2,n3 = n, n, n
    ax.set_xlim(0, n1)
    ax.set_ylim(0, n2)
    ax.set_zlim(0, n3)
    ax.set_xticks(np.linspace(0, n1, num=5))
    ax.set_yticks(np.linspace(0, n2, num=5))
    ax.set_zticks(np.linspace(0, n3, num=5))
    ax.set_xticklabels(np.round(np.linspace(0, 1, 5), 2))
    ax.set_yticklabels(np.round(np.linspace(0, 1, 5), 2))
    ax.set_zticklabels(np.round(np.linspace(0, 1, 5), 2))
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title, fontsize=10)

# Initialize Fourier kernel
def initialize_kernel(n1, n2, n3):
    xx, yy, zz = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False), np.linspace(0,np.pi,n3,False))
    kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy)) + 2*n3*n3*(1-np.cos(zz))
    kernel[0,0,0] = 1     # to avoid dividing by zero
    return kernel

# 3d DCT
def dct3(a):
    return dctn(a, norm='ortho')

# 3d IDCT
def idct3(a):
    return idctn(a, norm='ortho')

def update_potential(phi, rho, nu, kernel, sigma):
    n1, n2, n3 = nu.shape

    rho -= nu
    workspace = dct3(rho) / kernel
    workspace[0,0,0] = 0
    workspace = idct3(workspace)

    phi += sigma * workspace
    h1 = np.sum(workspace * rho) / (n1*n2*n3) 

    return h1

def compute_w2(phi, psi, mu, nu):
  n1, n2, n3 = mu.shape
  x, y, z = np.meshgrid(np.linspace(1/n1,1-1/n1,n1), np.linspace(1/n2,1-1/n2,n2),np.linspace(1/n3,1-1/n3,n3))
  return np.sum(0.5 * (x*x+y*y+z*z) * (mu + nu) - nu*phi - mu*psi)/(n1*n2*n3)

def compute_ot(phi, psi, bf,mu, nu, sigma, inner ):
    n3, n2, n1 = np.shape(phi)
    kernel = initialize_kernel(n1, n2, n3)
    rho = np.copy(mu)

    x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1), np.linspace(0.5/n2,1-0.5/n2,n2), np.linspace(0.5/n3,1-0.5/n3,n3))

    old_w2 = compute_w2(phi, psi, mu, nu)
    for k in range(inner):
        rho = np.zeros((n3,n2,n1))
        bf.pushforward(rho, phi, nu)
        gradSq = update_potential(psi, rho, mu, kernel, sigma)

        bf.ctransform(phi, psi)
        bf.ctransform(psi, phi)

        bf.ctransform(psi, phi)
        bf.ctransform(phi, psi)

        new_w2 = compute_w2(phi, psi, mu, nu)

    return new_w2


def frechet_mean(dists, n_iter,name = '', init_lr1 = 1e-2, init_lr2 = 1, plot_option = False,save_option = True, return_option = False,  inner = 1):
  n3, n2, n1 = np.shape(dists[0])
  x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2),
                    np.linspace(0.5/n3,1-0.5/n3,n3))
  weights, dists = np.array(weights), np.array(dists) 
  weights, dists = weights[weights > 0], dists[weights > 0]
  n_dist, id, rd = len(dists), 1/2*(x**2 + y**2 + z**2), dists[0]
  id -= np.mean(id)
  sigma, w2_list = init_lr1 * np.ones(n_dist), np.zeros(n_dist)
  phi, psi = np.array([id] * n_dist), np.array([id] * n_dist)
  bf = BFM3D2(n1, n2, n3, rd)
  for i in range(n_iter):
    prev_psi = psi
    for j in range(n_dist):
      new_w2 = compute_ot(phi[j], psi[j], bf, rd, dists[j], sigma[j], inner = inner)
      if new_w2 < w2_list[j]:
        sigma[j] *= 0.95
      w2_list[j] = new_w2
    lr = np.exp(-(i+1)*init_lr2/n_iter)
    rho = np.ones_like(rd)
    bf.pushforward(rho, id+ lr*(np.mean(prev_psi,axis=0)-id), rd)
    rd = rho
    if (i+1)%500 == 0:
      print(f"Num_Iter : {i+1}")
      print(w2_list)
      fig = plt.figure(figsize=(8, 8))
      ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
      plot_isosurface(ax, rho, "", 'purple', n1, iso_level=1.5)
      ax.view_init(elev=0, azim=0) 
      if save_option:
        plt.savefig(f"/youraddress/Case3D/{name}/{i+1}.png", dpi=300)
        np.save(f"/youraddress/Case3D/{name}/npy{i+1}.npy", rho)
      if plot_option:
        plt.show()
  if return_option == True:
    return rd
  




def update_potential_weighted(phi, rho, nu, kernel, sigma, weight):
    n1, n2, n3 = nu.shape

    rho -= nu
    workspace = dct3(rho) / kernel
    workspace[0,0,0] = 0
    workspace = idct3(workspace)

    phi += sigma * weight * workspace
    h1 = np.sum(workspace * rho) / (n1*n2*n3)

    return h1

def compute_ot_weighted(phi, psi, bf,mu, nu, sigma, inner, weight):
    n3, n2, n1 = np.shape(phi)
    kernel = initialize_kernel(n1, n2, n3)
    rho = np.copy(mu)

    x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2), np.linspace(0.5/n3,1-0.5/n3,n3))

    old_w2 = compute_w2(phi, psi, mu, nu)
    for k in range(inner):
        rho = np.zeros((n3,n2,n1))
        bf.pushforward(rho, phi, nu)
        gradSq = update_potential_weighted(psi, rho, mu, kernel, sigma, weight)

        bf.ctransform(phi, psi)
        bf.ctransform(psi, phi)

        bf.ctransform(psi, phi)
        bf.ctransform(phi, psi)

        new_w2 = compute_w2(phi, psi, mu, nu)

    return new_w2

import time
def frechet_mean_weighted(dists, weight, n_iter,output_dir = '', init_lr1 = 1e-2, init_lr2 = 1, plot_option = False,save_option = True, return_option = False,  inner = 1):
  weight /= np.sum(weight)
  n3, n2, n1 = np.shape(dists[0])
  x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2),
                    np.linspace(0.5/n3,1-0.5/n3,n3))
  n_dist, id, rd = len(dists), 1/2*(x**2 + y**2 + z**2), dists[np.argmax(weight)]
  id -= np.mean(id)
  sigma, w2_list = init_lr1 * np.ones(n_dist), np.zeros(n_dist)
  phi, psi = np.array([id] * n_dist), np.array([id] * n_dist)
  bf = BFM3D2(n1, n2, n3, rd)
  os.makedirs(output_dir, exist_ok=True) 
  time1 = time.time()
  for i in range(n_iter):
    prev_psi = psi
    for j in range(n_dist):
      new_w2 = compute_ot_weighted(phi[j], psi[j], bf, rd, dists[j], sigma[j], inner = inner, weight = weight[j])
      if new_w2 < w2_list[j]:
        sigma[j] *= 0.95
      w2_list[j] = new_w2
    lr = np.exp(-(i+1)*init_lr2/n_iter)
    rho = np.ones_like(rd)
    bf.pushforward(rho, id+ lr*(np.average(prev_psi,axis=0,weights=weight)-id), rd)
    rd = rho
  time2 = time.time()
      # Create a figure with 4 subplots
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
  plot_isosurface(ax, rho, "Isosurface of rho", 'purple', n1, iso_level=1.5)
  elapsed = time2 - time1
  if save_option:
    plt.savefig("/youraddress/SGA/Case3D/"+output_dir+"final.png", dpi=300)
    np.save("/youraddress/SGA/Case3D/"+output_dir+"final.npy", rho)
    param_file = os.path.join(output_dir, "config.txt")
    with open(param_file, "w") as f:
      f.write(f"Elapsed Time: {elapsed}\n")
      f.write(f"Number of Grids : {n1} x {n2}\n")
      f.write(f"Averagew2 : {np.average(w2_list, weights=weight)}\n")
      f.write(f"Initial lr : {init_lr1}, {init_lr2}\n")
      for l in range(len(weight)):
        f.write(f"Weight{l}: {weight[l]}\n")
  if plot_option:
    plt.show()
  if return_option == True:
    return rd
  

def frechet_mean_weighted(dists, weight, n_iter,output_dir = '', init_lr = 5e-3 , plot_option = False,save_option = True, return_option = False,  inner = 1):
  weight /= np.sum(weight)
  n3, n2, n1 = np.shape(dists[0])
  x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2),
                    np.linspace(0.5/n3,1-0.5/n3,n3))
  n_dist, id, rd = len(dists), 1/2*(x**2 + y**2 + z**2), dists[np.argmax(weight)]
  id -= np.mean(id)
  phi, psi = np.array([id] * n_dist), np.array([id] * n_dist)
  bf = BFM3D2(n1, n2, n3, rd)
  os.makedirs(output_dir, exist_ok=True) 
  time1 = time.time()
  for i in range(n_iter):
    prev_psi = psi
    w2dist = []
    for j in range(n_dist):
      new_w2 = compute_ot_weighted(phi[j], psi[j], bf, rd, dists[j], init_lr / np.sqrt(i+1), inner = inner, weight = weight[j])
      w2dist.append(new_w2)
    lr = init_lr / np.sqrt(i+1)
    rho = np.ones_like(rd)
    bf.pushforward(rho, id+ lr*(np.average(prev_psi,axis=0,weights=weight)-id), rd)
    rd = rho
  time2 = time.time()
      # Create a figure with 4 subplots
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
  plot_isosurface(ax, rho, "Isosurface of rho", 'purple', n1, iso_level=1.5)
  elapsed = time2 - time1
  if save_option:
    plt.savefig("/youraddress/SGA/Case3D/"+output_dir+"final.png", dpi=300)
    np.save("/youraddress/SGA/Case3D/"+output_dir+"final.npy", rho)
    param_file = os.path.join(output_dir, "config.txt")
    with open(param_file, "w") as f:
      f.write(f"Elapsed Time: {elapsed}\n")
      f.write(f"Number of Grids : {n1} x {n2}\n")
      f.write(f"Averagew2 : {np.average(w2dist, weights=weight)}\n")
      f.write(f"Initial lr : {init_lr}\n")
      for l in range(len(weight)):
        f.write(f"Weight{l}: {weight[l]}\n")
  if plot_option:
    plt.show()
  if return_option == True:
    return rd

def objective(dists, weights, potentials, bf):
  n3, n2, n1 = np.shape(dists[0])
  m = len(weights)
  x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2),
                    np.linspace(0.5/n3,1-0.5/n3,n3))
  id = 1/2 * (x**2 + y**2 + z**2)
  convexm, conjugatem = id - np.tensordot(-weights[:-1]/weights[-1], potentials, axes=([0], [0])), np.zeros((n3,n2,n1))
  bf.ctransform(conjugatem, convexm)
  ctransformm = id - conjugatem
  value = weights[-1] * np.sum(dists[-1] * ctransformm) / (n1*n2*n3)
  for j in range(m-1):
    convexj, conjugatej = id - potentials[j], np.zeros((n3,n2,n1))
    bf.ctransform(conjugatej, convexj)
    ctransformj = id - conjugatej
    value += weights[j] * np.sum(dists[j] * ctransformj) / (n1*n2*n3)
    
  return value 

import time

def sga3D_sqrt(dists, weights, n_iter,lr, output_dir, record_option = False, save_option = True, return_option = False):
   n3, n2, n1 = np.shape(dists[0])
   x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2),
                    np.linspace(0.5/n3,1-0.5/n3,n3))
   kernel, id = initialize_kernel(n1, n2, n3), 1/2 * (x**2 + y**2 + z**2)
   bf = BFM3D2(n1, n2, n3, np.ones((n3, n2, n1)))
   weights, dists = np.array(weights), np.array(dists)
   
   weights, dists = weights[weights > 0], dists[weights > 0]
   order = np.argsort(np.argsort(weights))[::-1]
   weights, dists = weights[order], dists[order]
   cconcaves, distm = np.zeros((len(dists) - 1,n3,n2,n1)), dists[-1]
   output_dir = output_dir
   os.makedirs(output_dir+'/error', exist_ok=True) 
   
   if record_option:
    error_value = objective(dists, weights, cconcaves, bf)
    print(error_value)
    error = [error_value]
   time1 = time.time()
   record_time = 0
   old_cconcaves = np.copy(cconcaves)
   for i in range(n_iter):
    sigma = lr/np.sqrt(i+1)
    new_cconcaves = []
    for j in range(len(dists)-1):
      rhom, rhoj, distj, old_cconcaves_ = np.zeros((n3,n2,n1)), np.zeros((n3,n2,n1)), dists[j], np.copy(old_cconcaves)
        
      cconcavem, cconcavej = np.tensordot(-weights[:-1]/weights[-1], old_cconcaves_, axes=([0], [0])), old_cconcaves_[j]
        ## Change to convex function
      convexm, convexj = id - cconcavem, id - cconcavej

        # Gradient Ascent
      conjugatem, conjugatej = np.zeros((n3,n2,n1)), np.zeros((n3,n2,n1))
      bf.ctransform(conjugatem, convexm)
      bf.ctransform(conjugatej, convexj)
        
      bf.pushforward(rhom, conjugatem, distm)
      bf.pushforward(rhoj, conjugatej, distj)
      workspace = dct3((rhom-rhoj)) / kernel
      workspace[0,0,0] = 0
      workspace = idct3(workspace)
      workspace *= weights[j]
      cconcavej += sigma  * workspace
      new_cconcaves.append(cconcavej)
    old_cconcaves = np.array(new_cconcaves)
    if ((i+1) % 100 == 0) & record_option:
      time1_record = time.time()
      error_value = objective(dists, weights, old_cconcaves, bf)
      print(error_value)
      error.append(error_value)
      np.save(output_dir + '/error/error'+str(i+1)+'.npy',error)
      time2_record = time.time()
      record_time += (time2_record - time1_record)
      
    cconcaves = old_cconcaves

   barycenter = np.zeros((n3,n2,n1))
   convex_potential = id - cconcaves[0]
   bf.ctransform(convex_potential, convex_potential)
   bf.pushforward(barycenter, convex_potential, dists[0])
   if save_option:
    time2 = time.time()
    elapsed = int(time2 - time1 - record_time)
    print(elapsed)
    param_file = os.path.join(output_dir, "config.txt")
    with open(param_file, "w") as f:
      f.write(f"Elapsed Time: {elapsed}\n")
      f.write(f"Number of Grids : {n1} x {n2}\n")
      if not record_option:
        f.write(f"Elapsed Time : {elapsed}\n")
      f.write(f"Initial lr : {sigma}\n")
      for l in range(len(weights)):
        f.write(f"Weight{l}: {weights[l]}\n")
    np.save(output_dir + '/result.npy',barycenter)

    
   if save_option: 
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    plot_isosurface(ax, barycenter, "Isosurface of rho", 'purple', n1, iso_level=1.5)
    plt.savefig(output_dir+"/image.png", dpi=300)
   if return_option == True:
    return barycenter
   
def sga3D_exponential_decay(dists, weights, n_iter,sigma, output_dir, record_option = False, save_option = True, return_option = False):
   n3, n2, n1 = np.shape(dists[0])
   x, y, z = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2),
                    np.linspace(0.5/n3,1-0.5/n3,n3))
   kernel, id = initialize_kernel(n1, n2, n3), 1/2 * (x**2 + y**2 + z**2)
   bf = BFM3D2(n1, n2, n3, np.ones((n3, n2, n1)))
   weights, dists = np.array(weights), np.array(dists)
   
   weights, dists = weights[weights > 0], dists[weights > 0]
   order = np.argsort(np.argsort(weights))[::-1]
   weights, dists = weights[order], dists[order]
   cconcaves, distm = np.zeros((len(dists) - 1,n3,n2,n1)), dists[-1]
   output_dir = output_dir
   os.makedirs(output_dir+'/error', exist_ok=True) 
   
   if record_option:
    error_value = objective(dists, weights, cconcaves, bf)
    print(error_value)
    error = [error_value]
   time1 = time.time()
   record_time = 0
   old_cconcaves = np.copy(cconcaves)
   for i in range(n_iter):
    new_cconcaves = []
    for j in range(len(dists)-1):
      rhom, rhoj, distj, old_cconcaves_ = np.zeros((n3,n2,n1)), np.zeros((n3,n2,n1)), dists[j], np.copy(old_cconcaves)
        
      cconcavem, cconcavej = np.tensordot(-weights[:-1]/weights[-1], old_cconcaves_, axes=([0], [0])), old_cconcaves_[j]
        ## Change to convex function
      convexm, convexj = id - cconcavem, id - cconcavej

        # Gradient Ascent
      conjugatem, conjugatej = np.zeros((n3,n2,n1)), np.zeros((n3,n2,n1))
      bf.ctransform(conjugatem, convexm)
      bf.ctransform(conjugatej, convexj)
        
      bf.pushforward(rhom, conjugatem, distm)
      bf.pushforward(rhoj, conjugatej, distj)
      workspace = dct3((rhom-rhoj)) / kernel
      workspace[0,0,0] = 0
      workspace = idct3(workspace)
      workspace *= weights[j]
      cconcavej += sigma  * workspace
      new_cconcaves.append(cconcavej)
    old_cconcaves = np.array(new_cconcaves)
    sigma *= 0.99
    if ((i+1) % 100 == 0) & record_option:
      time1_record = time.time()
      error_value = objective(dists, weights, old_cconcaves, bf)
      print(error_value)
      error.append(error_value)
      np.save(output_dir + '/error/error'+str(i+1)+'.npy',error)
      time2_record = time.time()
      record_time += (time2_record - time1_record)
      
    cconcaves = old_cconcaves

   barycenter = np.zeros((n3,n2,n1))
   convex_potential = id - cconcaves[0]
   bf.ctransform(convex_potential, convex_potential)
   bf.pushforward(barycenter, convex_potential, dists[0])
   if save_option:
    time2 = time.time()
    elapsed = int(time2 - time1 - record_time)
    print(elapsed)
    param_file = os.path.join(output_dir, "config.txt")
    with open(param_file, "w") as f:
      f.write(f"Elapsed Time: {elapsed}\n")
      f.write(f"Number of Grids : {n1} x {n2}\n")
      if not record_option:
        f.write(f"Elapsed Time : {elapsed}\n")
      f.write(f"Initial lr : {sigma}\n")
      for l in range(len(weights)):
        f.write(f"Weight{l}: {weights[l]}\n")
    np.save(output_dir + '/result.npy',barycenter)

    
   if save_option: 
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    plot_isosurface(ax, barycenter, "Isosurface of rho", 'purple', n1, iso_level=1.5)
    plt.savefig(output_dir+"/image.png", dpi=300)
   if return_option == True:
    return barycenter