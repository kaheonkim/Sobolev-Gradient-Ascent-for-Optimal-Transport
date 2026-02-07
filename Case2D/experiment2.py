import numpy as np
import numpy.ma as ma
import time
import matplotlib.pyplot as plt
import kagglehub

from time import time
from functions import *
from w2 import BFM
from metric import *
from PIL import Image

plt.rcParams['figure.figsize'] = (13, 8)
plt.rcParams['image.cmap'] = 'viridis'


# Download latest version
path = kagglehub.dataset_download("trainingdatapro/electric-scooters-tracking")

print("Path to dataset files:", path)

img = []
for i in range(16):
    if i<10:
        image = Image.open(path+'/1/images/1_frame_0'+str(i)+'.png')
        image = image.convert("L")
        image = image.resize((400,400), Image.Resampling.LANCZOS)
        image = np.array(image).astype(float)
        image *= 400**2/np.sum(image)
        img.append(np.array(image))
    else:
        image = Image.open(path+'/1/images/1_frame_'+str(i)+'.png').convert("L")
        image = image.convert("L")
        image = image.resize((400,400), Image.Resampling.LANCZOS)
        image = np.array(image).astype(float)
        image *= 400**2/np.sum(image)
        img.append(np.array(image))
img = np.array(img)

weight = [1/len(img)] * len(img)
wdha_constant(img, weight, 300, 1e-1, 'experiment2/WDHA1e-1', save_option = True, origin = 'upper', return_option = False)
sga(img, weight, 300, 0.1, 'experiment2/sga01', save_option = True, origin = 'upper', return_option = False)
frechet_mean_pot(img, weight, 300, 5e-4,'experiment2/cwb5e-4',plot_option = True,origin = 'upper', save_option=True, return_option = False)
frechet_mean_pot_debiased(img, weight, 500, 5e-4,'experiment2/dsb5e-4',plot_option = True,origin = 'upper', save_option=True, return_option = False)





