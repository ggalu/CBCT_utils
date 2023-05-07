""" preprocess projections of radiation intensity
    assumption: low values mean high material density,
                high values mean air

    data is saved as numpy uint16 format

"""
from __future__ import print_function
from __future__ import division
 
from os import makedirs
from os.path import join, exists
import numpy as np
from rich.progress import track
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich import print
import pickle
import scipy
import tifffile
import config as cfg
import matplotlib.pyplot as plt

console = Console()
 
## Dark frame.
#dark_frame = \
#  imread(join(cfg.raw_dir, 'dark_frame.tif')).astype(float)
# 
## Flat fields.
#pre_flat_field = \
#  imread(join(cfg.raw_dir, 'pre_flat_field.tif')).astype(float)
#pre_flat_field -= dark_frame
#post_flat_field = \
#  imread(join(cfg.raw_dir, 'post_flat_field.tif')).astype(float)
#post_flat_field -= dark_frame
 
num_proj = cfg.num_projections
 
# Determine maximum of projections.
M = -np.inf  # max

air_values = np.zeros(num_proj)

first = True
for p in track(range(num_proj), description="reading image data (step 0/2)..."):
    im = tifffile.imread(join(cfg.raw_dir, cfg.header+'{:04d}.tif'.format(p+1)))
    if first:
        nv, nu = im.shape
        print("image dimensions:", nv, nu)
        P = np.zeros((cfg.num_projections, nv, nu), dtype=np.uint16)
        print("min, max:", np.min(im), np.max(im))
        first = False
    P[p,:,:] = im + 1

for proj in track(range(num_proj), description="Processing data (step 1)..."):

    im = P[proj,:,:].astype(float) #imageio.v2.imread(join(cfg.raw_dir, cfg.header+'{:04d}.tif'.format(proj)))

    #im -= dark_frame
    # Compute interpolated flat field.
    #flat_field = \
    #  ((num_proj - 1 - proj) * pre_flat_field + proj * post_flat_field) / \
    #  (num_proj - 1)
    #im /= flat_field
    
    # air: left and right margins of image
    #I0 = np.mean([np.mean(im[:, : cfg.margin]), np.mean(im[:, -cfg.margin :])]) # left and right margins
    I0 = np.mean(im[:, :cfg.margin]) # only left margin
    air_values[proj] = I0

I0_global = np.mean(air_values)    

for proj in track(range(num_proj), description="Processing data (step 1.5)..."):
    im = P[proj,:,:].astype(float)
    I0 = np.mean(im[:, :cfg.margin]) # only left margin

    # Values above I0 are due to noise and wil produce negative densities later, remove them
    im[im > I0] = I0
    # normalize I0 such that equals the mean I0 over all images
    im /= I0/I0_global
    im = -np.log(im / I0_global) # convert from attenuation to material density
    if np.max(im) > M:
        M = np.max(im) # find the maximum value across all projections



console.print(Panel(\
"maximum intensity in dataset: %g \n \
 mean air intensity:           %g " % (M, I0_global), expand=False))
  
 
# Convert raw images to projections.
for proj in track(range(num_proj), description="Processing data (step 2)..."):
    im = P[proj,:,:].astype(float)

    #im -= dark_frame
    # Compute interpolated flat field.
    #flat_field = \
    #  ((num_proj - 1 - proj) * pre_flat_field + proj * post_flat_field) / \
    #  (num_proj - 1)
    #im /= flat_field
    
    #I0 = np.mean([np.mean(im[:, : cfg.margin]), np.mean(im[:, -cfg.margin :])])
    I0 = np.mean(im[:, :cfg.margin]) # only left margin
    
    # Values above I0 are due to noise and wil produce negative densities later.
    im[im > I0] = I0

    im /= I0/I0_global
    
    im = -np.log(im/I0_global)
    im /= M
    P[proj,:,:] = (im * 2**16).astype(np.uint16)

    #print("intensity shift:", I0, air_mean, I0/air_mean)
    
    #save_tiff(join(cfg.proj_dir, 'proj{:04d}.tif'.format(proj)), im)
    #tifffile.imwrite(join(cfg.proj_dir, 'proj{:04d}.tif'.format(proj)), im.astype(np.uint16), photometric='minisblack')
#tifffile.imwrite("proj.tif", P, photometric='miniswhite')

# crop and threshold
#threshold = 13601
#P[P < threshold] = 0

#topcrop = 300 
#botcrop = 300
#P = P[:,botcrop:-topcrop,:]

print("saving data to numpy array")
with open('projections.npy', 'wb') as f:
    np.save(f, P)
