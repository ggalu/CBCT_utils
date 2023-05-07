#!/home/gcg/miniconda3/envs/tigre/bin/python3
""" find drift in projections based on centroid
"""
import config as cfg
import scipy

from  scipy import ndimage
import numpy as np
from numpy import cos, sin
#import imageio
from os.path import join
from rich.progress import track
import matplotlib.pyplot as plt
import pickle
import sys



def load_projections():

    print("arguments", sys.argv)
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("need to run this tool like:")
        print("./drift.py projections.npy")
        sys.exit()

    with open(filename, 'rb') as f:
        P = np.load(f).astype(np.float32)

    dims = P.shape
    nv, nu = dims[1], dims[2]
    return P, dims[0], nv, nu
    
P, N, nv, nu = load_projections()

with open("shifts.npy", "rb") as f:
    shifts = np.load(f)

print("shape of shifts", shifts.shape)

# apply image shifts

for p in track(range(cfg.num_projections), description="applying shifts ..."):
    im = P[p, :, :]

    # intensity in air region (left + right margins)
    I0 = np.mean([np.mean(im[:, : cfg.margin]), np.mean(im[:, -cfg.margin :])])

    #i = N - p - 1
    shift = shifts[:,p] # this is correct, no negative sign here


    #print("shift:", shift)
    
    # apply horizontal centroid shift
    P[p,:,:] = scipy.ndimage.shift(im, shift, order=3, cval=I0)

print("... saving shifted projections")
with open( "shifted_projections.p", "wb" ) as f:
    np.save("shifted_projections.npy", P)
     



