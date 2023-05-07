from __future__ import division
from __future__ import print_function

import numpy as np
import tigre
import tigre.algorithms as algs
from tigre.utilities import sample_loader
from tigre.utilities.Measure_Quality import Measure_Quality
import tigre.utilities.gpu as gpu
import matplotlib.pyplot as plt
#import sys
#import pickle
import tifffile
import config as cfg

# load data
#with open('projections.npy', 'rb') as f:
#    P = np.load(f)

with open('shifted_projections.npy', 'rb') as f:
    P = np.load(f)

nangles, nv, nu = P.shape
print("dimensions of P, ", P.shape)

listGpuNames = gpu.getGpuNames()
if len(listGpuNames) == 0:
    print("Error: No gpu found")
else:
    for id in range(len(listGpuNames)):
        print("{}: {}".format(id, listGpuNames[id]))

gpuids = gpu.getGpuIds(listGpuNames[0])
print(gpuids)

geo = tigre.geometry(mode="cone", high_resolution=True)

#geo.DSD = cfg.SDD * 0.7667238400553809 # this did work with SDD=450, SOD=20!
#geo.DSO = cfg.SOD / 0.7667238400553809

calib_factor = 0.9857877900716536
geo.DSD = cfg.SDD * calib_factor
geo.DSO = cfg.SOD / calib_factor

geo.accuracy = 1.0
pxs = cfg.detector_pixel_size
geo.dDetector = np.array([pxs, pxs]) # size of each pixel            (mm)
geo.nDetector = np.array([nv, nu])
geo.sDetector = geo.dDetector * geo.nDetector
geo.nVoxel=np.array([nv, nu, nu]) # downscale image for faster computation
# size of one image pixel:
pxi = pxs * geo.DSO / geo.DSD
geo.sVoxel=np.array([nv*pxi, nu*pxi, nu*pxi])
geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel            (mm)
geo.offDetector = np.array([2.58 * pxs, -23.37 * pxs])



print(geo)

angles = cfg.angles
proj = P.astype(np.float32)

#print("... OSSART start")
#niter = 50
#imgOSSART = algs.ossart(proj, geo, angles, niter, gpuids=gpuids)
#print("... OSSART stop")



print("... FDK start")
fdkout = algs.fdk(proj, geo, angles, gpuids=gpuids)
print("... FDK stop")

# normalize image
fdkout[fdkout < 0] = 0
minval, maxval = np.min(fdkout), np.max(fdkout)
print("image range BEFORE scaling:", minval, maxval)
fdkout -= minval
fdkout = fdkout * (2**8 / (maxval-minval))
minval, maxval = np.min(fdkout), np.max(fdkout)
print("image range AFTER scaling:", minval, maxval)


fdkout = fdkout.astype(np.uint8)
plt.imshow(fdkout[geo.nVoxel[0] // 2 + 1])
plt.show()


#save data
print("... saving reconstruction as reco.npy")
with open('reco.npy', 'wb') as f:
    np.save(f, fdkout)

print("... saving reconstruction as reco.tif")
tifffile.imwrite("reco.tif", fdkout, photometric='minisblack')

