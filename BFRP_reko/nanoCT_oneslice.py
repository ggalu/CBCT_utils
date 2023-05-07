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
with open('projections.npy', 'rb') as f:
    P = np.load(f)

#with open('shifted_projections.npy', 'rb') as f:
#    P = np.load(f)

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

FULL_RECONSTRUCTION = True # reconstruct entire volume (True), or just central slice (False)

geo = tigre.geometry(mode="cone", high_resolution=True)
geo.DSD = cfg.SDD
geo.DSO = cfg.SOD
geo.accuracy = 1.0
pxs = cfg.detector_pixel_size
geo.dDetector = np.array([pxs, pxs]) # size of each pixel            (mm)
geo.nDetector = np.array([nv, nu])
geo.sDetector = geo.dDetector * geo.nDetector

# size of one image pixel:
pxi = pxs * geo.DSO / geo.DSD
if FULL_RECONSTRUCTION:
    geo.nVoxel=np.array([nv, nu, nu]) # downscale image for faster computation
    geo.sVoxel=np.array([nv*pxi, nu*pxi, nu*pxi])
else:
    geo.nVoxel=np.array([1, nu, nu]) # downscale image for faster computation
    geo.sVoxel=np.array([1*pxi, nu*pxi, nu*pxi])
geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel            (mm)

shiftu0 = -37.31 * pxs #(-40.81 + 0.7*5) # px units
shiftv0 = 2.38719226e-05 * pxs # px units
geo.offDetector = np.array([shiftv0, shiftu0]) # shift of +5 determined 7 May 2023

angles = cfg.angles
proj = P.astype(np.float32)

roll = 0
pitch = 0
yaw = 0
DSO0 = geo.DSO
DSD0 = geo.DSD
count = 0
start = 0
stop = 0
for q in range(start, stop + 1, 1):
    #factor = (1+0.005*q)
    print("####################################################################")
    print("step: ", count)
    print("####################################################################")


    filename = "vanilla.tif" # not changing any parameters

    #print("factor:", factor)
    #geo.DSO = DSO0 * factor # change source-object
    #geo.DSD = DSD0 / factor

    
    
    
    # horizontal shift (u)
    #geo.offDetector = np.array([0 * pxs, (-40.81 + 0.7*q) * pxs])
    #filename = "shiftu_%d_%d_.tif" % (count, q)

    # vertical shift (v)
    #geo.offDetector = np.array([(shiftv0 + q) * pxs, shiftu0 * pxs])
    #filename = "shiftv_%d_%d_.tif" % (count, q)
    
    # ROLL
    #roll = q * 0.5 * np.pi / 180.0
    #geo.rotDetector = np.array([roll, pitch, yaw])
    #filename = "roll_%d_%d_.tif" % (count, q)

    # PITCH
    #pitch = q * 0.5 * np.pi / 180.0
    #geo.rotDetector = np.array([roll, pitch, yaw])
    #filename = "pitch_%d_%d_.tif" % (count, q)

    # YAW
    #yaw = q * 0.5 * np.pi / 180.0
    #geo.rotDetector = np.array([roll, pitch, yaw])
    #filename = "yaw_%d_%d_.tif" % (count, q)


    print(geo)

    print("... FDK start")
    fdkout = algs.fdk(proj, geo, angles, gpuids=gpuids)
    print("... FDK stop")

    # normalize image and convert to uint8 (0-255) grayscale
    fdkout[fdkout < 0] = 0
    minval, maxval = np.min(fdkout), np.max(fdkout)
    print("image range BEFORE scaling:", minval, maxval)
    fdkout -= minval
    fdkout = fdkout * (2**8 / (maxval-minval))
    minval, maxval = np.min(fdkout), np.max(fdkout)
    print("image range AFTER scaling:", minval, maxval)

    fdkout = fdkout.astype(np.uint8)
    
    tifffile.imwrite(filename, fdkout[geo.nVoxel[0] // 2], photometric='minisblack')
    count += 1
    print("... wrote file ", filename)




#plt.imshow(fdkout[geo.nVoxel[0] // 2 + 1])
#plt.show()


#save data
print("... saving reconstruction as reco.npy")
with open('reco.npy', 'wb') as f:
    np.save(f, fdkout)

print("... saving reconstruction as reco.tif")
tifffile.imwrite("reco.tif", fdkout, photometric='minisblack')
