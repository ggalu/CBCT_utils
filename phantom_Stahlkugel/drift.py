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
    

def compute_centroid():

    mu = np.zeros(N)
    mv = np.zeros(N)

    for p in track(range(N), description="... computing mu and mv ..."):
        vc = nv//2
        #s = scipy.ndimage.center_of_mass(P[p,vc,:]) # for all vertical coordinates
        s = scipy.ndimage.center_of_mass(P[p,:,:])
        
        mv[p], mu[p] = s[0], s[1] # for all vertical coordinates
    return mv, mu

def ideal_trajectory(param,  theta):

    r0, z0, theta0 = param
    scale1 = scale2 = 1.0

    x = sin(theta - theta0) * r0
    y = cos(theta - theta0) * r0
    
    RFD = SDD * L2px * scale1
    RFI = SOD * L2px * scale2

    u_id = y * RFD / (RFI + x)
    v_id = z0 * RFD / (RFI + x)
    return np.asarray([v_id, u_id])

def score_function_u(params):
    traj = ideal_trajectory(params, thetas)
    diff = (traj[1,:] - centroid_u)**2 # only fit horizontal coordinate
    return np.sum(diff)

def score_function_v(z0):
    params = (r0, z0, theta0)
    traj = ideal_trajectory(params, thetas)
    diff = (traj[0,:] - centroid_v)**2 # only fit vertical coordinate
    return np.sum(diff)



thetas = np.flip(cfg.angles)
angles = 180 * thetas / np.pi 
SOD = cfg.SOD
SDD = cfg.SDD
px2L = cfg.detector_pixel_size
L2px = 1.0 / px2L # ... and its inverse 

#
P, N, nv, nu = load_projections()
centroid_v, centroid_u = compute_centroid()
np.savetxt("schwerpunkt_phantom.dat", np.column_stack((centroid_u, centroid_v)))

plt.plot(centroid_u, centroid_v)
plt.show()

detector_center_u = nu/2
print("detector center:", detector_center_u)
mean_centroid_u = np.mean(centroid_u)
mean_shift_u =  mean_centroid_u - detector_center_u
print("mean horizontal centroid:", mean_centroid_u)
print("mean horizontal shift:", mean_shift_u)
centroid_u -= mean_centroid_u # shift trajectory such that it revolves around zero
mean_centroid_v = np.mean(centroid_v)
centroid_v -= mean_centroid_v # shift trajectory such that it revolves around zero


from scipy.optimize import minimize
r0, z0, theta0 = 22.1, 3.0, 0.7
params0 = (r0, z0, theta0)
res = minimize(score_function_u, params0, method='Nelder-Mead', tol=1e-8)
r0, z0, theta0  = res.x
print("Excenter Radius r0:", r0)
print("Phasenshift theta0:", theta0)

res = minimize(score_function_v, (z0,), method='Nelder-Mead', tol=1e-4)
z0 = res.x
print("z0:", z0)
params = (r0, z0, theta0)
traj = ideal_trajectory(params, thetas)

plt.plot(angles, centroid_u, "rx", markevery=5, label="Projektion horizontal")
plt.plot(angles, traj[1,:], "r-", label="gefittete horizontale Trajektorie")
plt.plot(angles, centroid_v, "gx", markevery=5, label="Projektion vertikal")
plt.plot(angles, traj[0,:], label="gefittete vertikale Trajektorie")
plt.xlabel("Rotationswinkel / °")
plt.ylabel("Schwerpunkt der Projektion [px]")
plt.legend()
plt.grid()
plt.show()

#---------------------------------------

deviation_u = centroid_u - traj[1,:]
deviation_v = centroid_v - traj[0,:]
plt.plot(angles, deviation_u, label="Abweichung Proj. - ideal (horiz)")
plt.plot(angles, deviation_v, label="Abweichung Proj. - ideal (vert)")
plt.xlabel("Rotationswinkel / °")
plt.ylabel("Schwerpunkt der Projektion [px]")
plt.legend()
plt.grid()
plt.show()

sys.exit()


# apply image shifts

for p in track(range(cfg.num_projections), description="applying shifts ..."):
    im = P[p, :, :]

    # intensity in air region (left + right margins)
    I0 = np.mean([np.mean(im[:, : cfg.margin]), np.mean(im[:, -cfg.margin :])])
    shift = (-deviation_v[p] , -deviation_u[p] - mean_shift_u)
    
    # apply horizontal centroid shift
    P[p,:,:] = scipy.ndimage.shift(im, shift, order=3, cval=I0)

print("... saving shifted projections")
with open( "shifted_projections.p", "wb" ) as f:
    np.save("shifted_projections.npy", P)
     



