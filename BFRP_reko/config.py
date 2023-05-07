# filename: config.py
import numpy as np

# BFRP nano-CT Datensatz "/home/gcg/Archiv/BFRP/binned_recs/"
SDD = 445.9691 # source to detecter
SOD = 8.1867 # source to object
detector_pixel_size = 2*0.0495  # [mm], 2 x gebinnt
num_projections = 600
header = 'reverted'
angles = np.linspace(0, 2*np.pi, num_projections, endpoint=True)

margin = 10
print("length of angles:", len(angles))
raw_dir = 'raw'
preproc_dir = 'preprocessed'
proj_dir = 'projections'
reco_dir = 'reconstruction'





