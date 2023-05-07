# filename: config.py
import numpy as np

# nano CT Stahlkugeln 3. Mai Nr. 2
SOD = 20 # source to object
SDD = 350 # source to detecter
detector_pixel_size = 0.0495 * 4608 / 1000  # [mm]
num_projections = 360
horizontal_shift = 0
header = 'proj'
angles = np.linspace(2 * np.pi, 0, num_projections, endpoint=True)

margin = 10
print("length of angles:", len(angles))
#raw_dir = 'phantom'
raw_dir = 'raw'
preproc_dir = 'preprocessed'
proj_dir = 'projections'
reco_dir = 'reconstruction'





