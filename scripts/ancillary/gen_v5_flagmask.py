"""gen_v5_flagmask.py"""

import numpy as np
from netCDF4 import Dataset

# NH
nh_anc = Dataset("./ecdr-ancillary-psn25.nc")
nh_surf = np.array(nh_anc.variables["surface_type"])
ydim, xdim = nh_surf.shape

nh_flag = np.zeros((ydim, xdim), dtype=np.uint8)
nh_flag[:] = 255
nh_flag[nh_surf == 50] = 0  # Ocean
nh_flag[nh_surf == 75] = 252  # Lake
nh_flag[nh_surf == 200] = 253  # Coast
nh_flag[nh_surf == 250] = 254  # Land
assert np.all(nh_flag != 255)
ofn = "flagmask_psn25_v05.dat"
nh_flag.tofile(ofn)
print(f"Wrote: {ofn}")

# SH
sh_anc = Dataset("./ecdr-ancillary-pss25.nc")
sh_surf = np.array(sh_anc.variables["surface_type"])
ydim, xdim = sh_surf.shape

sh_flag = np.zeros((ydim, xdim), dtype=np.uint8)
sh_flag[:] = 255
sh_flag[sh_surf == 50] = 0  # Ocean
sh_flag[sh_surf == 75] = 252  # Lake
sh_flag[sh_surf == 200] = 253  # Coast
sh_flag[sh_surf == 250] = 254  # Land
assert np.all(sh_flag != 255)
ofn = "flagmask_pss25_v05.dat"
sh_flag.tofile(ofn)
print(f"Wrote: {ofn}")
