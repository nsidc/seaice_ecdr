"""Routines to determine the AMSR-E pole hole.

"""

import datetime as dt

import numpy as np
from netCDF4 import Dataset
from scipy.signal import convolve2d

had_missing = np.zeros((896, 608), dtype=np.uint16)
year = 2005

for doy in range(365):
    date = dt.date(
        2005,
        1,
        1,
    ) + dt.timedelta(days=doy)
    fn = f'/ecs/DP4/AMSA/AE_SI12.003/{date.strftime("%Y.%m.%d")}/AMSR_E_L3_SeaIce12km_V15_{date.strftime("%Y%m%d")}.hdf'
    ds = Dataset(fn, "r")
    siconc = np.array(ds.variables["SI_12km_NH_ICECON_DAY"])
    # is_missing = siconc == 28160
    # Seems to use an interpreted data value?
    is_missing = siconc == 110
    had_missing[is_missing] = had_missing[is_missing] + 1
    n_polehole_vals = np.sum(np.where(is_missing, 1, 0))

# had_missing.tofile('had_amsre_polehole.dat')
had_many_missing = had_missing >= 3
# had_many_missing.tofile('had_many_missing.dat')

# By inspection, it looks like we can declare the pole hole to
# be any place where there were at least three values of 110
# in 2005 "near the pole".  But...there are a couple of places
# *not* near the pole where there were also more than 3 missing
# value occurrences.  Exclude those.
# Indexes of "near pole": i: 295-322 j: 455-480

is_missing_nearpole = np.zeros(had_missing.shape, dtype=bool)

imin, imax = 295, 322
jmin, jmax = 455, 480
pole_count_subset = had_missing[jmin:jmax, imin:imax]
near_pole_subset = is_missing_nearpole[jmin:jmax, imin:imax]

near_pole_subset[pole_count_subset >= 3] = True
# is_missing_nearpole.tofile('is_missing_nearpole.dat')

# Compute the convolution of this so that the polehole mask incorporates
# one pixel out in any direction -- including diagonal -- from what we
# found in year (here 2005)
kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
convolved = convolve2d(
    is_missing_nearpole,
    kernel,
    mode="same",
    boundary="fill",
    fillvalue=0,
)
is_amsre_polehole = convolved > 0
# is_amsre_polehole.tofile('is_amsre_polehole.dat')
