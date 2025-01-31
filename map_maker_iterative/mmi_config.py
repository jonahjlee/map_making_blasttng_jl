# ============================================================================ #
# mmi_config.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Map Maker Iterative configuration options script. 
# ============================================================================ #

from enum import Enum

roaches = [1]

class ScanPass(Enum):
    PASS_0 = 0
    PASS_1 = 1
    PASS_2 = 2
    ALL = 3

pass_to_map = ScanPass.PASS_0

maps_to_build = ['DF']  # options: ['A', 'P', 'DF']

# platescale, band, psf
platescale = 5.9075e-6  # deg/um = 21.267 arcsec/mm
band_dict = {1:500, 2:250, 3:350, 4:250, 5:250}  # um
psf_dict = {1:0.0150, 2:0.0075, 3:0.0105, 4:0.0075, 5:0.0075}  # deg/beam
# arcsec: 54, 27, 38, 27, 27
# f/# = 3.87953; D=2.33 (w/ lyot stop)

# map pixel bin sizes to use (determines final resolution)
# beam is 0.01 degrees or 36 arcsec
pixels_per_beam = 2 # 2 in x and y = 4 pixel sampling of beam
# x_bin = y_bin = psf/pixels_per_beam/platescale # um
x_bin = y_bin = 0.0150/pixels_per_beam/platescale # bin size override

# KID to use as the reference for shift table calculations
kid_ref_dict = {
    1: '0100',
    2: '0103',
    3: '0003',
    4: '0001',
    5: '0001',
}

# source name for SkyCoord
source_name = 'RCW 92'

# data indices (old, includes more of the start)
# scan of object RCW 92
# slice_i = {1:37_125_750, 2:37_144_000, 3:37_141_250, 4:37_138_750, 5:37_139_000}[roach] # RCW 92
# cal_i   = slice_i + 516_000 # cal lamp
# cal_f   = slice_i + 519_000

# data indices (new, starts at top of rectangular scan; see roach_slicing.ipynb)
slice_i_dict = {
    1: 37_134_600,
    2: 37_149_500,
    3: 37_150_150,
    4: 37_147_350,
    5: 37_147_800,
}

# index offset for start/end
# e.g. second pass on roach 2 should be sliced at [37_149_500+169450:37_149_500+340400]
# pass_indices = [0, 169450, 340400, 511900]
pass_indices = [0, 169450, 340400, 507100]  # override pass 3 to end before cal lamp

# calibration lamp used to normalize tods
# note: cal lap turns on before end of pass 3 so it should be cut short
cal_i_offset = 507100  # from slice_i
cal_f_offset = 3000  # from cal_i


# common-mode loop iterations
ct_its = 2

# base data directories
dir_root   = '/media/player1/blast2020fc1/fc1/'   # control computer
dir_conv   = dir_root + "converted/"              # control computer

# data directories and files
dir_master = dir_conv + 'master_2020-01-06-06-21-22/'

dir_targ_dict = {
    1: dir_root + 'roach_flight/roach1/targ/Tue_Jan__7_00_55_50_2020/',
    2: dir_root + 'roach_flight/roach2/targ/Tue_Jan__7_00_55_50_2020/',
    3: dir_root + 'roach_flight/roach3/targ/Tue_Jan__7_00_55_51_2020/',
    4: dir_root + 'roach_flight/roach4/targ/Tue_Jan__7_00_55_50_2020/',
    5: dir_root + 'roach_flight/roach5/targ/Tue_Jan__7_00_55_51_2020/',
}

dir_roach_dict = {
    1: dir_conv +'roach1_2020-01-06-06-22-01/',
    2: dir_conv +'roach2_2020-01-06-06-22-01/',
    3: dir_conv +'roach3_2020-01-06-06-21-56/',
    4: dir_conv +'roach4_2020-01-06-06-22-01/',
    5: dir_conv +'roach5_2020-01-06-06-22-01/',
}

# KID rejects list
file_rejects_dict = {
    1: dir_root + f'map_making/kid_rejects/kid_rejects_roach1.dat',
    2: dir_root + f'map_making/kid_rejects/kid_rejects_roach2.dat',
    3: dir_root + f'map_making/kid_rejects/kid_rejects_roach3.dat',
    4: dir_root + f'map_making/kid_rejects/kid_rejects_roach4.dat',
    5: dir_root + f'map_making/kid_rejects/kid_rejects_roach5.dat',
}

# common-mode file
file_commonmode = lambda roach: f'common_mode_roach{roach}.dat'

# log file
log_file = 'map_making.log'

# single KID maps output directory
dir_single = 'single_maps/'

# map aligning parameters output dir and file
dir_xform = 'align/'
# file_xform = dir_xform + f'align_roach{roach}.npy'
# file_source_coords = dir_xform + f'source_coords_roach{roach}.npy'

# first unused KID channel (2469 total used channels)
kid_max_dict = {1:380, 2:473, 3:667, 4:497, 5:450}

# TOD peak properties for find_peaks for source search
peak_s = 3   # prominence [multiple of noise]
peak_w = 100 # width [indices]
