# ============================================================================ #
# mmi_config.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Map Maker Iterative configuration options script. 
# ============================================================================ #


roach = 1

maps_to_build = ['DF'] # options: ['A', 'P', 'DF']

# platescale, band, psf
platescale = 5.9075e-6 # deg/um = 21.267 arcsec/mm
band = {1:500, 2:250, 3:350, 4:250, 5:250} # um
psf = {1:0.0150, 2:0.0075, 3:0.0105, 4:0.0075, 5:0.0075}[roach] # deg/beam
# arcsec: 54, 27, 38, 27, 27
# f/# = 3.87953; D=2.33 (w/ lyot stop)

# map pixel bin sizes to use (determines final resolution)
# beam is 0.01 degrees or 36 arcsec
pixels_per_beam = 2 # 2 in x and y = 4 pixel sampling of beam
# x_bin = y_bin = psf/pixels_per_beam/platescale # um
x_bin = y_bin = 0.0150/pixels_per_beam/platescale # bin size override

# KID to use as the reference for shift table calculations
kid_ref = {1:'0100', 2:'0001', 3:'0003', 4:'0001', 5:'0001'}[roach]

# source name for SkyCoord
source_name = 'RCW 92'

# data indices
# scan of object RCW 92
slice_i = {1:37_125_750, 2:37_144_000, 3:37_141_250, 4:37_138_750, 5:37_139_000}[roach] # RCW 92
cal_i   = slice_i + 516_000 # cal lamp
cal_f   = slice_i + 519_000

# common-mode loop iterations
ct_its = 2

# high pass filter cutoff frequency, in spatial scale
fc_high_scale = 40*psf # deg
print(f"fc_high_scale={fc_high_scale}")

# base data directories
dir_root   = '/media/player1/blast2020fc1/fc1/'   # control computer
dir_conv   = dir_root + "converted/"              # control computer

# data directories and files
dir_master = dir_conv + 'master_2020-01-06-06-21-22/'
dir_roach  = dir_conv
dir_targ   = dir_root + f'roach_flight/roach{roach}/targ/'
if roach == 1:
    dir_roach   += 'roach1_2020-01-06-06-22-01/'
    dir_targ    += 'Tue_Jan__7_00_55_50_2020/'
elif roach == 2:
    dir_roach   += 'roach2_2020-01-06-06-22-01/'
    dir_targ    += 'Tue_Jan__7_00_55_50_2020/'
elif roach == 3:
    dir_roach   += 'roach3_2020-01-06-06-21-56/'
    dir_targ    += 'Tue_Jan__7_00_55_51_2020/'
elif roach == 4:
    dir_roach   += 'roach4_2020-01-06-06-22-01/'
    dir_targ    += 'Tue_Jan__7_00_55_50_2020/'
elif roach == 5:
    dir_roach   += 'roach5_2020-01-06-06-22-01/'
    dir_targ    += 'Tue_Jan__7_00_55_51_2020/'

# detector layout file
file_layout = dir_root + f'map_making/detector_layouts/layout_roach{roach}.csv'

# KID rejects list
file_rejects = dir_root + f'map_making/kid_rejects/kid_rejects_roach{roach}.dat'

# common-mode file
file_commonmode = f'common_mode_roach{roach}.dat'

# log file
log_file = 'map_making.log'

# single KID maps output directory
dir_single = 'single_maps/'

# map aligning parameters output dir and file
dir_xform = 'align/'
# file_xform = dir_xform + f'align_roach{roach}.npy'
file_source_coords = dir_xform + f'source_coords_roach{roach}.npy'

# first unused KID channel (2469 total used channels)
kid_max = {1:380, 2:474, 3:667, 4:498, 5:450}[roach]

# TOD peak properties for find_peaks for source search
peak_s = 3   # prominence [multiple of noise]
peak_w = 100 # width [indices] 

# Noise finding highpass parameters
noise_cutoff_freq = 10 # Hz
noise_order       = 3
