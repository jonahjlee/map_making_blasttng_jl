# ============================================================================ #
# save_maps.py
#
# Jonah Lee
#
# Save Maps
# Python script to create human-viewable (.png) maps from mmi.py output.
# Allows for creation of maps on remote SSH connection via CLI.
# ============================================================================ #

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

ignore_directories = ['map_maker_iterative']

if __name__ == "__main__":

    # ===== parse command-line args ===== #

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map-dir",
                        help="Directory with map files. Opens most recent by default.",
                        default=None)
    parser.add_argument("-i", "--iter-num",
                        help="Common-mode iteration to view. Default: highest iter.",
                        default=None, type=int)
    parser.add_argument("-o", "--offset",
                        help="Constant offset to add to map. Small values (~0.01) can improve color scaling.",
                        default=0, type=float)
    parser.add_argument("-l", "--linthresh",
                        help="Linear threshold for symmetric logarithmic scaling.",
                        default=0, type=float)
    args = parser.parse_args()

    # ===== determine best defaults ===== #

    if args.map_dir is None:
        dirs = [directory for directory in os.listdir("..")
                if os.path.isdir(directory) and directory.startswith('map_') and directory not in ignore_directories]
        latest_ctime = max(dirs, key=os.path.getctime)
        map_dir = latest_ctime
    else:
        map_dir = args.map_dir

    if args.iter_num is None:
        subdirectories = os.listdir(map_dir)
        iter_num = max([int(it[3]) for it in subdirectories if it.startswith('it_')])
    else:
        iter_num = args.iter_num

    # ===== load image data ===== #

    combined_map_path = os.path.join(map_dir, f'it_{iter_num}', 'combined_map.npy')

    blasttng_map = np.load(combined_map_path)
    blasttng_x_offset_um = blasttng_map[0]  # um offset (x) on sensor
    blasttng_y_offset_um = blasttng_map[1]  # um offset (y) on sensor
    blasttng_df = blasttng_map[2]  # signal strength, df

    # ===== plot and save results ===== #

    norm = colors.SymLogNorm(linthresh=0.01)

    plt.imshow(blasttng_df + args.offset, cmap='viridis')
    plt.colorbar(label='DF')
    plt.title(f"combined map, it_{iter_num}.\nBuilt from folder: {map_dir}")
    lin_map_name = f'it_{iter_num}_combined_map.png'
    plt.savefig(os.path.join(map_dir, lin_map_name))
    print(f'Saved map {lin_map_name} to folder {map_dir}')
    plt.close()

    plt.imshow(blasttng_df + args.offset, norm=norm, cmap='viridis')
    plt.colorbar(label='DF')
    plt.title(f"combined map, it_{iter_num}, log scale.\nBuilt from folder: {map_dir}")
    log_map_name = f'it_{iter_num}_combined_map_log.png'
    plt.savefig(os.path.join(map_dir, log_map_name))
    print(f'Saved map {log_map_name} to folder {map_dir}')
    plt.close()
