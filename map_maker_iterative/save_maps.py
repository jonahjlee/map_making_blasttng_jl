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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map-dir",
                        help="Directory with map files. Opens most recent by default.",
                        default=None)
    parser.add_argument("-i", "--iterations", nargs='*',
                        help="Common-mode iterations to view. Default: highest iter. only.",
                        default=None, type=int)
    parser.add_argument("-l", "--linthresh",
                        help="Linear threshold for symmetric logarithmic scaling.",
                        default=0.01, type=float)
    parser.add_argument('-s', '--single-kid-maps', nargs='*',
                        help='Renders maps for single KIDs in selected iteration. Leave value empty to build all single maps.',
                        default=None, type=int)
    parser.add_argument('-ct', '--common-mode',
                        help='Produces the map of the common-mode for the iteration if data is present.',
                        action='store_true')
    return parser.parse_args()


def save_iter_plots(args, iter_num, map_dir):

    # ===== locate iteration directory ===== #
    iter_dir = os.path.join(map_dir, f'it_{iter_num}')


    # ===== load image data ===== #
    combined_map_path = os.path.join(iter_dir, 'combined_map.npy')
    blasttng_map = np.load(combined_map_path)
    blasttng_df = blasttng_map[2]  # signal strength, df


    # ===== combined linear & log maps ===== #
    norm = colors.SymLogNorm(linthresh=args.linthresh)
    plt.imshow(blasttng_df, cmap='viridis')
    plt.colorbar(label='DF')
    plt.title(f"combined map, it_{iter_num}.\nBuilt from folder: {map_dir}")
    lin_map_name = f'it_{iter_num}_combined_map.png'
    plt.savefig(os.path.join(map_dir, lin_map_name))
    print(f'Saved map {lin_map_name} to folder {map_dir}')
    plt.close()
    plt.imshow(blasttng_df, norm=norm, cmap='viridis')
    plt.colorbar(label='DF')
    plt.title(f"combined map, it_{iter_num}, log scale.\nBuilt from folder: {map_dir}")
    log_map_name = f'it_{iter_num}_combined_map_log.png'
    plt.savefig(os.path.join(map_dir, log_map_name))
    print(f'Saved map {log_map_name} to folder {map_dir}')
    plt.close()


    # ===== common-mode maps ===== #
    if args.common_mode:
        cmmap = np.load(os.path.join(iter_dir, 'common_mode_map.npy'), allow_pickle=True)
        plt.imshow(cmmap, cmap='viridis')
        plt.colorbar(label='DF')
        plt.title(f'common-mode, it_{iter_num}\nBuilt from folder: {map_dir}')
        map_name = name = f'it_{iter_num}_common_mode_map.png'
        plt.savefig(os.path.join(map_dir, map_name))
        print(f'Saved map {map_name} to folder {map_dir}')
        plt.close()


    # ===== single kid maps ===== #
    if args.single_kid_maps is not None:
        singles_dir = os.path.join(iter_dir, 'single_maps')
        out_dir = os.path.join(singles_dir, 'single_plots')
        os.makedirs(out_dir, exist_ok=True)

        if len(args.single_kid_maps) == 0:
            kid_maps = [fname for fname in os.listdir(singles_dir)
                        if not os.path.isdir(os.path.join(singles_dir, fname))
                        and fname.startswith('map_kid_')]
            print(f'Building maps for all {len(kid_maps)} KIDs.')
        else:
            kid_maps = [f'map_kid_{kid_id:04}.npy' for kid_id in args.single_kid_maps]
            print(f'Building maps for {len(kid_maps)} KIDs.')

        for fname in kid_maps:
            kid_map = np.load(os.path.join(singles_dir, fname), allow_pickle=True)
            plt.imshow(kid_map[2], cmap='viridis')
            plt.title(f'{fname}, {map_dir} it_{iter_num}')
            map_name = f'{fname[:-4]}.png'
            plt.savefig(os.path.join(out_dir, map_name))
            print(f'Saved map {map_name} to folder {out_dir}')
            plt.close()


if __name__ == "__main__":

    # ===== parse command-line args ===== #
    args = parse_args()


    # ===== determine best defaults ===== #
    if args.map_dir is None:
        dirs = [directory for directory in os.listdir()
                if os.path.isdir(directory)
                and directory.startswith('map_')
                and directory not in ignore_directories]
        latest_ctime = max(dirs, key=os.path.getctime)
        map_dir = latest_ctime
    else:
        map_dir = args.map_dir

    if args.iterations is None:
        subdirectories = os.listdir(map_dir)
        iters = [max([int(it[3]) for it in subdirectories if it.startswith('it_')])]
    else:
        iters = args.iterations


    # ===== save plots for selected iterations ===== #
    for iter_num in iters:
        save_iter_plots(args, iter_num, map_dir)