import numpy as np
import matplotlib.pyplot as plt
import pygetdata as gd
import subprocess
import os
from scipy.ndimage import shift
from sys import getsizeof
import pickle


def targ(chan, dir):
    '''resonant frequency calibration sweep for given chan'''
    # assumes global dir_data and dir_targ

    # get list of targ files 
     # sort and remove bottom 4 (not relevant) files
    targ_file_list = sorted(os.listdir(dir))[:-4] 
    # load all targ files data
    dat = np.array([
        np.fromfile(f"{dir}/{f}", dtype = '<f')
        for f in targ_file_list
    ])
    
    # filter data to I and Q data for given channel
    # I = dat[:,chan,1]
    # Q = dat[:,chan,2]
    I = dat[::2,chan]
    Q = dat[1::2,chan]
    
    return I,Q

def Δfx_grad(Q, I, Qf, If):
    '''Calculate Δfx from 'gradient' method
    I: I(t): timestream S21 real component
    Q: Q(t): timestream S21 imaginary component
    If: I(f): frequency sweep S21 real component
    Qf: Q(f): frequency sweep S21 imaginary component'''
    
    dIfdf = np.diff(If)/1e3        # Δx is const. so Δy=dI/df
    dQfdf = np.diff(Qf)/1e3        # /1e3 for units
    dIfdff0 = dIfdf[len(dIfdf)//2] # dI(f)/df at f0
    dQfdff0 = dQfdf[len(dQfdf)//2] # assume f0 is centre index
    I_n = I - np.mean(I)           # centre values on 0
    Q_n = Q - np.mean(Q)           #
    
    den = dIfdff0**2 + dQfdff0**2  # 
    
    numx = ((I_n*dIfdff0 + Q_n*dQfdff0))
    Δfx = numx/den
    
    numy = ((Q_n*dIfdff0 - I_n*dQfdff0))
    Δfy = numy/den
    
    return Δfx


def view_data():
    m_diff = np.diff(master_time)
    r_diff = np.diff(roach_time)
    print('RA & DEC shape: ', RAm.shape, DECm.shape)
    print('Master time: ')
    print(f"    min: {np.min(master_time)}, max: {np.max(master_time)}, avg. diff: {np.mean(m_diff)}, clock speed: {1/np.mean(m_diff)}")
    print('Detector time: ')
    print(f"    min: {np.min(roach_time)}, max: {np.max(roach_time)}, avg. diff: {np.median(r_diff)}, clock speed: {1/np.median(r_diff)}")
    return


#populate map

def fill_coverage_map():
    Z = np.zeros((len(d_pix),len(r_pix)))
    for i in range(len(d_pix)):
        for j in range(len(r_pix)): #loop over pixel map
            det_inds = (RA >= r_pix[j]) & (RA < (r_pix[j] + ra_bin)) & (DEC >= d_pix[i]) & (DEC < (d_pix[i] + dec_bin)) # bining RA and DEC into pixel
            pix_val = det_inds.sum() #counts ra&dec for each pixel
            Z[i,j] = pix_val 
    return Z

def fill_map(datastream, kid_n):
    """datastream : PH or AMPS or DF"""
    Z = np.zeros((len(d_pix),len(r_pix)))
    for i in range(len(d_pix)):
        for j in range(len(r_pix)): #loop over pixel map
            det_inds = (RA >= r_pix[j]) & (RA < (r_pix[j] + ra_bin)) & (DEC >= d_pix[i]) & (DEC < (d_pix[i] + dec_bin)) # bining RA and DEC into pixel
            det_vals = datastream[kid_n, det_inds] #data values at each pixel from ts
            pix_val = np.nanmean(det_vals)
            Z[i,j] = pix_val
    return Z

def standardize_images(listofimages):
    "min-max normalization, 0-1"
    return [(image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image)) for image in listofimages]

def flip_images(listofimages):
    def cond1(im):
        #condition for flip, tbd
        return np.nanmedian(im) > 0.5
    return [-1*image + 1 if cond1(image) else image for image in listofimages]

def find_brightest_pixel(image):
    # Find the indices of the maximum value in the 2D array
    indices = np.unravel_index(np.nanargmax(image, axis=None), image.shape)
    return indices

def remove_kid(removekids):
    "enter list like XXXX, i.e [0110, 0555]"
    k_remove_index = []
    for k in removekids:
        try:
            k_remove_index.append(kid_nums.index(k))
            print(f"Removing kid: {removekids}")
        except ValueError:
            print(f'kid{k} already removed')

    for k in k_remove_index:
        kid_nums.pop(k) if k<len(kid_nums)-1 else kid_nums.pop()
        single_channel_maps.pop(k) if k<len(single_channel_maps)-1 else single_channel_maps.pop()
    return


def align_images(images, fit_func):
    # Find the brightest pixel in the reference image
    reference_image = images[0]
    #brightest_pixel_ref = find_brightest_pixel(reference_image)
    #reference_center = fit_gaussian(reference_image)
    reference_center = fit_func(reference_image)

    # Initialize a list to store aligned images
    aligned_images = [reference_image]

    # Iterate through the rest of the images and align them
    shift_amounts = [np.array([0,0])]
    for i in range(1, len(images)):
        current_image = images[i]

        # Find the brightest pixel in the current image
        current_center = fit_func(current_image)
        #print(f"current center : {current_center}")

        # Calculate the shift needed to align the current image with the reference image
        shift_amount = np.array(reference_center) - np.array(current_center)
        #print(f"shift amount : {shift_amount}")

        # Shift the current image
        aligned_image = shift(current_image, shift_amount, cval=np.nan, order=0)

        # Append the aligned image to the list
        aligned_images.append(aligned_image)
        shift_amounts.append(shift_amount)

    return aligned_images, shift_amounts

def show_list_of_images(images, ts_type):
    fig, axs1 = plt.subplots(1, len(images), sharex=True, sharey=True, figsize=(20,4))
    for i,ax in enumerate(axs1):
        a = axs1[i].pcolor(images[i])
        axs1[i].set_title(f'kid{kid_nums[i]}')
        fig.colorbar(a)
    axs1[0].set_ylabel(f"{ts_type}")

    fig.tight_layout()
    plt.show()
    return

def show_scan_area():
    plt.scatter(RAm[7000000:7972200], DECm[7000000:7972200])
    plt.scatter(RAm[start:stop], DECm[start:stop])
    plt.show()

if __name__ == "__main__":

    #locations and folder
    dir_base = '/home/triv/Desktop/localBLASTTNG/'
    dir_master = dir_base + 'master_2020-01-06-06-21-22'
    dir_roach3 = dir_base + 'roach3_2020-01-06-06-21-56'
    dir_targ = dir_base + 'targ_sweep/Mon_Jan__6_06_00_34_2020'

    roach_num = 3 #using roach 3 rn

    mode_dirfile = gd.RDONLY
    data_master = gd.dirfile(dir_master, mode_dirfile)
    data_roach3 = gd.dirfile(dir_roach3)

    #getting data
    RAm = data_master.getdata(b'RA')
    DECm = data_master.getdata(b'DEC')
    master_time = data_master.getdata(b'TIME') + data_master.getdata(b'TIME_USEC')*1e-6
    roach_time = data_roach3.getdata(b'ctime_built_roach3')

    #setting scan area from RA and DEC master arrays
    start = int(7.85e6)
    stop = int(7.9722e6)

    RA = RAm[start:stop]
    DEC = DECm[start:stop]

    #Scaling Master and Detector arrays by time
    DETinds = np.array([np.searchsorted(roach_time, t, side='left') for t in master_time]) #finds closest index in roach time for each master time
    DETinds[DETinds == len(roach_time)] = len(roach_time) - 1 # fixed max index bug from np.searchsorted

    print(f"start: {start}, stop: {stop}")
    print("Master: ", RA.shape, DEC.shape) 

    np.savez(f"master_data_roach{roach_num}.npz", RA=RA, DEC=DEC)

    kid_nums_big = [f"{k:04}" for k in range(0,680)]
        
    kid_nums_split = np.array_split(kid_nums_big, 20)
    for kn, kid_nums in enumerate(kid_nums_split):
        
        #[f"{k:04}" for k in range(0,680)]
        #detectors
        many_phase_vals = []
        many_absIQ_vals = []
        many_Δf_vals = []
        for n,k_num in enumerate(kid_nums):
            I_vals = data_roach3.getdata(f"i_kid{k_num}_roach3".encode('UTF-8'))
            Q_vals = data_roach3.getdata(f"q_kid{k_num}_roach3".encode('UTF-8'))
            If, Qf = targ(int(k_num) ,dir_targ)
            many_absIQ_vals.append(np.sqrt(I_vals**2 + Q_vals**2)[DETinds][start:stop])
            many_phase_vals.append(np.arctan2(Q_vals, I_vals)[DETinds][start:stop])
            many_Δf_vals.append(Δfx_grad(Q_vals, I_vals, Qf, If)[DETinds][start:stop])

            del I_vals
            del Q_vals
            
        with open(f"detector_timestream_{kn}_ts.pickle", "wb") as f:
            pickle.dump(many_phase_vals, f)
            pickle.dump(many_absIQ_vals, f)
            pickle.dump(many_Δf_vals, f)
            
        print('finished chunk')
        
        del many_absIQ_vals
        del many_phase_vals
        del many_Δf_vals
        
    print('Done')
