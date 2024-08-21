# ============================================================================ #
# mmi_map_lib.py
#
# James Burgoyne jburgoyne@phas.ubc.ca 
# CCAT Prime 2024
#
# Map Maker Iterative map making library. 
# ============================================================================ #


from typing import NamedTuple
import functools
import warnings
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.ndimage import shift, gaussian_filter

# import mmi_data_lib as dlib
import mmi_tod_lib as tlib



# ============================================================================ #
# logThis
def logThis(func):
    '''Decorator to mark func as one to be logged when run.
    '''

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.logThis = True  # Add an attribute to mark the function
    return wrapper



# ============================================================================ #
# sourceCoordsAzEl
@logThis
def sourceCoordsAzEl(source_name, lat, lon, alt, time):
    '''Tod of az/el coordinates of the source (telescope frame).

    source_name: (str) The source name for SkyCoord.
    lat, lon, alt, time: (1D arrays) Tods to define telescope location.
    '''

    # Define the coordinates of the source
    source = SkyCoord.from_name(source_name)

    # Define the location of the telescope (tod)
    telescope_location = EarthLocation(
        lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m)

    # Create an astropy Time object of Unix time array
    time = np.nan_to_num(time) # deals with any nan or inf
    time_array = Time(time, format='unix_tai') # timestamp w/ us

    # Altitude-Azimuth frame tod
    altaz_frame = AltAz(obstime=time_array, location=telescope_location)

    # Calculate the coordinates of source in el/az frame (tod)
    source_altaz = source.transform_to(altaz_frame)

    # Extract azimuth and elevation from the resulting AltAz object
    az = source_altaz.az.to(u.deg).value
    el = source_altaz.alt.to(u.deg).value
    
    # return as NamedTuple
    class SourceAzEl(NamedTuple): 
        az: np.ndarray
        el: np.ndarray
    return SourceAzEl(az, el)


# ============================================================================ #
# azElOffsets
@logThis
def azElOffsets(source_azel, az, el):
    '''Generate az/el offset tods from az/el pointing and source coords.

    source_azel: (SourceAzEl) Source az/el tods.
    az, el: (1D array) Pointing az/el tods.
    '''
    
    # generate offsets
    offset_az = source_azel.az - az
    offset_el = source_azel.el - el
    
    # determine correction to pointing
    # since boresight differs from pointing data
    cor_az = np.mean(offset_az)
    cor_el = np.mean(offset_el)
    offset_az -= cor_az
    offset_el -= cor_el
          
    # log.info(f"Boresight correction: az={cor_az}, el={cor_el}")
    
    return offset_az, offset_el


# ============================================================================ #
# offsetsTanProj
@logThis
def offsetsTanProj(x, y, platescale):
    '''Convert offsets in degrees to image plane in um.

    x, y: (1D array of floats) The offsets arrays in degrees.
    platescale: (float) Platescale in deg/um.
    '''
    
    from astropy.wcs import WCS
    
    w = WCS(naxis=2)
    w.wcs.crpix = [0, 0]   # center of the focal plane is tangent point
    w.wcs.crval = [0., 0.] # source is at center in offsets map
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [platescale, platescale]
    
    xp, yp = w.wcs_world2pix(x, y, 0)

    return xp, yp


# ============================================================================ #
# xyFromAb
def xyFromAb(ab, platescale, pixels_per_beam, psf):
    '''Convert ab detector offset coordinates [um] to xy shifts [pix].

    ab = {kid: (a_kid, b_kid)}
    '''
    
    # conversion factor
    T = platescale*pixels_per_beam/psf

    # apply conversion to ab from layout file
    xy = {
        f"{int(kid):04}": (T*ab[kid][0], T*ab[kid][1])
        for kid in ab.keys()
    }
    
    return xy


# ============================================================================ #
# genMapAxesAndBins
@logThis
def genMapAxesAndBins(x, y, x_bin, y_bin):
    '''Generate the 1D bin/edge arrays and 2D map arrays.

    x, y: (1D array; floats) x,y tods, e.g. az/el.
    x_bin, y_bin: (float) x/y bin size for map.
    '''

    # generate map bin arrays
    x_bins = np.arange(np.min(x), np.max(x), x_bin)
    y_bins = np.arange(np.min(y), np.max(y), y_bin)

    # generate map bin edge arrays
    x_edges = np.arange(np.min(x), np.max(x) + x_bin, x_bin)
    y_edges = np.arange(np.min(y), np.max(y) + y_bin, y_bin)

    # generate meshgrid 2D map bin arrays
    xx, yy = np.meshgrid(x_bins, y_bins)

    return xx, yy, x_bins, y_bins, x_edges, y_edges


# ============================================================================ #
# calcAst
def calcAst(combined_map, az, el, x_edges, y_edges):
    '''

    combined_map: (2D array of floats) Map including all KID data.
    az/el: (1D array of floats) The az/el offset tods, 
        already shifted for this KID.
    x_edges/y_edges: (1D array of floats) The map bin edges.
    '''

    # loop through each az/el sample
    # get the value associated to the bin for that az/el

    indices_x, indices_y = azelToMapPix(az, el, x_edges, y_edges)

    ast = combined_map[indices_y, indices_x]

    return ast


# ============================================================================ #
# commomMode
@logThis
def commomModeLoop(kids, dat_targs, Ff, dat_align_indices, 
                   roach, dir_roach, i_i, i_cal, i_f, 
                   az, el, x_edges, y_edges, source_xy, combined_map):
    '''Calculate common mode estimate.
    Computationally and I/O expensive.

    kid: (str) The KID number, e.g. '0001'.
    dat_targs: (tuple of 2 1D arrays of floats) Target sweep I and Q.
    Ff: (1D array of floats) Frequency axis for target sweep.
    dat_align_indices: (list of ints) Indices to align tods.
    roach: (int) The roach number.
    dir_roach: (string) The directory that roach data is in.
    i_i: (int) Desired data first index.
    i_cal: (int) First index of calibration lamp data.
    i_f: (int) Final index.
    combined_map: (2D array of floats) Map including all KID data.
    '''

    tod_sum = np.zeros(i_cal - i_i)

    for kid in kids:
            
        # get the calibrated df for this kid
        tod = tlib.getCalKidDf(kid, dat_targs, Ff, dat_align_indices, 
                          roach, dir_roach, i_i, i_cal, i_f)
        
        # remove astronomical signal estimate
        if combined_map is not None:
            Δaz, Δel = source_xy[kid]
            ast = calcAst(combined_map, az+Δaz, el+Δel, x_edges, y_edges)
            tod -= ast

        # add this tod to summation tod
        tod_sum += tod
            
    commonmode = tod_sum/len(kids)

    return commonmode


# ============================================================================ #
# buildSingleKIDMap
def buildSingleKIDMap(tod, x, y, x_edges, y_edges):
    '''Build a 2D map from time ordered data for a single KID.

    tod: (1D array of floats) The time ordered data (df).
    x/y: (1D arrays; floats) The time ordered positional data, e.g. az/el.
    x_edges/y_edges: (1D array of floats) The map bin edges.
    '''

    zz0, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    zz, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=tod)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        zz = np.divide(zz, zz0)
    # zz /= zz0

    zz[zz == 0] = np.nan

    return zz.T


# ============================================================================ #
# sourceCoords
def sourceCoords(xx, yy, zz):
    '''Find source (in image coords); smooth then find max pixel.
    
    xx, yy, zz: (2D arrays of floats) The image data (meshgrid style).
    '''
    
    # gaussian smoothing
    smoothing_kernel = 2 # in pixel coords
    nonan_array = np.where(np.isnan(zz), np.nanmedian(zz), zz)
    smoothed_array = gaussian_filter(nonan_array, sigma=smoothing_kernel)
    
    # identify source in smoothed map by max value (pixel coords)
    max_coords = np.unravel_index(np.argmax(smoothed_array), smoothed_array.shape) 
    
    # convert max point to image coords
    x_im, y_im = xx[max_coords], yy[max_coords]

    # im coords have origin at center
    return x_im, y_im


# ============================================================================ #
# sourceCoordsToPixShift
def sourceCoordsToPixShift(x_im, y_im, xx, yy):
    '''Convert source coords (az/el) to map pixel shift to (0,0).

    x_im/y_im: (float) Source coords in image (az/el).
    xx/yy: (2D array of floats) The image data (meshgrid).
    '''
    
    # bin sizes
    dx = np.abs(xx[0,1] - xx[0,0])
    dy = np.abs(yy[1,0] - yy[0,0])

    # calculate shift in pix coords
    # shift is negative of position
    # divide by bin size to convert to pix deltas
    Δx = -x_im/dx    
    Δy = -y_im/dy

    return Δx, Δy


# ============================================================================ #
# sourceCoordsToPixShift
def azelToMapPix(az, el, x_edges, y_edges):
    '''Convert az/el coords to map pix coords.

    az/el: (1D array of floats) Array of az/el coordinates.
    x_edges/y_edges: (1D array of floats) The map az/el bin edges.
    '''
    
    indices_x = np.searchsorted(x_edges, az, side='right') - 1
    indices_y = np.searchsorted(y_edges, el, side='right') - 1

    # Correct the indices if they go out of bounds
    indices_x = np.clip(indices_x, 0, len(x_edges) - 2)
    indices_y = np.clip(indices_y, 0, len(y_edges) - 2)

    return indices_x, indices_y


# ============================================================================ #
# nanaverage
def nanaverage(arr, weights=None, axis=None):
    '''Compute the [weighted] average of an array, ignoring NaN values.
    
    arr: (array_like) Input array containing numbers to average.
    weights: (array_like) Weights to compute a weighted average.
    axis: (int) Axis along which to compute the average.
    '''

    # Create a mask where NaN values are True
    mask = np.isnan(arr)
    
    # Replace NaN values with zeros for calculation if weights are provided
    if weights is not None:
        arr = np.where(mask, 0, arr)
    
    # Calculate the sum along the specified axis, ignoring NaN values
    total = np.nansum(arr, axis=axis)
    
    # Count the non-NaN values along the specified axis
    count = np.sum(~mask, axis=axis)
    
    # Handle the case where count is zero to avoid division by zero
    count[count == 0] = 1
    
    # Calculate the average
    if weights is None:
        avg = total / count.astype(float)
    else:
        avg = np.average(arr, weights=weights, axis=axis)
    
    return avg


# ============================================================================ #
# combineMaps
def combineMaps(kids, single_maps, shifts):
    '''Combine single KID maps to make a final map.
    Use a weighted average, where weights are std of each single map.

    kids: (list of strings) The KIDs identifiers.
    single_maps: (list of 2D arrays of floats) The single KID maps.
    shifts: (list of 2-tuples of ints) The map alignment shifts.
    '''

    # translate maps ([::-1] to deal with numpy reversed x/y?)
    zz_xformed = [
        shift(single_maps[kid], shifts[kid][::-1], cval=np.nan, order=0)
        for kid in kids]

    # combine maps
    # zz_combined = np.nanmean(zz_xformed, axis=0)
    zz_weights = [1 / np.nanstd(a) for a in zz_xformed]
    zz_combined = nanaverage(zz_xformed, axis=0, weights=zz_weights)

    return zz_combined


# ============================================================================ #
# combinedMapLoop
@logThis
def combineMapsLoop(kids, dat_targs, Ff, dat_align_indices, 
                    roach, dir_roach, i_i, i_cal, i_f, 
                    x, y, x_edges, y_edges, xx, yy, common_mode, 
                    save_singles_func=None, shifts=None):
    '''Calculate the combined map.
    Computationally and I/O expensive.

    kid: (str) The KID number, e.g. '0001'.
    dat_targs: (tuple of 2 1D arrays of floats) Target sweep I and Q.
    Ff: (1D array of floats) Frequency axis for target sweep.
    dat_align_indices: (list of ints) Indices to align tods.
    roach: (int) The roach number.
    dir_roach: (string) The directory that roach data is in.
    i_i: (int) Desired data first index.
    i_cal: (int) First index of calibration lamp data.
    i_f: (int) Final index.
    x/y: (1D arrays; floats) The time ordered positional data, e.g. az/el.
    x_edges/y_edges: (1D array of floats) The map bin edges.
    xx/yy: (2D array of floats) The image data (meshgrid).
    common_mode: (1D array of floats) The common mode tod.
    save_singles_func: (func) Handles saving out the single maps.
    shifts: (1D array of floats) Pixel shift to align maps.
    '''

    single_maps = {}
    shifts_source = {}
    source_xy = {}

    for kid in kids:

        # get the calibrated df for this kid
        tod = tlib.getCalKidDf(kid, dat_targs, Ff, dat_align_indices, 
                          roach, dir_roach, i_i, i_cal, i_f)
        
        # remove common mode
        tod -= common_mode

        # build the binned pixel map
        zz  = buildSingleKIDMap(tod, x, y, x_edges, y_edges)
        single_maps[kid] = zz

        # find the source's coords
        xy = sourceCoords(xx, yy, zz) # x_im, y_im
        source_xy[kid] = xy

        # find shift required to center source in map
        shifts_source[kid] = sourceCoordsToPixShift(xy[0], xy[1], xx, yy)

        # use source determined shifts if not given as input
        shifts = shifts_source if shifts is None else shifts

        # output single map to file
        if save_singles_func is not None:
            save_singles_func(kid, np.array([xx, yy, zz]))

    # create combined map
    combined_map = combineMaps(kids, single_maps, shifts)

    return combined_map, shifts_source, source_xy
    