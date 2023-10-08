
'''
======================================================================
 Title:                   3D Reconstruction Pipeline
                          Interpolation
 Author:                  Janan Arslan
 Creation Date:           21 JUN 2022
 Latest Modification:     22 AUG 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.0
======================================================================

Interpolation portion of the MALMO 3D Reconstruction pipeline.

The interpolation was designed to fill in the gaps between whole slide images (WSIs).

In this code, you can specify how many interpolated images you want by changing the value
of n (currently set to 5 by default). You also have the option of running (1) whole image-level
interpolation (output_directory), and (2) patch-level interpolation (output_directory_2).

Interpolated images are also given a naming convention that follows immediately after the original
filename to maintain consistency in image sequence. 


Sections where your input is required have been marked with **Modifiable section**.

'''


import os
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interpn
import cv2
import numpy as np
from PIL import Image


''' Modifiable Section '''
## Directory paths
image_directory = './input'  ## Path to the folder containing input images
output_directory = './output/whole/image/level'  ## Output folder for whole image level interpolation
output_directory_2 = './output/patch/level'  ## Output folder for patch-based interpolation


''' Modifiable Section '''
## Set the number of interpolations between pairs
n = 5

''' Modifiable Section '''
## Patch parameters
patch_size = (512, 512)  # Specify desired patch size
stride = 256  # Based on step size when sliding the patch


def bwperim(bw, n=4):
    """
    Compute the perimeter of objects in binary images.
    
    Parameters:
    - bw: Binary image
    - n: Connectivity, can be 4 or 8.
    
    Returns:
    - Binary image of perimeter.
    """
    ## Checking for valid connectivity
    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    
    rows,cols = bw.shape
    
    ## Define directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))
    
    ## Shift image in each direction
    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    
    ## Get perimeter based on differences in each direction
    idx = (north == bw) & (south == bw) & (west  == bw) & (east  == bw)
    
    ## If 8-connectivity is used, consider diagonals
    if n == 8:
        ## Define diagonal directions
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        
        ## Shift image in each diagonal direction
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        
        ## Update perimeter definition to include diagonals
        idx &= (north_east == bw) & (south_east == bw) & (south_west == bw) & (north_west == bw)
    
    return ~idx * bw


def signed_bwdist(im):
    """
    Compute the signed distance transform of an image.
    
    Parameters:
    - im: Binary image
    
    Returns:
    - Signed distance transform of the image.
    """
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im



def bwdist(im):
    """
    Compute the distance transform of an image.
    
    Parameters:
    - im: Binary image
    
    Returns:
    - Distance transform of the image.
    """
    dist_im = distance_transform_edt(1-im)
    return dist_im



def interp_shape(top, bottom, slice_num, total_slices):
    """
    Interpolate between two binary images at a given slice number.
    
    Parameters:
    - top: Top image (binary)
    - bottom: Bottom image (binary)
    - slice_num: Slice number for which interpolation is required
    - total_slices: Total number of slices
    
    Returns:
    - Interpolated binary image.
    """
    ## Calculate interpolation precision
    precision = slice_num / (total_slices + 1)
    
    ## Compute signed distance transforms for both images
    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)
    
    ## Interpolate between the distance maps of the two images
    r, c = top.shape
    top_and_bottom = np.stack((bottom, top))
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))
    xi = np.c_[np.full((r*c),precision), xi]
    
    ''' Modifiable Section '''
    out = interpn(points, top_and_bottom, xi, method='linear', bounds_error=True, fill_value=0)
    out = out.reshape((r, c))
    out = out > 0
    return out



def interpolate_patches(image_1, image_2, patch_size, stride, n):
    """
    Perform patch-level interpolation between two images.
    
    Parameters:
    - image_1: First image
    - image_2: Second image
    - patch_size: Size of the patch (tuple of height and width)
    - stride: Stride for extracting patches
    - n: Number of interpolations
    
    Returns:
    - Interpolated image.
    """
    height, width = image_1.shape
    result = np.zeros_like(image_1)

    ## Iterate over the image in strides to get patches
    for i in range(0, height - patch_size[0] + 1, stride):
        for j in range(0, width - patch_size[1] + 1, stride):
            patch_1 = image_1[i:i + patch_size[0], j:j + patch_size[1]]
            patch_2 = image_2[i:i + patch_size[0], j:j + patch_size[1]]
            interpolated_patch = interp_shape(patch_1, patch_2, n, n)  ## Interpolate the two patches
            result[i:i + patch_size[0], j:j + patch_size[1]] = interpolated_patch

    return result



## Get a sorted list of all image filenames in the directory
all_images = sorted([img for img in os.listdir(image_directory) if img.endswith(('.jpg', '.png', '.tif'))])

## Check to ensure an even number of images for pairs
if len(all_images) % 2 != 0:
    raise ValueError("Odd number of images in directory. Please ensure you have pairs of images.")


## Process images in pairs whole image level
for i in range(0, len(all_images), 2):
    ## Load the two images
    image_1_path = os.path.join(image_directory, all_images[i])
    image_2_path = os.path.join(image_directory, all_images[i+1])
    image_1 = cv2.imread(image_1_path, 0)
    image_2 = cv2.imread(image_2_path, 0)

    ## Create interpolations between the pair
    for j in range(1, n+1):
        out = interp_shape(image_1, image_2, j, n)
        filename = f"interpolated_between_{all_images[i]}_and_{all_images[i+1]}_slice_{j}.tif"
        output_path = os.path.join(output_directory, filename)
        im = Image.fromarray((out*255).astype(np.uint8))
        im.save(output_path)



## Process images in pairs at a patch-level
for i in range(0, len(all_images), 2):
    # Load the two images
    image_1_path = os.path.join(image_directory, all_images[i])
    image_2_path = os.path.join(image_directory, all_images[i+1])
    image_1 = cv2.imread(image_1_path, 0)
    image_2 = cv2.imread(image_2_path, 0)

    ## Create interpolations between the pair
    for j in range(1, n+1):
        out = interpolate_patches(image_1, image_2, patch_size, stride, j)
        ''' Modifiable Section '''
        filename = f"patch_interpolated_between_{all_images[i]}_and_{all_images[i+1]}_slice_{j}.tif"
        output_path_2 = os.path.join(output_directory_2, filename)
        im = Image.fromarray((out*255).astype(np.uint8))
        im.save(output_path_2)
