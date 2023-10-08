'''
============================================================================
 Title:                   Extracting WSI Masks for Foreground Extraction
 Author:                  Janan Arslan
 Creation Date:           19 JUL 2023
 Latest Modification:     19 JUL 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.0
============================================================================

Pre-processing step used in the 3D Reconstruction pipeline of the MALMO Project.

This step is part of the artificant removal process. It involves simply creating
simple masks from the WSIs, with these masks later being used for the foreground
extraction process. 

The standalone code for the centering is provided here. But this code, along
with all other pre-processing steps, have been included in a GUI-based package
readily downloadable and accessible for ease of use.

Sections where your input is required have been marked with **Modifiable section**.


'''


import numpy as np
import cv2
from scipy import ndimage, misc
import os
from PIL import Image
import itertools



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    ## Turn strings into lists and number chunks
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def wsi_mask(path, outPath):
    ## Select path
    global path, outPath

    for root, dirs, files in os.walk(path):
        ## Remove ASCII sorting and replace with natural sorting
        files.sort(key=alphanum_key)
        print(files)
        for file1 in files:
            input_image = os.path.join(path, file1)

            ## Read image
            image = cv2.imread(input_image)

            ## Make a copy of the image
            original = image.copy()

            ## Convert image to grayscale for thresholding
            img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            ## Treshold the image to get contour to calculate center
            (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

          
            ## Define path where images will be saved and save images
            outwardPath = os.path.join(outPath, file1)
            cv2.imwrite(outwardPath, im_bw)


## Set paths and run method
''' Modifiable section'''
input_path = 'set/your/input/path'
output_path = 'set/your/output/path'

wsi_mask(input_path, output_path)

