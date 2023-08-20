'''
=======================================================
 Title:                   Add White Pixeled Border to WSI
 Author:                  Janan Arslan
 Creation Date:           22 AUG 2022
 Latest Modification:     20 AUG 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.0
=======================================================

Pre-processing step in the MALMO 3D reconstruction pipeline.

Designed to ensure all tissue sections of WSIs are centered within their
respective images. This step was utilized to ease the process of registration.

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


def add_border(path, outPath):

    for root, dirs, files in os.walk(path):
        ## Remove ASCII sorting and replace with natural sorting
        files.sort(key=alphanum_key)
        print(files)
        for file1 in files:
            input_image = os.path.join(path, file1)

            ## Read image
            im = cv2.imread(input_image)

            ## Extract dimensions of original image
            row, col = im.shape[:2]
            bottom = im[row-2:row, 0:col]
            mean = cv2.mean(bottom)[0]

            ## Specify the border size you want to add
            ''' Modifiable section '''
            bordersize = 2000

            ## Add the border, and include the colour of the border in value
            border = cv2.copyMakeBorder(
                im,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )


            ## Define path where images will be saved and save images
            outwardPath = os.path.join(outPath, file1)
            cv2.imwrite(outwardPath, border)

## Set paths and run method
''' Modifiable section '''
input_path = 'set/your/input/path'
output_path = 'set/your/output/path'

add_border(input_path, output_path)

