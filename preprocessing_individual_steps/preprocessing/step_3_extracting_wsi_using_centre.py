'''
=======================================================
 Title:                   WSI Tissue Centering
 Author:                  Janan Arslan
 Creation Date:           07 AUG 2022
 Latest Modification:     20 AUG 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.0
=======================================================

Pre-processing step used in the 3D Reconstruction pipeline of the MALMO Project.

In our dataset, two tissue sections were given per slide. When cutting these into
their respective top and bottom tissues, there would be considerable differences
in terms of the location of tissues across the image. This led to
problems in our registration. Thus, our pre-processing pipeline involves
the centering of the tissue sections in preparation for registration.

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


def wsi_center(path, outPath):
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

            ## Invert the image  
            im_bw =~ im_bw

            ## Extract contours
            contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE,   cv2.CHAIN_APPROX_SIMPLE)
            
            ## Select the largest contour (i.e., the tissue)
            c = max(contours, key = cv2.contourArea)

            ## Using moments, get the centre of the tissue
            M = cv2.moments(c)

            cX = (int(M["m10"] / M["m00"]))
            cY = (int(M["m01"] / M["m00"]))

            ## Establish dimensions of the final image
            ''' Modifiable section'''
            dimX = 6000
            dimY = 5000

            ## Establish the starting point of bounding box
            X = int(cX - (dimX/2))
            Y = int(cY - (dimY/2))

            ## In the event X or Y is a negative, reset it to 0
            if X < 0:
                X = 0
            if Y < 0:
                Y = 0

            ## Crop the bounding box
            crop = original[Y:Y+dimY, X:X+dimX]
          
            ## Define path where images will be saved and save images
            outwardPath = os.path.join(outPath, file1)
            cv2.imwrite(outwardPath, crop)


## Set paths and run method
''' Modifiable section'''
input_path = 'set/your/input/path'
output_path = 'set/your/output/path'

wsi_center(input_path, output_path)

