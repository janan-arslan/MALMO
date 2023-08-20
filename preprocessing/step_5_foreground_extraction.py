'''
=======================================================
 Title:                   Foreground Extraction
 Author:                  Janan Arslan
 Creation Date:           07 AUG 2022
 Latest Modification:     20 AUG 2022
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.0
=======================================================

Pre-processing step used in the 3D Reconstruction pipeline of the MALMO Project.

Using the mask generated as a result of Step 4, the mask acts as a 'map'
to which regions we want to extract as the foreground. The tissue section is then
lifted off of the original image. Once the foreground has been extracted,
the background will originally appear as black. To reinstate the original look,
but with a clean background, the black background is converted to white. 

The standalone code for the centering is provided here. But this code, along
with all other pre-processing steps, have been included in a GUI-based package
readily downloadable and accessible for ease of use.

Sections where your input is required have been marked with **Modifiable section**.

'''

import numpy as np
import cv2
from scipy import ndimage, misc
import os
import operator
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave


def forgeound_extraction(path_image, path_mask, path_output):
    ## Select path
    global path1, path2, outPath1, outPath2 
    path1 = path_image
    path2 = path_mask
    outPath1 = path_output


    for image_path1, image_path2 in zip(os.listdir(path1),os.listdir(path2)):
        ## Get input image from chosen file
        input_image1 = os.path.join(path1, image_path1)
        input_image2 = os.path.join(path2, image_path2)
        
        ## Read image and mask
        img = cv2.imread(input_image1, 1)
        seg = cv2.imread(input_image2, 1)
        
        ## Create fg/bg mask 
        seg_gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        _,fg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        _,bg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

        ## Convert mask to 3-channels
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

        ## cv2.bitwise_and to extract the region
        fg = cv2.bitwise_and(img, fg_mask)
        bg = cv2.bitwise_and(img, bg_mask)

        
        ## Create a white background free of image artefacts to paste the foreground
        bk = np.full(img.shape, 255, dtype=np.uint8)

        ## Combine foreground and background
        final = cv2.bitwise_or(fg, bk)


        ## Define path where images will be saved and save images
        outwardPath1 = os.path.join(outPath1, image_path1)
        cv2.imwrite(outwardPath1, fg)
                

## Set paths and run method
''' Modifiable section'''
input_path1 = 'set/your/input/images/path'
input_path2 = 'set/your/input/masks/path'
output_path = 'set/your/output/path'

forgeound_extraction(input_path1, input_path1, output_path)
