'''
=======================================================
 Title:                   Foreground Extraction
 Author:                  Janan Arslan
 Creation Date:           07 AUG 2022
 Latest Modification:     20 AUG 2023
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
from PIL import Image


def color_converter(path, outPath):
    ## Select path
    global path, outPath

    for image_path in os.listdir(path):    
        ## Get input image from chosen file
        input_image = os.path.join(path, image_path)

        img = cv2.imread(input_image)

        ## Convert all black pixels to white pixels
        img[np.where((img==[0,0,0]).all(axis=2))] = [255,255,255]
  
      
        ## Define path where images will be saved and save images
        outwardPath = os.path.join(outPath, image_path)
        cv2.imwrite(outwardPath, img)


## Set paths and run method
''' Modifiable section'''
input_path = 'set/your/input/images/path'
output_path = 'set/your/output/path'

color_converter(input_path, output_path)

