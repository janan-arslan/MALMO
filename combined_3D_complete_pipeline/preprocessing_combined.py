'''
======================================================================
 Title:                   3D Reconstruction Pipeline Preprocessing
 Author:                  Janan Arslan
 Creation Date:           21 AUG 2023
 Latest Modification:     21 AUG 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 1.0a
======================================================================

Pre-processing steps used in the 3D Reconstruction pipeline of the MALMO Project.

This code provides a collective space for all components of the preprocessing. Using the
GUI interface, end-users select the input path where the images are contained, the
output path of interest, and then select the preprocessing button for completion.

This version of the code is designed for images, such as JPG, PNG, and TIFF (marked as 'a').

Future versions will include being able to automatically select Bio-Formats, such as
.MRXS.

GUI interface is preliminary. It will continue to be updated. 

Sections where your input is required have been marked with **Modifiable section**.

'''


import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from tkinter import*
from tkinter import Menu
import tkinter.filedialog as fdialog
import tkinter as tk
from tkinter import PhotoImage
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


def select_input():
    global path
    path = fdialog.askdirectory()


def select_output():
    global outPath
    outPath = fdialog.askdirectory()


def preprocess_comb():
    ## Select path
    global path, outPath

    for root, dirs, files in os.walk(path):
        ## Remove ASCII sorting and replace with natural sorting
        files.sort(key=alphanum_key)
        print(files)
        for file1 in files:
            input_image = os.path.join(path, file1)

            ## Read image
            im = cv2.imread(input_image)

            ##--------Add Border-------##

            ## Extract information regarding original image
            row, col = im.shape[:2]
            bottom = im[row-2:row, 0:col]
            mean = cv2.mean(bottom)[0]

            ## Specify the border size you want to add
            ''' Modifiable Section '''
            bordersize = 2000

            ## Add the border, and include the colour of the border in value
            ''' Modifiable Section '''
            border = cv2.copyMakeBorder(
                im,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )


            ##--------Extract WSI using Center-------##
            
            ## Make a copy of the image
            original = border.copy()

            ## Convert image to grayscale for thresholding
            img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            ## Treshold the image to get contour to calculate center
            (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            ## Invert the image  
            im_bw =~ im_bw

            ## Extract contours
            contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE,   cv2.CHAIN_APPROX_SIMPLE)[-2:]
            
            ## Select the largest contour (i.e., the tissue)
            c = max(contours, key = cv2.contourArea)

            ## Using moments, get the centre of the tissue
            M = cv2.moments(c)

            cX = (int(M["m10"] / M["m00"]))
            cY = (int(M["m01"] / M["m00"]))

            ## Establish dimensions of the final image
            ''' Modifiable Section '''
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
            

            ##--------Generating Masks for Foreground Extraction-------##

            ## Make a copy of the image
            original = crop.copy()

            ## Convert image to grayscale for thresholding
            img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            ## Treshold the image to getforeground
            (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            ## Invert to get final mask
            im_bw =~ im_bw
            

            ##--------Foreground Extraction-------##

            img = original
            seg_gray = im_bw

            ## Create fg/bg mask 
            _,fg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            _,bg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

            ## convert mask to 3-channels
            fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

            ## cv2.bitwise_and to extract the region
            fg = cv2.bitwise_and(img, fg_mask)
            bg = cv2.bitwise_and(img, bg_mask)


            ##--------Adding New Clean Background-------##


            ''' Modifiable Section '''
            fg[np.where((fg==[0,0,0]).all(axis=2))] = [255,255,255]



            ##--------Save Image-------##
            
            ## Define path where images will be saved and save images
            outwardPath = os.path.join(outPath, file1)
            cv2.imwrite(outwardPath, fg)

            print("Processed Image:", file1)


## A logo resizer to ensure logo fits in the GUI interface

''' Modifiable Section '''            
def resize_image(image, new_width, new_height):
    return image.subsample(int(image.width() / new_width), int(image.height() / new_height))



## Initialize the window
root = Tk()

## Window info
root.title("MALMO 3D")

Label(text="MALMO 3D Pipeline", font=('Arial', 15)).pack()
Label(text="Preprocessing", font=('Arial', 15)).pack()

#### Logo
##logo = PhotoImage(file="./malmo_vector_logo.png")
##
##''' Modifiable Section '''
##logo_label = tk.Label(root, image=logo)
##resized_logo = resize_image(logo, 110, 75)  # Adjust the subsampling factors as needed
##
##logo_label = tk.Label(root, image=resized_logo)
##logo_label.pack()

## Frame/window size
root.geometry('200x300')


## Buttons
btn = tk.Button(root, text="Preprocess", bg="gray", fg="black", command=preprocess_comb)
btn.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

btn = tk.Button(root, text="Output", bg="gray", fg="black", command=select_output)
btn.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")


btn = tk.Button(root, text="Input", bg="gray", fg="black", command=select_input)
btn.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

## Start GUI
root.mainloop()
