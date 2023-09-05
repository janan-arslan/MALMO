'''
======================================================================
 Title:                   3D Reconstruction Pipeline
                          Postprocessing
 Author:                  Janan Arslan
 Creation Date:           02 NOV 2022
 Latest Modification:     03 SEP 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.0
======================================================================

Post-processing steps used in the 3D Reconstruction pipeline of the MALMO Project.

During the 2D-level segmentation, the boundaries of the inpainted WSI images can be segmented
as blood vessels. Initially, it was assumed that residue epidermis could have been causing this,
but running 2D segmentation after trying to eliminate any residue border still resulted in
false positive segmentations around the border. This may be attributed to some of the blood vessel
images in which the UNet model was trained being void of any blood cells, only leaving the
perimeter of the blood vessel; this might mimic the appearance of the tissue edges.

Therefore, a postprocessing step was employed to clean up any residue before moving onto the
final 3D reconstruction. This was achieved similar to the foreground extraction method
introduced in the pre-processing step. However, the major difference here is we obtain
a threshold of the original WSI, but erode the edges as much as possible (as this will be used
in extracting only the blood vessels from the segmented images, leaving the borders behind).

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
from tkinter import scrolledtext
from scipy import ndimage, misc
import os
from PIL import Image
import itertools

from pathlib import Path
        

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    ## Turn strings into lists and number chunks
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def select_wsi():
    global path
    path = fdialog.askdirectory()

def select_segmentation():
    global segmPath
    segmPath = fdialog.askdirectory()

def select_output():
    global outPath
    outPath = fdialog.askdirectory()


def generate_mask():
    global path, mask_path

    ## Create a folder in the grandparent directory to store the predictions
    ## This will later be used for the inpainting portion.

    ## Original WSI path
    wsi_path = Path(path)

    ## Get the parent directory
    wsi_path_parent = wsi_path.parent

    ## Create a new folder in the grandparent directory
    ''' Modifiable section '''
    new_folder_name = "masks"

    ## Final storage for predictions
    mask_path = wsi_path_parent / new_folder_name

    ## Make the directory
    os.mkdir(mask_path)

    ## Confirm where the predictions will be stored
    print("Created directory:", mask_path)

    ## The total number of WSIs in the folder need to be counted

    ## List all files in the folder
    file_list = os.listdir(wsi_path)

    ## Count the number of files
    num_files = len(file_list)

    for root, dirs, files in os.walk(path):
        ## Remove ASCII sorting and replace with natural sorting
        files.sort(key=alphanum_key)
        print(files)
        for file1 in files:
            input_image = os.path.join(path, file1)

            ## Read image
            image = cv2.imread(input_image)

            ##--------Create a new threshold mask-------##

            ## Make a copy of the image
            original = image.copy()

            ## Convert image to grayscale for thresholding
            img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            ## Treshold the image to get contour to calculate center
            (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            ## Invert the image
            im_bw =~ im_bw

            ## Find the largest contour
            im_bw = im_bw.astype('uint8')

            ## Extract contours
            contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            ## Select the largest contour (i.e., the tissue)
            c = max(contours, key = cv2.contourArea)

            ## Draw the largest contour (i.e., the WSI), eliminating small marks
            cv2.drawContours(im_bw, [c],-1, color=(255, 255, 255), thickness=cv2.FILLED)

            ## Select kernel
            ''' Modifiable Section '''
            kernel = np.ones((3,3),np.uint8)

            ## Closing to fill in the tissue artefacts
            im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)

            ## Erosion to slightly remove the borders that have extraneous information
            ''' Modifiable Section '''
            im_bw = cv2.erode(im_bw,kernel,iterations = 25)

            ## Define path where masks will be saved and save masks
            maskPath = os.path.join(mask_path, file1)
            cv2.imwrite(maskPath, im_bw)

    print("Done!")
            
    return mask_path



def postprocess():
    ## Select path
    global segmPath, mask_path, outPath

    ## Begin post-processing. 

    for image_path1, image_path2 in zip(sorted(os.listdir(segmPath)),sorted(os.listdir(mask_path))):
        try:

            ## Get input image from chosen file
            input_image1 = os.path.join(segmPath, image_path1)
            input_image2 = os.path.join(mask_path, image_path2)
            print("[READING IMAGE]:", input_image1)
            
            ## Read images and resize to match
            ''' Modifiable Section '''
            img = cv2.imread(input_image1, 1)
            img = cv2.resize(img,(6000,5000))

            seg = cv2.imread(input_image2, 1)
            seg = cv2.resize(seg,(6000,5000))
            
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


            ## Define path where images will be saved and save images
            outwardPath1 = os.path.join(outPath, image_path1)
            cv2.imwrite(outwardPath1, fg)

            print("[PROCESSES COMPLETED FOR]:", image_path1)
        except:
            pass

    print("Done!")


## Exporting console text to scrolling text
class PrintRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)

## A logo resizer to ensure logo fits in the GUI interface

''' Modifiable Section '''            
def resize_image(image, new_width, new_height):
    return image.subsample(int(image.width() / new_width), int(image.height() / new_height))



## Initialize the window
root = Tk()

## Window info
root.title("MALMO 3D")

Label(text="MALMO 3D Pipeline", font=('Arial', 15)).pack()
Label(text="Postprocessing", font=('Arial', 15)).pack()

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
root.geometry('550x300')

# Create a Frame with a border
frame_with_border = tk.Frame(root, relief="solid", borderwidth=1)
frame_with_border.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# ScrolledText widget for displaying the output inside the Frame
output_text = scrolledtext.ScrolledText(frame_with_border, wrap=tk.WORD, height=10)
output_text.pack(fill=tk.BOTH, expand=True)

# Buttons
btn_input_path = tk.Button(root, text="WSI Path", bg="gray", fg="black", command=select_wsi)
btn_input_path.pack(side=tk.LEFT, padx=10, pady=10)

btn_output_path = tk.Button(root, text="Create Masks", bg="gray", fg="black", command=generate_mask)
btn_output_path.pack(side=tk.LEFT, padx=10, pady=10)

btn_output_path = tk.Button(root, text="Segmentation Path", bg="gray", fg="black", command=select_segmentation)
btn_output_path.pack(side=tk.LEFT, padx=10, pady=10)

btn_output_path = tk.Button(root, text="Output Path", bg="gray", fg="black", command=select_output)
btn_output_path.pack(side=tk.LEFT, padx=10, pady=10)

btn_preprocess = tk.Button(root, text="Postprocess", bg="gray", fg="black", command=postprocess)
btn_preprocess.pack(side=tk.LEFT, padx=10, pady=10)


## Configure grid weights to make the ScrolledText widget expand vertically
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

## Create a PrintRedirector instance and redirect sys.stdout to it
print_redirector = PrintRedirector(output_text)
sys.stdout = print_redirector

## Start GUI
root.mainloop()
