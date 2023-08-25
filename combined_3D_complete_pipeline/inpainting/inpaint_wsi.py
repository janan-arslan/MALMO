'''
======================================================================
 Title:                   3D Reconstruction Pipeline
                          Inpainting
 Author:                  Janan Arslan
 Creation Date:           22 AUG 2022
 Latest Modification:     22 AUG 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 3.0
======================================================================

This represents the inpainting step used in the 3D Reconstruction pipeline of the MALMO Project.

This code provides a collective space for all components of the preprocessing. Using the
GUI interface, end-users select the input path where the WSI images are contained, select
the pre-trained UNet model path, select output path of interest, then click "Segment" to
allow the pre-trained UNet model to predict the epidermis regions within the WSI, and finally
select "Inpaint" so that the prediction mask can be used to inpaint the epidermis. This step
leaves a nice, cleaned WSI images, which will be much easier to run image registration on in
the next step. 

This version of the code is designed for images, such as JPG, PNG, and TIFF (marked as 'a').

GUI interface is preliminary. It will continue to be updated.

Sections where your input is required have been marked with **Modifiable section**.

'''

import numpy as np
import cv2

from tkinter import*
from tkinter import Menu
import tkinter as tk
import tkinter.filedialog as fdialog
from tkinter import PhotoImage
from tkinter import scrolledtext

from scipy import ndimage, misc
import os
import itertools

import warnings
from PIL import Image, ImageFile

from model import *
from data import *

import matplotlib.pyplot as plt

from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")




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


def select_model():
    global modelPath, model
    modelPath = fdialog.askopenfilename()
    ## Read in trained UNet model.
    ''' Modifiable section '''
    model = unet(modelPath)



## Data info
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')




def segmentation_task():
    global path, segm_path

    ## Create a folder in the grandparent directory to store the predictions
    ## This will later be used for the inpainting portion.

    ## Original WSI path
    wsi_path = Path(path)

    ## Get the parent directory
    wsi_path_parent = wsi_path.parent

    ## Create a new folder in the grandparent directory
    ''' Modifiable section '''
    new_folder_name = "predictions"

    ## Final storage for predictions
    segm_path = wsi_path_parent / new_folder_name

    ## Make the directory
    os.mkdir(segm_path)

    ## Confirm where the predictions will be stored
    print("Created directory:", segm_path)

    ## The total number of WSIs in the folder need to be counted

    ## List all files in the folder
    file_list = os.listdir(wsi_path)

    ## Count the number of files
    num_files = len(file_list)

    ## Run prediction on the original WSIs
    testGene = testGenerator(wsi_path, num_files)
    results = model.predict_generator(testGene, num_files, verbose=1)
    saveResult(segm_path, results)

    return segm_path

    

def wsi_inpainting():
    ## Select path
    global path, segm_path, outPath

    print("Checking directory:", segm_path)

    for image_path1, image_path2 in zip(os.listdir(path),os.listdir(segm_path)):
        ## Image 1 should be the WSI to be inpainted.
        ## Image 2 the prediction mask.
        input_image1 = os.path.join(path, image_path1)
        input_image2 = os.path.join(segm_path, image_path2)
        
        ## Read original WSI  
        background = Image.open(input_image1)

        ## Read the prediction from model
        ## Run an additional threshold, as the prediction can sometimes
        ## leave a blurry outline, impacting the final inpainting.
        ## The additional thresholding ensures nice, clean
        ## lines in the segmentation mask.
        ## Initially reading the prediction in OpenCV format for preliminary
        ## processing.
        
        segm = cv2.imread(input_image2, 0)

        (thresh, segm) = cv2.threshold(segm, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        ## Create a color image with the same dimensions as the grayscale image
        color_image = np.zeros((segm.shape[0], segm.shape[1], 3), dtype=np.uint8)

        ## Set all color channels (R, G, B) to the grayscale values
        color_image[:, :, 0] = segm  # Red channel
        color_image[:, :, 1] = segm  # Green channel
        color_image[:, :, 2] = segm  # Blue channel

        ## Convert background to transparent, then save
        tmp = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(color_image)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)

        ## Convert OpenCV image to PIL format
        sm = Image.fromarray(dst)  

        ## Resize prediction to fit the original WSI image
        target_size = background.size

        sm = sm.resize(target_size, Image.ANTIALIAS)

        ## Superimpose the two images for the final inpainting
        background.paste(sm, (0,0), sm)

        background = np.array(background)

        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        ## Define path where images will be saved and save images
        outwardPath = os.path.join(outPath, image_path1)

        cv2.imwrite(outwardPath, background)


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
Label(text="Inpainting", font=('Arial', 15)).pack()

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


## Buttons
btn_input_path = tk.Button(root, text="Input Path", bg="gray", fg="black", command=select_input)
btn_input_path.pack(side=tk.LEFT, padx=10, pady=10)


btn_output_path = tk.Button(root, text="Model Path", bg="gray", fg="black", command=select_model)
btn_output_path.pack(side=tk.LEFT, padx=10, pady=10)

btn_output_path = tk.Button(root, text="Output Path", bg="gray", fg="black", command=select_output)
btn_output_path.pack(side=tk.LEFT, padx=10, pady=10)


btn_input_path = tk.Button(root, text="Segment", bg="gray", fg="black", command=segmentation_task)
btn_input_path.pack(side=tk.LEFT, padx=10, pady=10)

btn_output_path = tk.Button(root, text="Inpaint", bg="gray", fg="black", command=wsi_inpainting)
btn_output_path.pack(side=tk.LEFT, padx=10, pady=10)

# Configure grid weights to make the ScrolledText widget expand vertically
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create a PrintRedirector instance and redirect sys.stdout to it
print_redirector = PrintRedirector(output_text)
sys.stdout.write = print_redirector.write


# kick off the GUI
root.mainloop()


