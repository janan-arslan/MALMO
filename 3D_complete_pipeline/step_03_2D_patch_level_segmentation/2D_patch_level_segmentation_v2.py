
'''
======================================================================
 Title:                   3D Reconstruction Pipeline
                          2D Patch-Level Segmentation
 Author:                  Janan Arslan
 Creation Date:           06 JUL 2022
 Latest Modification:     30 AUG 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.0
======================================================================

This represents the segmentation step used in the 3D Reconstruction pipeline of the MALMO Project.

Here, the preprocessed and inpainted image can now undergo segmentation. In this code,
the patch size for the segmentation was reduced to 64x64, even though the UNet model
trained included training patches of 512x512. However, the model was trained with
patches extracted at the highest resolution. In the pipeline however, in to create
a manageable 3D model that isn't computationally expensive, the original images
were downsampled by 8. At a downsampling of 8, the spatial information contains more
surface area as compared to the surface area at the highest resolution (x40 magnification).
Therefore, to create an equivalent spatial resolution with a 8 downsampled images, patches
of size 64x64 were extracted instead.

Using the GUI interface, end-users select the input path where the WSI images are contained, select
the pre-trained UNet model path, select output path of interest, and then click "Segment BV" to
allow the pre-trained UNet model to segment the blood vessels within the WSI. An additional
post-processing step is required following this to remove false positives around the perimeter
of the WSI. 

This version of the code is designed for images, such as JPG, PNG, and TIFF.

GUI interface is preliminary. It will continue to be updated.

Sections where your input is required have been marked with **Modifiable section**.

'''



import numpy as np

import tifffile as tiff
import cv2
from keras.utils import normalize
import imutils

from tkinter import*
from tkinter import Menu
import tkinter as tk
import tkinter.filedialog as fdialog
from tkinter import PhotoImage
from tkinter import scrolledtext


from scipy import ndimage, misc
import os


from model import *
from data import *

import warnings
from PIL import Image, ImageFile

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



def bv_segmentation_task():
    global path, outPath

    for image_path in os.listdir(path):    
        ## Get input image from chosen file
        input_image = os.path.join(path, image_path)

        ## Separate original name from extension to assist with saving later
        file, ext = os.path.splitext(input_image)
        basename = os.path.basename(file)

        ## Set patch-size
        ''' Modifiable Section '''
        patch_size=64

        ## Read the input image## Read the input image
        image = cv2.imread(input_image)

        ## Initialize the segmentation image
        segm_img = np.zeros(image.shape)

        ## Initiate patch counter
        patch_num=1
        for i in range(0, image.shape[0], patch_size):   
            for j in range(0, image.shape[1], patch_size):  

                ## Extract a single patch from the image## Extract a single patch from the image
                single_patch = image[i:i+patch_size, j:j+patch_size]

                ## Normalize the pixel values of the patch
                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
                single_patch_shape = single_patch_norm.shape[:2]


                cv2.imwrite('./he_patch/0.tif', single_patch)

                ## Prepare the patch for testing the model
                testGene = testGenerator("./he_patch/",1)

                ## Get model predictions for the patch
                results = model.predict_generator(testGene,1,verbose=0)
                saveResult("./output",results)            

                single_patch_prediction = cv2.imread('./output/0_predict.tif')
                
                ## Accumulate the resized prediction back to the segmentation image
                segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
                    
                ## Print progress
                print("[PROCESSING IMAGE]: ", basename, "[PROCESSING PATCH NUMBER]: ", patch_num, " at position ", i,j)
                patch_num+=1

        print("[COMPLETED]: ", str(input_image))
        print("################################")

        ## Save image to selected output folder
        outwardPath = os.path.join(outPath, basename+'.png')
        cv2.imwrite(outwardPath, segm_img)



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
Label(text="2D Segmentation", font=('Arial', 15)).pack()

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
root.geometry('500x300')

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

''' Modifiable Section '''
''' Can change button name depending on biomarker of interest'''
btn_input_path = tk.Button(root, text="Segment BV", bg="gray", fg="black", command=bv_segmentation_task)
btn_input_path.pack(side=tk.LEFT, padx=10, pady=10)


# Configure grid weights to make the ScrolledText widget expand vertically
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create a PrintRedirector instance and redirect sys.stdout to it
print_redirector = PrintRedirector(output_text)
sys.stdout.write = print_redirector.write


# kick off the GUI
root.mainloop()
