'''
======================================================================
 Title:                   3D Reconstruction Pipeline
                          3D Reconstruction
 Author:                  Janan Arslan
 Creation Date:           12 NOV 2022
 Latest Modification:     03 SEP 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.0
======================================================================

This represents the final 3D reconstruction saved as OBJ file.

GUI interface is preliminary. It will continue to be updated.

Sections where your input is required have been marked with **Modifiable section**.

'''


from skimage import io
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure

from tkinter import*
from tkinter import Menu
import tkinter as tk
import tkinter.filedialog as fdialog
from tkinter import PhotoImage
from tkinter import scrolledtext

def select_input():
    global path
    path = fdialog.askdirectory()

def select_output():
    global outPath
    outPath = fdialog.askdirectory()

def threed_reconstruction():
    global path, outPath
    ## Read the series of images for reconstruction
    im_collection = io.imread_collection(path+'/*.png') ## If the image is a TIF, use plugin='tifffile' at the end of this line

    ## Concatenate images for 3D recostruction
    im_3d = im_collection.concatenate()

    ## Two versions of verts, faces and norms.
    ## One for visualisation in Python.
    ## One to be saved in OBJ format.

    ## Transpose concatenation
    im_3d = im_3d.transpose(2,1,0)

    ## Verts, faces and norms for Python viewing
    verts, faces, norm, val = measure.marching_cubes_lewiner(im_3d, 5.0)

    ## Verts, faces and norms modified to suit OBJ save
    verts1, faces1, norm1, val1 = measure.marching_cubes_lewiner(im_3d, 5.0)

    ## The addition of the constant 1 to faces ensures the OBJ
    ## shows the complete 3D model. Otherwise, you have a weird 3D model
    ## appearing.
    ## The problem was that Python considered the verts as starting
    ## from 0 and MeshLab or other similar applications consider them as starting from 1.
    ## An alternative method to create a OBJ is to use ImageJ.
    ## Use Image sequence, then save it as PGM, then save it as Wavefront OBJ with a threshold
    ## of 1 and a resampling of 2. 
    faces1=faces1 +1

    total_surface = measure.mesh_surface_area(verts, faces)
    print(total_surface)

    print("Drawing the OBJ file and saving now")

    ''' Modifiable Section '''
    file_name = '/choose/your/file/name.obj'

    thefile = open(outPath+file_name, 'w')
    for item in verts1:
      thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

    for item in norm1:
      thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

    for item in faces1:
      thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  

    thefile.close()


    print("Showing the 3D in visvis version")

    import visvis as vv
    vv.mesh(np.fliplr(verts), faces, norm, val)
    vv.use().Run()

    print("Done!")


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
Label(text="3D Reconstruction", font=('Arial', 15)).pack()

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

btn_output_path = tk.Button(root, text="Output Path", bg="gray", fg="black", command=select_output)
btn_output_path.pack(side=tk.LEFT, padx=10, pady=10)

btn_output_path = tk.Button(root, text="3D Reconstruct", bg="gray", fg="black", command=threed_reconstruction)
btn_output_path.pack(side=tk.LEFT, padx=10, pady=10)


# Configure grid weights to make the ScrolledText widget expand vertically
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create a PrintRedirector instance and redirect sys.stdout to it
print_redirector = PrintRedirector(output_text)
sys.stdout.write = print_redirector.write


# kick off the GUI
root.mainloop()



