'''
======================================================================
 Title:                   3D Reconstruction Pipeline
                          2D Patch-Level Segmentation (HPC version)
 Author:                  Janan Arslan
 Creation Date:           06 JUL 2022
 Latest Modification:     30 AUG 2023
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.2
======================================================================

This represents the segmentation step used in the 3D Reconstruction pipeline of the MALMO Project.

The difference between this and the original segmentation is that the original uses a GUI  
interface but does not use parallel processing (and thus much slower), whereas this version 
was designed to be executed on a high-performance cluster (HPC) such as the one at ICM or Jean Zay. 

Here, the preprocessed and inpainted image can now undergo segmentation. In this code,
the patch size for the segmentation was reduced to 64x64, even though the UNet model
trained included training patches of 512x512. However, the model was trained with
patches extracted at the highest resolution. In the pipeline, however, to create
a manageable 3D model that isn't computationally expensive, the original images
were downsampled by 8. At downsampling of 8, the spatial information contains more
surface area as compared to the surface area at the highest resolution (x40 magnification).
Therefore, to create an equivalent spatial resolution with 8 downsampled images, patches
of size 64x64 were extracted instead.

Sections where your input is required have been marked with **Modifiable section**.

'''

import numpy as np
import cv2
from keras.utils import normalize
import os
from model import *
from data import *
import warnings
from PIL import Image, ImageFile
import re
from multiprocessing import Pool

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def process_images(image_batch, model, outPath, path):
    for image_path in image_batch:
        input_image = os.path.join(path, image_path)
        file, ext = os.path.splitext(input_image)
        basename = os.path.basename(file)
        patch_size = 64
        image = cv2.imread(input_image)
        segm_img = np.zeros(image.shape)
        patch_num = 1
        for i in range(0, image.shape[0], patch_size):
            for j in range(0, image.shape[1], patch_size):
                single_patch = image[i:i + patch_size, j:j + patch_size]
                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
                single_patch_shape = single_patch_norm.shape[:2]
                cv2.imwrite('./he_patch/0.tif', single_patch)
                testGene = testGenerator("./he_patch/", 1)
                results = model.predict_generator(testGene, 1, verbose=0)
                saveResult("./output", results)
                single_patch_prediction = cv2.imread('./output/0_predict.tif')
                segm_img[i:i + single_patch_shape[0], j:j + single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
                print("[PROCESSING IMAGE]: ", basename, "[PROCESSING PATCH NUMBER]: ", patch_num, " at position ", i, j)
                patch_num += 1
        outwardPath = os.path.join(outPath, basename + '.png')
        cv2.imwrite(outwardPath, segm_img)


def bv_segmentation_task_parallel(path, modelPath, outPath):
    model = unet(modelPath)
    image_paths = [os.path.join(path, image) for image in os.listdir(path)]
    cpu_count = os.cpu_count()
    image_batches = [image_paths[i::cpu_count] for i in range(cpu_count)]
    with Pool(cpu_count) as pool:
        pool.starmap(process_images, [(batch, model, outPath, path) for batch in image_batches])


if __name__ == "__main__":
    input_path = input("Enter the input directory path: ").strip()
    model_path = input("Enter the model file path: ").strip()
    output_path = input("Enter the output directory path: ").strip()

    bv_segmentation_task_parallel(input_path, model_path, output_path)
