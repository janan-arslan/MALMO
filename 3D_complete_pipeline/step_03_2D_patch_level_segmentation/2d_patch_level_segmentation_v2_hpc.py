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
    model = unet(modelPath)  # Assuming the 'unet' function can directly accept the model path
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
