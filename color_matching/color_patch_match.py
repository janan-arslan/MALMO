'''
=======================================================
 Title:                   WSI Patch Color Matching
 Author:                  Janan Arslan
 Creation Date:           19 JUL 2023
 Latest Modification:     19 JUL 2023
 Version:                 2.0
=======================================================

Post-processing step following cycle-GAN implementation in the MALMO project.

This code is designed to ensure a uniformity in color amongst the patches
from a WSI. In some instances of cycle-GAN, the trained model may create patches
that have varying levels of color intensities. This code was designed to normalize
these color intensities. It also ensures that any white regions from the original
patch (i.e., prior to conversion) retains its original state. 

'''


import cv2
import numpy as np


def compute_histogram(image, mask=None, histSize=256, ranges=[0, 256]):
    image = np.float32(image)
    hist = cv2.calcHist([image], [0], mask, [histSize], ranges)
    hist /= np.sum(hist)
    return hist


def match_histogram(src_image, tar_image, white_thresh=245):
    # Convert the images from BGR to LAB color space
    matched = cv2.cvtColor(src_image, cv2.COLOR_BGR2Lab)
    target = cv2.cvtColor(tar_image, cv2.COLOR_BGR2Lab)

    # Create a mask for non-white pixels
    mask = (matched[:,:,0] < white_thresh) & (matched[:,:,1] < white_thresh) & (matched[:,:,2] < white_thresh)
    mask = mask.astype(np.uint8)  # Convert the mask to uint8

    # Process each channel separately
    for channel in range(3):
        # Compute the histograms
        src_hist = compute_histogram(matched[:,:,channel], mask=mask)
        tar_hist = compute_histogram(target[:,:,channel])

        # Create a lookup table
        table = np.zeros((1,256), dtype=np.uint8)

        # Compute the cumulative distribution functions
        src_cdf = np.cumsum(src_hist)
        tar_cdf = np.cumsum(tar_hist)

        # Create the lookup table (mapping from src_image to tar_image)
        j = 0
        for i in range(256):
            while tar_cdf[j] < src_cdf[i] and j < 255:
                j += 1
            table[0,i] = j

        # Apply the mapping to the non-white pixels in src_image
        matched_channel = matched[:,:,channel]
        matched[mask==1,channel] = cv2.LUT(matched_channel[mask==1].astype(np.uint8), table).flatten()

    # Convert the image back from LAB to BGR color space
    return cv2.cvtColor(matched, cv2.COLOR_Lab2BGR)



# Load source and target images (add your own paths here)
src_image = cv2.imread('image1.png')
tar_image = cv2.imread('image2.png')

# Resize where necssary
src_image = cv2.resize(src_image, (700,700))
tar_image = cv2.resize(tar_image, (700,700))

# Match histograms and save the output
matched_image = match_histogram(src_image, tar_image)
cv2.imwrite('matched_image.jpg', matched_image)
print("Done!")
