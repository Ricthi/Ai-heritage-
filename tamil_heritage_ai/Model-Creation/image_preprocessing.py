import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import os

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

def preprocess_image(image_path_or_image):
    """
    Performs complete preprocessing (skew correction, line removal, denoising, adaptive thresholding).
    Returns the binarized image.
    """
    if isinstance(image_path_or_image, str):
        image = cv2.imread(image_path_or_image)
        if image is None:
            raise FileNotFoundError(f"Could not read image from {image_path_or_image}")
    else:
        image = image_path_or_image
        
    # Correct skew
    angle, rotated = correct_skew(image)
    
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(rotated, [c], -1, (255, 255, 255), 5)
        
    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(rotated, [c], -1, (255, 255, 255), 5)
        
    # Refresh gray after drawing contours
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    
    # Median filter for Salt and pepper/impulse noise
    filter1 = cv2.medianBlur(gray, 5)
    
    # Gaussian blur to smoothen out the image edges
    filter2 = cv2.GaussianBlur(filter1, (5, 5), 0)
    
    # Non-localized means for final denoising
    dst = cv2.fastNlMeansDenoising(filter2, None, 17, 9, 17)
    
    # Binarize image using adaptive thresholding
    th1 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return th1

if __name__ == "__main__":
    # Example usage:
    # final_img = preprocess_image('Original.jpg')
    # cv2.imwrite('ImagePreProcessingFinal.jpg', final_img)
    pass
