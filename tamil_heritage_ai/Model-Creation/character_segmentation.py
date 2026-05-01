import cv2
import numpy as np
import imutils
import os

def segment_characters(image_path_or_image, output_dir="Images"):
    """
    Segments characters from a binarized (preprocessed) image.
    Extracts each character as a region of interest (ROI) and saves them in the output directory.
    Returns the original image with bounding boxes drawn, and a list of cropped ROIs.
    """
    if isinstance(image_path_or_image, str):
        image = cv2.imread(image_path_or_image)
        if image is None:
            raise FileNotFoundError(f"Could not read image from {image_path_or_image}")
    else:
        # Work on a copy so we don't modify the input array if it's reused
        image = image_path_or_image.copy()
        
    # We expect the input to be BGR format for drawing colored boxes, but works with grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh1, None, iterations=2)
    
    cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    # Sort contours by coordinates
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1])
    
    orig = image.copy()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    i = 0
    rois = []
    for cnt in sorted_ctrs:
        # Check the area of contour, if it is very small ignore it
        if cv2.contourArea(cnt) < 200:
            continue
            
        # Filtered contours are detected
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Taking ROI of the contour
        roi = image[y:y+h, x:x+w]
        
        # Mark them on the image
        cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save your contours or characters
        if output_dir:
            roi_path = os.path.join(output_dir, f"roi{i}.png")
            cv2.imwrite(roi_path, roi)
            
        rois.append(roi)
        i += 1
        
    return orig, rois

if __name__ == "__main__":
    # Example usage:
    # orig_with_boxes, rois = segment_characters("ImagePreProcessingFinal.jpg", output_dir="Images")
    # cv2.imwrite("box.jpg", orig_with_boxes)
    pass
