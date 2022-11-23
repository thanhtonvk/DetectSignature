import cv2
import numpy as np
from model import get_model
from preprocessing import img_preprocess
import os
import faiss
def get_mask(path):
    image = cv2.imread(path)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 38, 0])
    upper = np.array([145, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask==0] = (255, 255, 255)

    # Find contours on extracted mask, combine boxes, and extract ROI
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = np.concatenate(cnts)
    x,y,w,h = cv2.boundingRect(cnts)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = result[y:y+h, x:x+w]
    return ROI
