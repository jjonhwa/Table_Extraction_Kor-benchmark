import cv2
import os
import numpy as np

def detect_box(image, line_min_width = 15):
  gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  img_bin = cv2.threshold(gray_scale, 120, 225, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
  blur = cv2.GaussianBlur(img_bin, (3,3), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
  horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations = 1)
  horizontal_mask = cv2.dilate(horizontal_mask, horizontal_kernel, iterations = 1)

  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
  vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations = 1)
  vertical_mask = cv2.dilate(vertical_mask, vertical_kernel, iterations = 1)

  img_bin_final = horizontal_mask|vertical_mask

  ret, labels, stats, centroids = \
    cv2.connectedComponentsWithStats(~img_bin_final, connectivity = 8, ltype = cv2.CV_32S)
  return stats, labels