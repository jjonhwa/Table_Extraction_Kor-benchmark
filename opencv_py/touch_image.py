import cv2
import pandas as pd
import numpy as np

def image_scale(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    return thresh

def cut_image(scale_img) :
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
    detect_horizontal = cv2.morphologyEx(scale_img, cv2.MORPH_OPEN,
                                         horizontal_kernel, iterations = 2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for i in range(len(cnts)) :
      if i == 0 :
        continue
      first_line = cv2.boundingRect(cnts[i-1])[1]
      second_line = cv2.boundingRect(cnts[i])[1]
      # 파일을 자를 경우 800의 경계값의 오류가 있을 수 있음
      # 800보다 클 경우 파일을 자름
      if abs(first_line - second_line) >= 800 :
        start_line = second_line-5
        break
    clean = scale_img[start_line:, :]
    return start_line, clean

def remove_horizontal(scale_image) :
    clean = scale_image.copy()
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    detect_horizontal = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                         horizontal_kernel, iterations  = 2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts :
        cv2.drawContours(clean, [c], -1, 0, 3)
        
    return clean

def search_x(scale_cut_image) :
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    detect_vertical = cv2.morphologyEx(scale_cut_image, cv2.MORPH_OPEN,
                                      vertical_kernel, iterations = 3)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    x_list = []
    for i in range(len(cnts)) :
      x_list.append(list(cv2.boundingRect(cnts[i][0])))
    
    tmp = pd.DataFrame(x_list)
    max_x = np.max(tmp[0])
    min_x = np.min(tmp[0])
    return min_x, max_x

def remove_vertical(scale_image) :
    clean = scale_image.copy()
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    detect_vertical = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                       vertical_kernel, iterations = 3)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts :
        cv2.drawContours(clean, [c], -1,  0, 3)
        
    return clean

def dilate_and_erode(scale_image, dil_iterations = 5, erode_iterations = 5) :
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilate = cv2.dilate(scale_image, kernel, anchor = (-1, -1), iterations = dil_iterations)
    erode = cv2.erode(dilate, kernel, anchor = (-1, -1), iterations = erode_iterations)
    
    cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    return cnts