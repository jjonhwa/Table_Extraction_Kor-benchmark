import cv2 
import pandas as pd
import numpy as np

def preprocess_image(contour) :
    final_list = []
    for c in contour : 
        final_list.append(list(cv2.boundingRect(c)))
        
    final_data = pd.DataFrame()
    for i in range(len(final_list)) :
        new_row = final_list[i]
        new_row = pd.DataFrame(new_row).T
        
        final_data = pd.concat([final_data, new_row])
        
    final_data.reset_index(drop = True, inplace = True)
    final_data.columns = ['x', 'y', 'w', 'h']
    
    tmp = final_data.groupby('y').agg({'h' : 'max'})
    temp = tmp.reset_index()
    
    
    drop_list = []
    for i in range(len(temp)) :
        if i == 0 :
            continue
        if abs(temp['y'][i-1] - temp['y'][i]) <= 10 :
            if temp['h'][i-1] >= temp['h'][i] :
                drop_list.append(i)
            else :
                drop_list.append(i-1)
                
    temp = temp.drop(drop_list, axis = 0)
    temp.reset_index(drop = True, inplace = True)
    
    drop_list = []
    for i in range(len(temp)) :
        if i == 0 :
            continue
        if abs(temp['y'][i-1] - temp['y'][i]) <= 15 :
            if temp['h'][i-1] + temp['y'][i-1] >= temp['h'][i] + temp['y'][i] :
                drop_list.append(i)
            else :
                drop_list.append(i-1)
    temp = temp.drop(drop_list, axis = 0)
    temp.reset_index(drop = True, inplace = True)

    drop_list = []
    for i in range(len(temp)) :
        if i == 0 :
            continue
        if abs(temp['y'][i-1] - temp['y'][i]) <= 25 :
            if temp['h'][i-1] + temp['y'][i-1] >= temp['h'][i] + temp['y'][i] :
                drop_list.append(i)
            else :
                drop_list.append(i-1)
    temp = temp.drop(drop_list, axis = 0)
    temp.reset_index(drop = True, inplace = True)
    
    final = pd.merge(temp, final_data)
    return final

def draw_line(image, contour, data, min_x, max_x) :
    new_x_line = []
    new_y_line = []

    for c in contour :
        x, y, w, h = cv2.boundingRect(c)
        
        for i in range(len(data)) :
            if x == data['x'][i] and y == data['y'][i] and \
                w == data['w'][i] and h == data['h'][i] :
                area = cv2.contourArea(c)
                
                if area > 40 :
                    ROI = image[y:y+h, x:x+w]
                    ROI = cv2.GaussianBlur(ROI, (7,7), 0)
                    
                    cv2.line(image, (min_x, y+h-5), (max_x, y+h-5), (0, 0, 0), 3)
                    
    return image