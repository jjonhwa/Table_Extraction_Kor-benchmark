import cv2
import pandas as pd
from opencv_py import detect_box
import numpy as np

def table_detect(image, table_location) :
  '''
  이미지 및 테이블 좌표를 입력하여 이미지에서 떼어내기
  return : 테이블 이미지 array
  Table_Location to Extract_Table_List
  '''
  extract_table_list = []
  for item in table_location :
    src = cv2.imread(image, cv2.IMREAD_COLOR)
    if item[0] > 3 :
      item[0] = item[0] - 3
    if item[1] > 3 :
      item[1] = item[1] - 3
    dst = src[item[1]:item[3], item[0]:item[2]].copy()
    extract_table_list.append(dst)
  return extract_table_list

def cell_detect(extract_table) :
  '''
  추출된 테이블 이미지에서 셀을 인식한다.
  return : 셀이 그려진 테이블이미지, 셀의 좌표
  Extract_Table to (Drawing_Extract_Table, Box_Of_Cell)
  w, h의 범위는 하이퍼파라미터로 각 테이블의 맞게 수정하도록 한다.
  '''
  box = []
  stats, labels = detect_box(extract_table, line_min_width = 15)
  for x,y,w,h,ares in stats[2:] :
    if 50<w<10000 and 30<h<10000 :
      cv2.rectangle(extract_table, (x,y), (x+w,y+h), (0,255,0), 2)
      if x-5 < 0 and y-5 < 0 :
        box.append([x,y,x+w+5,y+h+5])
      elif x-5 < 0 :
        box.append([x,y-5,x+w+5,y+h+5])
      elif y-5 < 0 :
        box.append([x-5,y,x+w+5,y+h+5])
      else :
        box.append([x-5,y-5,x+w+5,y+h+5])
  return extract_table, box

def fix_box(box) :
  '''
  박스를 행단위로 끊어서 새로운 박스로 만들어준다.
  1D LIST to 2D LIST
  '''
  new_box = []
  for i, tmp in enumerate(box) :
    if i == 0 :
      new_tmp = []
      new_tmp.append(tmp)
      before_tmp = tmp
      continue
    if tmp[-1] == before_tmp[-1] :
      new_tmp.append(tmp) 
      before_tmp = tmp
    else :
      new_box.append(new_tmp)
      new_tmp = []
      new_tmp.append(tmp)
      before_tmp = tmp
    if i+1 == len(box) :
      new_box.append(new_tmp)
  return new_box

def find_col_len(box) :
  '''
  2D LIST box에서 각 행의 최대 길이를 return값으로 받는다.
  모든 행을 같은 길이로 만들어주기 위해 사용
  '''
  col_len = 0
  for i in range(len(box)) :
    if i == 0 :
      tmp_len = len(box[i])
      col_len = tmp_len
      continue
    tmp_len = len(box[i])
    if tmp_len > col_len :
      col_len = tmp_len
  return col_len

def text_to_list(item, box, col_len) :
  '''
  실행하기 앞서 : 이 함수의 경우 Naver OCR을 기준으로 만들어졌다.
  다른 OCR 및 텍스트 json파일을 활용할 경우 수정 필요.
  item : 추출된 텍스트 json 형식
  box : extact_table에서의 셀의 좌표
  col_len : 행의 최대길이
  return : text list
  '''
  outer = []
  for i in range(len(box)) :
    box_row = box[i]
    out = []
    for j in range(len(box_row)) :
      inner = ''
      for k in range(len(item)) :
        x1 = item[k]['boundingPoly']['vertices'][0]['x']
        x2 = item[k]['boundingPoly']['vertices'][1]['x']
        y1 = item[k]['boundingPoly']['vertices'][0]['y']
        y2 = item[k]['boundingPoly']['vertices'][2]['y']
        if 'inferText' not in item[i].keys() :
          continue
        if x1 > box_row[j][0] and x2 < box_row[j][2] and y1 > box_row[j][1] and y2 < box_row[j][3] :
          inner = inner + item[k]['inferText'] + ' '
      inner = inner.strip()
      out.append(inner)
    while True :
      if len(out) < col_len :
        out.append('')
      if len(out) == col_len :
        break
    for it in out :
      outer.append(it)
  return outer
        
def make_dataframe(text_list, col_len) :
  '''
  추출된 text로 이루어진 list, 열의 길의를 입력값으로 받는다.
  return : dataframe
  '''
  arr = np.array(text_list)
  dataframe = pd.DataFrame(arr.reshape(-1, int(col_len)))
  return dataframe