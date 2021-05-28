# Naver OCR & CascadeTabNet을 활용한 한국어 Table Extraction

영어의 경우 Deep Learning을 활용한 Table Detection(with Structure Recognition) - Cell Detection - Text Extract를 바탕으로 Table Extraction에 대한 연구가 진행되었으나 한국어 Table Extraction에 대한 연구 및 오픈소스가 없었다.
이에 대하여 한국어 Table Extraction을 진행하기 위해 Table Detection, Cell Detection, Text Extraction에 대한 오픈 소스를 검색 및 탐구하였고 이를 알맞게 추출하기 위하여 오픈 소스를 수정하여 최종 Table Extraction을 하였다.
전 과정은 PDF 원하는 페이지 추출 > PDF to Image > Image Table Detection > Image Cell Dectection > Text Extraction > Make DataFrame 순으로 진행하였으며 현 git에는 Table Extraction에 대한 과정만 담았다.

## 진행과정
모든 과정은 Colab에서 진행하였습니다. (GPU 환경)  
[1. CascadeTabNet을 활용한 Table Detection](#1-cascadetabnet을-활용한-table-detection)  
[2. OpenCV를 활용한 Cell Detection](#2-opencv를-활용한-cell-detection)  
[3. Naver OCR를 활용한 Text Extraction](#3-naver-ocr를-활용한-text-extraction)  
[4. DataFrame 만들기](#4-dataframe-만들기)  
[5. Demo](#5-demo)  

## Setup
TableDetection에 활용할 모델은 Pytorch based MMdetection framework(Version 1.2)에서 개발되었으며 이는 CascadeTabNet을 인용하였습니다. <https://github.com/DevashishPrasad/CascadeTabNet> 
더불어, MMdetection(Version 1.2) 및 MMcv(Version 0.4.3)를 참고 및 수정하여 진행하였습니다. <https://github.com/open-mmlab/mmdetection>, <https://github.com/open-mmlab/mmcv>
```
!pip install -q mmcv terminaltables
%cd '/mmdetection'
!pip install -r '/mmdetection/requirements/optional.txt'
!python setup.py install
!python setup.py develop
!pip install -r {'requirements.txt'}
!pip install pillow==6.2.1
```

## Dependency
Table Detection의 경우 `PyTorch = 1.4.0`, `Torchvision = 0.5.0`, `Cuda = 10.0`에서 개발된 코드를 사용하였습니다.(CascadeTabNet참고 - 링크 위 참조)
```
!pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

**Note :** 전 과정은 Colab에서 실시하였으며 Colab의 경우 cv2.imshow가 `from google.colab.patches import cv2_imshow`로 변경되었다.

## 1. CascadeTabNet을 활용한 Table Detection
CascadeTabNet의 경우 Structure Recognition 및 Cell Recognition의 기능까지 탑재되어 있으나 현 과정에서는 인식률의 문제 때문에(특히 Cell) Table Detection만 사용하기 위하여 코드를 수정하여 진행하였다.

![Table_Detection](https://user-images.githubusercontent.com/53552847/119606010-be91d980-be2c-11eb-8112-2579ec0c4e7b.PNG)

## 2. OpenCV를 활용한 Cell Detection
OpenCV를 활용하여 추출된 Table로부터 Rectangle을 인식하여 Cell로 인식하도록 하였으며 이에 대한 Box값 및 Rectangle Drawing을 하였다.

![Cell_Detection_1](https://user-images.githubusercontent.com/53552847/119606891-53490700-be2e-11eb-97de-18bd4963d719.PNG)
![Cell_Detection_2](https://user-images.githubusercontent.com/53552847/119606898-57752480-be2e-11eb-8bd9-8599399b1f82.PNG)
![Cell_Detection_3](https://user-images.githubusercontent.com/53552847/119606899-57752480-be2e-11eb-8ee7-10f3c19ca6da.PNG)
![Cell_Detection_4](https://user-images.githubusercontent.com/53552847/119606901-580dbb00-be2e-11eb-8c27-efff38a0a5c0.PNG)
![Cell_Detection_5](https://user-images.githubusercontent.com/53552847/119608171-7c6a9700-be30-11eb-8899-23bdbc03b392.PNG)

## 3. Naver OCR를 활용한 Text Extraction
**NOTE :** Naver OCR는 유료 프로그램(각 이미지당 3원)이며 **Service Key**, **URL**을 clova ai에서 내려받아 사용할 수 있다. (아래 링크 참조)

Naver OCR을 활용할 경우 거의 대부분의 텍스트들이 알맞게 추출이 되었으며 더불어 json파일의 경우 좌표를 얻을 수 있고 이를 데이터프레임을 만드는 데에 활용하도록 한다.
Naver OCR에 대한 자세한 설명은 Clova AI Research의 github [여기](https://github.com/clovaai/deep-text-recognition-benchmark), Clova ai의 OCR프로그램 사용에 대한 자세한 설명은 [여기](https://guide.ncloud-docs.com/docs/ko/ocr-ocr-1-1)에서 확인할 수 있다.(참고 - API 자동 호출을 위해서는 앞 링크에서 CLOVA OCR API 연동가이드, CLOVA OCR API 호출 가이드도 필독하다록 하자.)

<https://github.com/clovaai/deep-text-recognition-benchmark>
<https://www.ncloud.com/product/aiService/ocr>
<https://guide.ncloud-docs.com/docs/ko/ocr-ocr-1-1>

## 4. DataFrame 만들기
OpenCV로 부터 얻은 Box값과 Naver OCR로 얻은 좌표값을 활용하여 알맞은 데이터프레임 형태로 만들어준다.

|Table 이미지|추출된 DataFrame|
|-----|------|
|![Cell_Detection_1](https://user-images.githubusercontent.com/53552847/119606891-53490700-be2e-11eb-97de-18bd4963d719.PNG)|![Output_Detection_1](https://user-images.githubusercontent.com/53552847/119608032-41686380-be30-11eb-87b5-744a0ed01f42.PNG)|
|![Cell_Detection_2](https://user-images.githubusercontent.com/53552847/119606898-57752480-be2e-11eb-8bd9-8599399b1f82.PNG)|![Output_Detection_2](https://user-images.githubusercontent.com/53552847/119608034-4200fa00-be30-11eb-8e0a-315b4dc55980.PNG)|
|![Cell_Detection_3](https://user-images.githubusercontent.com/53552847/119606899-57752480-be2e-11eb-8ee7-10f3c19ca6da.PNG)|![Output_Detection_3](https://user-images.githubusercontent.com/53552847/119608035-42999080-be30-11eb-93ef-8c6bf872b387.PNG)|
|![Cell_Detection_4](https://user-images.githubusercontent.com/53552847/119606901-580dbb00-be2e-11eb-8c27-efff38a0a5c0.PNG)|![Output_Detection_4](https://user-images.githubusercontent.com/53552847/119608036-42999080-be30-11eb-8b8c-36f49a7d1fe8.PNG)|
|![Cell_Detection_5](https://user-images.githubusercontent.com/53552847/119608171-7c6a9700-be30-11eb-8899-23bdbc03b392.PNG)|![Output_Detection_5](https://user-images.githubusercontent.com/53552847/119608038-43322700-be30-11eb-8612-52ed4a417ad3.PNG)|
## 5. Demo
**Note :** 본 Demo는 Naver OCR 프로그램(유료)을 활용한 것으로 본인의 URL 및 Service key를 입력해야 실행할 수 있습니다.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FWrEKTMyGGPTDw-Mfh0qSXFhKLWOqGj9?authuser=1#scrollTo=shOdp0SZKzjZ)

