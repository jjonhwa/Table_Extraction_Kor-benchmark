# Naver OCR & CascadeTabNet을 활용한 한국어 Table Extraction

영어의 경우 Deep Learning을 활용한 Table Detection(with Structure Recognition) - Cell Detection - Text Extract를 바탕으로 Table Extraction에 대한 연구가 진행되었으나 한국어 Table Extraction에 대한 연구 및 오픈소스가 없었다.
이에 대하여 한국어 Table Extraction을 진행하기 위해 Table Detection, Cell Detection, Text Extraction에 대한 오픈 소스를 검색 및 탐구하였고 이를 알맞게 추출하기 위하여 오픈 소스를 수정하여 최종 Table Extraction을 하였다.
전 과정은 PDF 원하는 페이지 추출 > PDF to Image > Image Table Detection > Image Cell Dectection > Text Extraction > Make DataFrame 순으로 진행하였으며 현 git에는 Table Extraction에 대한 과정만 담았다.

## 진행과정
모든 과정을 Colab에서 진행하였습니다. (GPU 환경)
1. CascadeTabNet을 활용한 Table Detection
2. Opencv를 활용한 Cell Detection
3. Naver OCR를 활용한 Text Extraction
4. DataFrame 만들기

### Setup
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

### Dependency
Table Detection의 경우 `PyTorch = 1.4.0`, `Torchvision = 0.5.0`, `Cuda = 10.0`에서 개발된 코드를 사용하였습니다.(CascadeTabNet참고 - 링크 위 참조)
```
!pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

**Note :** 전 과정은 Colab에서 실시하였으며 Colab의 경우 cv2.imshow가 `from google.colab.patches import cv2_imshow`로 변경되었다.

### Demo
