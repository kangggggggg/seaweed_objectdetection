# 김(Seaweed) 이미지 기반 이물질 검출 및 품질 분류 AI 모델 개발
## 프로젝트 개요
본 프로젝트는 김 이미지에서 결함을 탐지하기 위해 **YOLO**(You Only Look Once) 모델을 활용하여 Google Colab에서 학습 및 예측을 수행한 과정입니다. 학습된 모델을 로컬 환경에서도 실행할 수 있도록 `requirements.txt` 파일을 제공하며, 데이터 전처리부터 모델 학습, 평가, 테스트까지의 과정을 다룹니다.
- README.md # 설명 파일
- data/ # 모델의 학습 및 평가에 사용된 데이터
- model/ # 학습된 모델 파일
- src/ # 데이터 정제/전처리, 모델 구조, 학습 및 평가를 위한 소스 코드 파일
- notebooks/ # Colab에서 실행한 Jupyter notebook 파일들
- scripts/ # 환경 설정 및 모델 학습을 위한 스크립트 파일
- demo/ # 테스트 실행 데모 영상
- figures/ # 모델 설명 및 실험 결과 설명을 위한 그림 파일
- requirements.txt # 로컬 환경에서 필요한 라이브러리 목록

## Colab 환경에서의 실행

모델 훈련, 평가, 테스트 예측 과정은 **Google Colab**을 사용하여 수행하였습니다. Colab은 GPU 가속을 지원하며, 프로젝트 실행에 적합한 환경을 제공합니다. 학습된 모델은 로컬 환경에서도 실행 가능하며, 이를 위해 필요한 환경 구성 정보를 제공합니다.

### Colab 실행 환경 (HW/SW)

- **Google Colab Pro 환경**
  - **GPU**: Tesla T4 (15GB)
  - **NVIDIA Driver 버전**: `535.104.05`
  - **Python 버전**: `3.10.12`
  - **PyTorch CUDA 버전**: `12.1`

### 로컬 환경 실행 방법

로컬 환경에서 YOLO 모델을 실행할 수 있도록 환경을 설정해야 합니다. 로컬에서 모델을 실행하기 위한 필수 라이브러리와 설정 방법은 아래와 같습니다.

1. **Conda 환경 구성**:
   ```bash
   conda create -n yolov8_env python=3.10
   conda activate yolov8_env
   ```bash
2. **필수 라이브러리 설치**: 로컬 환경에서 YOLO 모델을 실행하려면 requirements.txt 파일에 명시된 라이브러리를 설치해야 합니다.
  ```bash
  pip install -r requirements.txt
  ```
3. **필요한 환경**:
- CUDA: 12.2 이상
- NVIDIA Driver: 535.xx 이상
- Python 버전: 3.10.x
- PyTorch CUDA 버전: 12.1
4. **YOLO 모델 실행 예시**:
  ```bash
  python src/preprocessing.py  # 모델 훈련을 위한 전처리(YOLO 형식 변환)
  python src/train.py  # 모델 학습
  python src/evaluate.py  # 모델 평가
  python src/predict.py  # 테스트 데이터 예측
  ```
### 'yolo.yaml' 파일 설명
YOLO 모델 학습을 위해 yolo.yaml 파일은 데이터 경로 및 클래스 수와 같은 설정 정보를 포함합니다.
- 'yolo.yaml' 파일:
  '''yaml
  train: ../data/PREPROCESSED_TRAIN  # 학습 데이터 경로
  val: ../data/PREPROCESSED_VAL  # 검증 데이터 경로
  nc: 3  # 클래스 수
  names: [0, 1, 2]  # 클래스 이름 (aq : 0, st : 1, fl : 2)
  '''
이 파일은 scripts 폴더에 저장되어 있습니다. 또한, 모델 학습 시 자동으로 생성됩니다.


### 'requirements.txt' 내용
로컬 환경에서 YOLO 모델을 실행하기 위해 필요한 라이브러리 목록은 다음과 같습니다:
  ```makefile
  absl-py==1.4.0
  albumentations==1.4.15
  opencv-python==4.10.0.84
  matplotlib==3.7.1
  numpy==1.26.4
  Pillow==10.4.0
  torch==2.5.0+cu121  # PyTorch 버전, CUDA 12.1과 호환
  torchvision==0.20.0+cu121
  ultralytics==8.3.21
  scikit-learn==1.5.2
  tqdm==4.66.5
  ```

### 학습 및 평가 수행 방법
모든 과정은 Colab에서 실행되었으나, 로컬에서도 실행할 수 있도록 Jupyter 노트북과 Python 스크립트를 제공합니다. 각 파일의 설명은 아래와 같습니다.

1. **학습 (train.ipynb)**
- YOLO 모델을 사용해 학습 데이터를 기반으로 학습을 수행합니다.
- 학습 데이터는 data/TRAIN 폴더에 저장되어 있으며, 학습된 모델 파일은 model/ 폴더에 저장됩니다.
- Colab에서 Tesla T4 GPU를 활용하여 학습하였으며, 로컬에서는 GPU 환경이 필요합니다.
2. **데이터 전처리 (preprocess.ipynb)**
- 원본 데이터를 YOLO 학습 형식에 맞게 전처리합니다.
- YOLO 형식으로 변환된 이미지를 포함한 훈련 데이터는 data/PREPROCESSED_TRAIN, 검증 데이터는 data/PREPROCESSED_VAL 폴더에 저장됩니다.
- 전처리 과정은 Colab에서 진행하였지만, 로컬에서도 실행 가능합니다.
3. **평가 (evaluate.ipynb)**
- 학습된 YOLO 모델을 사용하여 성능 평가를 수행합니다.
- 평가 지표는 IoU 0.75 및 IoU 0.95에서의 Precision, Recall, mAP를 포함합니다.
4. **테스트 예측 (predict.ipynb)**
- 학습된 YOLO 모델을 테스트 이미지에 적용하여 예측을 수행합니다.
- 예측 결과는 data/predictions/ 폴더에 JSON 형식으로 저장되며, 시각화된 결과도 포함됩니다.

### 로컬 환경에서 실행하는 방법
Conda 환경을 구성한 후, requirements.txt에 명시된 라이브러리를 설치합니다.
notebooks/ 폴더에 있는 주피터 노트북 파일을 사용하여 학습, 평가, 테스트 과정을 실행할 수 있습니다.
필요한 데이터는 data/ 폴더에 저장되어 있어야 합니다.

### 데모 영상
테스트 데이터에 대한 예측 과정을 담은 데모 영상은 demo/ 폴더에 저장되어 있습니다. 이 영상을 통해 실제로 모델이 예측을 수행하는 과정을 확인할 수 있습니다.

### 저작권 문제
본 프로젝트에서 사용한 데이터는 경진대회에서 제공된 데이터로, 외부 데이터를 사용하지 않았으며, 코드 및 라이브러리도 저작권 문제가 없음을 확인하였습니다.
