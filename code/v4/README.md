# V4: Heart Sound Classification - Wavelet Features

## 프로젝트 개요
1차원 심장소리 데이터에서 웨이브릿 변환을 통한 특징 추출 방식의 정확도 향상 효과를 검증합니다.

## 목표
1. **특징 추출 방식 비교**: 원본 .wav 데이터 vs 웨이브릿 변환 데이터
2. **정확도 향상 검증**: 웨이브릿 변환의 성능 개선 효과 측정
3. **안정적인 전처리**: 일관된 특성 벡터 길이와 정확한 라벨링

## 데이터셋
- **소스**: PhysioNet Challenge 2016 Heart Sound Database
- **구성**: training-a, training-b, training-c, training-d, training-e, training-f
- **라벨링**: REFERENCE.csv 파일 기반 (-1: normal → 0, 1: abnormal → 1)

## 파일 구조
```
/workspace/code/v4/
├── preprocess_wavelet.py    # 웨이브릿 전처리 스크립트
├── training_nn.ipynb        # 신경망 훈련 노트북
└── README.md               # 이 파일

/workspace/wavelet_v4/       # 전처리된 웨이브릿 데이터 저장소
└── *.npz                   # 웨이브릿 특성 + 라벨 파일들
```

## 사용 방법

### 1. 웨이브릿 전처리 실행
```bash
cd /workspace/code/v4
python preprocess_wavelet.py
```

### 2. 신경망 훈련
```bash
# Jupyter Notebook에서 training_nn.ipynb 실행
```

## 전처리 특징
- **웨이브릿 변환**: Daubechies 4 웨이브릿 1-레벨 DWT
- **특성 벡터**: 근사계수(cA) + 세부계수(cD) 결합
- **일관된 길이**: 모든 특성을 2000차원으로 패딩/트렁케이션
- **정확한 라벨**: REFERENCE.csv 파일에서 실제 라벨 추출

## 저장 데이터 형식
각 .npz 파일 포함 내용:
- `features`: 2000차원 웨이브릿 특성 벡터
- `label`: 분류 라벨 (0: normal, 1: abnormal)
- `original_cA_shape`: 원본 근사계수 길이 (참조용)
- `original_cD_shape`: 원본 세부계수 길이 (참조용)
- `set_name`: 출처 훈련세트 이름
- `filename`: 원본 파일명

## 신경망 구조
- **Input**: 2000차원 웨이브릿 특성
- **Hidden Layers**: 128 → 64 → 32 뉴런
- **Activation**: ReLU (은닉층), Sigmoid (출력층)
- **Regularization**: Dropout (0.5, 0.3, 0.2)
- **Output**: 이진 분류 (normal/abnormal)

## 성능 지표
- 정확도 (Accuracy)
- 정밀도 (Precision)
- 재현율 (Recall)
- F1-Score
- 혼동 행렬 (Confusion Matrix)

## 버전 정보
- **V4 특징**:
  - 정확한 REFERENCE.csv 기반 라벨링
  - 일관된 2000차원 특성 벡터
  - 개선된 신경망 구조 (더 깊은 네트워크)
  - 상세한 성능 분석 및 시각화
