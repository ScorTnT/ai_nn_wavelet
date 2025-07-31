# Heart Sound Classification with Wavelet Transform

## 프로젝트 개요

이 프로젝트는 PhysioNet Challenge 2016 Heart Sound Database를 사용하여 **심장 소리의 정상/비정상 분류**를 수행하는 머신러닝 연구입니다. 주요 목표는 **웨이블릿 변환(Wavelet Transform) 기반 특징 추출 방법의 효과성을 검증**하고, 다양한 전처리 및 모델링 접근법을 비교 분석하는 것입니다.

## 데이터셋

### 소스 데이터
- **데이터셋**: PhysioNet Challenge 2016 Heart Sound Database
- **데이터 구성**: 
  - `training-a/` : 409
  - `training-b/` : 490
  - `training-c/` : 31
  - `training-d/` : 55
  - `training-e/` : 2141
  - `training-f/` : 114
  - total: 3240 (6개 훈련 세트)
  - `validation/` (검증 세트)
- **데이터 형식**: 
  - `.wav` 파일 (심장소리 녹음)
  - `REFERENCE.csv` (라벨 정보: -1=정상, 1=비정상)
- **분류 문제**: 이진 분류 (Normal/Abnormal Heart Sounds)

### 데이터 라벨링
- **정상 (Normal)**: -1 → 0으로 변환
- **비정상 (Abnormal)**: 1 → 1로 유지
- 각 훈련 세트마다 `REFERENCE.csv` 파일에서 라벨 정보 추출

## 프로젝트 목적

### 1. 특징 추출 방법 비교
- **원본 데이터**: 직접 `.wav` 파일에서 특징 추출
- **웨이블릿 변환 데이터**: 사전 처리된 웨이블릿 계수에서 특징 추출
- **성능 비교**: 웨이블릿 변환의 분류 정확도 향상 효과 검증

### 2. 다양한 모델링 접근법 실험
- **전통적 머신러닝**: SVM, Random Forest
- **딥러닝**: Multi-Layer Perceptron (MLP) 신경망
- **특징 벡터 차원**: 10차원 통계적 특징 vs 2000차원 웨이블릿 계수

## 현재 진행 중인 연구

### 웨이블릿 변환 전처리 과정 개선

현재 **5가지 고급 특징 추출 방법**을 통한 웨이블릿 전처리 최적화 연구를 진행하고 있습니다:

#### 5가지 특징 추출 방법 (각 레벨별)
1. **절대값 평균 (Mean Absolute Value)**
   - 각 레벨 안에 있는 모든 계수들에 대한 절대값의 평균값
   
2. **제곱 평균 (Mean Square Value)**
   - 각 레벨 안에 있는 모든 계수들을 제곱하여 구한 평균값
   
3. **표준편차 (Standard Deviation)**
   - 각 레벨 안에 있는 모든 계수들의 표준편차
   
4. **레벨간 비율 (Inter-level Ratio)**
   - 인접한 레벨간의 레벨 안에 있는 모든 계수들에 대한 평균값의 절대값 비율
   
5. **중앙값 (Median Value)**
   - 각 레벨 안에 있는 모든 계수들의 중앙값

#### 다중 레벨 분석
- **5개 레벨**: 웨이블릿 분해를 통한 다중 해상도 분석
- **총 30개 특징**: 5개 특징 × 5개 레벨 = 30차원 특징 벡터
- **목표**: 더 풍부한 주파수-시간 정보 활용으로 분류 성능 향상

## 프로젝트 구조

```
/workspace/
├── training-a/              # 훈련 데이터 세트 A
├── training-b/              # 훈련 데이터 세트 B
├── training-c/              # 훈련 데이터 세트 C
├── training-d/              # 훈련 데이터 세트 D
├── training-e/              # 훈련 데이터 세트 E
├── training-f/              # 훈련 데이터 세트 F
├── validation/              # 검증 데이터 세트
├── code/                    # 소스 코드 버전별 관리
│   ├── v1/                  # V1: SVM + MFCC
│   ├── v2/                  # V2: Random Forest + 향상된 특징
│   ├── v3/                  # V3: 신경망 + 웨이블릿 통계 특징
│   ├── v4/                  # V4: 개선된 웨이블릿 전처리
│   ├── v5/                  # V5: 강화된 신경망 + 데이터 증강
│   └── v6/                  # V6: 최신 개선 버전
├── wavelet/                 # 전처리된 웨이블릿 데이터 (v3)
├── wavelet_v4/              # 전처리된 웨이블릿 데이터 (v4)
├── wavelet_v5/              # 전처리된 웨이블릿 데이터 (v5)
└── papers/                  # 관련 연구 논문들
```

## 기술적 접근법

### 웨이블릿 변환 설정
- **웨이블릿 종류**: Daubechies 4 (db4)
- **분해 레벨**: 1레벨 (현재) → 5레벨 (연구 진행 중)
- **특징 추출**: 통계적 특징 (평균, 표준편차, 왜도, 첨도, 에너지)

### 모델 아키텍처 (최신 버전)
- **입력층**: 웨이블릿 특징 벡터
- **은닉층**: 128 → 64 → 32 뉴런
- **활성화 함수**: ReLU (은닉층), Sigmoid (출력층)
- **정규화**: Dropout (0.6, 0.5, 0.4)
- **출력층**: 이진 분류 확률

### 성능 평가 지표
- **정확도 (Accuracy)**
- **정밀도 (Precision)**
- **재현율 (Recall)**
- **F1-Score**
- **혼동 행렬 (Confusion Matrix)**

## 현재 연구 상태

### 완료된 작업
- [x] V1-V3: 기본 머신러닝 모델 및 신경망 구현
- [x] V4-V5: 웨이블릿 전처리 파이프라인 구축
- [x] 데이터 증강 및 정규화 기법 적용
- [x] 모델 성능 비교 분석 시스템 구축

### 진행 중인 작업
- [ ] **5레벨 웨이블릿 분해**: 다중 해상도 특징 추출
- [ ] **30차원 특징 벡터**: 5가지 특징 × 5레벨 최적화
- [ ] **성능 비교**: 기존 10차원 vs 새로운 30차원 특징
- [ ] **데이터 전처리 연습**: 다양한 특징 추출 방법론 실험

### 향후 계획
- [ ] 5레벨 웨이블릿 분해 결과 분석
- [ ] 최적 특징 조합 선택 알고리즘 개발
- [ ] 교차 검증을 통한 모델 일반화 성능 평가
- [ ] 실시간 분류 시스템 구현

## 실행 방법

### 환경 설정
```bash
# 필요한 라이브러리 설치
pip install numpy pandas scikit-learn librosa pywt tensorflow matplotlib seaborn
```

### 데이터 전처리 (V5 예시)
```bash
cd /workspace/code/v5
python preprocess_wavelet.py
```

### 모델 훈련
```bash
# Jupyter Notebook에서 실행
jupyter notebook training_nn.ipynb
```

## 기여자 및 참고 문헌

### 관련 연구
- PhysioNet Challenge 2016: "Classification of Normal/Abnormal Heart Sound Recordings"
- 웨이블릿 변환 기반 생체신호 분석 연구
- 딥러닝을 활용한 심장소리 분류 연구

### 개발 환경
- **운영체제**: Docker Container (Debian GNU/Linux 12)
- **프로그래밍 언어**: Python 3.x
- **주요 라이브러리**: TensorFlow, scikit-learn, PyWavelets, librosa
- **개발 도구**: VS Code, Jupyter Notebook

---

**Last Updated**: July 30, 2025  
**Repository**: ai_nn_wavelet  
**Branch**: main
