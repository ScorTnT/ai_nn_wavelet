FROM tensorflow/tensorflow:2.15.0-gpu

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    tar \
    gzip \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip

# 머신러닝 관련 패키지 설치
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    librosa \
    PyWavelets \
    scipy \
    jupyter \
    notebook

# 작업 디렉토리 구조 생성
RUN mkdir -p /workspace/code/main \
    && mkdir -p /workspace/Train \
    && mkdir -p /workspace/Test \
    && mkdir -p /workspace/training \
    && mkdir -p /workspace/validation

# 환경변수 설정
ENV PYTHONPATH=/workspace
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# GPU 메모리 증가 설정
ENV TF_MEMORY_GROWTH=true

# 권한 설정
RUN chmod -R 755 /workspace

# 포트 노출 (Jupyter 사용시)
EXPOSE 8888

# 기본 명령어
CMD ["/bin/bash"]