from keras.models import load_model
import numpy as np

# 모델 로드
model_path = '/workspace/model/segL-1024/best_model_7345_m-val_acc_b-128_lr-004.h5'  # 실제 경로로 변경
model = load_model(model_path)

# 1. 모델 요약 정보
print("=== 모델 요약 ===")
model.summary()