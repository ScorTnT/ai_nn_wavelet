from keras import models
from keras import layers
import numpy as np
import tensorflow as tf

# GPU 설정 및 확인
print("=== GPU 설정 확인 ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# GPU 메모리 증가 설정 (GPU가 있는 경우)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU 메모리 증가 설정 완료: {len(gpus)}개 GPU 사용 가능")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")
else:
    print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")

# 데이터 로드
data4train_path = '/workspace/Train/Train30.csv'
data4test_path = '/workspace/Test/Test30.csv'
data4train = np.loadtxt(data4train_path, delimiter=',', dtype=np.float32)
data4test = np.loadtxt(data4test_path, delimiter=',', dtype=np.float32)

train_images = data4train[:,0:-1]
train_labels = data4train[:,-1]
test_images = data4test[:,0:-1]
test_labels = data4test[:,-1]

# GPU 사용을 위한 mixed precision 설정 (선택사항)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 모델 구성 - GPU에서 더 효율적인 배치 크기 사용
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    network = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(30,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

# 모델 컴파일
network.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print(f"\n=== 모델 정보 ===")
network.summary()

# GPU에 최적화된 배치 크기 설정
batch_size = 128 if gpus else 64  # GPU가 있으면 더 큰 배치 크기 사용
epochs = 25
print(f"사용 디바이스: {'GPU' if gpus else 'CPU'}")

# 학습 시작
print(f"\n=== 학습 시작 ===")
history = network.fit(
    train_images, 
    train_labels, 
    epochs=epochs, 
    batch_size=batch_size,
    validation_split=0.2,  # 검증 데이터 분할
    verbose=1
)

# 평가
print(f"\n=== 모델 평가 ===")
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')
print(f'테스트 손실: {test_loss:.4f}')

# GPU 사용 통계 (GPU가 있는 경우)
if gpus:
    print(f"\n=== GPU 메모리 사용량 ===")
    for i, gpu in enumerate(gpus):
        memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
        print(f"GPU {i}: {memory_info['current'] / 1024**2:.1f} MB / {memory_info['peak'] / 1024**2:.1f} MB (peak)")