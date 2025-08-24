from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

data4train_path = '/workspace/Train/Train30v15_1.csv'
data4test_path = '/workspace/Test/Test30-1024.csv'
data4train = np.loadtxt(data4train_path, delimiter=',', dtype=np.float32)
data4test = np.loadtxt(data4test_path, delimiter=',', dtype=np.float32)

# 1. 데이터 전처리 개선
train_images = data4train[:,0:-1]
train_labels = data4train[:,-1]
test_images = data4test[:,0:-1]
test_labels = data4test[:,-1]

# 2. 데이터 정규화 추가
scaler = StandardScaler()  # 또는 MinMaxScaler()
train_images = scaler.fit_transform(train_images)
test_images = scaler.transform(test_images)

# 3. 언더샘플링 개선 (stratified sampling)
from sklearn.utils import resample

# 클래스별 분리
train_0 = train_images[train_labels == 0]
train_1 = train_images[train_labels == 1]
labels_0 = train_labels[train_labels == 0]
labels_1 = train_labels[train_labels == 1]

# 균형 맞추기
n_min = min(len(train_0), len(train_1))
print(f"클래스 0: {len(train_0)}개, 클래스 1: {len(train_1)}개")
print(f"균형 맞춤: 각 클래스 {n_min}개씩 사용")

# 리샘플링
train_0_balanced = resample(train_0, n_samples=n_min, random_state=42)
train_1_balanced = resample(train_1, n_samples=n_min, random_state=42)
labels_0_balanced = resample(labels_0, n_samples=n_min, random_state=42)
labels_1_balanced = resample(labels_1, n_samples=n_min, random_state=42)

# 데이터 합치기
train_images_balanced = np.vstack([train_0_balanced, train_1_balanced])
train_labels_balanced = np.hstack([labels_0_balanced, labels_1_balanced])

# 섞기
indices = np.random.permutation(len(train_images_balanced))
train_images = train_images_balanced[indices]
train_labels = train_labels_balanced[indices]

# 4. 검증 데이터 분할 개선
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_labels
)

# 5. 모델 구조 개선
def create_improved_model(input_dim=30):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(8, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])
    
    return model

network = create_improved_model()

# 6. 최적화기 개선
optimizer = Adam(
    learning_rate=0.004,  # 학습률 조정
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

network.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 라벨 변환
y_train_cat = to_categorical(y_train, 2)
y_val_cat = to_categorical(y_val, 2)
test_labels_cat = to_categorical(test_labels, 2)

# 8. 콜백 개선
callbacks = [
    # EarlyStopping(
    #     monitor='val_accuracy',  # val_loss 대신 val_accuracy 모니터링
    #     patience=10,             # patience 증가
    #     restore_best_weights=True,
    #     verbose=1
    # ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    # ,
    # ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.5,              # 학습률을 50%로 감소
    #     patience=5,
    #     min_lr=1e-7,
    #     verbose=1
    # )
]

# 9. 모델 요약 출력
print("=== 모델 구조 ===")
network.summary()

print(f"\n=== 데이터 정보 ===")
print(f"훈련 데이터: {X_train.shape}")
print(f"검증 데이터: {X_val.shape}")
print(f"테스트 데이터: {test_images.shape}")

# 10. 훈련 개선
history = network.fit(
    X_train, y_train_cat,
    epochs=100,              # 최대 에포크 증가
    batch_size=128,           # 배치 크기
    validation_data=(X_val, y_val_cat),
    callbacks=callbacks,
    verbose=1
)

# 11. 평가 및 분석
test_loss, test_acc = network.evaluate(test_images, test_labels_cat, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')

# 12. 상세 분석
y_pred = network.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_labels_cat, axis=1)

print("\n=== 분류 보고서 ===")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=['정상', '비정상']))

print("\n=== 혼동 행렬 ===")
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

# 13. 훈련 과정 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('/workspace/training_history.png')
plt.show()