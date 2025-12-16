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

# 3. Conv1D를 위한 데이터 reshape
# (samples, features) -> (samples, timesteps, features)
# 30개 특징을 30개의 timesteps으로 취급하고 각 timestep은 1개의 feature
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], 1))
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], 1))

print(f"Reshaped train data: {train_images.shape}")  # (samples, 30, 1)
print(f"Reshaped test data: {test_images.shape}")

# 4. 언더샘플링 개선 (stratified sampling)
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

# 5. 검증 데이터 분할 개선
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_labels
)

# 6. Conv1D 모델 구조 개선
def create_conv1d_model(input_shape=(30, 1)):
    model = models.Sequential([
        # 첫 번째 Conv1D 블록
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # 두 번째 Conv1D 블록
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # 세 번째 Conv1D 블록
        layers.Conv1D(filters=16, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Global Average Pooling (Flatten 대신)
        layers.GlobalAveragePooling1D(),
        
        # Dense 레이어들
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        # 출력층
        layers.Dense(2, activation='softmax')
    ])
    
    return model

network = create_conv1d_model()

# 7. 최적화기 개선
optimizer = Adam(
    learning_rate=0.003,  # Conv1D에 맞게 학습률 조정
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

network.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 8. 라벨 변환
y_train_cat = to_categorical(y_train, 2)
y_val_cat = to_categorical(y_val, 2)
test_labels_cat = to_categorical(test_labels, 2)

# 9. 콜백 개선
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_conv1d.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
]

# 10. 모델 요약 출력
print("=== Conv1D 모델 구조 ===")
network.summary()

print(f"\n=== 데이터 정보 ===")
print(f"훈련 데이터: {X_train.shape}")
print(f"검증 데이터: {X_val.shape}")
print(f"테스트 데이터: {test_images.shape}")

# 11. 훈련
history = network.fit(
    X_train, y_train_cat,
    epochs=150,
    batch_size=64,  # Conv1D에 맞게 배치 크기 조정
    validation_data=(X_val, y_val_cat),
    callbacks=callbacks,
    verbose=1
)

# 12. 평가 및 분석
test_loss, test_acc = network.evaluate(test_images, test_labels_cat, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')

# 13. 상세 분석
y_pred = network.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_labels_cat, axis=1)

print("\n=== 분류 보고서 ===")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=['정상', '비정상']))

print("\n=== 혼동 행렬 ===")
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

# 14. 민감도와 특이도 계산
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # 민감도
specificity = tn / (tn + fp)  # 특이도
precision = tp / (tp + fp)    # 정밀도
accuracy = (tp + tn) / (tp + tn + fp + fn)  # 정확도

print(f"\n=== 성능 지표 ===")
print(f"정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"민감도 (Sensitivity): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"특이도 (Specificity): {specificity:.4f} ({specificity*100:.2f}%)")
print(f"정밀도 (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"F1-Score: {2 * (precision * sensitivity) / (precision + sensitivity):.4f}")

# 15. 훈련 과정 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy (Conv1D)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss (Conv1D)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 혼동행렬 히트맵
plt.subplot(1, 3, 3)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['예측: 정상', '예측: 비정상'],
            yticklabels=['실제: 정상', '실제: 비정상'])
plt.title('Confusion Matrix (Conv1D)')

plt.tight_layout()
plt.savefig('/workspace/conv1d_training_results.png', dpi=150, bbox_inches='tight')
plt.show()

# 16. 결과 요약 저장
results_summary = f"""
=== Conv1D 모델 결과 요약 ===
모델 타입: 1D Convolutional Neural Network
입력 형태: {X_train.shape}
훈련 샘플: {len(X_train)}
검증 샘플: {len(X_val)}
테스트 샘플: {len(test_images)}

=== 최종 성능 ===
테스트 정확도: {test_acc:.4f} ({test_acc*100:.2f}%)
민감도: {sensitivity:.4f} ({sensitivity*100:.2f}%)
특이도: {specificity:.4f} ({specificity*100:.2f}%)
정밀도: {precision:.4f} ({precision*100:.2f}%)
F1-Score: {2 * (precision * sensitivity) / (precision + sensitivity):.4f}

=== 혼동행렬 ===
               예측
실제     정상    비정상
정상     {tn}     {fp}
비정상   {fn}     {tp}

=== 모델 특징 ===
- Conv1D 레이어를 사용하여 30개 특징의 순차적 패턴 학습
- BatchNormalization과 Dropout을 통한 정규화
- GlobalAveragePooling1D로 특징 압축
- 학습률 스케줄링과 Early Stopping 적용
"""

with open('/workspace/conv1d_model_results.txt', 'w', encoding='utf-8') as f:
    f.write(results_summary)

print(results_summary)
print(f"\n결과가 '/workspace/conv1d_model_results.txt'에 저장되었습니다.")

# 17. 특징 맵 시각화 (선택사항)
def visualize_feature_maps(model, sample_data, layer_names=None):
    """Conv1D 모델의 특징 맵 시각화"""
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers if 'conv1d' in layer.name]
    
    plt.figure(figsize=(15, len(layer_names) * 3))
    
    for i, layer_name in enumerate(layer_names):
        # 중간 모델 생성
        intermediate_model = models.Model(inputs=model.input,
                                        outputs=model.get_layer(layer_name).output)
        
        # 특징 맵 추출
        feature_maps = intermediate_model.predict(sample_data[:1])  # 첫 번째 샘플만
        
        plt.subplot(len(layer_names), 1, i+1)
        plt.imshow(feature_maps[0].T, aspect='auto', cmap='viridis')
        plt.title(f'Feature Maps from {layer_name}')
        plt.xlabel('Time Steps')
        plt.ylabel('Filters')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('/workspace/conv1d_feature_maps.png', dpi=150, bbox_inches='tight')
    plt.show()

# 특징 맵 시각화 실행
print("\n=== 특징 맵 시각화 ===")
visualize_feature_maps(network, X_val[:5])  # 검증 데이터의 첫 5개 샘플