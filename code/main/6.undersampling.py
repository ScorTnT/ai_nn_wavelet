from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np

data4train_path = '/workspace/Train/Train30.csv'
data4test_path = '/workspace/Test/Test30.csv'
data4train = np.loadtxt(data4train_path, delimiter=',', dtype=np.float32)
data4test = np.loadtxt(data4test_path, delimiter=',', dtype=np.float32)

# 라벨 분포 확인 및 언더샘플링
train_images = data4train[:,0:-1]
train_labels = data4train[:,-1]

# 클래스별 인덱스 추출
idx_0 = np.where(train_labels == 0)[0]
idx_1 = np.where(train_labels == 1)[0]

# 두 클래스 중 적은 개수로 맞춤
n_min = min(len(idx_0), len(idx_1))
np.random.shuffle(idx_0)
np.random.shuffle(idx_1)
sel_idx = np.concatenate([idx_0[:n_min], idx_1[:n_min]])
np.random.shuffle(sel_idx)

# 언더샘플링된 데이터로 재구성
train_images = train_images[sel_idx]
train_labels = train_labels[sel_idx]

test_images = data4test[:,0:-1]
test_labels = data4test[:,-1]

network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(30,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dropout(0.4))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(4, activation='relu'))
network.add(layers.Dense(2, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=15, batch_size=64)

test_loss, test_acc = network.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)