from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np

data4train_path = '/workspace/Train/Train60.csv'
data4test_path = '/workspace/Test/Test60.csv'
data4train = np.loadtxt(data4train_path, delimiter=',', dtype=np.float32)
data4test = np.loadtxt(data4test_path, delimiter=',', dtype=np.float32)

train_images = data4train[:,0:-1]
train_labels = data4train[:,-1]
test_images = data4test[:,0:-1]
test_labels = data4test[:,-1]

network = models.Sequential()
network.add(layers.Dense(256, activation='relu', input_shape=(60,)))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(2, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=15, batch_size=64)

test_loss, test_acc = network.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)
