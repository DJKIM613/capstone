import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.utils import to_categorical

total_datas = eval(open("./input.txt", "r").read())
train_datas = total_datas[:180]
test_datas = total_datas[180:]

dimension = 44
train_images = []
train_labels = []
for train_data in train_datas:
    train_images.append(train_data[0:dimension])
    train_labels.append(train_data[dimension:])

test_images = []
test_labels = []
for test_data in test_datas:
    if test_data[dimension] == 0 : continue
    test_images.append(test_data[0:dimension])
    test_labels.append(test_data[dimension:])

print(test_labels)
print(len(test_images), len(test_labels))
train_images = np.asarray(train_images, dtype=np.float32)
train_labels = np.asarray(train_labels, dtype=np.float32)

test_images = np.asarray(test_images, dtype=np.float32)
test_labels = np.asarray(test_labels, dtype=np.float32)
print(type(train_images))
print(train_images.shape)

network = models.Sequential()
network.add(layers.Dense(100, activation='relu', input_shape=(dimension,)))
network.add(layers.Dense(2, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


train_labels = to_categorical(train_labels, 2)
test_labels = to_categorical(test_labels, 2)

k = 4
num_val_samples = len(train_datas) // k
for i in range(k):
    val_images = train_images[i * num_val_samples : (i + 1) * num_val_samples]
    val_labels = train_labels[i * num_val_samples : (i + 1) * num_val_samples]

    partial_train_images = np.concatenate([train_images[:i * num_val_samples], train_images[:(i + 1) * num_val_samples]], axis = 0)
    partial_train_labels = np.concatenate([train_labels[:i * num_val_samples], train_labels[:(i + 1) * num_val_samples]], axis = 0)
    #print(partial_train_labels[0])
    network.fit(partial_train_images, partial_train_labels, epochs = 100, batch_size = 1, verbose = 0)

print(test_images[0], test_labels[0])
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)
