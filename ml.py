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
    test_images.append(test_data[0:dimension])
    test_labels.append(test_data[dimension:])

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


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

history = network.fit(train_images, train_labels, epochs=100, batch_size= 30, validation_data=(train_images, train_labels))

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

history_dict = history.history
loss = history_dict['loss']
print(history_dict.keys())
valloss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, valloss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()