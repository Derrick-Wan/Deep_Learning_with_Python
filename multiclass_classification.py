from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1
    # The way below works fine too.
    # result2 = np.zeros((len(sequences), dimension))
    # for i in range(len(sequences)):
    #     result2[i, sequences[i]] = 1
    return result


X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

train_labels_OH = to_categorical(train_labels)
test_labels_OH = to_categorical(test_labels)

# def to_one_hot(labels, dimension=46):
#     result = np.zeros(len(labels), dimension)
#     for i, label in enumerate(labels):
#         result[i, label] = 1
#     return result
# y_train = to_one_hot(train_labels)
# y_test = to_one_hot(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

X_val = X_train[:1000]
partial_X_train = X_train[1000:]

y_val = train_labels_OH[:1000]
partial_y_train = train_labels_OH[1000:]

output = model.fit(partial_X_train, partial_y_train, epochs=20, batch_size=512, validation_data=(X_val, y_val))

# Draw the picture of the train.

import matplotlib.pyplot as plt

loss = output.history['loss']
val_loss = output.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Testing loss')
plt.title('Training and testing loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Training and validation accuracy
acc = output.history['acc']
val_acc = output.history['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Testing acc')
plt.title('Training and testing acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
