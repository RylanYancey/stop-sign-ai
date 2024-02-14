
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense
from keras import Input

from load import load_data

train_labels, train_images = load_data('train')

train_labels = tf.constant(train_labels)
train_images = tf.constant(train_images)

model = Sequential()
model.add(Input(shape=(224,224,3)))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=20, batch_size=10)
acc, loss = model.evaluate(train_images, train_labels)

print(f'Accuracy: {acc}')