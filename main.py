
import tensorflow as tf
from keras.models import Sequential
from keras import Input, layers
from keras.applications import ResNet50

import load

train_labels, train_images = load.load_data('train')

train_labels = tf.constant(train_labels)
train_images = tf.constant(train_images)

model = ResNet50(
    classes=1,
    classifier_activation="sigmoid",
    weights=None,
    include_top=False
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

acc, loss = model.evaluate(train_images, train_labels)

print(f'Test Accuracy: {acc}')