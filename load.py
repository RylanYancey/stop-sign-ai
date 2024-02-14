
import csv
import os
import re

from pathlib import Path

import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, smart_resize

import numpy as np

def load_data(file: str):
    indices = []
    labels = []
    images = []

    regex = re.compile(r'\d+')

    with open(file + '/labels.tsv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            else:
                indices.append([int(x) for x in regex.findall(row[0])][0])

    directory = Path(file + '/').glob('*')
    for filename in directory:
        filename = str(filename)
        if filename.__contains__('.tsv') or filename.__contains__('DS'):
            continue
        else:
            index = [int(x) for x in regex.findall(filename)][0]
            img = tf.io.read_file(filename)
            img = tf.io.decode_image(img, channels=3)
            # Use Smart Resize to preserve aspect ratio
            img = smart_resize(img, size=(224,224))
            img = img_to_array(img)
            # Normalize pixel values to be in the range [0,1]
            img = img / 255.0

            images.append(tf.constant(img))    
            if indices.__contains__(index):
                labels.append(tf.constant(1.0))
            else:
                labels.append(tf.constant(0.0))

    return labels, images
    