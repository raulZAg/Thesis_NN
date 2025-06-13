import tensorflow as tf
import numpy as np
import cv2
import os
import shutil

model = tf.keras.models.load_model('FinalModel.h5')

def process_image(image_path):
    image = cv2.imread(image_path)
    input = cv2.resize(image, (299, 299))
    input = np.array(input, dtype=np.float32)
    input /= 255.0
    input = input.reshape((1, 299, 299, 3))
    output = model.predict(input)
    return output

def load_categories():
    categories = []
    with open('data/name of the animals.txt', 'r') as file:
        for line in file:
            categories.append(line.strip())
    return categories

image_folders = './save/videos/'
categories = load_categories()
output_folder = './result/'

for root, _, files in os.walk(image_folders):
    for image_file in files:
        if image_file.endswith('.png'):
            image_path = os.path.join(root, image_file)
            output = process_image(image_path)
            predicted_class_idx = np.argmax(output)
            confidence_score = output[0][predicted_class_idx]

            print(confidence_score)
            print(categories[predicted_class_idx])

            if confidence_score >= 2:
                category_folder = os.path.join(output_folder, categories[predicted_class_idx])
                os.makedirs(category_folder, exist_ok=True)
                shutil.copy(image_path, os.path.join(category_folder, image_file))