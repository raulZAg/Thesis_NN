import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
from glob import glob

model = tf.keras.models.load_model('FinalModel.h5')

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")

def save_frame(video_path, save_dir, gap=10):
    name = video_path.split("/")[-1].split(".")[0]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break

        if idx == 0:
            cv2.imwrite(f"{save_path}/{video_name}_{idx}.png", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{video_name}_{idx}.png", frame)

        idx += 1

    return save_path

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

def delete_frame_data(video_path):
    shutil.rmtree(video_path)

if __name__ == "__main__":
    video_paths = glob("videos/*")
    save_dir = "save"
    categories = load_categories()
    output_folder = './result/'

    for path in video_paths:
        save_path = save_frame(path, save_dir, gap=1)

        for root, _, files in os.walk(save_path):
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

        delete_frame_data(save_path)