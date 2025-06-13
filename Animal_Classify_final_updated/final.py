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

def process_video(video_path):
    frames_data = []
    consecutive_frames = []
    prev_category_idx = None

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    idx = 0
    first_frame_number = None

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break

        input_frame = cv2.resize(frame, (299, 299))
        input_frame = np.array(input_frame, dtype=np.float32)
        input_frame /= 255.0
        input_frame = input_frame.reshape((1, 299, 299, 3))
        output = model.predict(input_frame)
        predicted_class_idx = np.argmax(output)
        confidence_score = output[0][predicted_class_idx]

        print(confidence_score)
        print(categories[predicted_class_idx])

        if confidence_score >= 0.8:
            if prev_category_idx == predicted_class_idx:
                consecutive_frames.append((frame, idx, categories[predicted_class_idx]))
            else:
                if first_frame_number is not None and len(consecutive_frames) >= 10:
                    last_frame_number = idx - 1
                    frames_data.append((consecutive_frames, first_frame_number, last_frame_number, categories[prev_category_idx]))
                consecutive_frames = [(frame, idx, categories[predicted_class_idx])]
                first_frame_number = idx

            prev_category_idx = predicted_class_idx

        idx += 1

    if first_frame_number is not None and len(consecutive_frames) >= 10:
        last_frame_number = idx - 1
        frames_data.append((consecutive_frames, first_frame_number, last_frame_number, categories[prev_category_idx]))

    cap.release()
    return frames_data, video_name

def load_categories():
    categories = []
    with open('data/name of the animals.txt', 'r') as file:
        for line in file:
            categories.append(line.strip())
    return categories

if __name__ == "__main__":
    video_paths = glob("videos/*")
    categories = load_categories()
    output_folder = './result/'

    for path in video_paths:
        frames_data, video_name = process_video(path)

        for consecutive_frames, first_frame_number, last_frame_number, category in frames_data:
            save_path = os.path.join(output_folder, category)
            create_dir(save_path)

            image_folder = os.path.join(save_path, f"{video_name}_{category}_{first_frame_number}-{last_frame_number}")
            create_dir(image_folder)

            for frame, frame_number, _ in consecutive_frames:
                cv2.imwrite(f"{image_folder}/{video_name}_{category}_{frame_number}.png", frame)

            height, width, _ = consecutive_frames[0][0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_video_name = f"{video_name}_{category}_{first_frame_number}-{last_frame_number}.avi"
            output_video_path = os.path.join(save_path, output_video_name)
            out = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

            for frame, _, _ in consecutive_frames:
                out.write(frame)

            out.release()

        # Delete the processed video after processing and saving its frames and videos
        try:
            os.remove(path)
            print(f"Successfully deleted processed video: {path}")
        except Exception as e:
            print(f"ERROR: Failed to delete video {path}: {str(e)}")