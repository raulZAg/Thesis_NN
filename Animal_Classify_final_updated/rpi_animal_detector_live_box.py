#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
CATEGORIES_FILE   = 'data/name of the animals.txt'
MODEL_PATH        = 'FinalModel.tflite'
OUTPUT_DIR        = './result/'
FRAME_SIZE        = (640, 480)
DETECTION_THRESH  = 0.5    # now a probability threshold
MIN_FRAMES        = 10
BOX_COLOR         = (0, 255, 0)  # green
TEXT_COLOR        = (255, 255, 255)  # white
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE        = 0.7
THICKNESS         = 2
# ──────────────────────────────────────────────────────────────────────────────

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def load_categories(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_tflite_model(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def main():
    create_dir(OUTPUT_DIR)

    # Load labels and model
    print("Loading categories...")
    categories = load_categories(CATEGORIES_FILE)
    print(f"Loaded {len(categories)} categories")

    print("Loading TFLite model...")
    interpreter = load_tflite_model(MODEL_PATH)
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_h, in_w = input_details[0]['shape'][1:3]
    print("Model loaded.")

    # Camera init
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={'size': FRAME_SIZE, 'format': 'RGB888'}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("Camera started.")

    # Loop
    consecutive = []
    prev_idx = None
    frame_count = 0

    print("Starting detection... (press 'q' in window to quit)")
    try:
        while True:
            frame = picam2.capture_array()  # (H,W,3)

            # Preprocess & infer
            img = cv2.resize(frame, (in_w, in_h))
            inp = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            logits = interpreter.get_tensor(output_details[0]['index'])[0]
            probs  = softmax(logits)
            idx    = int(np.argmax(probs))
            prob   = float(probs[idx])

            # Draw full-frame box & label if above threshold
            if prob >= DETECTION_THRESH:
                label = f"{categories[idx]}: {prob*100:.1f}%"
                # full-frame rectangle
                cv2.rectangle(frame, (0,0), (FRAME_SIZE[0]-1, FRAME_SIZE[1]-1), BOX_COLOR, THICKNESS)
                # text background
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
                cv2.rectangle(frame, (5,5), (5+tw+4,5+th+4), BOX_COLOR, cv2.FILLED)
                # text
                cv2.putText(frame, label, (7, 7+th), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS)

                # console log
                print(f"Detected: {label}")

                # buffer logic
                if idx == prev_idx:
                    consecutive.append((frame.copy(), frame_count))
                else:
                    if len(consecutive) >= MIN_FRAMES:
                        save_detection(consecutive, categories[prev_idx])
                    consecutive = [(frame.copy(), frame_count)]
                prev_idx = idx

            # Show live feed
            cv2.imshow("Live Feed", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        if len(consecutive) >= MIN_FRAMES:
            save_detection(consecutive, categories[prev_idx])
        print("Cleanup done")

def save_detection(buffer, label):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    label_dir = os.path.join(OUTPUT_DIR, label)
    create_dir(label_dir)
    img_dir = os.path.join(label_dir, f"{label}_{timestamp}")
    create_dir(img_dir)

    # save frames
    for frame, idx in buffer:
        cv2.imwrite(os.path.join(img_dir, f"frame_{idx}.png"), frame)

    # save video
    h, w, _ = buffer[0][0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_path = os.path.join(label_dir, f"{label}_{timestamp}.avi")
    out = cv2.VideoWriter(vid_path, fourcc, 10, (w, h))
    for frame, _ in buffer:
        out.write(frame)
    out.release()

    print(f"Saved {len(buffer)} frames and video: {vid_path}")

if __name__ == '__main__':
    main()