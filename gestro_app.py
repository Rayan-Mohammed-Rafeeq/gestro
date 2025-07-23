import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

# Load saved model
MODEL_DIR = "Tensorflow/workspace/exported-models/my_model/saved_model"
detection_model = tf.saved_model.load(MODEL_DIR)
detect_fn = detection_model.signatures['serving_default']

# Load label map
category_index = {
    1: {'id': 1, 'name': 'Hello'},
    2: {'id': 2, 'name': 'Yes'},
    3: {'id': 3, 'name': 'Victory'},
    4: {'id': 4, 'name': 'Rock On'},
}

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img)[tf.newaxis, ...]
    return input_tensor, img

def run_inference(image):
    input_tensor, img_rgb = preprocess(image)
    detections = detect_fn(input_tensor)

    # Extract detection data
    bboxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    h, w, _ = image.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = bboxes[i]
            start_point = (int(xmin * w), int(ymin * h))
            end_point = (int(xmax * w), int(ymax * h))
            color = (0, 255, 0)
            cv2.rectangle(image, start_point, end_point, color, 2)
            label = category_index[class_ids[i]]['name']
            cv2.putText(image, f"{label}: {int(scores[i]*100)}%", (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def infer_from_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        result_img = run_inference(frame)
        return result_img
    else:
        return np.zeros((480, 640, 3), dtype=np.uint8)

def infer_from_image(uploaded_img):
    image = cv2.cvtColor(uploaded_img, cv2.COLOR_RGB2BGR)
    result_img = run_inference(image)
    return result_img

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Hand Gesture Recognition with TensorFlow & OpenCV")

    with gr.Tab("Live Webcam"):
        webcam_btn = gr.Button("Capture and Predict")
        webcam_output = gr.Image(label="Detected Gestures")
        webcam_btn.click(fn=infer_from_webcam, outputs=webcam_output)

    with gr.Tab("Upload Image"):
        upload_input = gr.Image(type="numpy", label="Upload Image")
        upload_output = gr.Image(label="Detected Gestures")
        upload_input.change(fn=infer_from_image, inputs=upload_input, outputs=upload_output)

# Launch
demo.launch()