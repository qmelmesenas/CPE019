import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("violence_detection_cnn.h5")

# Frame extraction function
def extract_frames(video_path, skip=7, max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("⚠️ Could not open video.")
        return []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if frame_count % skip == 0:
            frame = cv2.resize(frame, (128, 128))
            frame = frame.astype('float32') / 255.0
            frames.append(frame)
        frame_count += 1
    cap.release()
    return np.array(frames)

# Predict video violence
def predict_video(frames):
    if len(frames) == 0:
        return "No frames extracted."
    preds = model.predict(frames)
    avg_pred = np.mean(preds)
    label = "Violent" if avg_pred > 0.5 else "Non-Violent"
    return f"Prediction: **{label}** ({avg_pred:.2f} confidence)"

# Streamlit UI
st.title("🎥 Violence Detection in Video")
st.write("Upload a short video and let the model predict if it's violent or not.")

uploaded_file = st.file_uploader("Upload .mp4 video", type=["mp4"])

if uploaded_file is not None:
    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.video(uploaded_file)

    if st.button("Predict Violence"):
        with st.spinner("Processing video..."):
            frames = extract_frames(tmp_path)
            result = predict_video(frames)
        st.success(result)
