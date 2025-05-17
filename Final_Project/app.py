import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model
import gdown

# Google Drive link for the model
model_path = 'violence_detection_model.h5'
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/uc?id=1KAZdlLzkrLRi5BoTE2EatRqH30MiU_TN", output=model_path, quiet=False)
# Load the model
model = load_model(model_path)

# Extract frames and return frames + FPS
def extract_frames(video_path, skip=7, max_frames=100):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        st.error("⚠️ Could not open video.")
        return [], 0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if frame_count % skip == 0:
            resized = cv2.resize(frame, (64, 64))
            resized = resized.astype('float32') / 255.0
            frames.append(resized)
        frame_count += 1
    cap.release()
    return np.array(frames), fps

# Predict video violence using frame sequences
def predict_video(frames, original_fps, skip, sequence_length=16):
    if len(frames) < sequence_length:
        return "Video too short to extract full sequence.", [], []

    sequences = []
    for i in range(len(frames) - sequence_length + 1):
        clip = frames[i:i + sequence_length]
        sequences.append(clip)

    sequences = np.array(sequences)  
    preds = model.predict(sequences)

    violent_indices = [i for i, p in enumerate(preds) if p > 0.5]
    avg_pred = np.mean(preds)
    label = "Violent" if avg_pred > 0.5 else "Non-Violent"

    # Use the center frame of each detected violent sequence
    violent_frames = [frames[i + sequence_length // 2] for i in violent_indices]
    timestamps = [round((i + sequence_length // 2) * skip / original_fps, 2) for i in violent_indices]

    result_text = f"Prediction: **{label}** ({avg_pred:.2f} confidence)"
    return result_text, violent_frames, timestamps

# Streamlit UI
st.set_page_config(page_title="Violence Detection", layout="centered")
st.title("Violence Detection in Video")
st.write("Upload a short video and let the model predict if it's violent or not. Frames with violence will be shown below.")

uploaded_file = st.file_uploader("Upload .mp4 video", type=["mp4"])

if uploaded_file is not None:
    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.video(uploaded_file)

    if st.button("Predict Violence"):
        with st.spinner("Processing video..."):
            frames, fps = extract_frames(tmp_path)
            result, violent_frames, timestamps = predict_video(frames, fps, skip=7)

        st.success(result)

        if violent_frames:
            st.subheader("🛑 Violent Frames Detected:")
            for i, (frame, time_sec) in enumerate(zip(violent_frames, timestamps)):
                # Convert frame to displayable image
                frame_bgr = (frame * 255).astype(np.uint8)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Frame {i+1} at {time_sec} sec", use_container_width=True)
        else:
            st.info("✅ No specific violent frames detected.")
