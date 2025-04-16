import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tempfile
import pygame
import time

# Initialize Pygame for audio
pygame.mixer.init()

# Load the trained model (Update path as needed)
try:
    model = load_model("Model.h5")
    st.success(" Model loaded successfully!")
except Exception as e:
    st.error(f" Error loading model: {str(e)}")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Define emotion labels (same order as model output)
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# Mapping emotions to multiple emojis
emotion_to_emoji = {
    "angry": ["😠", "😡", "🤬"],
    "disgust": ["🤢", "🤮", "😷"],
    "fear": ["😨", "😱", "😰"],
    "happy": ["😊", "😁", "😃"],
    "sad": ["😢", "😭", "☹️"],
    "surprise": ["😲", "😯", "🤯"],
    "neutral": ["😐", "😑", "🤨"]
}

# Audio files for each emotion (Add correct paths)
#audio_files = {
 #   'angry': 'audio/angry.wav',
  #  'disgust': 'audio/disgust.wav',
   # 'fear': 'audio/fear.wav',
    #'happy': 'audio/happy.wav',
    #'sad': 'audio/sad.wav',
    #'surprise': 'audio/surprise.wav',
    #'neutral': 'audio/neutral.wav'
#}
# Function to play audio for emotion
def play_audio(emotion):
    try:
        pygame.mixer.music.load(audio_files[emotion])
        pygame.mixer.music.play()
        time.sleep(3)
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")

# Function to get random emoji for each emotion
def get_random_emoji(emotion):
    """Returns a random emoji for the predicted emotion."""
    return random.choice(emotion_to_emoji[emotion])

# Function to preprocess image for prediction
def preprocess_image(image):
    """Preprocess image for model prediction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.resize(gray, (48, 48))  # Resize to 48x48
    gray = img_to_array(gray)  # Convert to array
    gray = np.expand_dims(gray, axis=0)  # Add batch dimension
    gray /= 255.0  # Normalize pixel values (0-1)
    return gray

# Function to detect emotion
def detect_emotion(frame):
    """Detect face and predict emotion."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return 'No Face Detected', ['']

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        processed_roi = preprocess_image(roi)
        prediction = model.predict(processed_roi)
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        return emotion, emotion_to_emoji[emotion]

    return None, None

# Streamlit App
st.title("🎭 **Facial Expression Recognition with Emojis**")
st.sidebar.title("📂 **Input Options**")
option = st.sidebar.selectbox("Select Input Type", ("Image", "Video", "Live Feed"))

# Image Upload Section
if option == "Image":
    uploaded_file = st.file_uploader("📤 **Upload an Image**", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        st.image(frame, caption="Uploaded Image", use_column_width=True)
        emotion, emojis = detect_emotion(frame)
        if emotion:
            st.markdown(f"## 🎯 **Emotion:** {emotion}")
            for emoji in emojis:
                st.markdown(f"<h1 style='font-size:100px'>{emoji}</h1>", unsafe_allow_html=True)
            #play_audio(emotion)
        else:
            st.warning("⚠️ No emotion detected in the image.")

# Video Upload Section
elif option == "Video":
    uploaded_file = st.file_uploader("📤 **Upload a Video**", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_counter = 0
        skip_frames = 5

        if not cap.isOpened():
            st.error("⚠️ Error opening the video file.")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_counter % skip_frames == 0:
                    emotion, emojis = detect_emotion(frame)
                    if emotion:
                        st.markdown(f"## 🎯 **Emotion:** {emotion}")
                        for emoji in emojis:
                            st.markdown(f"<h1 style='font-size:100px'>{emoji}</h1>", unsafe_allow_html=True)
                        #play_audio(emotion)
                frame_counter += 1
                stframe.image(frame, channels="BGR")
            cap.release()

# Live Feed Section
elif option == "Live Feed":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    frame_counter = 0
    skip_frames = 5

    if not cap.isOpened():
        st.error("⚠️ Error accessing the webcam.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_counter % skip_frames == 0:
                emotion, emojis = detect_emotion(frame)
                if emotion:
                    st.markdown(f"## 🎯 **Emotion:** {emotion}")
                    for emoji in emojis:
                        st.markdown(f"<h1 style='font-size:100px'>{emoji}</h1>", unsafe_allow_html=True)
                    #play_audio(emotion)
            frame_counter += 1
            stframe.image(frame, channels="BGR")
        cap.release()

# Footer
st.markdown("---")  # Horizontal line for separation
st.markdown("<h4 style='text-align: center;'>👨‍💻 Developed by RSR</h4>", unsafe_allow_html=True)
