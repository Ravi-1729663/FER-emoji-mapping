# 😊 Facial Emotion Recognition & Emoji Mapping System

This project is a **Facial Expression Recognition System** that detects human emotions from facial expressions using deep learning, and maps each detected emotion to a corresponding emoji. The system supports **image input**, **video input**, and **live camera feed**, and outputs both **emotion classification results** and **visual emoji representation**. Voice feedback is also integrated for accessibility and fun!

---

## 🚀 Features

- 🎭 Real-time facial emotion recognition using CNN / Pretrained Models (VGG16/ResNet50/EfficientNet).
- 📸 Accepts input from:
  - Uploaded images
  - Uploaded videos
  - Live webcam feed
- 🧠 Trained on datasets like **CK+** and **AffectNet** for high accuracy.
- 😄 Maps multiple emojis to each emotion randomly for variety.
- 🔊 Emotion-specific **audio feedback** (e.g., laughter for happy, sigh for sad).
- 📊 Detailed **EDA and performance visualization**.
- 📱 Deployed using **Streamlit** for a responsive UI.

---

## 🧠 Supported Emotions & Emojis

| Emotion       | Sample Emojis                     |
|---------------|-----------------------------------|
| Happy 😀      | 😄 😊 😁 😂 🤩                     |
| Sad 😢        | 😢 😞 😔 😟 😿                     |
| Angry 😠      | 😠 😡 🤬 😤                        |
| Surprise 😮   | 😮 😲 😯 🤯                        |
| Disgust 🤢    | 🤢 🤮 😖 🤧                        |
| Fear 😱       | 😱 😨 😰 😬                        |
| Neutral 😐    | 😐 😑 🙃 🫥                        |

---


---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** – Model building and training
- **OpenCV** – Video & webcam processing
- **Streamlit** – Web app deployment
- **Pandas / Matplotlib / Seaborn** – EDA
- **gTTS / Pygame / Playsound** – Audio output
- **Haarcascade / Dlib** – Face detection

---

## 📈 Dataset

- **CK+ (Extended Cohn-Kanade)**
- **AffectNet**  
(Combined for better accuracy, class balancing, and real-world applicability.)

---

## 🧪 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/facial-emotion-emoji-mapper.git
   cd facial-emotion-emoji-mapper/streamlit_app


📊 Model Accuracy
Training Accuracy: ~95%

Validation Accuracy: ~92%

Confusion matrix & detailed plots included in the app's EDA section.

