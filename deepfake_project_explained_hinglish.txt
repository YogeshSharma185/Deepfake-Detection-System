
🧠 Deepfake Detection Project - Full Explanation (Hinglish)

📁 Project Overview:
Yeh project image, video, aur audio ke deepfake detection ke liye banaya gaya hai. User koi bhi file upload karta hai, aur humara AI model batata hai ki yeh "Real" hai ya "Fake".

---------------------------------------------
🔧 app.py (Main Streamlit App)
---------------------------------------------
- Ye file Streamlit UI ka main entry point hai.
- User yaha pe image, video, ya audio upload karta hai.
- `detect_image()`, `detect_video()` aur `detect_audio()` functions ko call karta hai jo ki `detector` folder me hain.

👇 Example:
```python
from detector.image_detector import detect_image
```
Yeh line `image_detector.py` file ka function import karti hai.

- UI ke liye Streamlit ka layout use hota hai:
  - `st.radio()` -> user select karega: Image, Video ya Audio
  - `st.file_uploader()` -> file upload karne ke liye
  - `st.image()`, `st.video()` -> results dikhane ke liye

- Har section (Image/Video/Audio) ke liye ek if-block likha gaya hai.

---------------------------------------------
🧠 image_detector.py (Image Detection)
---------------------------------------------
- User jo image upload karta hai, usse PIL image me convert kiya jata hai.
- `facenet-pytorch` ka `MTCNN` use karke faces detect karte hain.
- Har detected face ko crop karke HuggingFace model ko dete hain:
```python
model_id = "prithivMLmods/deepfake-detector-model-v1"
```
- Model batata hai face fake hai ya real, aur confidence score deta hai.
- Faces ke bounding boxes red color se draw kiye jaate hain.
- Cropped faces return kiye jaate hain UI me dikhane ke liye.

Return values:
- `verdict`, `confidence`, `face_count`, `annotated_image`, `face_crops`

---------------------------------------------
🎥 video_detector.py (Video Detection)
---------------------------------------------
- Upload hui video ko temp file me save kiya jata hai.
- `cv2.VideoCapture()` se har 15th frame ko read karte hain.
- Har frame ko image banake HuggingFace model pe detection karte hain.
- Agar verdict "Fake" hota hai to frame ko highlight karke list me store kar lete hain.
- Last me summary generate hoti hai:
  - Total frames analyzed
  - Fake frames count
  - Fake percentage
- Fake frames UI pe grid me dikhaye jaate hain.

Return values:
- `verdict`, `total_frames`, `fake_count`, `summary`, `fake_frames`

---------------------------------------------
🎵 audio_detector.py (Audio Detection)
---------------------------------------------
- User se `.mp3`, `.wav`, etc file lete hain.
- `torchaudio` se audio load karke 16kHz me resample karte hain.
- `MelodyMachine/Deepfake-audio-detection` model ko audio diya jata hai.
- Model verdict (Fake/Real) aur confidence return karta hai.

Optional:
- In future, audio segments (like 00:12–00:16) pe bhi verdict show kar sakte hain.

---------------------------------------------
📁 requirements.txt
---------------------------------------------
Yeh file sari required libraries list karti hai:

- `streamlit`: UI ke liye
- `opencv-python`: video reading ke liye
- `torch`, `transformers`: deep learning models ke liye
- `facenet-pytorch`: face detection ke liye
- `huggingface_hub`, `datasets`: model download/train ke liye
- `torchaudio`: audio processing ke liye
- `matplotlib`: audio visualization ke liye
- `python-dotenv`: .env se token load karne ke liye

---------------------------------------------
🔐 Token System (HuggingFace)
---------------------------------------------
- `.env` file me `HF_TOKEN=your_token_here` likha jata hai.
- `python-dotenv` se token load hota hai:
```python
token = os.getenv("HF_TOKEN")
```

---------------------------------------------
🔗 Flow Summary:
1. User file upload karta hai via UI (`app.py`)
2. Depending on file type, respective detector function call hota hai.
3. Model file ko analyze karta hai aur result deta hai.
4. Result (verdict + confidence + visuals) Streamlit pe show hote hain.

Project modular hai — har media type ke liye alag detector likha gaya hai.

✅ All ready for presentation or deployment!
