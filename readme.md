# 🕵️‍♂️ Deepfake Detection System

A comprehensive Deepfake Detection System built using Python and Streamlit. This project supports **image, video, and audio** deepfake detection using pre-trained models from Hugging Face. It features real-time analysis, altered frame detection, segment-based audio analysis, and a clean UI.

---

## 🚀 Features

- ✅ **Image Deepfake Detection**
  - Upload JPG/PNG images
  - Frame-by-frame manipulation analysis
  - Visual highlighting of fake regions
  - Verdict summary (Real/Fake) with metrics

- ✅ **Video Deepfake Detection**
  - MP4 file upload
  - Frame extraction and analysis
  - Flagging of altered frames
  - Summary verdict with frame count and fake ratio

- ✅ **Audio Deepfake Detection**
  - WAV/MP3 upload
  - Audio chunking and analysis 
  - Timeline-based detection (e.g. 00:12–00:16 marked fake)
  - Overall verdict and visualization

- ✅ **Unified Streamlit UI**
  - Simple tab-based interface: Image, Video, Audio
  - Upload file ➝ View verdict ➝ Explore detailed analysis
  - Expanded metrics and visualization tools
  - Clean, modern, and responsive design

---

## 🧠 Models Used

| Modality | Model Name | Source |
|----------|------------|--------|
| Image    | `prithivMLmods/deepfake-detector-model-v1` | Hugging Face |
| Video    | Frame-level analysis via image model | Internal |
| Audio    | `mo-thecreator/Deepfake-audio-detection` | Hugging Face |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.13, PyTorch
- **Libraries**:
  - `torch`, `torchvision`, `torchaudio`
  - `transformers`, `facenet-pytorch`
  - `opencv-python`, `Pillow`, `numpy`, `matplotlib`


## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YogeshSharma185/Deepfake-Detection-System.git
   cd deepfake-detector
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Output Highlights

- ✅ Verdict: **Real** / **Fake**
- 🎯 Frame-wise or segment-wise marking
- 📌 Visuals: Annotated image frames, timeline segments, and more
- 📈 Metrics: Confidence score, fake ratio, altered segments

---

## 🙌 Acknowledgements

- Hugging Face for model hosting
- PyTorch ecosystem
- Streamlit for rapid UI development

---

## ✨ Future Improvements

- Add real-time webcam/video stream detection
- Extend to multilingual audio detection
- Add model comparison feature

---

> Built with ❤️ by Yogesh