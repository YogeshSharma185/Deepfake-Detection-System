import streamlit as st
from detector.image_detector import detect_image
from detector.video_detector import detect_video
import base64  # For video display using html and st.video
st.set_page_config(page_title="🧠 Deepfake Detection System", layout="wide")

# === HEADER ===
st.markdown("<h1 style='text-align: center;'>🎭 Deepfake Detection System</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 18px;'>
🎭 <strong>Deepfake or Real? Upload and Know in Seconds.</strong><br>
We analyze <strong>faces</strong>, <strong>frames</strong>, and <strong>voices</strong> using cutting-edge AI to detect even the most subtle manipulations.
</p>
""", unsafe_allow_html=True)
# === DISCLAIMER ===
with st.expander("⚠️ Disclaimer: About Model Accuracy", expanded=False):
    st.markdown("""
**Please Note:**

- Our deepfake detection models are pre-trained on public datasets like CelebDF, DFD, and others that mostly contain celebrity videos and high-quality recordings.
- If you upload **personal recordings** (e.g. from webcam or phone), especially with **background noise**, **low lighting**, or **local accents**, they **may be falsely flagged as "Fake"**.
- Audio clips with distortion, echo, or low clarity may also be detected as fake due to **training bias**.

**Why this happens:**
- The model hasn't seen your unique voice, environment, or device type before.
- It may misclassify unfamiliar inputs as manipulated.

**We're working to improve this!**
- Future versions will include fine-tuning on diverse real-world data.
- You can still use the confidence score and visual cues to interpret results more carefully.

""")

# === INPUT TYPE SELECTION ===
st.markdown("---")
option = st.radio("📁 Select input type:", ["Image", "Video","Audio"], horizontal=True)

# === IMAGE HANDLING ===
if option == "Image":
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.markdown("#### ✅ Input Received: Image")
        verdict, confidence, face_count, result_img, face_thumbs = detect_image(uploaded)

        verdict_color = "#dc3545" if verdict == "Fake" else "#28a745"
        st.markdown(f"<h3 style='color: {verdict_color};'>🧠 Final Verdict: {verdict}</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{confidence}")
        col2.metric("Faces Detected", face_count)
        col3.metric("Status", verdict)

        # Display fixed-size full image (small & neat)
        st.markdown("### 📷 Annotated Input Image")
        st.image(result_img, width=400)

        # Display cropped faces if verdict is Fake
        if verdict == "Fake" and face_thumbs:
            st.markdown("### 🖼️ Detected Fake Face Regions")
            cols = st.columns(4)
            for i, thumb in enumerate(face_thumbs):
                with cols[i % 4]:
                    st.image(thumb, caption=f"Face {i+1}", use_container_width=True)


# === VIDEO HANDLING ===
elif option == "Video":
    uploaded = st.file_uploader("Upload a video (MP4/AVI)", type=["mp4", "avi", "mov"])
    if uploaded:
        st.markdown("#### ✅ Input Received: Video")

        # Read and show the original uploaded video (resized preview)
        video_bytes = uploaded.read()

        import base64
        b64_video = base64.b64encode(video_bytes).decode()
        video_html = f"""
            <video controls style='width: 400px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        """
        st.markdown("### 🎞️ Uploaded Video Preview")
        st.markdown(video_html, unsafe_allow_html=True)

        # Analyze the video
        with st.spinner("🔍 Analyzing video frame-by-frame..."):
            from io import BytesIO
            fake_stream = BytesIO(video_bytes)
            verdict, total_frames, fake_frame_count, summary, fake_frames, _ = detect_video(fake_stream)

        # Verdict Display
        verdict_color = "#dc3545" if verdict == "Fake" else "#28a745"
        st.markdown(f"<h3 style='color: {verdict_color};'>🧠 Final Verdict: {verdict}</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Frames Analyzed", total_frames)
        col2.metric("Fake Frames", fake_frame_count)
        col3.metric("Fake %", summary["fake_percentage"])

        st.markdown("### 📊 Detection Summary")
        st.dataframe(summary, use_container_width=True)

        if fake_frames:
            st.markdown("### 🖼️ Detected Fake Frames")
            cols = st.columns(4)
            for i, frame in enumerate(fake_frames):
                with cols[i % 4]:
                    st.image(frame, use_container_width=True, caption=f"Fake Frame {i+1}")
        else:
            st.success("✅ No fake frames detected in analyzed frames.")

# handling audio detection 

elif option == "Audio":
    uploaded = st.file_uploader("Upload an audio file (WAV/MP3/FLAC)", type=["wav", "mp3", "flac"])
    if uploaded:
        st.markdown("#### ✅ Input Received: Audio")
        st.audio(uploaded, format='audio/wav')

        # Save to file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        from detector.audio_detector import detect_audio, get_waveform_image
        verdict, confidence, segment_preds = detect_audio(tmp_path)
        waveform_img = get_waveform_image(tmp_path)

        verdict_color = "#dc3545" if verdict == "Fake" else "#28a745"
        st.markdown(f"<h3 style='color: {verdict_color};'>🧠 Final Verdict: {verdict}</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Confidence", confidence)
        col2.metric("Status", verdict)

        st.markdown("### 🎵 Audio Waveform")
        st.image(waveform_img, use_container_width=True)

        st.markdown("### 📊 Time-based Analysis")
        for seg in segment_preds:
            start = f"{int(seg['start']//60):02}:{int(seg['start']%60):02}"
            end = f"{int(seg['end']//60):02}:{int(seg['end']%60):02}"
            label = seg["label"]
            conf = seg["confidence"]
            emoji = "🟥" if label == "Fake" else "🟩"
            st.markdown(f"{emoji} `{start} – {end}` → **{label}** ({conf})")
