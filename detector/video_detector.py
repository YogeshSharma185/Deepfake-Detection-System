import os
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
from transformers import AutoImageProcessor, SiglipForImageClassification
from dotenv import load_dotenv

# Load token and model
load_dotenv()
token = os.getenv("HF_TOKEN")
model_id = "prithivMLmods/deepfake-detector-model-v1"
processor = AutoImageProcessor.from_pretrained(model_id, token=token)
model = SiglipForImageClassification.from_pretrained(model_id, token=token)
model.eval()

def detect_video(video_file):
    # Save uploaded video to disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        temp_path = temp_file.name

    cap = cv2.VideoCapture(temp_path)
    frame_count = 0
    real_count = 0
    fake_count = 0
    fake_frames = []
    display_frames = []
    annotated_frames = []

    label_map = {"0": "Fake", "1": "Real"}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 15 != 0:
            continue

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            verdict = label_map[str(label_id)]
            confidence = probs[label_id].item()

        # Add verdict text on frame
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"{verdict} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if verdict == "Fake" else (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        annotated_frames.append(annotated)
        display_frames.append(annotated)
        if verdict == "Fake":
            fake_count += 1
            fake_frames.append(annotated)
        else:
            real_count += 1

    cap.release()
    os.remove(temp_path)

    verdict = "Fake" if fake_count >= 2 else "Real"
    total_analyzed = real_count + fake_count

    summary = {
        "total_analyzed_frames": total_analyzed,
        "real_frames": real_count,
        "fake_frames": fake_count,
        "fake_percentage": f"{(fake_count / total_analyzed * 100):.1f}%" if total_analyzed else "0.0%"
    }

    # ⬅️ Return 6 values now
    return verdict, total_analyzed, fake_count, summary, fake_frames, annotated_frames
