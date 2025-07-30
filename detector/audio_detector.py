# detector/audio_detector.py

import torch
import torchaudio
import os
from dotenv import load_dotenv
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Load token
load_dotenv()
token = os.getenv("HF_TOKEN")

# Load model
model_id = "mo-thecreator/Deepfake-audio-detection"
extractor = AutoFeatureExtractor.from_pretrained(model_id, token=token)
model = AutoModelForAudioClassification.from_pretrained(model_id, token=token)
model.eval()

label_map = {int(k): v for k, v in model.config.id2label.items()}

def detect_audio(audio_file, chunk_seconds=3):
    waveform, sr = torchaudio.load(audio_file)

    # Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    waveform = waveform.squeeze()
    duration = waveform.shape[0] / sr
    chunk_samples = int(chunk_seconds * sr)

    segment_preds = []
    fake_segments = 0

    for i in range(0, len(waveform), chunk_samples):
        chunk = waveform[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            break  # skip too short

        inputs = extractor(chunk.numpy(), sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            confidence = probs[label_id].item()
            label = label_map[label_id].capitalize()

        start = i / sr
        end = min((i + chunk_samples) / sr, duration)
        segment_preds.append({
            "start": start,
            "end": end,
            "label": label,
            "confidence": f"{confidence:.2f}"
        })

        if label.lower() == "fake":
            fake_segments += 1

    # Final overall label
    verdict = "Fake" if fake_segments >= 1 else "Real"
    avg_conf = np.mean([float(seg["confidence"]) for seg in segment_preds])

    return verdict, f"{avg_conf:.2f}", segment_preds


def get_waveform_image(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy()

    plt.figure(figsize=(10, 2))
    plt.plot(waveform, color="skyblue")
    plt.title("Audio Waveform", fontsize=12)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Image.open(buf)
