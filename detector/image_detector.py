import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from facenet_pytorch import MTCNN
from dotenv import load_dotenv

# Load token
load_dotenv()
token = os.getenv("HF_TOKEN")

# Load HuggingFace model
model_id = "prithivMLmods/deepfake-detector-model-v1"
processor = AutoImageProcessor.from_pretrained(model_id, token=token)
model = SiglipForImageClassification.from_pretrained(model_id, token=token)
model.eval()
model.to("cpu")  # Force CPU

label_map = {"0": "Fake", "1": "Real"}

# Face detector
mtcnn = MTCNN(keep_all=True, device="cpu")

def detect_image(input_img):
    if isinstance(input_img, np.ndarray):
        img = Image.fromarray(input_img).convert("RGB")
    else:
        img = Image.open(input_img).convert("RGB")

    # Clone original image to draw on
    annotated_img = img.copy()

    # Detect faces
    boxes, _ = mtcnn.detect(img)
    face_count = len(boxes) if boxes is not None else 0

    faces = []
    face_thumbs = []

    if boxes is not None:
        draw = ImageDraw.Draw(annotated_img)
        for box in boxes:
            draw.rectangle(box.tolist(), outline="red", width=3)
            face_crop = img.crop(box)
            faces.append(face_crop)
            face_thumbs.append(face_crop.resize((224, 224)))  # uniform size

    # If no faces, fallback to full image
    if not faces:
        faces = [img]
        face_thumbs = [img.resize((224, 224))]

    all_confidences = []
    all_preds = []

    for face in faces:
        inputs = processor(images=face, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            all_preds.append(label_id)
            all_confidences.append(probs[label_id].item())

    fake_votes = sum(1 for pred in all_preds if pred == 0)
    real_votes = sum(1 for pred in all_preds if pred == 1)

    final_label = 0 if fake_votes >= real_votes else 1
    avg_conf = np.mean(all_confidences)
    verdict = label_map[str(final_label)]

    return verdict, f"{avg_conf:.2f}", face_count, annotated_img, face_thumbs
