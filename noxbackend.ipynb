from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image

# Load a better model for AI-generated image detection
model_name = "microsoft/dit-large"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def predict(image):
    """Predicts if an image is AI-generated or real."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Adjust labels if needed
    labels = ["AI-Generated", "Real"]  # Modify based on actual model output
    confidence, pred = torch.max(probs, dim=-1)

    return f"{labels[pred.item()]} ({confidence.item() * 100:.2f}% confidence)"
