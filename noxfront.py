import gradio as gr
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image

# Load the model
model_name = "microsoft/dit-large"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def predict(image):
    """Predicts if an image is AI-generated or real."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["AI-Generated", "Real"]
    confidence, pred = torch.max(probs, dim=-1)

    return f"{labels[pred.item()]} ({confidence.item() * 100:.2f}% confidence)"

# Simplified CSS with Poppins font
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

body {
    font-family: 'Poppins', sans-serif !important;
    background: linear-gradient(135deg, #0D1B2A, #1B263B) !important;
}

.gradio-container {
    max-width: 800px !important;
    margin: 0 auto !important;
    background: none !important;
}

.gr-interface {
    background: rgba(31, 41, 55, 0.8) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
}

.gr-button-primary {
    background: #1E40AF !important;
    border: none !important;
    font-family: 'Poppins', sans-serif !important;
}

.title {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(to right, #3B82F6, #60A5FA) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
}

.description {
    color: rgba(255, 255, 255, 0.8) !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
}
"""

# Create the interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Result"),
    title="NOXIAN",
    description="Upload an image to detect if it's AI-generated or real",
    theme="default",
    css=custom_css
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
