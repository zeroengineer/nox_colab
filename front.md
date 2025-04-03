# AI-Generated Image Detection

This repository contains a simple implementation of an AI-generated image detection model using Hugging Face's `microsoft/dit-large` model. The script loads the model and processes images to classify whether they are AI-generated or real.

## Features
- Utilizes `microsoft/dit-large`, a Vision Transformer model for image classification.
- Uses `AutoImageProcessor` for preprocessing images.
- Provides a confidence score along with the prediction.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install transformers torch pillow
```

## Usage
### Import necessary libraries
```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
```

### Load the model and processor
```python
model_name = "microsoft/dit-large"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
```

### Define the prediction function
```python
def predict(image):
    """Predicts if an image is AI-generated or real."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["AI-Generated", "Real"]  # Modify based on actual model output
    confidence, pred = torch.max(probs, dim=-1)
    
    return f"{labels[pred.item()]} ({confidence.item() * 100:.2f}% confidence)"
```

### Example Usage
```python
image = Image.open("path/to/your/image.jpg")
prediction = predict(image)
print(prediction)
```

## Acknowledgments
This project utilizes the `microsoft/dit-large` model from Hugging Face Transformers.

## License
This project is open-source and available under the MIT License.

