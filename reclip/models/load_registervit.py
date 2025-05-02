
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests

# Example RegisterViT model from HuggingFace if hosted; else we need to define it ourselves
# We'll use a compatible ViT-B/16 backbone for now as placeholder

def load_registervit_model():
    model_name = "facebook/deit-base-distilled-patch16-224"  # temporary stand-in
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def embed_image(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state

if __name__ == "__main__":
    model, processor = load_registervit_model()
    sample_image = "data/sample_images/apple.png"
    embedding = embed_image(model, processor, sample_image)
    print("✅ RegisterViT-compatible embedding shape:", embedding.shape)
