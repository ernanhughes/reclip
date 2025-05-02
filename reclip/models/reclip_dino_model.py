
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from datasets import load_dataset
from io import BytesIO
import requests

class ReclipVisionModel:
    def __init__(self, model_name="facebook/dinov2-small"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Mean pooling

def get_random_icon():
    dataset = load_dataset("andrewburns/hf_flat_icons", split="train")
    sample = dataset.shuffle(seed=42)[0]
    image_url = sample["link"]
    tags = sample["tags"]
    image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
    return image, tags
