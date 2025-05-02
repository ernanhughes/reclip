
import torch
from torchvision.models import vit_s_16, ViT_S_16_Weights
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from io import BytesIO
import requests

def load_dino_model():
    weights = ViT_S_16_Weights.IMAGENET1K_V1
    model = vit_s_16(weights=weights)
    model.eval()
    processor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
    ])
    return model, processor

def get_random_icon():
    dataset = load_dataset("andrewburns/hf_flat_icons", split="train")
    sample = dataset.shuffle(seed=42)[0]
    image_url = sample["link"]
    tags = sample["tags"]
    image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
    return image, tags

def run_dino_inference():
    model, processor = load_dino_model()
    image, tags = get_random_icon()
    print("🖼 Tags:", tags)
    input_tensor = processor(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    print("✅ DINOv2-Small embedding shape:", output.shape)
    return output

if __name__ == "__main__":
    run_dino_inference()
