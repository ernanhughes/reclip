from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests

def load_hf_flat_icons(split="train", limit=None):
    dataset = load_dataset("andrewburns/hf_flat_icons", split=split)
    if limit:
        dataset = dataset.select(range(limit))
    return dataset

def get_image_and_label(example):
    url = example["link"]
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    tags = example["tags"]
    prompt = ", ".join(tags) if isinstance(tags, list) else str(tags)
    return image, prompt