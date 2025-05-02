import torch
from PIL import Image
from reclip.models.register_vit import RegisterViT


def test_registervit():
    model = RegisterViT()
    dummy_image = Image.new("RGB", (224, 224), color="gray")
    pixel_values = model.preprocess(dummy_image)
    with torch.no_grad():
        embedding = model(pixel_values)
    print("✅ RegisterViT embedding shape:", embedding.shape)

    
if __name__ == "__main__":
    test_registervit()
