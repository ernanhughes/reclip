
import torch
from torchvision import transforms
from PIL import Image
from reclip.models.text_encoder import TextEncoder
from reclip.models.register_vit import RegisterViT

def compute_similarity(image_path, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = RegisterViT().to(device)
    text_encoder = TextEncoder()
    image_emb = model(image_tensor)
    text_emb = text_encoder.encode(prompt).to(device)

    similarity = torch.nn.functional.cosine_similarity(image_emb, text_emb.unsqueeze(0)).item()
    return similarity
