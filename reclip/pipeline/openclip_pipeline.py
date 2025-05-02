
import torch
from PIL import Image
import open_clip
from torchvision import transforms

# Load model and tokenizer globally (to avoid repeated loading)
model_name = 'ViT-H-14'
pretrained = 'laion2b_s32b_b79k'
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)
model.eval()

def compute_similarity_openclip(image_path, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer([prompt])
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    return similarity
