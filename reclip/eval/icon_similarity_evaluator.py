
import torch
from PIL import Image
from torchvision import transforms
from reclip.data.hf_flat_icons_loader import load_hf_flat_icons, get_image_and_label

# Optional imports, will be dynamically loaded
open_clip = None
RegisterViT = None
TextEncoder = None

class IconSimilarityEvaluator:
    def __init__(self, model_name="openclip"):
        self.model_name = model_name.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_name == "openclip":
            global open_clip
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14', pretrained='laion2b_s32b_b79k'
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
            self.model.eval().to(self.device)

        elif self.model_name == "registervit":
            global RegisterViT, TextEncoder
            from reclip.models.register_vit import RegisterViT
            from reclip.models.text_encoder import TextEncoder
            self.model = RegisterViT().to(self.device).eval()
            self.text_encoder = TextEncoder()
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def compute_similarity(self, image: Image.Image, prompt: str) -> float:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model_name == "openclip":
                image_emb = self.model.encode_image(image_tensor)
                text_emb = self.model.encode_text(self.tokenizer([prompt]))
            elif self.model_name == "registervit":
                image_emb = self.model(image_tensor)
                text_emb = self.text_encoder.encode(prompt).to(self.device)
            else:
                raise ValueError("Invalid model_name")

            # Normalize
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            return torch.nn.functional.cosine_similarity(image_emb, text_emb.unsqueeze(0)).item()

    def evaluate_batch(self, dataset, limit=5):
        results = []
        for i, example in enumerate(dataset.select(range(limit))):
            image, label = get_image_and_label(example)
            prompt = f"An icon about {label}"
            try:
                sim = self.compute_similarity(image, prompt)
                results.append((label, sim))
            except Exception as e:
                results.append((label, f"ERROR: {e}"))
        return results
