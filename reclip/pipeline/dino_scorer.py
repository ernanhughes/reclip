
import torch
import torch.nn.functional as F
from reclip.models.reclip_dino_model import ReclipVisionModel, get_random_icon
from reclip.utils.model_recaption import recaption_with_model  # Assume this returns a string caption
from transformers import AutoTokenizer, AutoModel

class ReclipScorer:
    def __init__(self, text_model_name="intfloat/e5-small"):
        self.vision = ReclipVisionModel()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_model.eval()

    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)

    def score_similarity(self, image, text: str) -> float:
        image_emb = self.vision.embed_image(image)
        text_emb = self.embed_text(text)
        similarity = F.cosine_similarity(image_emb, text_emb, dim=0)
        return similarity.item()

if __name__ == "__main__":
    scorer = ReclipScorer()
    image, tags = get_random_icon()
    prompt = recaption_with_model(", ".join(tags))
    print("🧠 Prompt:", prompt)
    sim = scorer.score_similarity(image, prompt)
    print(f"🔍 Cosine similarity (image vs prompt): {sim:.4f}")
