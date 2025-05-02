
from sentence_transformers import SentenceTransformer
import torch

class TextEncoder:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        with torch.no_grad():
            return torch.tensor(self.model.encode(text)).float()
