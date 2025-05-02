
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from typing import Optional

class RegisterViT(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small", num_register_tokens=4):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.num_register_tokens = num_register_tokens

        # Determine embedding size from base model config
        hidden_size = self.base_model.config.hidden_size
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size))

        # Final projection (optional)
        self.projector = nn.Identity()

    def forward(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.base_model(**inputs, output_hidden_states=True)
        patch_tokens = outputs.last_hidden_state  # [B, N, D]

        # Repeat register tokens across batch
        batch_size = patch_tokens.size(0)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        # Concatenate register tokens to the end
        extended = torch.cat([patch_tokens, register_tokens], dim=1)

        # Optional: apply attention over extended tokens (future)
        return self.projector(extended[:, -self.num_register_tokens:, :].mean(dim=1))  # Average register tokens

def test_registervit():
    model = RegisterViT()
    image = Image.new("RGB", (224, 224), color="gray")
    with torch.no_grad():
        emb = model(image)
    print("✅ RegisterViT embedding shape:", emb.shape)

if __name__ == "__main__":
    test_registervit()
