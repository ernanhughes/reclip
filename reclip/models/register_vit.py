import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoImageProcessor


class RegisterViT(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small", num_register_tokens=4):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.num_register_tokens = num_register_tokens

        hidden_size = self.base_model.config.hidden_size
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size))

        self.projector = nn.Identity()

    def preprocess(self, image):
        """Converts a PIL image into a tensor suitable for the model"""
        return self.processor(images=image, return_tensors="pt")["pixel_values"]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(pixel_values=pixel_values, output_hidden_states=True)
        patch_tokens = outputs.last_hidden_state  # shape [B, N, D]

        B = patch_tokens.size(0)
        register_tokens = self.register_tokens.expand(B, -1, -1)

        extended = torch.cat([patch_tokens, register_tokens], dim=1)
        return self.projector(extended[:, -self.num_register_tokens:, :].mean(dim=1))