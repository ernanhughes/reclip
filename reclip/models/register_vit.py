
import torch
import torch.nn as nn
import timm

class RegisterViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, num_registers=4):
        super().__init__()
        self.base_vit = timm.create_model(model_name, pretrained=pretrained)
        self.num_registers = num_registers
        self.hidden_dim = self.base_vit.embed_dim
        self.registers = nn.Parameter(torch.randn(1, num_registers, self.hidden_dim))
        total_tokens = self.base_vit.pos_embed.shape[1] + num_registers
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.base_vit.pos_embed = None

    def forward(self, x):
        B = x.shape[0]
        x = self.base_vit.patch_embed(x)
        cls_token = self.base_vit.cls_token.expand(B, -1, -1)
        reg_tokens = self.registers.expand(B, -1, -1)
        x = torch.cat((cls_token, reg_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.base_vit.pos_drop(x)
        for blk in self.base_vit.blocks:
            x = blk(x)
        x = self.base_vit.norm(x)
        return torch.nn.functional.normalize(x[:, 0], dim=-1)
