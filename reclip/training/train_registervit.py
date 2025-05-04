import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os
import time
import requests
from registervit_model import RegisterViT  # Adjust this if needed to: from reclip.models.register_vit import RegisterViT

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class IconDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", max_len=512):
        self.dataset = load_dataset("andrewburns/hf_flat_icons", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        image = Image.open(BytesIO(requests.get(entry["link"]).content)).convert("RGB")
        text = ", ".join(entry["tags"])
        return image, text

def collate_fn(batch):
    images, texts = zip(*batch)
    return list(images), list(texts)

class RegisterTrainer:
    def __init__(self, model_name="facebook/dinov2-small", text_model_name="intfloat/e5-small", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = RegisterViT(model_name=model_name).to(device)
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        self.text_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.optimizer = optim.AdamW([self.model.register_tokens], lr=1e-3)
        self.loss_fn = nn.CosineEmbeddingLoss()
        os.makedirs("checkpoints", exist_ok=True)

    def train_one_epoch(self, dataloader, epoch=0, use_wandb=False):
        self.model.train()
        total_loss = 0
        for step, (images, texts) in enumerate(dataloader):
            pixel_values = torch.cat([self.model.preprocess(img) for img in images]).to(self.device)
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                text_emb = self.text_model(**inputs).last_hidden_state.mean(dim=1)

            image_emb = self.model(pixel_values)
            targets = torch.ones(image_emb.size(0), device=self.device)

            loss = self.loss_fn(image_emb, text_emb, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch+1} Step {step}: Loss = {loss.item():.4f}")
                if use_wandb:
                    wandb.log({"loss": loss.item(), "epoch": epoch+1, "step": step})

        avg_loss = total_loss / len(dataloader)
        print(f"📦 Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        self.save_checkpoint(epoch+1)
        return avg_loss

    def save_checkpoint(self, epoch):
        ckpt_path = f"checkpoints/registervit_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "register_tokens": self.model.register_tokens.detach().cpu(),
        }, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

def run_training(use_wandb=False):
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="reclip-registervit", name=f"run_{int(time.time())}", config={"model": "dinov2-small", "batch_size": 8})
    elif use_wandb:
        print("⚠️ wandb not installed. Logging disabled.")

    dataset = IconDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    trainer = RegisterTrainer()

    for epoch in range(3):
        trainer.train_one_epoch(dataloader, epoch=epoch, use_wandb=use_wandb)

if __name__ == "__main__":
    run_training(use_wandb=True)
