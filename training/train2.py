import torch
from torch.utils.data import Dataset, DataLoader
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from models.fusion_transformer2 import MultimodalFusionLLM
from models.loss import MultimodalLoss
from utils.model_utils import load_llm_and_tokenizer
from utils.latent_utils import get_video_latents

class MultimodalDataset(Dataset):
    def __init__(self, aligned_data_path, video_latents_dir, tokenizer, max_length=128):
        self.aligned_data = []
        with open(aligned_data_path, 'r') as f:
            for line in f:
                self.aligned_data.append(json.loads(line))
        
        self.video_latents_dir = video_latents_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.aligned_data)

    def __getitem__(self, idx):
        entry = self.aligned_data[idx]
        text = entry['text']
        associated_video_files = entry['associated_video_files']

        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        video_latents = get_video_latents(associated_video_files, self.video_latents_dir)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'video_latents': video_latents,
            'target_text_ids': input_ids.clone()
        }

def train_model(model, dataset, loss_fn, optimizer, num_epochs=3, batch_size=4, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            video_latents = batch['video_latents'].to(device)
            target_text_ids = batch['target_text_ids'].to(device)

            optimizer.zero_grad()

            fused_embeddings = model(input_ids, attention_mask, video_latents)
            lm_logits = model.llm.lm_head(fused_embeddings)

            with torch.no_grad():
                llm_outputs_for_contrastive = model.llm.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                text_embeddings_for_contrastive = llm_outputs_for_contrastive.last_hidden_state

            total_batch_loss, lm_batch_loss, contrastive_batch_loss = loss_fn(
                fused_embeddings=lm_logits,
                target_text_ids=target_text_ids,
                text_attention_mask=attention_mask,
                video_latents=video_latents,
                text_embeddings_from_llm=text_embeddings_for_contrastive
            )

            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Total Loss: {total_batch_loss.item():.4f}, LM Loss: {lm_batch_loss.item():.4f}, Contrastive Loss: {contrastive_batch_loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"./model_epoch_{epoch+1}.pt")
        print(f"Model saved after epoch {epoch+1}")

if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    aligned_data_path = os.path.join(root_dir, "preprocessing", "aligned_data.jsonl")
    video_latents_dir = os.path.join(root_dir, "preprocessing", "video_latents")

    llm_model_name = 'gpt2'
    video_latent_dim = 512
    projection_dim = 768
    num_epochs = 3
    batch_size = 2
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm, tokenizer = load_llm_and_tokenizer(llm_model_name, device)
    model = MultimodalFusionLLM(llm=llm, video_latent_dim=video_latent_dim, projection_dim=projection_dim)
    loss_fn = MultimodalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataset = MultimodalDataset(aligned_data_path, video_latents_dir, tokenizer)

    print(f"Starting training on {device}...")
    train_model(model, dataset, loss_fn, optimizer, num_epochs, batch_size, device)
    print("Training complete.")
