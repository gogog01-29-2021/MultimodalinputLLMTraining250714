
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os
import numpy as np

from models.fusion_transformer import MultimodalFusionLLM
from models.loss import MultimodalLoss

class MultimodalDataset(Dataset):
    def __init__(self, aligned_data_path, video_latents_dir, tokenizer, max_length=128):
        self.aligned_data = []
        with open(aligned_data_path, \'r\') as f:
            for line in f:
                self.aligned_data.append(json.loads(line))
        
        self.video_latents_dir = video_latents_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.aligned_data)

    def __getitem__(self, idx):
        entry = self.aligned_data[idx]
        text = entry[\'text\']
        associated_video_files = entry[\'associated_video_files\']

        # Tokenize text
        encoding = self.tokenizer(text, return_tensors=\'pt\', padding=\'max_length\', truncation=True, max_length=self.max_length)
        input_ids = encoding[\'input_ids\'].squeeze()
        attention_mask = encoding[\'attention_mask\'].squeeze()

        # Load video latents
        video_latents_list = []
        for video_file in associated_video_files:
            latent_filename = os.path.basename(video_file).replace(\".mp4\", \"_latents.npy\")
            latent_path = os.path.join(self.video_latents_dir, latent_filename)
            if os.path.exists(latent_path):
                video_latents_list.append(torch.tensor(np.load(latent_path), dtype=torch.float32))
            else:
                print(f"Warning: Latent file not found for {video_file}. Skipping.")
        
        if not video_latents_list:
            # If no video latents are found, provide a dummy tensor of appropriate shape
            # This assumes a single video latent vector per text entry for simplicity
            video_latents = torch.zeros(1, 512) # Dummy data for one frame
        else:
            # Average video latents if multiple are associated with one text entry
            # This is a simplification; more sophisticated pooling or sequence modeling could be used
            video_latents = torch.mean(torch.stack(video_latents_list), dim=0)

        return {
            \'input_ids\': input_ids,
            \'attention_mask\': attention_mask,
            \'video_latents\': video_latents,
            \'target_text_ids\': input_ids.clone() # For LM loss, target is usually the input itself shifted
        }

def train_model(model, dataset, loss_fn, optimizer, num_epochs=3, batch_size=4, device=\'cpu\'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch[\'input_ids\'].to(device)
            attention_mask = batch[\'attention_mask\'].to(device)
            video_latents = batch[\'video_latents\'].to(device)
            target_text_ids = batch[\'target_text_ids\'].to(device)

            optimizer.zero_grad()

            # Forward pass through the multimodal fusion LLM
            lm_logits, text_embeddings_from_llm = model(input_ids, attention_mask, video_latents)

            # Calculate loss
            total_batch_loss, lm_batch_loss, contrastive_batch_loss = loss_fn(
                lm_logits=lm_logits,
                target_text_ids=target_text_ids,
                video_latents=video_latents,
                text_embeddings_from_llm=text_embeddings_from_llm
            )

            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Total Loss: {total_batch_loss.item():.4f}, LM Loss: {lm_batch_loss.item():.4f}, Contrastive Loss: {contrastive_batch_loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"./model_epoch_{epoch+1}.pt")
        print(f"Model saved after epoch {epoch+1}")

if __name__ == \'__main__\':
    # Configuration
    aligned_data_path = \'/home/ubuntu/natural_video_llm/preprocessing/aligned_data.jsonl\'
    video_latents_dir = \'/home/ubuntu/natural_video_llm/preprocessing\'
    llm_model_name = \'gpt2\'
    video_latent_dim = 512 # This should match the output of extract_video_latents.py
    projection_dim = 768 # This should match the hidden size of the LLM (e.g., GPT-2)
    num_epochs = 3
    batch_size = 2 # Small batch size for demonstration
    learning_rate = 5e-5
    device = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')

    # Initialize tokenizer, model, loss, and optimizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = MultimodalFusionLLM(llm_model_name=llm_model_name, video_latent_dim=video_latent_dim, projection_dim=projection_dim)
    loss_fn = MultimodalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create dataset
    dataset = MultimodalDataset(aligned_data_path, video_latents_dir, tokenizer)

    # Start training
    print(f"Starting training on {device}...")
    train_model(model, dataset, loss_fn, optimizer, num_epochs, batch_size, device)

    print("Training complete.")


