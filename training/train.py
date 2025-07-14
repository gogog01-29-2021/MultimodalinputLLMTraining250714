
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
            # Assuming video_file is like ".mp4" and latents are ".npy"
            latent_filename = os.path.basename(video_file).replace(\".mp4\", \"_latents.npy\")
            latent_path = os.path.join(self.video_latents_dir, latent_filename)
            if os.path.exists(latent_path):
                video_latents_list.append(torch.tensor(np.load(latent_path), dtype=torch.float32))
            else:
                # Handle missing latent files (e.g., return a zero tensor or skip)
                print(f"Warning: Latent file not found for {video_file}. Using dummy data.")
                video_latents_list.append(torch.zeros(1, 512)) # Dummy data
        
        if not video_latents_list:
            # If no video latents are found, provide a dummy tensor
            video_latents = torch.zeros(1, 512) # Single dummy frame
        else:
            # Pad or truncate video latents to a fixed number of frames if necessary
            # For simplicity, let's just take the first one or concatenate if multiple
            video_latents = torch.cat(video_latents_list, dim=0) # Concatenate all latents
            # You might want to implement padding/truncation here to a fixed number of frames
            # For now, let's just take the first frame's latent if there are many
            if video_latents.shape[0] > 1:
                video_latents = video_latents[0].unsqueeze(0) # Take first frame's latent
            

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
            # The MultimodalFusionLLM returns fused_embeddings
            fused_embeddings = model(input_ids, attention_mask, video_latents)

            # To calculate LM loss, we need to pass fused_embeddings through the LLM's LM head
            # This is a simplified example. In a real scenario, you'd integrate this more deeply.
            # For GPT-2, the LM head is usually `model.llm.lm_head`
            lm_logits = model.llm.lm_head(fused_embeddings)

            # Calculate loss
            # The loss function expects `text_embeddings_from_llm` for contrastive loss.
            # We can get this from the base LLM output before fusion.
            with torch.no_grad(): # Don't compute gradients for this part if it's just for contrastive loss
                llm_outputs_for_contrastive = model.llm.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                text_embeddings_for_contrastive = llm_outputs_for_contrastive.last_hidden_state

            total_batch_loss, lm_batch_loss, contrastive_batch_loss = loss_fn(
                fused_embeddings=lm_logits, # Pass logits for LM loss
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


