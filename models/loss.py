
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100) # -100 is default for ignoring padding

    def forward(self, fused_embeddings, target_text_ids, text_attention_mask, video_latents, text_embeddings_from_llm):
        # 1. Next Token Prediction Loss (Language Modeling Loss)
        # Assuming fused_embeddings are the enhanced representations for next token prediction
        # In a real scenario, fused_embeddings would be passed through the LLM's LM head
        # For simplicity, let's assume fused_embeddings are directly comparable to target_text_ids
        
        # To calculate next token prediction loss, we need logits. 
        # This simplified example assumes fused_embeddings are already logits or can be mapped to them.
        # In a full LLM integration, you'd pass fused_embeddings through the LLM's final linear layer (LM head).
        
        # For demonstration, let's assume fused_embeddings are the output of the LLM's hidden states
        # and we need to project them to vocabulary size for next token prediction.
        # This part would typically be handled within the LLM's forward pass or a custom LM head.
        
        # Dummy projection to vocab size for illustration (replace with actual LM head)
        # vocab_size = 50257 # Example for GPT-2
        # lm_logits = torch.nn.Linear(fused_embeddings.shape[-1], vocab_size)(fused_embeddings)
        
        # For now, let's assume fused_embeddings are directly used for LM loss (conceptual)
        # The actual LM head application would be in the training loop or a more complex model forward.
        
        # Shift so that tokens <eos> are predicted by the model
        # This is standard for causal language modeling
        # shifted_logits = lm_logits[:, :-1, :].contiguous()
        # shifted_labels = target_text_ids[:, 1:].contiguous()
        # lm_loss = self.cross_entropy_loss(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))

        # Since we are returning fused_embeddings from the model, the LM loss calculation
        # will depend on how these embeddings are used to predict the next token.
        # For now, we'll focus on the contrastive loss.
        lm_loss = torch.tensor(0.0) # Placeholder for LM loss

        # 2. CLIP-style Contrastive Loss
        # Normalize embeddings
        video_latents_norm = F.normalize(video_latents.mean(dim=1), p=2, dim=-1) # Mean over frames
        text_embeddings_norm = F.normalize(text_embeddings_from_llm.mean(dim=1), p=2, dim=-1) # Mean over tokens

        # Calculate similarity scores
        similarity_scores = torch.matmul(video_latents_norm, text_embeddings_norm.T) / self.temperature

        # Create labels for contrastive loss (identity matrix for positive pairs)
        labels = torch.arange(similarity_scores.shape[0]).to(similarity_scores.device)

        # Calculate contrastive loss for video-to-text and text-to-video
        loss_v2t = F.cross_entropy(similarity_scores, labels)
        loss_t2v = F.cross_entropy(similarity_scores.T, labels)

        contrastive_loss = (loss_v2t + loss_t2v) / 2

        # Total loss (you can weight these as needed)
        total_loss = lm_loss + contrastive_loss

        return total_loss, lm_loss, contrastive_loss

if __name__ == '__main__':
    loss_fn = MultimodalLoss()

    # Dummy inputs for testing
    batch_size = 4
    seq_len = 10
    video_frames = 5
    video_latent_dim = 512
    llm_hidden_size = 768

    fused_embeddings = torch.randn(batch_size, seq_len, llm_hidden_size)
    target_text_ids = torch.randint(0, 50257, (batch_size, seq_len)) # Example vocab size for GPT-2
    text_attention_mask = torch.ones(batch_size, seq_len)
    video_latents = torch.randn(batch_size, video_frames, video_latent_dim)
    text_embeddings_from_llm = torch.randn(batch_size, seq_len, llm_hidden_size)

    total_loss, lm_loss, contrastive_loss = loss_fn(fused_embeddings, target_text_ids, text_attention_mask, video_latents, text_embeddings_from_llm)

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"LM Loss (placeholder): {lm_loss.item():.4f}")
    print(f"Contrastive Loss: {contrastive_loss.item():.4f}")


