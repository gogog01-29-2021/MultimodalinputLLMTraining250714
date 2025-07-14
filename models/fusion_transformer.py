
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class MultimodalFusionLLM(nn.Module):
    def __init__(self, llm_model_name="gpt2", video_latent_dim=512, projection_dim=768):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
        # Ensure the LLM tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.tokenizer.eos_token_id

        self.video_projection = nn.Linear(video_latent_dim, projection_dim)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.llm.config.hidden_size,
            num_heads=8, # Example number of heads
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.llm.config.hidden_size)

    def forward(self, input_ids, attention_mask, video_latents):
        # Project video latents to LLM embedding space
        projected_video_latents = self.video_projection(video_latents)
        
        # Get LLM embeddings
        llm_outputs = self.llm.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        text_embeddings = llm_outputs.last_hidden_state

        # Apply cross-attention: query=text, key=value=video
        # Permute to (sequence_length, batch_size, embed_dim) for MultiheadAttention
        # Or use batch_first=True in MultiheadAttention constructor
        attn_output, _ = self.cross_attention(
            query=text_embeddings,
            key=projected_video_latents,
            value=projected_video_latents
        )
        
        # Add and normalize
        fused_embeddings = self.norm(text_embeddings + attn_output)

        # Pass fused embeddings through the LLM's language modeling head
        # This requires manually passing through the LM head, as we've bypassed the usual forward pass
        # For simplicity, we'll just return the fused embeddings for now, 
        # and the loss function will handle the next token prediction.
        # In a full implementation, you'd re-integrate this into the LLM's forward pass or use a custom LM head.
        
        # A more complete approach would involve feeding fused_embeddings back into the LLM's decoder layers
        # or using them to modify the attention mechanism within the LLM.
        # For this initial setup, we'll assume the fused_embeddings are the enhanced representations
        # that will be used for downstream tasks (e.g., next token prediction in the training loop).
        
        return fused_embeddings

if __name__ == '__main__':
    # Example usage
    model = MultimodalFusionLLM()
    
    # Dummy inputs
    input_ids = torch.randint(0, model.tokenizer.vocab_size, (2, 10)) # Batch size 2, sequence length 10
    attention_mask = torch.ones(2, 10)
    video_latents = torch.randn(2, 5, 512) # Batch size 2, 5 video frames, latent dim 512

    fused_output = model(input_ids, attention_mask, video_latents)
    print("Fused output shape:", fused_output.shape)
    # Expected output shape: torch.Size([2, 10, 768]) (batch_size, sequence_length, hidden_size)


