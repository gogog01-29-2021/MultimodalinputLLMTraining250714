
import cv2
import torch
import numpy as np
import time
from datetime import datetime

from transformers import AutoTokenizer
from models.fusion_transformer import MultimodalFusionLLM
from preprocessing.extract_video_latents import extract_latents

# Configuration
LLM_MODEL_NAME = "gpt2"
VIDEO_LATENT_DIM = 512 # Must match the output of extract_video_latents.py
PROJECTION_DIM = 768 # Must match the hidden size of the LLM (e.g., GPT-2)
MODEL_CHECKPOINT_PATH = "./model_epoch_3.pt" # Path to your trained model checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = MultimodalFusionLLM(llm_model_name=LLM_MODEL_NAME, video_latent_dim=VIDEO_LATENT_DIM, projection_dim=PROJECTION_DIM)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using a randomly initialized model.")
    
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def get_video_latent_from_frame(frame, model_name=\'r3d_18\'):
    # This is a simplified version of extract_latents for single frame processing
    # It requires the video model to be loaded and used directly.
    # For real-time, we should load the video model once.
    
    # Load pre-trained video model (this should ideally be loaded once outside this function)
    # For demonstration, let's assume a simplified direct call or a pre-loaded model instance
    
    # Placeholder for actual video feature extraction logic
    # In a real system, you'd pass the frame through the video encoder
    # For now, let's return a dummy latent
    return torch.randn(1, VIDEO_LATENT_DIM).to(DEVICE) # Dummy latent

# --- Main Real-time Interaction Loop ---
def realtime_interaction():
    tokenizer, model = setup_model(MODEL_CHECKPOINT_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time interaction. Type your message and press Enter. Press 'q' to quit.")

    # Placeholder for video feature extractor model
    # In a real scenario, you'd load the video feature extractor here once
    # e.g., video_feature_extractor = video.r3d_18(weights=video.R3D_18_Weights.DEFAULT).to(DEVICE)
    # video_feature_extractor = torch.nn.Sequential(*(list(video_feature_extractor.children())[:-1]))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Live Webcam Feed", frame)

        # Get video latent for the current frame
        # In a real system, you'd process the 'frame' using your video feature extractor
        # current_video_latent = process_frame_to_latent(frame, video_feature_extractor)
        current_video_latent = get_video_latent_from_frame(frame) # Using dummy for now

        # Simulate text input
        user_input = input("You: ")
        if user_input.lower() == 'q':
            break

        # Tokenize user input
        input_encoding = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        input_ids = input_encoding["input_ids"]
        attention_mask = input_encoding["attention_mask"]

        # Expand video latent to match batch size and sequence length if necessary
        # For simplicity, assuming 1 video latent per text input
        # The model expects video_latents to be (batch_size, num_video_frames, video_latent_dim)
        # Here, num_video_frames is 1 for a single frame's latent
        video_latents_batch = current_video_latent.unsqueeze(0) # (1, 1, VIDEO_LATENT_DIM)

        # Generate response
        with torch.no_grad():
            # The generate method expects input_ids or inputs_embeds
            # We need to get the fused embeddings first
            # This is a simplified call, as the `generate` method of HuggingFace models
            # typically works with `input_ids` and then internally manages embeddings.
            # To use `inputs_embeds`, we need to pass the initial text embeddings + video fusion.
            
            # Get initial text embeddings from the LLM's base model
            initial_text_embeddings = model.llm.base_model.get_input_embeddings()(input_ids)

            # Project video latents to LLM embedding space
            projected_video_latents = model.video_projection(video_latents_batch)

            # Apply cross-attention: query=text, key=value=video
            attn_output, _ = model.cross_attention(
                query=initial_text_embeddings,
                key=projected_video_latents,
                value=projected_video_latents
            )
            
            # Add and normalize the attention output with initial text embeddings
            fused_embeddings = model.norm(initial_text_embeddings + attn_output)

            generated_ids = model.llm.generate(
                inputs_embeds=fused_embeddings,
                max_length=input_ids.shape[1] + 50, # Generate up to 50 new tokens
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"LLM: {response}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Ensure you have a trained model checkpoint before running this.
    # For testing, you might need to create a dummy model_epoch_3.pt file.
    realtime_interaction()


