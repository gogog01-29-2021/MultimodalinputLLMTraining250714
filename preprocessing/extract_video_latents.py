
import torch
import torchvision.transforms as transforms
from torchvision.models import video
import cv2
import os
import numpy as np

def extract_latents(video_path, output_dir=".", model_name=\'r3d_18\'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load pre-trained video model
    if model_name == \'r3d_18\':
        model = video.r3d_18(weights=video.R3D_18_Weights.DEFAULT)
    elif model_name == \'mc3_18\':
        model = video.mc3_18(weights=video.MC3_18_Weights.DEFAULT)
    elif model_name == \'r2plus1d_18\':
        model = video.r2plus1d_18(weights=video.R2PLUS1D_18_Weights.DEFAULT)
    else:
        raise ValueError(f\"Unsupported model_name: {model_name}\")

    # Remove the classification head to get features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval() # Set model to evaluation mode

    # Define transformations for video frames
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.221459, 0.216321])
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    latents = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        input_tensor = preprocess(frame_rgb)
        input_batch = input_tensor.unsqueeze(0) # Add a batch dimension

        with torch.no_grad():
            feature = model(input_batch)
            latents.append(feature.squeeze().cpu().numpy())

    cap.release()
    
    if latents:
        latents_array = np.array(latents)
        output_filename = os.path.join(output_dir, os.path.basename(video_path).replace(\".mp4\", \"_latents.npy\"))
        np.save(output_filename, latents_array)
        print(f"Extracted latents saved to {output_filename}")
    else:
        print(f"No latents extracted from {video_path}")

if __name__ == \'__main__\':
    # Example usage: You would replace this with actual video files
    # For testing, you might need a dummy video file or integrate with webcam_video_logger
    print("This script extracts video latents. Provide a video path to run it.")
    # Example: extract_latents(\"/path/to/your/video.mp4\", output_dir=\"/home/ubuntu/natural_video_llm/preprocessing\")





import time
import watchgod

def watch_for_new_videos(capture_dir, output_dir, model_name='r3d_18'):
    """Watches the capture directory for new video files and processes them."""
    print(f"Watching for new video files in {capture_dir}...")
    processed_files = set()

    for changes in watchgod.watch(capture_dir):
        for change_type, path in changes:
            if change_type == watchgod.Change.added and path.endswith('.mp4'):
                if path not in processed_files:
                    print(f"New video detected: {path}")
                    # Wait a moment to ensure the file is fully written
                    time.sleep(1)
                    extract_latents(path, output_dir, model_name)
                    processed_files.add(path)

if __name__ == '__main__':
    # To run in continuous mode:
    # 1. Start webcam_video_logger.py in one terminal.
    # 2. In another terminal, run this script to watch for new videos.
    capture_directory = '/home/ubuntu/natural_video_llm/capture'
    latents_output_directory = '/home/ubuntu/natural_video_llm/preprocessing'
    # watch_for_new_videos(capture_directory, latents_output_directory)

    # The original manual execution is still possible:
    # Example: extract_latents("/path/to/your/video.mp4", output_dir="/home/ubuntu/natural_video_llm/preprocessing")
    print("This script can now be run in watch mode to automatically process new videos.")
    print("Uncomment the 'watch_for_new_videos' call in the main block to enable it.")


