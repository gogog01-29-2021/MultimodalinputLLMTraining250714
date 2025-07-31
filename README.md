# Natural Video LLM

This project aims to train an LLM using real-time natural human facial expressions captured from live video during interaction, without using any symbolic labels like "AU" or "emotion class". The goal is to build a conversational model that learns to adapt and align responses subconsciously, the way humans co-regulate tone and meaning in real life.

## Project Structure

```
natural_video_llm/
├── capture/
│   ├── webcam_video_logger.py      # records video + timestamp in segments
│   ├── text_sync_logger.py         # records chat text + timestamps
│   └── session.jsonl               # time-aligned (video, text) logs
├── preprocessing/
│   ├── extract_video_latents.py    # ViT/TimeSformer encoder (can watch for new files)
│   ├── align_text_video.py         # token-level alignment for video segments
├── models/
│   ├── fusion_transformer.py       # LLM + cross-modal adapter (refined)
│   ├── loss.py                     # contrastive / fusion loss (refined)
├── training/
│   ├── train.py                    # core training loop (refined)
│   └── eval.py                     # coherence, tone, response eval (refined)
├── inference/
│   └── realtime_interaction.py     # Real-time webcam and text interaction with LLM
├── logs/
│   └── metrics/
└── README.md
```

## Setup

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <repository_url>
    cd natural_video_llm
    ```
2.  **Install dependencies:**
    ```bash
    pip install opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers watchgod
    ```

## Usage

### 1. Data Capture (for Training)

To capture live webcam video and synchronize text input for training data, run the following scripts in separate terminal sessions:

**Terminal 1 (Video Capture):**
```bash
python capture/webcam_video_logger.py
```

**Terminal 2 (Text Input):**
```bash
python capture/text_sync_logger.py
```

Press `q` in the video capture window to stop recording. Type `quit` in the text input terminal to stop.

This will generate video segments and update `session.jsonl` in the `natural_video_llm/capture/` directory.

### 2. Preprocessing (for Training)

After capturing data, extract video latents and align text with video:

**Extract Video Latents (Continuous Processing):**

To automatically process new video segments as they are created by `webcam_video_logger.py`, run this in a separate terminal:
```bash
python preprocessing/extract_video_latents.py
```
*Note: Ensure the `watch_for_new_videos` function call is uncommented in `extract_video_latents.py` for continuous processing.*

**Align Text and Video:**
```bash
python preprocessing/align_text_video.py
```
This will create `aligned_data.jsonl` in the `natural_video_llm/preprocessing/` directory.

### 3. Training

To train the multimodal LLM:

```bash
python training/train.py
```

This will save model checkpoints (e.g., `model_epoch_X.pt`) in the `natural_video_llm/training/` directory.

### 4. Evaluation

To evaluate the trained model:

```bash
python training/eval.py
```

*Note: Ensure `model_checkpoint_path` in `eval.py` points to your trained model file.*

### 5. Real-time Interaction (Deployment Example)

To run the real-time interaction demo:

```bash
python inference/realtime_interaction.py
```

*Note: Ensure `MODEL_CHECKPOINT_PATH` in `realtime_interaction.py` points to your trained model file. This script uses dummy video latents for now; integrate your actual video feature extractor for full functionality.*




## Placeholders and Further Development

While significant progress has been made, some components are still placeholders or require further development for a fully robust and production-ready system:

### 1. Real-time Video Feature Extraction

-   **Current State**: In `inference/realtime_interaction.py`, the `get_video_latent_from_frame` function currently returns a `torch.randn` (dummy) tensor. This is a critical placeholder.
-   **Required Development**: You need to integrate the actual video feature extraction logic here. This involves:
    1.  Loading the pre-trained video model (e.g., `r3d_18`) once at the start of `realtime_interaction.py`.
    2.  Modifying `get_video_latent_from_frame` to take a raw `cv2` frame, preprocess it (resize, normalize, convert to tensor), and pass it through the loaded video feature extractor to obtain the real latent vector.
    *Self-correction*: The `extract_latents` function in `preprocessing/extract_video_latents.py` already contains the core logic for this. You can adapt parts of it to work with single frames in real-time.

### 2. `MultimodalDataset` Video Latent Handling

-   **Current State**: In `training/train.py`, the `MultimodalDataset.__getitem__` method currently averages video latents if multiple are associated with one text entry (`torch.mean(torch.stack(video_latents_list), dim=0)`). This is a simplification.
-   **Required Development**: For more sophisticated modeling of temporal dynamics in facial expressions, you might consider:
    -   **Pooling**: Using max pooling or attention pooling over the sequence of video latents.
    -   **Sequence Models**: Passing the sequence of video latents through a small recurrent neural network (RNN) or transformer encoder before projecting them for fusion with text.

### 3. GPT Judge for Evaluation

-   **Current State**: In `training/eval.py`, the `evaluate_with_gpt_judge` function is a **placeholder** that returns dummy scores.
-   **Required Development**: To get meaningful evaluation, you need to integrate with a real LLM API (e.g., OpenAI GPT-4, Anthropic Claude). This involves:
    1.  Obtaining an API key for your chosen LLM.
    2.  Modifying the function to make API calls, sending the `generated_text`, `reference_text`, and `tone_context` (derived from video latents/analysis) as part of a prompt.
    3.  Parsing the LLM's response to extract coherence and tone alignment scores.

### 4. Deployment

Deployment of such a multimodal LLM system can be complex and depends heavily on the desired application (e.g., a web application, a desktop application, an API service).

-   **For a Web Application**: You would typically build a frontend (e.g., using React, as supported by Manus utilities) that captures webcam feed and sends text/video data to a backend. The backend (e.g., a Flask application, as supported by Manus utilities) would host the `MultimodalFusionLLM` and handle real-time inference.
    -   **Backend (Flask)**: You could adapt `inference/realtime_interaction.py` into a Flask API endpoint that receives video frames and text, performs inference, and returns the generated response.
    -   **Frontend (React)**: A React app would handle webcam access (using browser APIs like `getUserMedia`), send video frames and text to your Flask backend, and display the LLM's responses.
    -   **Deployment Tools**: Manus provides `service_deploy_frontend` and `service_deploy_backend` tools to deploy these components.

-   **For a Desktop Application**: You would integrate the `realtime_interaction.py` logic directly into a desktop application framework (e.g., PyQt, Kivy) that can access the webcam and provide a user interface.

-   **Key Considerations for Deployment**: 
    -   **Latency**: Real-time interaction requires low-latency video processing and LLM inference. This might necessitate GPU acceleration and optimized model serving frameworks (e.g., ONNX Runtime, TensorRT).
    -   **Scalability**: If serving many users, consider distributed inference and load balancing.
    -   **Privacy**: Handling live webcam data requires careful consideration of user privacy and data security.

## How to Train and Evaluate

Refer to the `Usage` section above and the detailed comments within `training/train.py` and `training/eval.py`.

1.  **Data Generation**: Run `capture/webcam_video_logger.py` and `capture/text_sync_logger.py` simultaneously to create `session.jsonl` and video segments.
2.  **Preprocessing**: Run `preprocessing/extract_video_latents.py` (in watch mode for convenience) to generate `.npy` latent files, then run `preprocessing/align_text_video.py` to create `aligned_data.jsonl`.
3.  **Training**: Execute `python training/train.py`. This will train the `MultimodalFusionLLM` and save model checkpoints.
4.  **Evaluation**: Execute `python training/eval.py`. This will load a trained model and run a (currently placeholder) evaluation using the GPT judge concept.

By following these steps and addressing the identified placeholders, you can progressively build and refine your multimodal LLM system.


