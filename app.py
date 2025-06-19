import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import streamlit as st
import tempfile
from pathlib import Path

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🎥",
    layout="wide"
)

# --- Configuration ---
MODEL_PATH = 'C:\deepfake\model_ep10.pth' # Path to your saved model
NUM_FRAMES = 6               # Number of frames used during training
IMG_SIZE = 224               # Image size used during training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {DEVICE}")

# --- Model Definition (MUST match your training code EXACTLY) ---
class ResNet50BiLSTM(nn.Module):
    def __init__(self, hidden=256): # Ensure hidden size matches training if specified
        super().__init__()
        # Load ResNet50 base - Use updated weights parameter if needed
        try:
            # Newer torchvision versions
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
             # Older torchvision versions (may raise warning)
             st.warning("Warning: Using older ResNet50 weights API. Consider updating torchvision.")
             base = models.resnet50(pretrained=True)

        self.cnn = nn.Sequential(*list(base.children())[:-1]) # Remove final FC layer
        # Adjust input features to LSTM if ResNet output changes (usually 2048 for ResNet50)
        lstm_input_features = base.fc.in_features # Get feature size before removing fc layer

        self.lstm = nn.LSTM(lstm_input_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, 128),  # hidden*2 because bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W) - B=batch, T=time/frames
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W) # -> (B*T, C, H, W)

        # Disable gradient calculation for CNN part if it wasn't trained
        # If you fine-tuned ResNet, keep gradients enabled.
        # Usually, we freeze the CNN backbone initially.
        with torch.no_grad():
            feats = self.cnn(x).view(B, T, -1) # -> (B, T, Features)
            # Example shape: (B, T, 2048) for ResNet50

        out, _ = self.lstm(feats) # -> (B, T, Hidden*2)
        # We typically use the output of the last time step for classification
        # Or sometimes average/max pool over time
        # Your training code uses out[:, -1, :] which takes the last time step's output
        return self.head(out[:, -1, :]).squeeze() # -> (B,)

# --- Load Model ---
@st.cache_resource
def load_model():
    st.write(f"Loading model from {MODEL_PATH}...")
    model = ResNet50BiLSTM().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Set model to evaluation mode (important!)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Preprocessing Components ---
# Initialize MTCNN for face detection (consider device placement)
# Keep_all=False ensures only the most probable face is kept if multiple are found.
mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, keep_all=False, device=DEVICE, post_process=False)

# Define the exact same transforms used during training
preprocess_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), # Ensure resize happens
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Use training normalization constants
])

# Fallback transform if no face is detected (matches your training code snippet)
fallback_transform = T.Compose([
    T.ToTensor(), # Convert the cropped full image to tensor first
    T.Resize((IMG_SIZE, IMG_SIZE)), # Then resize
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def crop_center_square(img):
    """Crops the center square area of a PIL image."""
    w, h = img.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    right = (w + size) // 2
    bottom = (h + size) // 2
    return img.crop((left, top, right, bottom))

# --- Video Preprocessing Function ---
def preprocess_video(video_path):
    """
    Opens a video, extracts frames, detects/crops faces, applies transforms.
    Returns a tensor ready for the model or None if processing fails.
    """
    frames_list = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        st.error(f"Error: Video {video_path} has no frames.")
        cap.release()
        return None

    step = max(total_frames // NUM_FRAMES, 1)
    frames_extracted = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(NUM_FRAMES):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            # If we run out of frames, duplicate the last one (or handle differently)
            if not frames_list: # No frames extracted at all
                 st.error(f"Warning: Could not read any frames from {video_path}")
                 cap.release()
                 return None
            st.warning(f"Warning: Could not read frame index {frame_idx}, duplicating last frame.")
            frames_list.append(frames_list[-1].clone()) # Duplicate the last tensor
            continue

        # Convert frame BGR (OpenCV) to RGB (PIL/PyTorch)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Detect face - MTCNN expects PIL Image or numpy array
        face_tensor = mtcnn(img) # Returns cropped face tensor on DEVICE or None

        if face_tensor is not None:
            # Convert tensor back to PIL to use standard torchvision transforms
            face_pil = T.ToPILImage()(face_tensor.cpu()) # Move to CPU for PIL
            processed_frame = preprocess_transform(face_pil) # Apply resize/norm
        else:
            # Fallback: Crop center square and apply transforms (matching training)
            st.warning(f"Warning: No face detected in frame {frames_extracted}. Using center crop.")
            center_cropped_img = crop_center_square(img)
            # Apply fallback transform (includes ToTensor, Resize, Normalize)
            processed_frame = fallback_transform(center_cropped_img)

        frames_list.append(processed_frame)
        frames_extracted += 1
        
        # Update progress
        progress = (i + 1) / NUM_FRAMES
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {i + 1}/{NUM_FRAMES}")

    cap.release()
    progress_bar.empty()
    status_text.empty()

    if frames_extracted == 0:
        st.error(f"Error: No frames could be processed for video {video_path}")
        return None
    elif frames_extracted < NUM_FRAMES:
         st.warning(f"Warning: Only {frames_extracted}/{NUM_FRAMES} frames processed. Duplicating last frame.")
         # Pad by duplicating the last frame
         last_frame_tensor = frames_list[-1].clone()
         for _ in range(NUM_FRAMES - frames_extracted):
             frames_list.append(last_frame_tensor)

    # Stack frames into a single tensor (T, C, H, W)
    video_tensor = torch.stack(frames_list, dim=0)
    # Add batch dimension (B, T, C, H, W) -> B=1
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor.to(DEVICE) # Ensure final tensor is on the correct device

# --- Streamlit UI ---
st.title("🎥 Deepfake Video Detection")
st.write("Upload a video to check if it's real or fake")

# Load model
model = load_model()
if model is None:
    st.error("Failed to load model. Please check the model path and try again.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    # Display video
    st.video(uploaded_file)

    # Process video and make prediction
    if st.button("Analyze Video"):
        with st.spinner("Processing video..."):
            input_tensor = preprocess_video(temp_video_path)
            
            if input_tensor is not None:
                # Make prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    probability = output.item() # Get the scalar probability

                # Determine prediction based on threshold (usually 0.5)
                prediction = 'FAKE' if probability > 0.5 else 'REAL'
                confidence = probability if prediction == 'FAKE' else 1.0 - probability

                # Display results
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction", prediction)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Progress bar for visualization
                st.progress(confidence if prediction == 'FAKE' else 1 - confidence)
                
                # Clean up temporary file
                os.unlink(temp_video_path)
            else:
                st.error("Failed to process video. Please try another video file.")