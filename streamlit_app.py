import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as F
import tempfile
import os
import shutil
from pathlib import Path

# Load a pre-trained FasterRCNN model with the correct number of classes
num_classes = 2  # 1 class (object) + background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# Replace the classifier to match your custom number of classes (2 in your case)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
# Load the model weights
model_weights_file = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_weights_file, weights_only=True, map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Function to display image with bounding boxes
def show(img, boxes, ax, color=(255, 0, 0)):
    boxes = boxes.detach().cpu().numpy().astype(np.int32)
    sample = img.permute(1, 2, 0).numpy().copy()
    
    for box in boxes:
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), color, 3)
    
    ax.axis("off")
    ax.imshow(sample)

# Extracting frames from the input video
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        frame_path = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        
        # Update progress
        progress = int(frame_count / total_frames * 100)
        progress_bar.progress(progress)
        status_text.text(f"Extracting frames: {frame_count}/{total_frames}")
    
    video.release()
    return frame_count

# Detect trapped people in a single frame
def detect_people_on_frame(frame_path, model):
    image = Image.open(frame_path).convert("RGB")
    img_tensor = F.to_tensor(image).to(device)
    
    with torch.no_grad():
        outputs = model([img_tensor])
    
    # Get predictions with high confidence
    predictions = outputs[0]
    boxes = predictions["boxes"][predictions["scores"] >= 0.5]
    scores = predictions["scores"][predictions["scores"] >= 0.5]
    
    # Apply non-maximum suppression
    nms = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    final_boxes = boxes[nms].cpu().numpy()
    final_scores = scores[nms].cpu().numpy()
    
    return final_boxes, final_scores

# Draw bounding boxes on an image
def draw_boxes(image, boxes, scores, threshold=0.5):
    for box, score in zip(boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Person: {score:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

# Convert processed frames back to video
def frames_to_video(input_folder, output_video_path, frame_rate=30):
    frames = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                     if f.endswith('.jpg')])
    
    if not frames:
        st.error("No frames found to create video")
        return False
    
    # Get dimensions from first frame
    frame = cv2.imread(frames[0])
    frame_height, frame_width, _ = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, 
                         (frame_width, frame_height))
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_frames = len(frames)
    
    # Write frames to video
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        out.write(frame)
        
        # Update progress
        progress = int((i + 1) / total_frames * 100)
        progress_bar.progress(progress)
        status_text.text(f"Creating video: {i+1}/{total_frames}")
    
    out.release()
    return True

# Process video function
def process_video(video_path, output_video_path):
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    
    try:
        # Step 1: Extract frames
        st.write("Step 1: Extracting frames from video...")
        num_frames = extract_frames(video_path, input_frames_dir)
        st.write(f"Extracted {num_frames} frames.")
        
        # Step 2: Process each frame
        st.write("Step 2: Detecting people in frames...")
        os.makedirs(output_frames_dir, exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        detection_count = st.empty()
        total_detections = 0
        
        for i in range(num_frames):
            frame_path = os.path.join(input_frames_dir, f'frame_{i:04d}.jpg')
            if not os.path.exists(frame_path):
                continue
                
            image = cv2.imread(frame_path)
            boxes, scores = detect_people_on_frame(frame_path, model)
            
            # Draw bounding boxes
            image_with_boxes = draw_boxes(image, boxes, scores)
            
            # Save the processed frame
            output_frame_path = os.path.join(output_frames_dir, f'output_frame_{i:04d}.jpg')
            cv2.imwrite(output_frame_path, image_with_boxes)
            
            # Count detections
            total_detections += len([s for s in scores if s >= 0.5])
            
            # Update progress
            progress = int((i + 1) / num_frames * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame: {i+1}/{num_frames}")
            detection_count.text(f"Total detections: {total_detections}")
        
        # Step 3: Convert frames back to video
        st.write("Step 3: Creating output video...")
        success = frames_to_video(output_frames_dir, output_video_path)
        
        if success:
            st.success(f"Video processing complete! Detected {total_detections} potential victims.")
            return total_detections
        else:
            st.error("Failed to create output video.")
            return 0
            
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

# Header
st.write("<h3 align='center'>Enhanced Search and Rescue Operations in Building Collapse using CNN</h3>", unsafe_allow_html=True)
st.image("images/repo-cover.jpg")
st.write("""### Inference""")

# Create tabs for image and video processing
tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

# Tab 1: Image Processing (existing functionality)
with tab1:
    st.markdown("Upload an image of a building collapse to detect people trapped in the debris...")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)
        
        # Check if the image has an alpha channel (RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')  # Convert to RGB by removing the alpha channel
        img = np.array(image)
        
        # Convert the image to tensor (C, H, W) format, scale to [0, 1]
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()  # Convert to tensor (C, H, W)
        img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
        
        # Add a batch dimension for the model
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Use st.spinner to show a loading animation during inference
        with st.spinner('Processing image...'):
            with torch.no_grad():
                output = model([img_tensor.squeeze(0)])  # Remove the batch dimension
        
        # Display the image with bounding boxes
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))  # 1 row, 1 column
        axes = [axes]
        
        # Predicted bounding boxes
        predictions = output[0]
        pp_boxes = predictions["boxes"][predictions["scores"] >= 0.5]
        scores = predictions["scores"][predictions["scores"] >= 0.5]
        nms = torchvision.ops.nms(pp_boxes, scores, iou_threshold=0.5)
        pp_boxes = pp_boxes[nms]
        
        # Show predicted bounding boxes on the image
        show(img_tensor[0], pp_boxes, axes[0])
        
        # Show the image
        st.pyplot(fig)
    else:
        st.warning("Please upload an image.")

# Tab 2: Video Processing (new functionality)
with tab2:
    st.markdown("Upload a video of a building collapse to detect people trapped in the debris...")
    
    # Video upload
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"], key="video_uploader")
    
    if uploaded_video is not None:
        # Create a temporary file to save the uploaded video
        temp_dir = tempfile.TemporaryDirectory()
        temp_video_path = os.path.join(temp_dir.name, "input_video.mp4")
        output_video_path = os.path.join(temp_dir.name, "output_video.mp4")
        
        # Save the uploaded video to the temporary file
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        # Show a preview of the uploaded video
        st.video(temp_video_path)
        
        # Process button
        if st.button("Process Video"):
            with st.spinner("Processing video... This may take a while."):
                total_detections = process_video(temp_video_path, output_video_path)
                
                if os.path.exists(output_video_path):
                    # Display the processed video
                    st.video(output_video_path)
                    
                    # Option to download the processed video
                    with open(output_video_path, "rb") as file:
                        st.download_button(
                            label="Download Processed Video",
                            data=file,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
    else:
        st.warning("Please upload a video.")