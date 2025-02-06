import streamlit as st

import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt

import torch
import torchvision


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


# Function to display image with bounding boxes
def show(img, boxes, ax, color=(255, 0, 0)):
    boxes = boxes.detach().cpu().numpy().astype(np.int32)
    sample = img.permute(1, 2, 0).numpy().copy()
    
    for box in boxes:
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), color, 3)
    
    ax.axis("off")
    ax.imshow(sample)


# Header
st.write("<h3 align='center'>Enhanced Search and Rescue Operations in Building Collapse using CNN</h3>", unsafe_allow_html=True)

st.image("images/repo-cover.jpg")


st.write("""### Inference""")


# Streamlit app layout
st.markdown("Upload an image of a building collapse to detect people trapped in the debris...")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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
        model.eval()
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
