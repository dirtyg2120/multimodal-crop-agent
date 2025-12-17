import asyncio
import os
import io
import torch
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from torchvision.ops import box_convert
from transformers import CLIPModel, CLIPProcessor
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict, load_model, annotate, load_image

# --- Your Agent Code ---
from app.agent.deps import AgronomyDeps
from app.agent.core import agronomy_agent
from app.vision.clip_labels import CLIP_LABEL_MAP
import app.agent.tools

# --- CONSTANTS & CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grounding DINO Config
GDINO_CONFIG = "app/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "app/groundingdino/weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT = "leaf . bug . worm ."
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

# CLIP Config
CLIP_MODEL_NAME = "Keetawan/clip-vit-large-patch14-plant-disease-finetuned"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- 1. MODEL LOADING (Cached) ---

@st.cache_resource
def load_gdino_model():
    """Load Grounding DINO model once."""
    print(f"Loading Grounding DINO on {DEVICE}...")
    model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
    return model

@st.cache_resource
def load_clip_model():
    """Load CLIP model once."""
    print(f"Loading CLIP on {DEVICE}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return model, processor

# --- 2. VISION HELPER FUNCTIONS ---

def extract_crops(image_source, boxes, phrases):
    """
    Extracts crops from the image based on Grounding DINO boxes.
    Returns a list of dictionaries containing the crop and metadata.
    """
    # image_source is a generic numpy array (H, W, 3)
    h_img, w_img, _ = image_source.shape
    crops = []

    for box, phrase in zip(boxes, phrases):
        # 1. Un-normalize and convert to pixels
        cx, cy, w, h = box.tolist()
        x1 = int((cx - 0.5 * w) * w_img)
        y1 = int((cy - 0.5 * h) * h_img)
        x2 = int((cx + 0.5 * w) * w_img)
        y2 = int((cy + 0.5 * h) * h_img)

        # 2. Safety Clip
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        # 3. Perform Crop
        crop_img = image_source[y1:y2, x1:x2]
        
        # 4. Filter tiny crops (noise)
        if crop_img.size == 0 or (x2-x1) < 10 or (y2-y1) < 10:
            continue

        crops.append({
            "label": phrase,
            "crop": crop_img, # RGB Numpy Array
            "bbox": (x1, y1, x2, y2)
        })
    return crops

def classify_crop_clip(image_crop, model, processor):
    """
    Runs CLIP on a single cropped image to find the disease label.
    """
    # Prepare Labels (From your CLIP_LABEL_MAP keys to text list)
    labels_text = list(CLIP_LABEL_MAP.values())
    
    # Convert numpy crop to PIL
    pil_image = Image.fromarray(image_crop)
    
    inputs = processor(
        text=labels_text,
        images=pil_image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image # [1, 42]
        probs = logits.softmax(dim=1)
    
    top_prob, top_idx = probs[0].max(dim=0)
    top_class_id = top_idx.item()
    top_confidence = top_prob.item()

    return top_class_id, top_confidence

# --- 3. STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Multimodal Crop Agent", layout="wide")
    st.title("🌾 Multimodal Crop Health Agent")

    # Load Models (Cached)
    with st.spinner("Loading AI Models..."):
        dino_model = load_gdino_model()
        clip_model, clip_processor = load_clip_model()

    uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"])

    caption = st.text_input("Caption", value=TEXT_PROMPT, help="Text prompt for GroundingDINO")
    col_a, col_b = st.columns(2)
    with col_a:
        box_threshold = st.slider("box_threshold", 0.00, 1.00, 0.30, 0.01)
    with col_b:
        text_threshold = st.slider("text_threshold", 0.00, 1.00, 0.25, 0.01)

    if uploaded_file:
        # Save temp file because load_image expects a path
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns([1, 1])
        
        # Display the uploaded file immediately
        with col1:
            st.image(uploaded_file, caption="Original Image", width="content")

        if st.button("Analyze Image", type="primary"):
            
            # --- STEP A: Grounding DINO ---
            with st.spinner("Detecting objects..."):
                # Use the official load_image utility
                # image_source: original numpy array (for display/cropping)
                # image_transformed: tensor (for model input)
                image_source, image_transformed = load_image(temp_path)
                
                boxes, logits, phrases = predict(
                    model=dino_model,
                    image=image_transformed,
                    caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=DEVICE
                )
                
                # Annotate and Show
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                with col2:
                    st.image(annotated_frame, caption="Detections (Grounding DINO)", width="content")

            # --- STEP B: Extract Crops ---
            # Note: extract_crops expects the numpy array (image_source)
            crops_data = extract_crops(image_source, boxes, phrases)
            st.success(f"Found {len(crops_data)} objects.")

            # --- STEP C: CLIP Verification Loop ---
            st.divider()
            st.subheader("🔍 Detailed Inspection (CLIP)")
            
            if not crops_data:
                st.warning("No leaves detected. Try a clearer image.")
            
            for i, item in enumerate(crops_data):
                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 2])
                    
                    # Show Crop
                    with c1:
                        st.image(item['crop'], caption=f"Object #{i+1}")
                    
                    # Run CLIP
                    class_id, confidence = classify_crop_clip(item['crop'], clip_model, clip_processor)
                    label_text = CLIP_LABEL_MAP.get(class_id, "Unknown")
                    
                    # Show Result
                    with c2:
                        st.markdown(f"**Diagnosis:** `{label_text}`")
                        st.progress(confidence, text=f"Confidence: {confidence:.2%}")

                    # --- STEP D: Agent (Connect Button) ---
                    with c3:
                        if confidence > 0.5: 
                            if st.button(f"Get Treatment Plan #{i+1}", key=f"btn_{i}"):
                                with st.spinner("Consulting Agronomy Agent..."):
                                    # Call the async pipeline (Placeholder)
                                    st.info("Request sent to Agent Core...")
                                    # result = asyncio.run(run_rag_pipeline(class_id, confidence))
                                    # st.json(result)

if __name__ == "__main__":
    main()