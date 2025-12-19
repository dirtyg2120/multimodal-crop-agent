import os
import io
import numpy as np
import cv2
import streamlit as st
import torch
import asyncio
# import nest_asyncio
# nest_asyncio.apply()

from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict, load_model, annotate, load_image

# --- Your Agent Code ---
# from app.agent.deps import AgronomyDeps
# from app.agent.core import agronomy_agent
# from app.vision.clip_labels import CLIP_LABEL_MAP
# import app.agent.tools
from app.pipe import VisionSystem, analyze_full_plant

# --- CONSTANTS & CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grounding DINO Config
# GDINO_CONFIG = "app/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GDINO_WEIGHTS = "app/groundingdino/weights/groundingdino_swint_ogc.pth"
GDINO_CONFIG = "app/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GDINO_WEIGHTS = "app/groundingdino/weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "leaf . bug . worm ."
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

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
    return VisionSystem()

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

# --- 3. STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Multimodal Crop Agent", layout="wide")
    st.title("🌾 Multimodal Crop Health Agent")

    # Load Models (Cached)
    with st.spinner("Loading AI Models..."):
        dino_model = load_gdino_model()
        vision_system = load_clip_model()

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
            with st.spinner("Analyzing plant health & consulting Agent..."):
                analysis_results = asyncio.run(analyze_full_plant(
                    crops_data, vision_system=vision_system
                ))
            
            st.divider()
            st.subheader("📝 Agronomist Diagnosis (Whole Plant)")
            
            # Extract data from the result dict
            agent_json = analysis_results["agent_response"]

            with st.expander("Input"):
                st.write(analysis_results["stats"].to_dict())
                st.write(analysis_results["detections"])
            
            # Create 3 columns for high-level stats
            k1, k2, k3 = st.columns(3)
            k1.metric("Overall Health", agent_json.overall_health_status)
            k2.metric("Severity Level", agent_json.severity_level)
            k3.metric("Infection Ratio", f"{agent_json.infection_ratio:.0%}")
            
            with st.expander("Treatment Plan", expanded=True):
                st.write("**Recommended Actions:**")
                for action in agent_json.recommended_actions:
                    st.write(f"- {action}")
                
                if agent_json.required_pesticides:
                    st.warning(f"💊 **Chemicals Required:** {', '.join(agent_json.required_pesticides)}")

            # 2. Show Detailed Inspection (The Loop)
            st.divider()
            st.subheader("🔍 Individual Leaf Inspection")
            with st.expander("Details"):
            
                detections = analysis_results["detections"]
                
                if not detections:
                    st.warning("No leaves detected.")
                
                # Use the data returned from main.py to draw the UI
                for obj in detections:
                    # Retrieve the original crop image using the ID
                    original_crop = crops_data[obj.crop_id]['crop']
                    
                    with st.container():
                        c1, c2, c3 = st.columns([1, 2, 2])
                        with c1:
                            st.image(original_crop, width=150)
                        with c2:
                            st.markdown(f"**{obj.label}**")
                            st.progress(obj.confidence, text=f"Confidence: {obj.confidence:.1%}")

            st.divider()

if __name__ == "__main__":
    main()