import os
import asyncio
import numpy as np
import streamlit as st
import torch
from PIL import Image

from groundingdino.util.inference import predict, load_model, annotate, load_image

from app.pipe import VisionSystem, analyze_full_plant, classify_crops, CLIP_CONFIDENCE_THRESHOLD
from app.few_shot_engine import FewShotEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GDINO_CONFIG = "app/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GDINO_WEIGHTS = "app/groundingdino/weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "leaf . bug . worm ."


@st.cache_resource
def load_gdino_model():
    return load_model(GDINO_CONFIG, GDINO_WEIGHTS)

@st.cache_resource
def load_vision_system():
    return VisionSystem()

@st.cache_resource
def load_few_shot_engine():
    return FewShotEngine()


def extract_crops(image_source, boxes, phrases):
    h, w, _ = image_source.shape
    crops = []
    for box, phrase in zip(boxes, phrases):
        cx, cy, bw, bh = box.tolist()
        x1, y1 = max(0, int((cx - bw/2) * w)), max(0, int((cy - bh/2) * h))
        x2, y2 = min(w, int((cx + bw/2) * w)), min(h, int((cy + bh/2) * h))
        crop = image_source[y1:y2, x1:x2]
        if crop.size > 0 and (x2-x1) >= 10 and (y2-y1) >= 10:
            crops.append({"label": phrase, "crop": crop, "bbox": (x1, y1, x2, y2)})
    return crops


def classify_with_fewshot(vision_system, few_shot_engine, crop_np):
    """Try CLIP first; if Unknown, check few-shot prototypes."""
    label, conf = vision_system.classify_leaf(crop_np)
    if label == "Unknown" and few_shot_engine.prototypes:
        pil = Image.fromarray(crop_np)
        fs_label, fs_score, matched = few_shot_engine.identify(pil)
        if matched:
            return fs_label, fs_score, True
    return label, conf, False


def annotate_with_clip_labels(image_source, crops_data, detections):
    """
    Re-draws bounding boxes using CLIP-resolved labels instead of DINO's
    generic 'leaf'/'bug' phrases.

    detections: list of DetectedObject (label, confidence, box pixel coords, crop_id)
    """
    h, w, _ = image_source.shape

    # Build tensors in the same format annotate() expects
    boxes_list, logits_list, phrases_list = [], [], []
    for obj in detections:
        x1, y1, x2, y2 = obj.box
        # Convert pixel bbox → normalised cx, cy, bw, bh
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        boxes_list.append([cx, cy, bw, bh])
        logits_list.append(obj.confidence)
        phrases_list.append(obj.label)

    if not boxes_list:
        return image_source  # nothing to draw

    boxes_t  = torch.tensor(boxes_list,  dtype=torch.float32)
    logits_t = torch.tensor(logits_list, dtype=torch.float32)

    return annotate(
        image_source=image_source,
        boxes=boxes_t,
        logits=logits_t,
        phrases=phrases_list,
    )


def main():
    st.set_page_config(page_title="Multimodal Crop Agent", layout="wide")
    st.title("🌾 Multimodal Crop Health Agent")

    dino_model      = load_gdino_model()
    vision_system   = load_vision_system()
    few_shot_engine = load_few_shot_engine()

    # Session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "crops_data" not in st.session_state:
        st.session_state.crops_data = None
    if "unknown_crops" not in st.session_state:
        st.session_state.unknown_crops = []

    # Sidebar: registered prototypes
    with st.sidebar:
        st.header("Few-Shot Prototypes")
        if few_shot_engine.prototypes:
            for label in few_shot_engine.prototypes:
                st.success(f"✅ {label}")
        else:
            st.caption("No prototypes registered yet.")

    uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        return

    col_box, col_txt = st.columns(2)
    box_threshold = col_box.slider("box_threshold", 0.0, 1.0, 0.30, 0.01)
    text_threshold = col_txt.slider("text_threshold", 0.0, 1.0, 0.25, 0.01)

    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original Image")

    # --- Analyze button: runs analysis and stores everything in session_state ---
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Detecting objects..."):
            image_source, image_transformed = load_image(temp_path)
            boxes, logits, phrases = predict(
                model=dino_model, image=image_transformed,
                caption=TEXT_PROMPT, box_threshold=box_threshold,
                text_threshold=text_threshold, device=DEVICE,
            )

        crops_data = extract_crops(image_source, boxes, phrases)
        st.success(f"Found {len(crops_data)} objects.")

        # Step 1: CLIP classification (fast) — show annotated image right away
        with st.spinner("Classifying leaves..."):
            detections = classify_crops(crops_data, vision_system, few_shot_engine)
            clip_annotated = annotate_with_clip_labels(image_source, crops_data, detections)

        with col2:
            st.image(clip_annotated, caption="Detections (CLIP labels)")

        # Step 2: Agent reasoning (slow) — uses pre-computed detections, no double CLIP
        with st.spinner("Agent is thinking..."):
            results = asyncio.run(analyze_full_plant(
                crops_data, vision_system=vision_system,
                few_shot_engine=few_shot_engine, detections=detections
            ))

        # Persist to session_state for reruns
        st.session_state.results = results
        st.session_state.crops_data = crops_data
        st.session_state.image_source = image_source
        st.session_state.clip_annotated = clip_annotated
        unknown_crops = [det for det in detections if det.label == "Unknown"]
        st.session_state.unknown_crops = [crops_data[obj.crop_id]["crop"] for obj in unknown_crops]

    # --- Results display: always shown if results exist in session_state ---
    results         = st.session_state.results
    crops_data      = st.session_state.crops_data
    clip_annotated  = st.session_state.get("clip_annotated")

    # # Re-display the CLIP-annotated image on subsequent reruns (e.g. after Register & Save)
    # if clip_annotated is not None:
    #     with col2:
    #         st.image(clip_annotated, caption="Detections (CLIP labels)")

    if results:
        agent_json = results["agent_response"]

        st.divider()
        st.subheader("📝 Agronomist Diagnosis")

        k1, k2, k3 = st.columns(3)
        k1.metric("Overall Health", agent_json.overall_health_status)
        k2.metric("Severity", agent_json.severity_level)
        k3.metric("Infection Ratio", f"{agent_json.infection_ratio:.0%}")

        with st.expander("Treatment Plan", expanded=True):
            st.info(f"**Reasoning:** {agent_json.reasoning}")
            for action in agent_json.recommended_actions:
                st.write(f"- {action}")
            if agent_json.required_pesticides:
                st.warning(f"💊 Chemicals: {', '.join(agent_json.required_pesticides)}")

        with st.expander("Individual Detections"):
            for obj in results["detections"]:
                crop_np = crops_data[obj.crop_id]["crop"]
                c1, c2 = st.columns([1, 3])
                c1.image(crop_np, width=120)
                label_text = "⚠️ Unknown" if obj.label == "Unknown" else obj.label
                c2.markdown(f"**{label_text}**")
                c2.progress(obj.confidence, text=f"{obj.confidence:.1%}")

    # --- Few-Shot Registration Panel: always shown if unknowns exist in session_state ---
    unknown_crops = st.session_state.unknown_crops
    if unknown_crops:
        st.divider()
        st.subheader("⚠️ Unknown Leaves — Teach the System")
        st.caption(
            f"{len(unknown_crops)} leaf(s) could not be identified by CLIP. "
            "Label them below — the system will remember for future runs."
        )

        cols = st.columns(min(len(unknown_crops), 4))
        for i, crop in enumerate(unknown_crops):
            cols[i % 4].image(crop, caption=f"Unknown #{i+1}", width=130)

        disease_name = st.text_input(
            "Disease name for these leaves",
            placeholder="e.g. Powdery Mildew, Anthracnose...",
            key="few_shot_disease_name"
        )
        if st.button("Register & Save", type="primary"):
            if not disease_name.strip():
                st.error("Enter a disease name.")
            else:
                pil_crops = [Image.fromarray(c) for c in unknown_crops]
                with st.spinner("Building prototype..."):
                    few_shot_engine.register(disease_name.strip(), pil_crops)
                st.session_state.unknown_crops = []
                st.success(
                    f"✅ **{disease_name}** registered from {len(pil_crops)} image(s). "
                    "Next time this disease appears, it will be detected automatically."
                )


if __name__ == "__main__":
    main()