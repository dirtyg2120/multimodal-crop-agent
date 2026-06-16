import os
import asyncio
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

from groundingdino.util.inference import predict, load_model, annotate, load_image

from app.pipe import VisionSystem, analyze_full_plant, classify_crops, CLIP_CONFIDENCE_THRESHOLD
from app.few_shot_engine import FewShotEngine
from app.vision.clip_labels import DISEASE_LABELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GDINO_CONFIG = "app/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GDINO_WEIGHTS = "app/groundingdino/weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "leaf . bug . worm ."


@st.cache_resource
def load_gdino_model():
    return load_model(GDINO_CONFIG, GDINO_WEIGHTS, device=DEVICE)

@st.cache_resource
def load_vision_system():
    return VisionSystem()

@st.cache_resource
def load_few_shot_engine():
    return FewShotEngine()


MAX_AREA_RATIO = 0.95  # drop DINO boxes that cover ≥85% of the image (whole-image false detections)

def extract_crops(image_source, boxes, phrases):
    h, w, _ = image_source.shape
    img_area = h * w
    crops = []
    for box, phrase in zip(boxes, phrases):
        cx, cy, bw, bh = box.tolist()
        x1, y1 = max(0, int((cx - bw/2) * w)), max(0, int((cy - bh/2) * h))
        x2, y2 = min(w, int((cx + bw/2) * w)), min(h, int((cy + bh/2) * h))
        box_area = (x2 - x1) * (y2 - y1)
        if box_area / img_area >= MAX_AREA_RATIO:
            continue  # skip whole-image boxes
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


def annotate_with_clip_labels(image_source: np.ndarray, crops_data: list, detections: list) -> np.ndarray:
    """
    Color-coded: green=healthy, red=disease, orange=unknown, yellow=pest.
    """
    if not detections:
        return image_source

    pil = Image.fromarray(image_source).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")

    # Use default PIL font (no external font needed)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for obj in detections:
        x1, y1, x2, y2 = obj.box
        label = obj.label
        conf  = obj.confidence

        # Box colour
        if label == "Unknown":
            color = (255, 140, 0)    # orange
        elif "Healthy" in label:
            color = (34, 197, 94)    # green
        elif any(kw in crops_data[obj.crop_id]["label"].lower() for kw in ("bug", "worm", "pest")):
            color = (234, 179, 8)    # yellow
        else:
            color = (239, 68, 68)    # red

        # Draw box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Format: "<Crop>, <disease> %"
        if label == "Unknown":
            text = f"Unknown {conf:.0%}"
        elif " leaf with " in label:
            # e.g. "Tomato leaf with Early blight" → "Tomato, Early blight 87%"
            crop, disease = label.split(" leaf with ", 1)
            text = f"{crop}, {disease} {conf:.0%}"
        elif label.startswith("Healthy "):
            # e.g. "Healthy Tomato leaf" → "Tomato, Healthy 87%"
            crop = label.replace("Healthy ", "").replace(" leaf", "")
            text = f"{crop}, Healthy {conf:.0%}"
        else:
            text = f"{label} {conf:.0%}"

        # Measure text size
        bbox  = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3

        # Label background: inside top-left of the box
        tx, ty = x1 + pad, y1 + pad
        draw.rectangle(
            [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
            fill=(*color, 200),   # semi-transparent
        )
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    return np.array(pil)


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
    if "active_disease_labels" not in st.session_state:
        st.session_state.active_disease_labels = DISEASE_LABELS[:]

    # Sidebar: registered prototypes only
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

    with st.expander("🏷️ Edit Disease Labels", expanded=False):
        st.caption(
            f"Active: **{len(st.session_state.active_disease_labels)}** labels  ·  "
            "One label per line  ·  Changes apply on next \'Analyze Image\'."
        )
        edited_text = st.text_area(
            "Disease labels",
            value="\n".join(st.session_state.active_disease_labels),
            height=250,
            key="label_editor",
            label_visibility="collapsed",
        )
        col_save, col_reset = st.columns([1, 1])
        if col_save.button("💾 Save labels", use_container_width=True):
            parsed = [ln.strip() for ln in edited_text.splitlines() if ln.strip()]
            if parsed:
                st.session_state.active_disease_labels = parsed
                st.success(f"✅ {len(parsed)} labels saved.")
            else:
                st.error("Label list cannot be empty.")
        if col_reset.button("↩️ Reset to defaults", use_container_width=True):
            st.session_state.active_disease_labels = DISEASE_LABELS[:]
            st.rerun()

    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original Image")

    # --- Analyze button: DINO + CLIP only, no agent ---
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

        with st.spinner("Classifying leaves..."):
            detections = classify_crops(
                crops_data, vision_system, few_shot_engine,
                disease_labels=st.session_state.active_disease_labels,
            )
            clip_annotated = annotate_with_clip_labels(image_source, crops_data, detections)

        with col2:
            st.image(clip_annotated, caption="Detections (CLIP labels)")

        # Persist detection results; reset stale agent output from previous image
        st.session_state.crops_data     = crops_data
        st.session_state.detections     = detections
        st.session_state.image_source   = image_source
        st.session_state.clip_annotated = clip_annotated
        st.session_state.results        = None
        unknown_crops = [det for det in detections if det.label == "Unknown"]
        st.session_state.unknown_crops  = [crops_data[obj.crop_id]["crop"] for obj in unknown_crops]

    # --- Re-display CLIP annotated image on reruns ---
    elif st.session_state.get("clip_annotated") is not None:
        with col2:
            st.image(st.session_state.clip_annotated, caption="Detections (CLIP labels)")

    # --- Agent button: shown only after detection has run ---
    if st.session_state.get("detections") is not None:
        st.divider()
        if st.button("🧠 Run Agent Diagnosis", type="secondary"):
            with st.spinner("Agent is thinking..."):
                results = asyncio.run(analyze_full_plant(
                    st.session_state.crops_data,
                    vision_system=vision_system,
                    few_shot_engine=few_shot_engine,
                    detections=st.session_state.detections,
                ))
            st.session_state.results = results

    # --- Results display ---
    results    = st.session_state.results
    crops_data = st.session_state.crops_data

    if results:
        agent_json = results["agent_response"]

        st.divider()
        st.subheader("📝 Agronomist Diagnosis")

        k1, k2, k3 = st.columns(3)
        k1.metric("Overall Health", agent_json.overall_health_status)
        k2.metric("Severity", agent_json.severity_level)
        k3.metric("Infection Ratio", f"{agent_json.infection_ratio:.0%}")

        with st.expander("Treatment Plan", expanded=True):
            # Normalize reasoning: split into readable paragraphs
            reasoning = agent_json.reasoning.strip()
            # Split on numbered steps (e.g. "1.", "2.") or double newlines
            import re
            paragraphs = re.split(r'\n{2,}|(?<=\.)\s*(?=\d+[\.\)])', reasoning)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            st.info("**Reasoning:**")
            for p in paragraphs:
                st.markdown(f"> {p}")

            # Render actions as a numbered list
            st.markdown("**Recommended Actions:**")
            for i, action in enumerate(agent_json.recommended_actions, 1):
                action = action.strip().lstrip("-•* ").strip()
                st.markdown(f"{i}. {action}")

            if agent_json.required_pesticides:
                st.warning(f"💊 **Chemicals:** {', '.join(agent_json.required_pesticides)}")

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