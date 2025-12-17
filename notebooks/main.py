from dino_clip_pipeline import (
    load_groundingdino,
    load_plant_disease_clip,
    detect_and_classify_leaf_disease,
)

dino = load_groundingdino(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth",
    device="cpu",
)

clip_model, clip_proc = load_plant_disease_clip(device="cpu")  # pass hf_token=... if needed

out = detect_and_classify_leaf_disease(
    image_path_or_url="./data/tomato.png",
    dino_model=dino,
    clip_model=clip_model,
    clip_processor=clip_proc,
    dino_text_prompt="leaf .",
    return_bytes=False,   # set True if you also want JPEG bytes per crop
)

for i, c in enumerate(out["crops"]):
    if c["clip_result"] is None:
        continue  # not routed to CLIP
    r = c["clip_result"]
    print(i, c["label"], c["bbox"], r["status"], r["label"], r["confidence"])

