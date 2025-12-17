from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# --------- Optional imports (kept lazy-ish to reduce import-time failures) ---------


def _lazy_import_groundingdino():
    from groundingdino.util.inference import load_model, load_image, predict  # type: ignore
    return load_model, load_image, predict


def _lazy_import_clip():
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    import torch  # type: ignore
    return CLIPModel, CLIPProcessor, torch


# --------- Data structures ---------


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class CropItem:
    label: str
    bbox: BBox
    score: float
    crop_np: np.ndarray  # HWC, RGB, uint8
    crop_pil: Image.Image
    crop_bytes: Optional[bytes] = None
    clip_result: Optional[Dict[str, Any]] = None


# --------- Utilities ---------


def _to_pil_rgb(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            return Image.fromarray(image, mode="L").convert("RGB")
        if image.shape[-1] == 4:
            return Image.fromarray(image, mode="RGBA").convert("RGB")
        return Image.fromarray(image, mode="RGB")
    # path/url string -> let GroundingDINO loader handle it elsewhere;
    # for crop extraction we expect PIL already.
    return Image.open(image).convert("RGB")


def _encode_jpeg(pil_img: Image.Image, quality: int = 92) -> bytes:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _clip_bbox(b: Sequence[float], w: int, h: int) -> BBox:
    # GroundingDINO returns xyxy in pixel coords for predict(..., ...), but we guard anyway.
    x1, y1, x2, y2 = b
    x1i = max(0, min(int(round(x1)), w - 1))
    y1i = max(0, min(int(round(y1)), h - 1))
    x2i = max(0, min(int(round(x2)), w))
    y2i = max(0, min(int(round(y2)), h))
    # ensure proper ordering and minimum size
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return (x1i, y1i, x2i, y2i)


# --------- Core: crop extraction ---------


def extract_crops(
    image_source: Union[str, np.ndarray, Image.Image],
    boxes_xyxy: np.ndarray,
    phrases: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    return_bytes: bool = False,
    jpeg_quality: int = 92,
) -> List[CropItem]:
    """
    Extract crops for every detection.

    Parameters
    ----------
    image_source:
        Path/URL, numpy array, or PIL Image.
    boxes_xyxy:
        Array of shape (N, 4) in pixel xyxy format.
    phrases:
        List of N phrases corresponding to boxes.
    scores:
        Optional N scores; if omitted, will default to 1.0.
    return_bytes:
        If True, also return JPEG bytes per crop in CropItem.crop_bytes.

    Returns
    -------
    List[CropItem]
        Each item includes crop_pil and crop_np, plus optional bytes.
    """
    img_pil = _to_pil_rgb(image_source)
    w, h = img_pil.size

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=float)
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must be (N,4); got {boxes_xyxy.shape}")

    if len(phrases) != len(boxes_xyxy):
        raise ValueError("phrases length must match boxes length")

    if scores is None:
        scores = [1.0] * len(boxes_xyxy)
    if len(scores) != len(boxes_xyxy):
        raise ValueError("scores length must match boxes length")

    out: List[CropItem] = []
    for b, phr, sc in zip(boxes_xyxy, phrases, scores):
        x1, y1, x2, y2 = _clip_bbox(b, w=w, h=h)
        crop_pil = img_pil.crop((x1, y1, x2, y2))
        crop_np = np.array(crop_pil, dtype=np.uint8)
        crop_bytes = _encode_jpeg(crop_pil, quality=jpeg_quality) if return_bytes else None
        out.append(
            CropItem(
                label=str(phr),
                bbox=(x1, y1, x2, y2),
                score=float(sc),
                crop_np=crop_np,
                crop_pil=crop_pil,
                crop_bytes=crop_bytes,
                clip_result=None,
            )
        )
    return out


# --------- GroundingDINO wrappers ---------


def load_groundingdino(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
) -> Any:
    """
    Load GroundingDINO model using the repo's inference helper.
    """
    load_model, _, _ = _lazy_import_groundingdino()
    return load_model(config_path, checkpoint_path, device=device)


def dino_predict(
    dino_model: Any,
    image_path_or_url: str,
    text_prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Run GroundingDINO prediction and return raw outputs.

    Returns keys:
      - image_source_np: numpy image from loader
      - boxes_xyxy: (N,4) float xyxy in pixel coords
      - phrases: list[str]
      - scores: list[float]
    """
    _, load_image, predict = _lazy_import_groundingdino()
    image_source_np, image = load_image(image_path_or_url)
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    # `boxes` is xyxy in pixel coords in current GroundingDINO util.inference
    boxes_xyxy = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
    scores = (
        logits.sigmoid().cpu().numpy().tolist()
        if hasattr(logits, "sigmoid")
        else np.asarray(logits).tolist()
    )
    # scores may be per-box scalar or per-token; keep scalar per box where possible
    # If it's a 2D array, take max for each box.
    scores_arr = np.asarray(scores)
    if scores_arr.ndim == 2:
        scores_list = scores_arr.max(axis=1).astype(float).tolist()
    elif scores_arr.ndim == 1:
        scores_list = scores_arr.astype(float).tolist()
    else:
        scores_list = [float(s) for s in scores]  # fallback
    return {
        "image_source_np": image_source_np,
        "boxes_xyxy": boxes_xyxy,
        "phrases": [str(p) for p in phrases],
        "scores": scores_list,
    }


# --------- CLIP helpers ---------


def load_clip(
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """
    Load a standard CLIP model + processor.

    You can swap `model_name` for a fine-tuned disease CLIP if you have one.
    """
    CLIPModel, CLIPProcessor, torch = _lazy_import_clip()
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_proc = CLIPProcessor.from_pretrained(model_name)
    clip_model.to(device)
    clip_model.eval()
    return clip_model, clip_proc


# --------- Leaf-only routing to your open-set logic ---------


def route_leaf_crops_to_clip(
    crops: List[CropItem],
    *,
    clip_model: Any,
    clip_processor: Any,
    classifier_fn: Callable[..., Dict[str, Any]],
    leaf_keyword: str = "leaf",
    classifier_kwargs: Optional[Dict[str, Any]] = None,
) -> List[CropItem]:
    """
    For every crop whose label contains leaf_keyword (case-insensitive),
    run classifier_fn(crop_pil, clip_model, clip_processor, **classifier_kwargs).

    classifier_fn is expected to return a dict, e.g.
        {"status": "known"/"unknown", "label": "...", "confidence": 0.87, ...}
    """
    if classifier_kwargs is None:
        classifier_kwargs = {}

    kw = leaf_keyword.lower()
    for c in crops:
        if kw in (c.label or "").lower():
            c.clip_result = classifier_fn(
                c.crop_pil,
                clip_model=clip_model,
                clip_processor=clip_processor,
                **classifier_kwargs,
            )
        else:
            c.clip_result = None
    return crops


# --------- End-to-end convenience ---------


def detect_and_classify_leaf_disease(
    image_path_or_url: str,
    *,
    dino_model: Any,
    clip_model: Any,
    clip_processor: Any,
    dino_text_prompt: str = "leaf .",
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    return_bytes: bool = False,
    jpeg_quality: int = 92,
    classifier_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    leaf_keyword: str = "leaf",
) -> Dict[str, Any]:
    """
    Full pipeline:
      1) GroundingDINO detect
      2) extract_crops(...) returns ALL crops
      3) route only "leaf" crops through classifier_fn (your CLIP open-set logic)

    If classifier_fn is None, this tries to import `classify_disease_with_open_set_logic`
    from the current Python path.
    """
    if classifier_fn is None:
        # Expect user to have this function somewhere importable.
        try:
            from classify import classify_disease_with_open_set_logic  # type: ignore
        except Exception:
            try:
                from clip_classifier import classify_disease_with_open_set_logic  # type: ignore
            except Exception as e:
                raise ImportError(
                    "classifier_fn was None and classify_disease_with_open_set_logic "
                    "could not be imported. Pass classifier_fn explicitly or provide "
                    "classify_disease_with_open_set_logic in classify.py / clip_classifier.py."
                ) from e
        classifier_fn = classify_disease_with_open_set_logic

    raw = dino_predict(
        dino_model=dino_model,
        image_path_or_url=image_path_or_url,
        text_prompt=dino_text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    crops = extract_crops(
        image_source=raw["image_source_np"],
        boxes_xyxy=raw["boxes_xyxy"],
        phrases=raw["phrases"],
        scores=raw["scores"],
        return_bytes=return_bytes,
        jpeg_quality=jpeg_quality,
    )

    crops = route_leaf_crops_to_clip(
        crops,
        clip_model=clip_model,
        clip_processor=clip_processor,
        classifier_fn=classifier_fn,
        leaf_keyword=leaf_keyword,
        classifier_kwargs=classifier_kwargs,
    )

    return {
        "image_path_or_url": image_path_or_url,
        "dino_text_prompt": dino_text_prompt,
        "detections": {
            "boxes_xyxy": raw["boxes_xyxy"],
            "phrases": raw["phrases"],
            "scores": raw["scores"],
        },
        "crops": [c.__dict__ for c in crops],  # serialize-friendly
    }
