"""
model_utils.py - Backend utilities for Mango Leaf Disease Classification
Supports GourNet V1 (Baseline, 683k params) and GourNet V2 (Enhanced, 98k params).
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf
import keras

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]

_robust_v2 = Path(__file__).resolve().parent / "Gournet_v2" / "gournet_v2_robust"
_best_v2 = Path(__file__).resolve().parent / "Gournet_v2" / "gournet_v2_best"

MODEL_PATHS = {
    "v1": Path(__file__).resolve().parent / "Gournet_v1" / "gournet_model",
    "v2": _robust_v2 if _robust_v2.exists() else _best_v2,
}

GRADCAM_LAYERS = {
    "v1": "Convolution-4",
    # Explain the features after GroupNormalization + ReLU, before pooling.
    "v2": "Block4_ReLU",
}

MODEL_SPECS = {
    "v1": {
        "name": "GourNet (Baseline)",
        "paper": "Alam et al. (2026), AdComSys 2025",
        "parameters": 683_656,
        "key_features": "4 Conv Blocks, Flatten + Dense(64), Standard Bottleneck",
        "norm": "None (No Normalization)",
        "pooling": "Flatten layer (589k dense params)",
        "regularization": "Data Augmentation only",
        "last_conv_layer": "Convolution-4",
    },
    "v2": {
        "name": "Enhanced GourNet V2 (Robust)",
        "paper": "Thesis Improvement (Robust Training)",
        "parameters": 98_280,
        "key_features": "GroupNormalization, GlobalAveragePooling2D, Dual Dropout, Robust Augmentation",
        "norm": "GroupNormalization (groups=8) after each block",
        "pooling": "GlobalAveragePooling2D (0-param transition)",
        "regularization": "SpatialDropout(0.2) + Head Dropout(0.4) + Robust Augmentation",
        "last_conv_layer": "Block4_ReLU",
    },
}

DISEASE_INFO = {
    "Anthracnose": {
        "pathogen": "Colletotrichum gloeosporioides (Fungus)",
        "symptoms": (
            "Irregular, sunken brown to black necrotic spots with dark margins on leaves. "
            "Lesions may coalesce to form large blighted patches; leaves curl and prematurely drop."
        ),
        "favorable_conditions": "Warm temperature (25-30°C), heavy rainfall, high relative humidity (>90%).",
        "management": [
            "Prune and destroy infected twigs and fallen leaves to reduce inoculum.",
            "Improve canopy aeration and sunlight penetration via structural pruning.",
            "Apply copper oxychloride (0.3%) or carbendazim / azoxystrobin before flowering and during wet flushes.",
        ],
        "badge_color": "#D32F2F",
    },
    "Bacterial Canker": {
        "pathogen": "Xanthomonas citri pv. mangiferaeindicae (Bacterium)",
        "symptoms": (
            "Water-soaked, angular black spots surrounded by a prominent translucent yellow chlorotic halo. "
            "Lesions crack open and exude bacterial ooze; leaves dry out and defoliate."
        ),
        "favorable_conditions": "High humidity, driving rain, windstorms, and pruning or insect wounds.",
        "management": [
            "Establish windbreaks to reduce windblown sand/rain damage.",
            "Prune diseased shoots during dry weather and sanitize pruning tools.",
            "Spray copper hydroxide or copper oxychloride combined with streptomycin sulfate (100-200 ppm).",
        ],
        "badge_color": "#C2185B",
    },
    "Cutting Weevil": {
        "pathogen": "Deporaus marginatus (Pest - Coleopteran Curculionid)",
        "symptoms": (
            "Fresh tender leaves are cleanly cut straight across the blade as if sliced with scissors. "
            "The severed apical halves drop to the orchard floor, stunting shoot development."
        ),
        "favorable_conditions": "Flushing periods with tender young vegetative leaves in monsoon/post-monsoon seasons.",
        "management": [
            "Rake and destroy fallen severed leaf portions to kill eggs and developing larvae.",
            "Shallow ploughing around tree bases to expose pupating larvae in soil to predators and sun.",
            "Apply neem seed kernel extract (NSKE 5%) or spinosad / bifenthrin at the emergence of new flush leaves.",
        ],
        "badge_color": "#7B1FA2",
    },
    "Die Back": {
        "pathogen": "Lasiodiplodia theobromae (Fungus)",
        "symptoms": (
            "Drying and browning of twigs beginning at the tip and progressing downward. "
            "Leaves turn dull brown, roll upward, and remain attached to dead branches (scorched look)."
        ),
        "favorable_conditions": "Drought stress, soil nutritional imbalance, physical wounds, stem-borer injuries.",
        "management": [
            "Prune infected branches at least 5-8 cm into healthy green wood.",
            "Immediately paste cut surfaces with Bordeaux paste or copper fungicide paint.",
            "Maintain balanced irrigation and spray thiophanate-methyl or propiconazole.",
        ],
        "badge_color": "#E65100",
    },
    "Gall Midge": {
        "pathogen": "Procontarinia matteiana / Procontarinia spp. (Insect - Diptera)",
        "symptoms": (
            "Wart-like, raised circular or conical blister galls on leaf blades. "
            "Severe infestation causes leaf crinkling, malformation, reduced photosynthesis, and leaf drop."
        ),
        "favorable_conditions": "Warm, humid weather coinciding with the initiation of young leaf flushes.",
        "management": [
            "Prune heavily infested twigs and bury or burn pruned foliage.",
            "Spray systemic insecticides such as imidacloprid (0.3 ml/L) or thiamethoxam upon young flush emergence.",
            "Encourage natural parasitoid wasps (Platygaster spp.) by avoiding broad-spectrum residual insecticides.",
        ],
        "badge_color": "#5D4037",
    },
    "Healthy": {
        "pathogen": "None (Healthy Foliage)",
        "symptoms": "Vibrant green, glossy leaf surface with uniform color, intact margins, and crisp venation.",
        "favorable_conditions": "Optimal agronomic care, proper soil moisture, balanced nutrition, pest-free orchard.",
        "management": [
            "Maintain scheduled balanced NPK and micronutrient (zinc, boron) foliar feeding.",
            "Ensure adequate drip irrigation, avoid waterlogging.",
            "Conduct routine scouting to catch early symptoms before pathogens establish.",
        ],
        "badge_color": "#2E7D32",
    },
    "Powdery Mildew": {
        "pathogen": "Oidium mangiferae (Fungus)",
        "symptoms": (
            "White to grayish powdery fungal patches on the upper/lower surface of tender leaves and blossoms. "
            "Infected leaves curl, distort, become leathery, and shed prematurely."
        ),
        "favorable_conditions": "Cool nights (10-15°C) followed by warm dry days (25-30°C) with high morning humidity.",
        "management": [
            "Prune interior deadwood to permit sunlight and air circulation throughout the tree crown.",
            "Apply wettable sulfur (0.2%) as a protective preventative spray.",
            "Apply systemic triazole fungicides like hexaconazole or penconazole at early panicle/flush emergence.",
        ],
        "badge_color": "#0288D1",
    },
    "Sooty Mould": {
        "pathogen": "Meliola mangiferae / Capnodium ramosum (Saprophytic Fungi)",
        "symptoms": (
            "Thick, velvety, black superficial coating over leaf blades and stems. "
            "Fungus does not penetrate plant tissue directly but significantly impairs photosynthesis and respiration."
        ),
        "favorable_conditions": "Presence of honeydew secreted by sucking pests (mango hoppers, mealybugs, scales).",
        "management": [
            "Target and eliminate sucking insect populations using horticultural oil, insecticidal soap, or imidacloprid.",
            "Spray a 1% to 2% dilute starch solution (boiled starch) which dries into flakes and peels off the soot.",
            "Wash tree crowns with water jets during dry spells.",
        ],
        "badge_color": "#37474F",
    },
}

_LOADED_MODELS: Dict[str, Any] = {}


def load_model_instance(model_key: str):
    """
    Load and return the Keras model instance for 'v1' or 'v2'.
    Caches model in memory.
    """
    if model_key not in MODEL_PATHS:
        raise ValueError(f"Unknown model_key: {model_key}. Expected 'v1' or 'v2'.")

    if model_key not in _LOADED_MODELS:
        path = str(MODEL_PATHS[model_key])
        model = keras.models.load_model(path)
        _LOADED_MODELS[model_key] = model

    return _LOADED_MODELS[model_key]


def preprocess_image(image_input) -> Tuple[np.ndarray, Image.Image]:
    """
    Takes a PIL Image, path, or file-like object, converts to RGB,
    resizes to (224, 224), and returns (batch_tensor, pil_display_image).
    Note: GourNet models contain an internal Rescaling(1./255) layer,
    so raw pixel values [0, 255] are passed to the model.
    """
    if not isinstance(image_input, Image.Image):
        pil_img = Image.open(image_input).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    resized_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32)
    batch_tensor = np.expand_dims(img_array, axis=0)
    return batch_tensor, pil_img


def predict(model, batch_tensor: np.ndarray) -> Dict[str, Any]:
    """
    Runs model inference and returns prediction metrics:
    - top_class: str
    - top_confidence: float (0.0 to 1.0)
    - top_index: int
    - all_probabilities: list of dicts [{'class': str, 'prob': float}, ...] sorted desc
    - latency_ms: float
    """
    start_time = time.perf_counter()
    preds = model(batch_tensor, training=False).numpy()[0]
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    top_confidence = float(preds[top_idx])

    ranked = []
    for idx in np.argsort(-preds):
        ranked.append({
            "class": CLASS_NAMES[idx],
            "probability": float(preds[idx]),
            "index": int(idx),
        })

    return {
        "top_class": top_class,
        "top_confidence": top_confidence,
        "top_index": top_idx,
        "probabilities": ranked,
        "latency_ms": latency_ms,
        "raw_logits": preds,
    }


def generate_gradcam_heatmap(
    model,
    layer_name: str,
    img_tensor: np.ndarray,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Explain one image using the pre-softmax score of a GourNet Dense head.

    Does not mutate the model's activation, weights, or prediction probabilities.
    Returns a normalized 2D map; all zeros means no positive Grad-CAM signal,
    NOT that the model ignored the leaf. Grad-CAM is not a segmentation mask.
    """
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    if img_tensor.shape.rank != 4 or img_tensor.shape[0] != 1:
        raise ValueError("Grad-CAM requires a single image batch (1, H, W, C).")
    classifier = model.layers[-1]
    if not isinstance(classifier, keras.layers.Dense):
        raise ValueError("Grad-CAM expects a final Dense classifier, as used by GourNet.")
    if keras.activations.serialize(classifier.activation) not in ("softmax", "linear"):
        raise ValueError("Grad-CAM supports a softmax or linear Dense classifier only.")
    if pred_index is not None and (
        not isinstance(pred_index, (int, np.integer))
        or not 0 <= pred_index < classifier.units
    ):
        raise ValueError("pred_index must be an integer within the classifier's class range.")

    target = model.get_layer(layer_name).output
    if len(target.shape) != 4:
        raise ValueError("Grad-CAM target must have spatial dimensions (B, H, W, C).")
    grad_model = keras.models.Model(
        model.inputs, [target, classifier.input],
    )

    with tf.GradientTape() as tape:
        # Watching inputs also supports models whose layers are all frozen.
        tape.watch(img_tensor)
        conv_outputs, head_features = grad_model(img_tensor, training=False)
        # Reconstruct the Dense pre-activation inside the tape. Never remove
        # softmax in-place: the app shares its cached model across requests.
        logits = tf.linalg.matmul(head_features, classifier.kernel)
        if classifier.use_bias:
            logits = tf.nn.bias_add(logits, classifier.bias)
        if pred_index is None:
            pred_index = int(tf.argmax(logits[0]))
        class_channel = logits[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Grad-CAM target is disconnected from the class score.")
    if not bool(tf.reduce_all(tf.math.is_finite(grads))):
        raise ValueError("Grad-CAM produced non-finite gradients.")
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    if not bool(tf.reduce_all(tf.math.is_finite(heatmap))):
        raise ValueError("Grad-CAM produced a non-finite heatmap.")
    return tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap)).numpy()


def overlay_gradcam(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Overlays a 2D Grad-CAM heatmap onto the original PIL image.
    Uses bicubic interpolation for smooth visual gradients.
    """
    # An empty explanation must not masquerade as a meaningful blue overlay.
    if not np.any(heatmap > 0):
        return original_pil.copy()

    # Resize heatmap to match original image dimensions
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_img.resize(original_pil.size, resample=Image.Resampling.BICUBIC)
    heatmap_resized_arr = np.array(heatmap_resized) / 255.0

    # Apply colormap
    import matplotlib
    color_map = matplotlib.colormaps.get(colormap_name, matplotlib.colormaps["jet"])
    colored_heatmap = color_map(heatmap_resized_arr)[:, :, :3]  # drop alpha
    colored_heatmap_uint8 = np.uint8(255 * colored_heatmap)

    orig_arr = np.array(original_pil)
    # Blend images
    blended = np.uint8((1.0 - alpha) * orig_arr + alpha * colored_heatmap_uint8)
    return Image.fromarray(blended)


def validate_leaf_image(
    pil_img: Image.Image,
    min_overall_ratio: float = 0.08,
    min_center_ratio: float = 0.15,
) -> Tuple[bool, str, Dict[str, float]]:
    """
    Verifies whether an input image possesses genuine botanical plant foliage
    and chlorophyll characteristics focused in the central subject area.

    Checks:
    1. Total variation (detects blank images, uniform blocks, and visual noise/static).
    2. Overall Foliage Signature (detects plant tissue pixels across the entire frame).
    3. Central Focus Verification (checks the center 50% region to reject outdoor
       photos where foliage is merely background behind people, cars, or objects).

    Returns:
        (is_valid, message, metrics_dict)
    """
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    # 1. Total variation test (detect blank images, solid blocks, or visual noise)
    tv = float(np.mean(np.abs(gray[1:, :] - gray[:-1, :])))
    if tv < 0.003:
        return False, "Image appears blank or is a uniform solid-color block.", {"foliage_ratio": 0.0, "center_ratio": 0.0, "tv": tv}
    if tv > 0.22:
        return False, "Image contains unstructured visual static or noise rather than photographic texture.", {"foliage_ratio": 0.0, "center_ratio": 0.0, "tv": tv}

    # 2. Overall Foliage Signature
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_dom = (g > r * 0.90) & (g > b * 1.10) & (g > 0.12)
    overall_ratio = float(np.mean(green_dom))

    # 3. Central Focal Area Verification (middle 50% width and height)
    center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    cr, cg, cb = center[:, :, 0], center[:, :, 1], center[:, :, 2]
    center_green = (cg > cr * 0.90) & (cg > cb * 1.10) & (cg > 0.12)
    center_ratio = float(np.mean(center_green))

    metrics = {
        "foliage_ratio": overall_ratio,
        "center_ratio": center_ratio,
        "tv": tv,
    }

    if overall_ratio < min_overall_ratio:
        return (
            False,
            f"No significant plant foliage detected (Overall foliage: {overall_ratio*100:.1f}%, minimum required: {min_overall_ratio*100:.1f}%).",
            metrics,
        )

    if center_ratio < min_center_ratio:
        return (
            False,
            f"Subject is not centered plant foliage. Foliage appears to be background only (Center foliage: {center_ratio*100:.1f}%, minimum required: {min_center_ratio*100:.1f}%).",
            metrics,
        )

    return (
        True,
        f"Botanical leaf specimen verified (Center: {center_ratio*100:.1f}%, Overall: {overall_ratio*100:.1f}%).",
        metrics,
    )


def run_batch_evaluation(
    v1_model,
    v2_model,
    items: list,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Executes batch inference across multiple leaf images for both GourNet V1 and V2,
    evaluates predictions against Ground Truth labels, and computes summary statistics
    strictly formatted as percentages.

    Returns:
        {
            "summary": {
                "total_images": int,
                "v1": {
                    "match_rate": float,
                    "avg_confidence": float,
                    "misclassification_rate": float,
                    "matches_count": int,
                },
                "v2": {
                    "match_rate": float,
                    "avg_confidence": float,
                    "misclassification_rate": float,
                    "matches_count": int,
                },
            },
            "details": list of dicts,
        }
    """
    total = len(items)
    if total == 0:
        return {"summary": {}, "details": []}

    rows = []
    for idx, item in enumerate(items):
        filename = item["filename"]
        img = item["image"]
        ground_truth = item["ground_truth"]

        batch_tensor, _ = preprocess_image(img)
        p1 = predict(v1_model, batch_tensor)
        p2 = predict(v2_model, batch_tensor)

        v1_pred = p1["top_class"]
        v1_conf = float(p1["top_confidence"])
        v1_match = (v1_pred == ground_truth)

        v2_pred = p2["top_class"]
        v2_conf = float(p2["top_confidence"])
        v2_match = (v2_pred == ground_truth)

        rows.append({
            "Filename": filename,
            "Ground Truth": ground_truth,
            "V1 Prediction": v1_pred,
            "V1 Confidence (%)": round(v1_conf * 100.0, 1),
            "V1 Match": "Match" if v1_match else "Misclassified",
            "V1 Match Bool": v1_match,
            "V1 Raw Conf": v1_conf,
            "V2 Prediction": v2_pred,
            "V2 Confidence (%)": round(v2_conf * 100.0, 1),
            "V2 Match": "Match" if v2_match else "Misclassified",
            "V2 Match Bool": v2_match,
            "V2 Raw Conf": v2_conf,
        })

        if progress_callback:
            progress_callback(idx + 1, total)

    v1_matches = sum(1 for r in rows if r["V1 Match Bool"])
    v1_match_rate = round((v1_matches / total) * 100.0, 1)
    v1_avg_conf = round((sum(r["V1 Raw Conf"] for r in rows) / total) * 100.0, 1)
    v1_misc_rate = round(100.0 - v1_match_rate, 1)

    v2_matches = sum(1 for r in rows if r["V2 Match Bool"])
    v2_match_rate = round((v2_matches / total) * 100.0, 1)
    v2_avg_conf = round((sum(r["V2 Raw Conf"] for r in rows) / total) * 100.0, 1)
    v2_misc_rate = round(100.0 - v2_match_rate, 1)

    return {
        "summary": {
            "total_images": total,
            "v1": {
                "match_rate": v1_match_rate,
                "avg_confidence": v1_avg_conf,
                "misclassification_rate": v1_misc_rate,
                "matches_count": v1_matches,
            },
            "v2": {
                "match_rate": v2_match_rate,
                "avg_confidence": v2_avg_conf,
                "misclassification_rate": v2_misc_rate,
                "matches_count": v2_matches,
            },
        },
        "details": rows,
    }



