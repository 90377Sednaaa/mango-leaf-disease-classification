"""
app.py - Mango Leaf Disease Classification Prototype
Side-by-Side Comparison: GourNet V1 (Baseline) vs GourNet V2 (Enhanced)
Theme-adaptive UI (optimized for both Light and Dark modes)
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

import model_utils

# Page configuration
st.set_page_config(
    page_title="Mango Leaf Disease Classification | GourNet V1 vs V2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-Adaptive CSS (seamless support for Dark & Light modes)
st.markdown(
    """
    <style>
    /* Content width and spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    
    /* Result Cards that adapt automatically to dark/light mode */
    .result-card {
        background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.06);
    }
    .diagnosis-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--secondary-text-color, #94a3b8);
        margin-bottom: 0.25rem;
    }
    .diagnosis-name {
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--text-color, #ffffff) !important;
        line-height: 1.2;
        margin-bottom: 0.65rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-v1-conf {
        background-color: rgba(59, 130, 246, 0.22);
        color: #60a5fa !important;
        border: 1px solid rgba(59, 130, 246, 0.45);
    }
    .badge-v2-conf {
        background-color: rgba(16, 185, 129, 0.22);
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .badge-latency {
        background-color: rgba(148, 163, 184, 0.15);
        color: var(--secondary-text-color, #cbd5e1) !important;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    /* Probabilities row styling */
    .prob-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .prob-name {
        font-weight: 500;
        color: var(--text-color, #ffffff) !important;
    }
    .prob-val {
        font-family: monospace;
        font-weight: 600;
        color: var(--text-color, #ffffff) !important;
    }
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 2rem !important;
        }
        .diagnosis-name {
            font-size: 1.4rem !important;
        }
        .result-card {
            padding: 1rem !important;
            margin-bottom: 1rem !important;
        }
        .badge-pill {
            margin-bottom: 0.3rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cached model loaders
@st.cache_resource(show_spinner="Loading GourNet V1 (Baseline)...")
def get_v1_model():
    return model_utils.load_model_instance("v1")


@st.cache_resource(show_spinner="Loading Enhanced GourNet V2...")
def get_v2_model():
    return model_utils.load_model_instance("v2")


v1_model = get_v1_model()
v2_model = get_v2_model()

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.header("Controls")

    st.subheader("Image Source")
    input_source = st.radio(
        "Select input method:",
        ("Pre-loaded Samples", "Upload Image", "Take Photo with Camera"),
        index=0,
        label_visibility="collapsed",
    )

    image_to_process = None
    image_title = ""

    # Dynamic directory scan: users can add custom images to samples/ anytime
    sample_dir = Path(__file__).resolve().parent / "samples"
    sample_dir.mkdir(exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    available_samples = sorted([
        f for f in sample_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ])

    if input_source == "Pre-loaded Samples":
        if available_samples:
            sample_options = {
                f.stem.replace("_", " ").title(): f
                for f in available_samples
            }
            selected_label = st.selectbox(
                "Select a sample specimen:",
                list(sample_options.keys()),
            )
            selected_path = sample_options[selected_label]
            image_to_process = Image.open(selected_path)
            image_title = f"{selected_label} ({selected_path.name})"
        else:
            st.warning("No sample files found in the 'samples/' directory.")

        st.caption("Tip: You can add custom sample images by dropping image files into the 'samples' folder.")

    elif input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose a leaf image (PNG, JPG, JPEG, WEBP):",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            image_title = f"Uploaded: {uploaded_file.name}"

    else:
        camera_img = st.camera_input("Point camera at mango leaf")
        if camera_img is not None:
            image_to_process = Image.open(camera_img)
            image_title = "Camera Capture"

    st.markdown("---")
    st.subheader("Input Validation")
    enable_leaf_guard = st.toggle(
        "Leaf-Only Validation Guard",
        value=True,
        help="Verifies biological leaf chlorophyll and central foliage signatures before model inference, rejecting non-leaf objects.",
    )

    if enable_leaf_guard:
        guard_strictness = st.select_slider(
            "Filter Strictness",
            options=["Permissive", "Balanced", "Strict"],
            value="Balanced",
            help="Permissive (10% center), Balanced (15% center), Strict (25% center).",
        )
        strictness_map = {
            "Permissive": (0.05, 0.10),
            "Balanced": (0.08, 0.15),
            "Strict": (0.12, 0.25),
        }
        min_overall, min_center = strictness_map[guard_strictness]
    else:
        min_overall, min_center = 0.08, 0.15

    st.markdown("---")
    st.subheader("Explainability")
    enable_gradcam = st.toggle("Enable Grad-CAM Heatmaps", value=True)

    if enable_gradcam:
        alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.10,
            max_value=0.85,
            value=0.45,
            step=0.05,
        )
        colormap = st.selectbox(
            "Colormap",
            ["jet", "viridis", "inferno", "plasma", "turbo"],
            index=0,
        )
    else:
        alpha = 0.45
        colormap = "jet"

    st.markdown("---")
    st.subheader("Model Architecture Specs")
    st.markdown(
        """
        - **GourNet V1 (Baseline)**
          - Reference: Alam et al. (2026)
          - Parameters: 683,656
          - Conv Layer: `Convolution-4`
        - **GourNet V2 (Enhanced)**
          - Thesis Improvement
          - Parameters: 98,280 (-85.6%)
          - Conv Layer: `Block4_Conv`
        """
    )


# ==============================================================================
# MAIN PAGE CONTENT
# ==============================================================================
st.title("Mango Leaf Disease Classification")
st.caption(
    "Side-by-side comparative evaluation of **GourNet V1** (Baseline, 683,656 parameters) "
    "and **Enhanced GourNet V2** (98,280 parameters) with class activation mapping."
)

if image_to_process is None:
    st.info("Select a pre-loaded sample from the sidebar or upload an image to begin classification.")
    st.stop()

# Validate leaf input if validation guard is active
val_metrics = {}
if enable_leaf_guard:
    is_leaf, val_message, val_metrics = model_utils.validate_leaf_image(
        image_to_process,
        min_overall_ratio=min_overall,
        min_center_ratio=min_center,
    )
    if not is_leaf:
        st.error(
            f"**Input Validation Rejected: Non-Leaf Specimen Detected**\n\n"
            f"{val_message}\n\n"
            f"Please upload or select a clear photograph of a mango leaf."
        )
        with st.expander("View Rejected Image", expanded=True):
            st.image(image_to_process, caption=f"Rejected Specimen: {image_title}", use_container_width=True)
            st.caption(
                f"Center foliage: {val_metrics.get('center_ratio', 0.0)*100:.1f}% (Required: {min_center*100:.0f}%) | "
                f"Overall foliage: {val_metrics.get('foliage_ratio', 0.0)*100:.1f}% (Required: {min_overall*100:.0f}%) | "
                f"Texture variation: {val_metrics.get('tv', 0.0):.4f}"
            )
        st.info("Tip: If you want to bypass this check to test out-of-distribution behavior, disable the 'Leaf-Only Validation Guard' in the sidebar.")
        st.stop()


# Preprocess image
batch_tensor, original_pil = model_utils.preprocess_image(image_to_process)


# Input overview expander
with st.expander("Input Image Information", expanded=False):
    col_img1, col_img2 = st.columns([1, 2])
    with col_img1:
        st.image(original_pil, caption=image_title, use_container_width=True)
    with col_img2:
        st.markdown(f"**Specimen Name:** `{image_title}`")
        st.markdown(f"**Source Resolution:** `{original_pil.width} x {original_pil.height}` pixels")
        st.markdown(f"**Model Input Dimensions:** `224 x 224` pixels (RGB)")
        if enable_leaf_guard and val_metrics:
            st.markdown(f"**Leaf Validation:** Verified (Foliage score: `{val_metrics.get('foliage_ratio', 0.0)*100:.1f}%`)")
        st.markdown(f"**Preprocessing:** Bilinear interpolation with internal `[0, 1]` rescaling.")

# Run Predictions
pred_v1 = model_utils.predict(v1_model, batch_tensor)
pred_v2 = model_utils.predict(v2_model, batch_tensor)

# High uncertainty / OOD alert
if pred_v1["top_confidence"] < 0.40 or pred_v2["top_confidence"] < 0.40:
    st.warning(
        "**High Uncertainty / Out-of-Distribution Warning**: Model confidence is below 40%. "
        "The image may be an atypical leaf specimen, degraded photo, or non-mango foliage."
    )


# Generate Grad-CAM overlays if toggled
gradcam_v1_overlay = None
gradcam_v2_overlay = None
if enable_gradcam:
    with st.spinner("Computing class activation heatmaps..."):
        heatmap_v1 = model_utils.generate_gradcam_heatmap(
            v1_model,
            model_utils.GRADCAM_LAYERS["v1"],
            batch_tensor,
            pred_v1["top_index"],
        )
        gradcam_v1_overlay = model_utils.overlay_gradcam(
            original_pil, heatmap_v1, alpha=alpha, colormap_name=colormap
        )

        heatmap_v2 = model_utils.generate_gradcam_heatmap(
            v2_model,
            model_utils.GRADCAM_LAYERS["v2"],
            batch_tensor,
            pred_v2["top_index"],
        )
        gradcam_v2_overlay = model_utils.overlay_gradcam(
            original_pil, heatmap_v2, alpha=alpha, colormap_name=colormap
        )

# ==============================================================================
# SIDE-BY-SIDE EVALUATION COLUMNS
# ==============================================================================
st.write("")
col_left, col_right = st.columns(2, gap="large")

# ----------------- LEFT COLUMN: V1 -----------------
with col_left:
    st.subheader("GourNet V1 (Baseline)")
    st.caption("Alam et al. (2026) Reproduction • 683,656 Parameters")

    # Diagnosis Card (with Blue Accent Border)
    st.markdown(
        f"""
        <div class="result-card" style="border-left: 4px solid #3b82f6;">
            <div class="diagnosis-label">Top Prediction</div>
            <div class="diagnosis-name">{pred_v1['top_class']}</div>
            <div>
                <span class="badge-pill badge-v1-conf">Confidence: {pred_v1['top_confidence']*100:.2f}%</span>
                <span class="badge-pill badge-latency">Latency: {pred_v1['latency_ms']:.1f} ms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Display
    if enable_gradcam and gradcam_v1_overlay is not None:
        st.image(
            gradcam_v1_overlay,
            caption=f"Grad-CAM Heatmap (Layer: {model_utils.GRADCAM_LAYERS['v1']})",
            use_container_width=True,
        )
    else:
        st.image(original_pil, caption="Input Leaf Image", use_container_width=True)

    # Top-3 Probabilities
    st.markdown("#### Top-3 Class Probabilities")
    for item in pred_v1["probabilities"][:3]:
        st.markdown(
            f"""
            <div class="prob-row">
                <span class="prob-name">{item['class']}</span>
                <span class="prob-val">{item['probability']*100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(item["probability"], 1.0))

    with st.expander("Full Probability Distribution (V1)"):
        df_v1 = pd.DataFrame(pred_v1["probabilities"])
        df_v1["probability (%)"] = (df_v1["probability"] * 100).round(2)
        st.dataframe(df_v1[["class", "probability (%)"]], use_container_width=True, hide_index=True)


# ----------------- RIGHT COLUMN: V2 -----------------
with col_right:
    st.subheader("Enhanced GourNet V2")
    st.caption("Thesis Improvement • 98,280 Parameters (-85.6%)")

    # Diagnosis Card (with Green Accent Border)
    st.markdown(
        f"""
        <div class="result-card" style="border-left: 4px solid #10b981;">
            <div class="diagnosis-label">Top Prediction</div>
            <div class="diagnosis-name">{pred_v2['top_class']}</div>
            <div>
                <span class="badge-pill badge-v2-conf">Confidence: {pred_v2['top_confidence']*100:.2f}%</span>
                <span class="badge-pill badge-latency">Latency: {pred_v2['latency_ms']:.1f} ms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Display
    if enable_gradcam and gradcam_v2_overlay is not None:
        st.image(
            gradcam_v2_overlay,
            caption=f"Grad-CAM Heatmap (Layer: {model_utils.GRADCAM_LAYERS['v2']})",
            use_container_width=True,
        )
    else:
        st.image(original_pil, caption="Input Leaf Image", use_container_width=True)

    # Top-3 Probabilities
    st.markdown("#### Top-3 Class Probabilities")
    for item in pred_v2["probabilities"][:3]:
        st.markdown(
            f"""
            <div class="prob-row">
                <span class="prob-name">{item['class']}</span>
                <span class="prob-val">{item['probability']*100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(item["probability"], 1.0))

    with st.expander("Full Probability Distribution (V2)"):
        df_v2 = pd.DataFrame(pred_v2["probabilities"])
        df_v2["probability (%)"] = (df_v2["probability"] * 100).round(2)
        st.dataframe(df_v2[["class", "probability (%)"]], use_container_width=True, hide_index=True)


# ==============================================================================
# COMPARATIVE BENCHMARK METRICS
# ==============================================================================
st.markdown("---")
st.subheader("Comparative Metrics")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

agreement = pred_v1["top_class"] == pred_v2["top_class"]
conf_delta = (pred_v2["top_confidence"] - pred_v1["top_confidence"]) * 100.0

with metric_col1:
    st.metric(
        label="Classification Agreement",
        value="Match" if agreement else "Differing",
        delta="Identical class" if agreement else "Different class",
    )

with metric_col2:
    st.metric(
        label="Confidence Delta (V2 - V1)",
        value=f"{pred_v2['top_confidence']*100:.2f}%",
        delta=f"{conf_delta:+.2f}%",
    )

with metric_col3:
    st.metric(
        label="Model Parameter Count",
        value="98,280",
        delta="-585,376 (-85.6%)",
        delta_color="normal",
    )

with metric_col4:
    st.metric(
        label="Inference Latency (V2)",
        value=f"{pred_v2['latency_ms']:.1f} ms",
        delta=f"{pred_v2['latency_ms'] - pred_v1['latency_ms']:+.1f} ms",
        delta_color="inverse",
    )

# ==============================================================================
# THESIS ARCHITECTURE SECTION
# ==============================================================================
with st.expander("Thesis Architectural Comparison (V1 vs V2)", expanded=False):
    st.markdown(
        """
        | Architectural Component | GourNet V1 (Baseline) | GourNet V2 (Enhanced Thesis Model) | Rationale & Effect |
        |---|---|---|---|
        | **Total Parameters** | 683,656 parameters | **98,280 parameters** | **85.6% reduction in model size**, improving edge-device feasibility. |
        | **Transition to Head** | `Flatten()` (9,216 features) | `GlobalAveragePooling2D()` (64 features) | Eliminates the 589,888-parameter dense bottleneck, reducing overfitting. |
        | **Normalization** | None | `GroupNormalization(groups=8)` | Prevents batch-statistic collapse on small batch sizes and stabilizes training. |
        | **Regularization** | Data Augmentation only | SpatialDropout(0.2) + Head Dropout(0.4) | Enforces distributed, redundant feature representations across channels. |
        | **Benchmark Test Accuracy** | ~97.00% (Paper reported) | **97.12%** | Matches/exceeds baseline accuracy with ~85% fewer weights. |
        """
    )
    st.caption("Note on Grad-CAM: GroupNormalization and GlobalAveragePooling2D encourage feature maps to focus specifically on localized lesion boundaries rather than whole-leaf geometry.")
