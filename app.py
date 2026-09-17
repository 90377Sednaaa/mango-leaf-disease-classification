"""
app.py - Mango Leaf Disease Classification Prototype
Side-by-Side Comparison: GourNet V1 (Baseline) vs GourNet V2 (Enhanced)
Theme-adaptive UI (optimized for both Light and Dark modes)
Includes Single Specimen Analysis and Batch Validation & Testing modes.
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
# SINGLE SPECIMEN VIEW
# ==============================================================================
def render_single_specimen_view(v1_model, v2_model):
    with st.sidebar:
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

    # Main page content for single specimen
    st.title("Mango Leaf Disease Classification")
    st.caption(
        "Side-by-side comparative evaluation of **GourNet V1** (Baseline, 683,656 parameters) "
        "and **Enhanced GourNet V2** (98,280 parameters) with class activation mapping."
    )

    if image_to_process is None:
        st.info("Select a pre-loaded sample from the sidebar or upload an image to begin classification.")
        return

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
            return

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

    # Side-by-side evaluation columns
    st.write("")
    col_left, col_right = st.columns(2, gap="large")

    # Left Column: V1
    with col_left:
        st.subheader("GourNet V1 (Baseline)")
        st.caption("Alam et al. (2026) Reproduction • 683,656 Parameters")

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

        if enable_gradcam and gradcam_v1_overlay is not None:
            st.image(
                gradcam_v1_overlay,
                caption=f"Grad-CAM Heatmap (Layer: {model_utils.GRADCAM_LAYERS['v1']})",
                use_container_width=True,
            )
        else:
            st.image(original_pil, caption="Input Leaf Image", use_container_width=True)

        if enable_gradcam and not np.any(heatmap_v1 > 0):
            st.warning(
                "V1: No positive Grad-CAM signal for this class. Showing the original "
                "image; this does not mean the leaf was ignored."
            )

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

    # Right Column: V2
    with col_right:
        st.subheader("Enhanced GourNet V2")
        st.caption("Thesis Improvement • 98,280 Parameters (-85.6%)")

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

        if enable_gradcam and gradcam_v2_overlay is not None:
            st.image(
                gradcam_v2_overlay,
                caption=f"Grad-CAM Heatmap (Layer: {model_utils.GRADCAM_LAYERS['v2']})",
                use_container_width=True,
            )
        else:
            st.image(original_pil, caption="Input Leaf Image", use_container_width=True)

        if enable_gradcam and not np.any(heatmap_v2 > 0):
            st.warning(
                "V2: No positive Grad-CAM signal for this class. Showing the original "
                "image; this does not mean the leaf was ignored."
            )

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

    # Comparative benchmark metrics
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

    # Thesis architecture comparison
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


# ==============================================================================
# BATCH VALIDATION & TESTING VIEW
# ==============================================================================
def render_batch_testing_view(v1_model, v2_model):
    with st.sidebar:
        st.subheader("Batch Testing Rules")
        st.markdown(
            """
            - **Minimum Upload Constraint:**
              Upload at least **10 images** (10, 11, 12, or more).
            - **Labeling Modes:**
              - **Mode 1 (Uniform):** One class applies to all uploaded leaves.
              - **Mode 2 (Mix-and-Match):** Per-image Ground Truth assignment.
            - **Dual-Model Inference:**
              GourNet V1 (683k params) and GourNet V2 (98k params) evaluated concurrently.
            - **Strict Metrics Display:**
              Results displayed strictly as **percentages** (1 decimal place). No raw fractional counts.
            """
        )
        st.markdown("---")
        st.subheader("Model Architecture Specs")
        st.markdown(
            """
            - **GourNet V1 (Baseline)**: 683,656 Parameters
            - **GourNet V2 (Enhanced)**: 98,280 Parameters (-85.6%)
            """
        )

    st.title("Batch Validation & Testing")
    st.caption(
        "Evaluate and benchmark **GourNet V1 (Baseline)** vs **Enhanced GourNet V2** "
        "across a multi-specimen batch. Enforces a strict minimum of at least 10 images."
    )

    # --------------------------------------------------------------------------
    # 1. Upload & Validation Rules
    # --------------------------------------------------------------------------
    st.subheader("1. Upload Batch Specimens")
    uploaded_files = st.file_uploader(
        "Upload leaf images (PNG, JPG, JPEG, WEBP) — Minimum 10 images required:",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Select at least 10 leaf image files.",
    )

    num_uploaded = len(uploaded_files) if uploaded_files else 0

    if num_uploaded < 10:
        st.warning(f"Please upload at least 10 images to run batch validation (Uploaded: {num_uploaded}/10).")
        run_disabled = True
    else:
        st.success(f"{num_uploaded} images uploaded. Batch requirement met (>= 10 images).")
        run_disabled = False

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 2. Labeling Modes
    # --------------------------------------------------------------------------
    st.subheader("2. Ground Truth Labeling Mode")
    label_mode = st.radio(
        "Select labeling workflow:",
        ("Mode 1: Uniform Class Batch", "Mode 2: Mix-and-Match (Per-Image Labeling)"),
        help="Mode 1 applies one class to all uploaded images. Mode 2 allows setting ground truth individually.",
    )

    ground_truth_map = {}

    if label_mode == "Mode 1: Uniform Class Batch":
        st.caption("All uploaded images in this batch will be evaluated against this Ground Truth label.")
        uniform_class = st.selectbox(
            "Select Ground Truth class for all uploaded images:",
            model_utils.CLASS_NAMES,
            index=model_utils.CLASS_NAMES.index("Healthy") if "Healthy" in model_utils.CLASS_NAMES else 0,
        )
        if uploaded_files:
            for f in uploaded_files:
                ground_truth_map[f.name] = uniform_class

    else:
        st.caption("Assign individual Ground Truth labels for each uploaded image using the table below:")
        if uploaded_files:
            if "per_image_labels" not in st.session_state:
                st.session_state["per_image_labels"] = {}

            # Quick batch preset helper
            col_preset1, col_preset2 = st.columns([2, 1])
            with col_preset1:
                preset_class = st.selectbox(
                    "Quick-set all rows to:",
                    model_utils.CLASS_NAMES,
                    key="quick_preset_select",
                )
            with col_preset2:
                st.write("")
                st.write("")
                if st.button("Apply to All Rows"):
                    for f in uploaded_files:
                        st.session_state["per_image_labels"][f.name] = preset_class
                    st.rerun()

            editor_data = []
            for f in uploaded_files:
                current_label = st.session_state["per_image_labels"].get(f.name, model_utils.CLASS_NAMES[0])
                editor_data.append({
                    "Filename": f.name,
                    "Ground Truth": current_label,
                })
            editor_df = pd.DataFrame(editor_data)

            edited_df = st.data_editor(
                editor_df,
                column_config={
                    "Filename": st.column_config.TextColumn("Filename", disabled=True),
                    "Ground Truth": st.column_config.SelectboxColumn(
                        "Ground Truth",
                        options=model_utils.CLASS_NAMES,
                        required=True,
                        help="Select ground truth class for this leaf specimen",
                    ),
                },
                hide_index=True,
                use_container_width=True,
                key="batch_data_editor",
            )

            for _, row in edited_df.iterrows():
                ground_truth_map[row["Filename"]] = row["Ground Truth"]
                st.session_state["per_image_labels"][row["Filename"]] = row["Ground Truth"]

            with st.expander("Inspect Uploaded Specimen Thumbnails", expanded=False):
                col_count = min(4, max(1, num_uploaded))
                cols = st.columns(col_count)
                for idx, f in enumerate(uploaded_files):
                    with cols[idx % col_count]:
                        try:
                            thumb_img = Image.open(f)
                            st.image(thumb_img, caption=f.name, use_container_width=True)
                        except Exception:
                            st.caption(f.name)
        else:
            st.info("Upload at least 10 images above to configure per-image ground truth labels.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. Execution Section
    # --------------------------------------------------------------------------
    st.subheader("3. Execution & Batch Inference")
    col_run, col_status = st.columns([1, 2])
    with col_run:
        run_batch = st.button(
            "Run Batch Validation",
            type="primary",
            disabled=run_disabled,
            use_container_width=True,
        )
    with col_status:
        if run_disabled:
            st.caption(f"Action blocked: Please upload at least 10 images (Current: {num_uploaded}/10).")
        else:
            st.caption(f"Ready: {num_uploaded} specimens queued for comparative evaluation.")

    if run_batch and uploaded_files and len(uploaded_files) >= 10:
        items = []
        errors = []
        for f in uploaded_files:
            try:
                # Seek to beginning of file in case it was read for thumbnails
                f.seek(0)
                img = Image.open(f)
                gt = ground_truth_map.get(f.name, model_utils.CLASS_NAMES[0])
                items.append({
                    "filename": f.name,
                    "image": img,
                    "ground_truth": gt,
                })
            except Exception as e:
                errors.append(f"{f.name}: {e}")

        if errors:
            st.error("Errors reading some uploaded images:\n" + "\n".join(errors))

        if len(items) >= 10:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(current, total):
                progress_bar.progress(float(current) / float(total))
                status_text.text(f"Evaluating image {current} of {total}...")

            with st.spinner("Executing dual-model comparative batch inference..."):
                results = model_utils.run_batch_evaluation(
                    v1_model,
                    v2_model,
                    items,
                    progress_callback=update_progress,
                )

            progress_bar.empty()
            status_text.empty()
            st.session_state["batch_results"] = results
            st.toast("Batch evaluation finished successfully!", icon="✅")

    # --------------------------------------------------------------------------
    # 4. Results & Metrics (PERCENTAGES STRICTLY ENFORCED)
    # --------------------------------------------------------------------------
    if "batch_results" in st.session_state and st.session_state["batch_results"] is not None:
        results = st.session_state["batch_results"]
        summary = results["summary"]
        v1_s = summary["v1"]
        v2_s = summary["v2"]

        st.markdown("---")
        st.subheader("4. Comparative Performance Dashboard")
        st.caption(
            "All batch performance metrics are presented strictly as percentages formatted to 1 decimal place. "
            "No raw fractional counts are displayed."
        )

        col_v1, col_v2 = st.columns(2, gap="large")

        # Column 1: GourNet V1
        with col_v1:
            st.markdown(
                f"""
                <div class="result-card" style="border-left: 4px solid #3b82f6;">
                    <div class="diagnosis-label">Model Architecture</div>
                    <div class="diagnosis-name" style="font-size: 1.35rem;">GourNet V1 (Baseline)</div>
                    <div style="color: var(--secondary-text-color, #94a3b8); font-size: 0.82rem; margin-bottom: 0.5rem;">
                        Alam et al. (2026) Reproduction • 683,656 Parameters
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    label="Batch Match Rate",
                    value=f"{v1_s['match_rate']:.1f}%",
                )
            with m2:
                st.metric(
                    label="Average Confidence",
                    value=f"{v1_s['avg_confidence']:.1f}%",
                )
            with m3:
                st.metric(
                    label="Misclassification Rate",
                    value=f"{v1_s['misclassification_rate']:.1f}%",
                )

        # Column 2: GourNet V2
        with col_v2:
            st.markdown(
                f"""
                <div class="result-card" style="border-left: 4px solid #10b981;">
                    <div class="diagnosis-label">Model Architecture</div>
                    <div class="diagnosis-name" style="font-size: 1.35rem;">Enhanced GourNet V2</div>
                    <div style="color: var(--secondary-text-color, #94a3b8); font-size: 0.82rem; margin-bottom: 0.5rem;">
                        Thesis Improvement • 98,280 Parameters (-85.6%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            with m1:
                delta_match = v2_s["match_rate"] - v1_s["match_rate"]
                st.metric(
                    label="Batch Match Rate",
                    value=f"{v2_s['match_rate']:.1f}%",
                    delta=f"{delta_match:+.1f}%",
                )
            with m2:
                delta_conf = v2_s["avg_confidence"] - v1_s["avg_confidence"]
                st.metric(
                    label="Average Confidence",
                    value=f"{v2_s['avg_confidence']:.1f}%",
                    delta=f"{delta_conf:+.1f}%",
                )
            with m3:
                delta_misc = v2_s["misclassification_rate"] - v1_s["misclassification_rate"]
                st.metric(
                    label="Misclassification Rate",
                    value=f"{v2_s['misclassification_rate']:.1f}%",
                    delta=f"{delta_misc:+.1f}%",
                    delta_color="inverse",
                )

        # Comparative Highlights Row
        st.markdown("")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric(
                label="Match Rate Advantage (V2 vs V1)",
                value=f"{v2_s['match_rate'] - v1_s['match_rate']:+.1f}%",
                delta="V2 Leading" if v2_s["match_rate"] >= v1_s["match_rate"] else "V1 Leading",
            )
        with col_c2:
            st.metric(
                label="Confidence Difference (V2 - V1)",
                value=f"{v2_s['avg_confidence'] - v1_s['avg_confidence']:+.1f}%",
            )
        with col_c3:
            st.metric(
                label="Parameter Reduction Efficiency",
                value="-85.6%",
                delta="98,280 vs 683,656 weights",
                delta_color="normal",
            )

        # ----------------------------------------------------------------------
        # 5. Detailed Breakdown Table
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader("5. Detailed Specimen Breakdown")

        details_list = results["details"]
        table_rows = []
        for r in details_list:
            table_rows.append({
                "Filename": r["Filename"],
                "Ground Truth": r["Ground Truth"],
                "V1 Prediction": r["V1 Prediction"],
                "V1 Confidence": f"{r['V1 Confidence (%)']:.1f}%",
                "V1 Match Status": r["V1 Match"],
                "V2 Prediction": r["V2 Prediction"],
                "V2 Confidence": f"{r['V2 Confidence (%)']:.1f}%",
                "V2 Match Status": r["V2 Match"],
            })
        df_display = pd.DataFrame(table_rows)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------------------------
        # 6. CSV Download
        # ----------------------------------------------------------------------
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Batch Results (CSV)",
            data=csv_data,
            file_name="mango_leaf_batch_validation_results.csv",
            mime="text/csv",
        )


# ==============================================================================
# MAIN ENTRYPOINT / ROUTER
# ==============================================================================
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio(
        "Select Operational Mode:",
        ("Single Specimen Analysis", "Batch Validation & Testing"),
        index=0,
    )
    st.markdown("---")

if app_mode == "Single Specimen Analysis":
    render_single_specimen_view(v1_model, v2_model)
else:
    render_batch_testing_view(v1_model, v2_model)
