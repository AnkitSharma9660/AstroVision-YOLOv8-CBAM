import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import pandas as pd
import json
import time
import io

# ── 1. PAGE CONFIGURATION & METADATA ──────────────────────────────────────────
st.set_page_config(
    page_title="AstroVision | Deep Space AI Detection",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. COSMIC DARK THEME & GLASSMORPHISM CSS ─────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Global Page Styling */
    .stApp {
        background: radial-gradient(circle at 50% 20%, #111827 0%, #0B0F19 50%, #030712 100%);
        color: #F3F4F6;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Cosmic Glow Header */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Metric Cards Custom Styling */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Custom Class Badge Pills */
    .badge-comet {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-galaxy {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(124, 58, 237, 0.2) 100%);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-cluster {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-nebula {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.2) 0%, rgba(225, 29, 72, 0.2) 100%);
        color: #F472B6;
        border: 1px solid rgba(244, 114, 182, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(129, 140, 248, 0.15) 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F17 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── 3. ASTRONOMICAL CLASS MAPPING & METADATA ──────────────────────────────────
CLASS_INFO = {
    0: {
        "name": "Comet",
        "icon": "☄️",
        "badge_class": "badge-comet",
        "color_hex": "#38BDF8",
        "rgb": (56, 189, 248),
        "desc": "An icy small Solar System body that heats up and releases gases when passing close to the Sun, forming a glowing coma and distinct dust/ion tail."
    },
    1: {
        "name": "Galaxy",
        "icon": "🌌",
        "badge_class": "badge-galaxy",
        "color_hex": "#C084FC",
        "rgb": (192, 132, 252),
        "desc": "A vast gravitationally bound system of stars, stellar remnants, interstellar gas, dust, and dark matter (e.g. Spiral, Elliptical, Irregular)."
    },
    2: {
        "name": "Globular Cluster",
        "icon": "🌟",
        "badge_class": "badge-cluster",
        "color_hex": "#FBBF24",
        "rgb": (251, 191, 36),
        "desc": "A densely packed spherical collection of hundreds of thousands of ancient stars tightly bound by gravity, orbiting a galactic core."
    },
    3: {
        "name": "Nebula",
        "icon": "🌫️",
        "badge_class": "badge-nebula",
        "color_hex": "#F472B6",
        "rgb": (244, 114, 182),
        "desc": "A massive interstellar cloud of dust, hydrogen, helium, and ionized gases—often serving as active star nurseries or supernova remnants."
    }
}

# ── 4. MODEL LOADING WITH BACKEND FALLBACK ────────────────────────────────────
@st.cache_resource
def load_astro_model():
    base_dir = Path(__file__).parent
    onnx_path = base_dir / "best_fixed.onnx"
    pt_path = base_dir / "best_fixed.pt"

    if onnx_path.exists():
        try:
            model = YOLO(str(onnx_path))
            return model, "ONNX Runtime (Optimized)"
        except Exception as e:
            st.warning(f"Could not load ONNX model directly: {e}. Falling back to PyTorch weights.")

    if pt_path.exists():
        try:
            model = YOLO(str(pt_path))
            return model, "PyTorch (.pt)"
        except Exception as e:
            st.error(f"Error loading PyTorch model: {e}")
            st.stop()

    st.error("Model file not found! Please ensure `best_fixed.onnx` or `best_fixed.pt` exists in the app folder.")
    st.stop()

model, model_backend = load_astro_model()

# ── 5. DEMO SYNTHETIC DEEP FIELD GENERATOR ────────────────────────────────────
def generate_synthetic_cosmic_field():
    """Generates a demo celestial deep-field image for instant testing."""
    width, height = 800, 600
    img = Image.new("RGB", (width, height), (5, 8, 20))
    draw = ImageDraw.Draw(img)

    # Random background stars
    np.random.seed(42)
    for _ in range(300):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        brightness = np.random.randint(100, 255)
        radius = int(np.random.choice([1, 1, 1, 2]))
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=(brightness, brightness, brightness))

    # Add glowing simulated nebula region
    nebula_overlay = Image.new("RGB", (width, height), (0, 0, 0))
    neb_draw = ImageDraw.Draw(nebula_overlay)
    neb_draw.ellipse([150, 100, 450, 400], fill=(120, 30, 90))
    neb_draw.ellipse([450, 250, 700, 500], fill=(20, 80, 140))
    nebula_overlay = nebula_overlay.filter(ImageFilter.GaussianBlur(radius=50))
    
    img = Image.blend(img, nebula_overlay, alpha=0.6)
    return img

# ── 6. SIDEBAR CONTROLS & SETTINGS ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔭")
    st.title("AstroVision Controls")
    st.caption(f"Engine Backend: `{model_backend}`")
    st.divider()

    st.subheader("⚙️ Detection Thresholds")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05,
                               help="Lower threshold detects faint cosmic objects; higher threshold reduces false positives.")
    iou_threshold = st.slider("IoU NMS Threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05,
                             help="Overlap threshold for Non-Maximum Suppression.")

    st.divider()
    st.subheader("🎯 Target Celestial Filter")
    selected_classes = st.multiselect(
        "Display Object Types",
        options=[v["name"] for v in CLASS_INFO.values()],
        default=[v["name"] for v in CLASS_INFO.values()],
        help="Select which astronomical object types to show in detection results."
    )

    st.divider()
    st.subheader("🎨 Custom Overlay Style")
    box_line_width = st.slider("Bounding Box Thickness", 1, 5, 2)
    show_labels = st.checkbox("Show Class Labels & Confidence", value=True)

    st.divider()
    st.markdown("### ℹ️ About Cosmica Dataset")
    st.info("""
    **AstroVision** uses a custom YOLOv8 model trained with **CBAM Attention Layers** on the **Cosmica Astronomical Dataset**, capable of detecting comets, galaxies, globular clusters, and nebulae in optical and astronomical surveys.
    """)

# ── 7. MAIN HEADER ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔭 AstroVision</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Deep Space Object Detection powered by YOLOv8 + CBAM Attention Networks</div>', unsafe_allow_html=True)

# ── 8. IMAGE INPUT SECTION ─────────────────────────────────────────────────────
input_col1, input_col2 = st.columns([2, 1])

with input_col1:
    uploaded_file = st.file_uploader(
        "Upload Deep Space Astronomical Image (JPG, PNG, TIFF)",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff"]
    )

with input_col2:
    st.markdown("#### Quick Test Options")
    use_sample = st.button("✨ Load Sample Cosmic Field Image", use_container_width=True)

image = None
image_source_label = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_source_label = f"Uploaded File: `{uploaded_file.name}`"
elif use_sample:
    image = generate_synthetic_cosmic_field()
    image_source_label = "Sample Cosmic Field Preset"

# ── 9. PREDICTION & DISPLAY TABS ───────────────────────────────────────────────
if image is not None:
    img_array = np.array(image)
    img_height, img_width = img_array.shape[:2]

    # Perform Model Inference
    start_time = time.time()
    with st.spinner("🚀 Running Deep Space AI Inference..."):
        results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold)
    proc_time_ms = (time.time() - start_time) * 1000

    raw_boxes = results[0].boxes
    detections = []
    
    selected_class_ids = [k for k, v in CLASS_INFO.items() if v["name"] in selected_classes]

    if raw_boxes is not None and len(raw_boxes) > 0:
        for b in raw_boxes:
            cid = int(b.cls[0])
            conf = float(b.conf[0])
            xyxy = b.xyxy[0].cpu().numpy().tolist() # [xmin, ymin, xmax, ymax]
            
            if cid in selected_class_ids:
                detections.append({
                    "class_id": cid,
                    "class_name": CLASS_INFO[cid]["name"],
                    "icon": CLASS_INFO[cid]["icon"],
                    "confidence": conf,
                    "box": xyxy,
                    "color": CLASS_INFO[cid]["rgb"]
                })

    # Custom Annotation Rendering
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        color = det["color"]
        
        # Bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=box_line_width)
        
        # Label pill
        if show_labels:
            label_text = f"{det['icon']} {det['class_name']} {det['confidence']*100:.1f}%"
            # Background pill rectangle for text
            text_size = len(label_text) * 7.5
            draw.rectangle([xmin, max(0, ymin - 22), xmin + text_size + 10, ymin], fill=color)
            draw.text((xmin + 5, max(2, ymin - 18)), label_text, fill=(255, 255, 255))

    # Create Navigation Tabs
    tab_dash, tab_gallery, tab_guide, tab_export = st.tabs([
        "🔭 Detection Dashboard",
        "🔍 Object Crop Gallery",
        "📘 Astronomical Field Guide",
        "💾 Export & Reports"
    ])

    # ── TAB 1: DETECTION DASHBOARD ─────────────────────────────────────────────
    with tab_dash:
        st.caption(image_source_label)

        # Summary Metrics Bar
        m1, m2, m3, m4 = st.columns(4)
        total_det = len(detections)
        unique_classes = len(set(d["class_name"] for d in detections))
        max_conf = max([d["confidence"] for d in detections]) * 100 if detections else 0.0

        m1.metric("Total Objects Detected", f"{total_det}")
        m2.metric("Unique Celestial Types", f"{unique_classes} / {len(CLASS_INFO)}")
        m3.metric("Highest Confidence", f"{max_conf:.1f}%" if detections else "N/A")
        m4.metric("Inference Latency", f"{proc_time_ms:.1f} ms")

        st.divider()

        # Side by Side Image Comparison
        col_orig, col_annot = st.columns(2)

        with col_orig:
            st.subheader("📷 Original Deep Space Image")
            st.image(image, use_container_width=True)

        with col_annot:
            st.subheader("✨ AI Detection Canvas")
            st.image(annotated_img, use_container_width=True)

        st.divider()
        st.subheader("📊 Detected Objects Breakdown")

        if detections:
            # Class Breakdown Summary Badges
            counts = Counter(d["class_id"] for d in detections)
            badge_html = "<div style='margin-bottom: 15px; display: flex; gap: 10px; flex-wrap: wrap;'>"
            for cid, count in counts.items():
                info = CLASS_INFO[cid]
                badge_html += f'<span class="{info["badge_class"]}">{info["icon"]} {info["name"]}: {count}</span>'
            badge_html += "</div>"
            st.markdown(badge_html, unsafe_allow_html=True)

            # Data Table
            table_data = []
            for idx, d in enumerate(detections, 1):
                box = d["box"]
                width_px = box[2] - box[0]
                height_px = box[3] - box[1]
                area_pct = (width_px * height_px) / (img_width * img_height) * 100

                table_data.append({
                    "Index": idx,
                    "Type": f"{d['icon']} {d['class_name']}",
                    "Confidence": f"{d['confidence']*100:.2f}%",
                    "Bounding Box [Xmin, Ymin, Xmax, Ymax]": f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]",
                    "Dimensions (Px)": f"{int(width_px)} × {int(height_px)}",
                    "Area Share": f"{area_pct:.2f}%"
                })

            df_results = pd.DataFrame(table_data)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
        else:
            st.info("🌌 No celestial objects detected with the selected thresholds. Try lowering the confidence slider in the sidebar.")

    # ── TAB 2: OBJECT CROP GALLERY ─────────────────────────────────────────────
    with tab_gallery:
        st.subheader("🔍 Cropped Celestial Object Gallery")
        st.write("Zoomed-in perspective of individual detected astronomical entities:")

        if detections:
            # Display grid of crops
            cols_per_row = 3
            for i in range(0, len(detections), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(detections):
                        det = detections[i + j]
                        box = [int(v) for v in det["box"]]
                        
                        # Pad box slightly
                        pad = 10
                        crop_box = [
                            max(0, box[0] - pad),
                            max(0, box[1] - pad),
                            min(img_width, box[2] + pad),
                            min(img_height, box[3] + pad)
                        ]
                        
                        cropped_img = image.crop(crop_box)

                        with row_cols[j]:
                            st.markdown(f'<span class="{CLASS_INFO[det["class_id"]]["badge_class"]}">{det["icon"]} #{i+j+1} {det["class_name"]}</span>', unsafe_allow_html=True)
                            st.image(cropped_img, use_container_width=True)
                            st.caption(f"**Confidence:** {det['confidence']*100:.2f}% | **BBox:** {box}")
        else:
            st.info("No object crops available.")

    # ── TAB 3: ASTRONOMICAL FIELD GUIDE ───────────────────────────────────────
    with tab_guide:
        st.subheader("📘 Deep Space Astronomical Guide")
        st.write("Learn more about the celestial structures recognized by AstroVision's YOLOv8-CBAM model:")

        guide_col1, guide_col2 = st.columns(2)

        for cid, info in CLASS_INFO.items():
            col = guide_col1 if cid % 2 == 0 else guide_col2
            with col:
                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="color: {info['color_hex']}; margin-top:0;">{info['icon']} {info['name']}</h3>
                    <p style="color: #D1D5DB; font-size: 0.95rem;">{info['desc']}</p>
                    <hr style="border-color: rgba(255,255,255,0.1);">
                    <small style="color: #9CA3AF;">Model Target Class Index: <code>{cid}</code></small>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 4: EXPORT & REPORTS ───────────────────────────────────────────────
    with tab_export:
        st.subheader("💾 Download & Export Detection Results")
        st.write("Export processed images, structured datasets, and metadata for scientific workflow integration:")

        ex_col1, ex_col2, ex_col3 = st.columns(3)

        # 1. Annotated Image Export
        buf_img = io.BytesIO()
        annotated_img.save(buf_img, format="PNG")
        ex_col1.download_button(
            label="🖼️ Download Annotated Image (PNG)",
            data=buf_img.getvalue(),
            file_name="astrovision_annotated.png",
            mime="image/png",
            use_container_width=True
        )

        # 2. CSV Report Export
        if detections:
            df_csv = pd.DataFrame([{
                "object_index": idx + 1,
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "confidence": round(d["confidence"], 4),
                "xmin": int(d["box"][0]),
                "ymin": int(d["box"][1]),
                "xmax": int(d["box"][2]),
                "ymax": int(d["box"][3])
            } for idx, d in enumerate(detections)])
            
            csv_bytes = df_csv.to_csv(index=False).encode('utf-8')
            ex_col2.download_button(
                label="📄 Export Detections (CSV)",
                data=csv_bytes,
                file_name="astrovision_detections.csv",
                mime="text/csv",
                use_container_width=True
            )

        # 3. JSON Summary Metadata
        json_report = {
            "application": "AstroVision",
            "model_backend": model_backend,
            "image_dimensions": {"width": img_width, "height": img_height},
            "parameters": {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            },
            "summary": {
                "total_detections": len(detections),
                "inference_time_ms": round(proc_time_ms, 2)
            },
            "detections": [{
                "id": idx + 1,
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "box": [round(x, 1) for x in d["box"]]
            } for idx, d in enumerate(detections)]
        }
        
        json_bytes = json.dumps(json_report, indent=2).encode('utf-8')
        ex_col3.download_button(
            label="📊 Export Full Report (JSON)",
            data=json_bytes,
            file_name="astrovision_report.json",
            mime="application/json",
            use_container_width=True
        )

else:
    # Initial Splash State when no image is loaded
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 40px;">
        <h2 style="color: #38BDF8;">🌌 Welcome to AstroVision</h2>
        <p style="color: #9CA3AF; font-size: 1.1rem; max-width: 600px; margin: 0 auto 20px auto;">
            Upload an optical sky survey or deep space photograph above, or click <b>Load Sample Cosmic Field Image</b> to explore real-time AI celestial object detection.
        </p>
    </div>
    """, unsafe_allow_html=True)