import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import time
import os

# ==========================================================
#  PROJECT: Chicken Disease Detection using CNN (EfficientNetB3)
#  DEPLOYMENT: single-file — everything (model + class map + img size)
#  is embedded inside chicken_disease_pipeline_selfcontained.pkl.
#  No separate .h5 / .keras file is needed on the server.
# ==========================================================

# ══════════════════════════════════════════════════════════════
#  PIPELINE CLASS
#  Imported (not redefined here) so pickle resolves it at the same
#  module path ("pipeline_class.ChickenDiseasePipelineSelfContained")
#  that was used when the .pkl was created. pipeline_class.py MUST
#  sit in this same folder alongside app.py.
# ══════════════════════════════════════════════════════════════
from pipeline_class import ChickenDiseasePipelineSelfContained  # noqa: F401  (needed for pickle.load)


# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The ONLY file this app depends on. Everything else (model weights,
# architecture, class names, image size) is embedded inside it.
PIPELINE_PATH = os.path.join(BASE_DIR, "chicken_disease_pipeline_selfcontained.pkl")

# Common info/action text for the well-known chicken-disease classes.
# If your dataset's folder names differ, add/edit entries here - anything
# not found below falls back to a generic message so the app never crashes.
CLASS_INFO = {
    "Healthy":
        "The bird shows no visible signs of disease in this sample. "
        "Droppings/imagery appear consistent with a normally functioning "
        "digestive and immune system.",
    "Coccidiosis":
        "Coccidiosis is a parasitic intestinal disease caused by Eimeria "
        "protozoa. It damages the intestinal lining, leading to bloody "
        "droppings, poor growth, and dehydration if untreated.",
    "New Castle Disease":
        "Newcastle Disease is a highly contagious viral infection affecting "
        "the respiratory, nervous, and digestive systems of poultry. It "
        "spreads rapidly and can cause high mortality in unvaccinated flocks.",
    "Salmonella":
        "Salmonellosis is a bacterial infection that affects the intestinal "
        "tract, causing diarrhea, weakness, and reduced egg production. It "
        "can also pose a food-safety risk to humans handling infected birds.",
}

CLASS_ACTION = {
    "Healthy":
        "✅ No treatment required. Continue routine flock monitoring, "
        "maintain clean water/feed, and keep up the regular vaccination "
        "schedule.",
    "Coccidiosis":
        "🔧 Isolate affected birds. Start an anticoccidial medication "
        "(e.g. amprolium) as advised by a vet. Improve litter dryness and "
        "sanitation to break the parasite's life cycle.",
    "New Castle Disease":
        "🔧 Quarantine the flock immediately and contact a veterinarian. "
        "Newcastle has no cure - focus on biosecurity, culling as required "
        "by local regulations, and vaccinating unaffected birds.",
    "Salmonella":
        "🔧 Isolate affected birds and consult a vet for antibiotic "
        "treatment. Disinfect housing/equipment thoroughly and practice "
        "strict hand hygiene, since this disease can spread to humans.",
}

CLASS_SEVERITY = {
    "Healthy":             ("NONE",   "#38A169"),
    "Coccidiosis":         ("HIGH",   "#E53E3E"),
    "New Castle Disease":  ("HIGH",   "#E53E3E"),
    "Salmonella":          ("MEDIUM", "#DD6B20"),
}

CLASS_ICONS = {
    "Healthy":             "✔",
    "Coccidiosis":         "🦠",
    "New Castle Disease":  "⚠",
    "Salmonella":          "☣",
}

# Fallback values used for any class name not listed above,
# so the app still works even if your dataset has different labels.
DEFAULT_INFO     = "No detailed description is available yet for this class."
DEFAULT_ACTION   = "🔧 Consult a veterinarian to confirm the diagnosis and decide on treatment."
DEFAULT_SEVERITY = ("UNKNOWN", "#64748B")
DEFAULT_ICON     = "❓"


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Chicken Disease Diagnosis | CNN",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #0D1117 !important;
    color: #E2E8F0 !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid #21262D !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.25rem 1rem !important; }

.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid #21262D;
    margin-bottom: 1.1rem;
}
.sb-logo-icon {
    width: 36px; height: 36px; border-radius: 9px;
    background: linear-gradient(135deg, #B45309, #F59E0B);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    box-shadow: 0 0 14px rgba(245,158,11,0.35);
}
.sb-logo-text { font-size: 0.8rem; font-weight: 700; color: #F1F5F9; line-height: 1.2; }
.sb-logo-sub  { font-size: 0.68rem; color: #64748B; font-weight: 400; }

.sb-nav-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4B5563; margin: 1rem 0 0.45rem;
}

.sb-fault-item {
    display: flex; align-items: center; gap: 10px;
    padding: 0.5rem 0.7rem; border-radius: 8px;
    background: #1C2333; margin-bottom: 5px;
    border: 1px solid #21262D;
    transition: border-color 0.15s;
}
.sb-fault-item:hover { border-color: #F59E0B; }
.sb-fault-icon { font-size: 1.15rem; width: 26px; text-align: center; flex-shrink: 0; }
.sb-fault-name { font-size: 0.79rem; font-weight: 600; color: #E2E8F0; line-height: 1.2; }
.sb-fault-sev  { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.05em; }

.sb-footer {
    margin-top: 1.25rem; padding-top: 0.85rem;
    border-top: 1px solid #21262D;
    font-size: 0.68rem; color: #4B5563; text-align: center; line-height: 1.7;
}

/* ══ TOPBAR ══ */
.topbar {
    background: linear-gradient(135deg, #0F172A 0%, #5F3A1E 60%, #B45309 100%);
    border: 1px solid #F59E0B33;
    border-radius: 14px;
    padding: 1.75rem 1.75rem;
    margin-bottom: 1.25rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 0 30px rgba(245,158,11,0.15);
    gap: 1rem; position: relative; overflow: hidden;
    color: white;
}
.topbar::before {
    content: '🐔'; position: absolute; right: 1.75rem; top: 50%;
    transform: translateY(-50%); font-size: 7rem; opacity: 0.08;
    line-height: 1; pointer-events: none;
}
.topbar-left h1 {
    font-size: 1.45rem; font-weight: 800; color: #FFFFFF;
    margin: 0 0 0.2rem; letter-spacing: -0.4px;
}
.topbar-left p {
    font-size: 0.855rem; color: rgba(255,255,255,0.72);
    margin: 0; font-weight: 400; line-height: 1.5;
}
.topbar-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.topbar-badge {
    padding: 0.28rem 0.8rem; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600;
    background: rgba(255,255,255,0.12); color: #FFFFFF;
    border: 1px solid rgba(255,255,255,0.22);
}
.topbar-badge.green {
    background: rgba(74,222,128,0.18); color: #4ADE80;
    border-color: rgba(74,222,128,0.35);
}

/* ══ STATUS BAR ══ */
.status-bar {
    background: #14532D22; border: 1px solid #16A34A33; border-radius: 9px;
    padding: 0.55rem 1.1rem; margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 0.85rem;
    font-size: 0.78rem; color: #4ADE80; font-weight: 500; flex-wrap: wrap;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22C55E; flex-shrink: 0;
    box-shadow: 0 0 6px #22C55E;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.status-divider { color: #16A34A66; }

/* ══ CARDS ══ */
.card {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 12px; padding: 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3); margin-bottom: 1rem;
}
.card-header {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #64748B;
    margin-bottom: 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid #21262D;
}
.card-header-icon { font-size: 0.9rem; }
.card-header-lg {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.92rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #94A3B8;
    margin-bottom: 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid #21262D;
}

/* ══ CHIP ROW ══ */
.chip-row { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 1rem; }
.chip {
    display: flex; align-items: center; gap: 5px;
    background: #1C2333; border: 1px solid #21262D;
    border-radius: 7px; padding: 0.3rem 0.75rem;
    font-size: 0.75rem; font-weight: 500; color: #94A3B8;
}
.chip-icon { font-size: 0.8rem; }

/* ══ RESULT CARD ══ */
.result-card {
    border-radius: 12px; padding: 1.5rem 1.25rem 1.25rem;
    text-align: center; border: 1.5px solid;
    margin-bottom: 0; position: relative; overflow: hidden;
    display: flex; flex-direction: column; align-items: center;
    height: 100%;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: currentColor;
}
.result-icon   { font-size: 2.5rem; margin-bottom: 0.5rem; display: block; line-height: 1; }
.result-eyebrow {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem;
}
.result-class  { font-size: 1.35rem; font-weight: 700; margin-bottom: 0.6rem; line-height: 1.2; }
.result-conf   { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.result-conf-sub { font-size: 0.7rem; color: #64748B; margin-top: 0.2rem; margin-bottom: 0.75rem; }
.sev-pill {
    display: inline-block; padding: 0.22rem 0.9rem; border-radius: 20px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; border: 1.5px solid;
}

/* ══ INFO & ACTION BOX ══ */
.info-box {
    background: #5F3A1E22; border-left: 3px solid #F59E0B;
    border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem; font-size: 0.855rem;
    color: #CBD5E1; line-height: 1.75;
}
.action-box {
    background: #14532D22; border-left: 3px solid #22C55E;
    border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem;
    font-size: 0.855rem; color: #CBD5E1; line-height: 1.75;
}
.box-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.35rem; display: block;
}
.box-label.orange { color: #F59E0B; }
.box-label.green  { color: #22C55E; }

/* ══ FAULT GRID ══ */
.fault-grid-wrapper {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px; align-items: stretch;
}
.fault-grid-item {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 11px; padding: 1.1rem 0.85rem; text-align: center;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100%; transition: border-color 0.15s, transform 0.15s;
}
.fault-grid-item:hover { border-color: #F59E0B; transform: translateY(-2px); }
.fault-grid-icon-box {
    width: 54px; height: 54px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.65rem; line-height: 1; font-weight: 700;
    margin: 0 auto 0.65rem;
    background: #1C2333; border: 1px solid #30374A;
}
.fault-grid-name { font-size: 0.79rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.35rem; }
.fault-grid-sev {
    display: inline-block; padding: 0.18rem 0.6rem;
    border-radius: 12px; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
}

/* ══ STEP CARDS ══ */
.step-grid-wrapper {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 10px; align-items: stretch; margin-bottom: 1rem;
}
.step-card {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 11px; padding: 1.4rem 1.1rem; text-align: center;
    display: flex; flex-direction: column; align-items: center; justify-content: flex-start;
    height: 100%;
}
.step-num {
    width: 38px; height: 38px; border-radius: 11px;
    background: linear-gradient(135deg, #B45309, #F59E0B);
    color: white; font-size: 0.95rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.85rem;
    box-shadow: 0 0 12px rgba(245,158,11,0.35);
}
.step-title { font-size: 0.88rem; font-weight: 700; color: #F1F5F9; margin-bottom: 0.3rem; }
.step-desc  { font-size: 0.76rem; color: #64748B; line-height: 1.55; }

/* ══ EXPANDER ══ */
[data-testid="stExpander"] {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #64748B !important; font-size: 0.82rem !important; }
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0D1117 !important; border-radius: 7px !important;
    font-size: 0.78rem !important; color: #94A3B8 !important;
    border: 1px solid #21262D !important;
}

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploader"] {
    background: #161B22 !important;
    border: 1.5px dashed #21262D !important;
    border-radius: 10px !important; padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: #F59E0B !important; }
.stSpinner > div { border-top-color: #F59E0B !important; }

/* ══ FOOTER ══ */
.footer {
    text-align: center; padding: 1.25rem; color: #374151; font-size: 0.73rem;
    border-top: 1px solid #21262D; margin-top: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  VERSION-SAFE IMAGE HELPER
#  Newer Streamlit versions use `use_container_width`; older
#  versions only support `use_column_width`. This wrapper tries
#  the modern kwarg first and falls back automatically so the
#  app doesn't crash regardless of the Streamlit version installed.
# ══════════════════════════════════════════════════════════════
def show_image(data, caption=None):
    try:
        st.image(data, caption=caption, use_container_width=True)
    except TypeError:
        st.image(data, caption=caption, use_column_width=True)


# ══════════════════════════════════════════════════════════════
#  LOAD PIPELINE
#  The single self-contained pkl carries the model's raw bytes,
#  the class-index mapping, and the trained image size — nothing
#  else needs to exist on the server for this app to run.
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        st.error(
            f"Could not find '{os.path.basename(PIPELINE_PATH)}'. "
            "Make sure it was generated with convert_to_selfcontained.py "
            "and uploaded alongside app.py."
        )
        st.stop()

    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)

    pipeline._load_model()   # load the Keras model now so errors surface immediately
    return pipeline


def get_info(cls):     return CLASS_INFO.get(cls, DEFAULT_INFO)
def get_action(cls):   return CLASS_ACTION.get(cls, DEFAULT_ACTION)
def get_severity(cls): return CLASS_SEVERITY.get(cls, DEFAULT_SEVERITY)
def get_icon(cls):     return CLASS_ICONS.get(cls, DEFAULT_ICON)


# ══════════════════════════════════════════════════════════════
#  PREPROCESS
#  Matches the training pipeline: images are resized only, no
#  extra /255 rescaling, since EfficientNetB3 already includes
#  its own normalization layer internally.
# ══════════════════════════════════════════════════════════════
def preprocess(uploaded_file, img_h, img_w) -> np.ndarray:
    raw_bytes = uploaded_file.getvalue()
    img = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_h, img_w])
    img = tf.cast(img, tf.float32)          # keep 0-255 range, EfficientNet normalizes internally
    img = tf.expand_dims(img, axis=0)       # add batch dimension
    return img.numpy()


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.spinner("Initialising model..."):
    pipeline = load_pipeline()
    model = pipeline._load_model()
    class_names = [pipeline.idx_to_class[i] for i in sorted(pipeline.idx_to_class)]
    IMG_H, IMG_W = pipeline.img_size

with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>
        <div class='sb-logo-icon'>🐔</div>
        <div>
            <div class='sb-logo-text'>Chicken Disease Diagnosis</div>
            <div class='sb-logo-sub'>CNN · EfficientNetB3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-nav-label'>Disease Reference</div>", unsafe_allow_html=True)
    for cls in class_names:
        severity, sev_color = get_severity(cls)
        icon = get_icon(cls)
        st.markdown(
            "<div class='sb-fault-item'>"
            f"<span class='sb-fault-icon'>{icon}</span>"
            "<div>"
            f"<div class='sb-fault-name'>{cls}</div>"
            f"<div class='sb-fault-sev' style='color:{sev_color}'>{severity} SEVERITY</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("""
    <div class='sb-footer'>
        Chicken Disease Diagnosis System<br>
        CNN-based Image Classification
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN PAGE: DISEASE DIAGNOSIS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class='topbar'>
    <div class='topbar-left'>
        <h1>Chicken Disease Diagnosis System</h1>
        <p>CNN-based image analysis for early poultry disease detection and flock health monitoring.</p>
    </div>
    <div class='topbar-right'>
        <span class='topbar-badge green'>🟢 System Ready</span>
        <span class='topbar-badge'>🐔 EfficientNetB3</span>
    </div>
</div>
""", unsafe_allow_html=True)

# model, class_names, IMG_H, IMG_W were already loaded once above (before the sidebar)

st.markdown(
    "<div class='status-bar'>"
    "<div class='status-dot'></div>"
    "<span>Model loaded successfully</span>"
    "<span class='status-divider'>|</span>"
    f"<span>Input: {model.input_shape}</span>"
    "<span class='status-divider'>|</span>"
    f"<span>Parameters: {model.count_params():,}</span>"
    "<span class='status-divider'>|</span>"
    f"<span>Classes: {len(class_names)}</span>"
    "<span class='status-divider'>|</span>"
    "<span>✅ Ready for inference</span>"
    "</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div class='card'>
    <div class='card-header-lg'><span style='font-size:1rem'>📤</span>&nbsp; Upload Chicken Image</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a chicken/droppings image (PNG / JPG) to check for disease",
    type=["png", "jpg", "jpeg"],
    label_visibility="visible"
)

# ── EMPTY STATE ───────────────────────────────────────────
if uploaded is None:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='step-grid-wrapper'>
        <div class='step-card'>
            <div class='step-num'>1</div>
            <div class='step-title'>Upload Image</div>
            <div class='step-desc'>Select a clear PNG/JPG photo of the chicken or its droppings.</div>
        </div>
        <div class='step-card'>
            <div class='step-num'>2</div>
            <div class='step-title'>CNN Analysis</div>
            <div class='step-desc'>The EfficientNetB3 model automatically extracts disease-related visual features.</div>
        </div>
        <div class='step-card'>
            <div class='step-num'>3</div>
            <div class='step-title'>Get Diagnosis</div>
            <div class='step-desc'>View the predicted condition, confidence score, explanation, and recommended action.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='card-header-lg'><span style='font-size:1rem'>📋</span>&nbsp; Detectable Classes</div>
    """, unsafe_allow_html=True)

    fault_html = "<div class='fault-grid-wrapper'>"
    for cls in class_names:
        severity, sev_color = get_severity(cls)
        icon = get_icon(cls)
        fault_html += (
            "<div class='fault-grid-item'>"
            f"<div class='fault-grid-icon-box'>{icon}</div>"
            f"<div class='fault-grid-name'>{cls}</div>"
            f"<span class='fault-grid-sev' style='background:{sev_color}18;color:{sev_color};border:1px solid {sev_color}44'>"
            f"{severity}"
            "</span>"
            "</div>"
        )
    fault_html += "</div>"
    st.markdown(fault_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── PREDICTION STATE ──────────────────────────────────────
else:
    with st.spinner("🔍 Analysing image..."):
        t0      = time.time()
        arr     = preprocess(uploaded, IMG_H, IMG_W)
        preds   = model.predict(arr, verbose=0)[0]
        elapsed = time.time() - t0

    pred_idx             = int(np.argmax(preds))
    pred_class           = class_names[pred_idx]
    confidence           = float(preds[pred_idx]) * 100
    severity, sev_color  = get_severity(pred_class)
    icon                 = get_icon(pred_class)

    st.markdown(
        "<div class='chip-row'>"
        f"<span class='chip'><span class='chip-icon'>📁</span>{uploaded.name}</span>"
        f"<span class='chip'><span class='chip-icon'>⏱️</span>{elapsed*1000:.0f} ms inference</span>"
        f"<span class='chip'><span class='chip-icon'>📐</span>{IMG_W} × {IMG_H} px input</span>"
        "<span class='chip'><span class='chip-icon'>🧠</span>CNN · Softmax output</span>"
        "</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("""
        <div class='card'>
            <div class='card-header'><span class='card-header-icon'>🖼️</span> Uploaded Image</div>
        """, unsafe_allow_html=True)
        show_image(
            uploaded.getvalue(),
            caption=f"{uploaded.name}  |  Resized to {IMG_W}×{IMG_H} for inference"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            "<div class='card' style='height:100%;display:flex;flex-direction:column;'>"
            "<div class='card-header'><span class='card-header-icon'>🔍</span> Diagnosis Result</div>"
            f"<div class='result-card' style='border-color:{sev_color};background:{sev_color}0D;color:{sev_color};flex:1;'>"
            f"<span class='result-icon'>{icon}</span>"
            "<div class='result-eyebrow'>Detected Condition</div>"
            f"<div class='result-class' style='color:{sev_color}'>{pred_class}</div>"
            f"<div class='result-conf' style='color:{sev_color}'>{confidence:.1f}%</div>"
            "<div class='result-conf-sub'>Model Confidence Score</div>"
            f"<div class='sev-pill' style='background:{sev_color}18;color:{sev_color};border-color:{sev_color}55'>"
            f"{severity} SEVERITY"
            "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='card'>"
        "<div class='card-header'><span class='card-header-icon'>📖</span> Explanation &amp; Recommended Action</div>"
        f"<span class='box-label orange'>Diagnosis — {icon} {pred_class}</span>"
        f"<div class='info-box'>{get_info(pred_class)}</div>"
        "<span class='box-label green'>Recommended Action</span>"
        f"<div class='action-box'>{get_action(pred_class)}</div>"
        "</div>",
        unsafe_allow_html=True
    )

    with st.expander("🔬 Technical Details — Raw Prediction Data"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Preprocessed Tensor Info**")
            st.code(
                f"Shape      : {arr.shape}\n"
                f"Dtype      : {arr.dtype}\n"
                f"Pixel min  : {arr.min():.4f}\n"
                f"Pixel max  : {arr.max():.4f}\n"
                f"Pixel mean : {arr.mean():.4f}\n"
                f"Inference  : {elapsed*1000:.1f} ms"
            )
        with d2:
            st.markdown("**Raw Softmax Probabilities**")
            for cls, p in zip(class_names, preds):
                bar = "█" * int(p * 28)
                st.code(f"{cls:<22}: {p*100:>6.3f}%  {bar}")

st.markdown("""
<div class='footer'>
    Chicken Disease Diagnosis System &nbsp;·&nbsp; Built with TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)