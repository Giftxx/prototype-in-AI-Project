# app.py (Refactored and Formatted)

import io
import urllib.request
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_img_label import st_img_label
import matplotlib.pyplot as plt

# ===================== Helper Functions =====================
def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load image from given bytes and convert to RGB PIL Image."""
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """Convert a PIL Image to OpenCV BGR format."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image to a PIL Image."""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def ensure_odd(n: int) -> int:
    """Ensure the given number n is odd (used for kernel sizes)."""
    n = int(n)
    return n if n % 2 == 1 else n + 1

def clamp(v: int, lo: int, hi: int) -> int:
    """Clamp value v to the range [lo, hi]."""
    return max(lo, min(hi, v))

def blend_with_original(orig_bgr: np.ndarray, fx_bgr: np.ndarray, strength_pct: float) -> np.ndarray:
    """Blend the effect image (fx_bgr) with the original image (orig_bgr) using the given strength percentage."""
    alpha = float(strength_pct) / 100.0
    return cv2.addWeighted(fx_bgr, alpha, orig_bgr, 1.0 - alpha, 0.0)

def make_display_image(bgr: np.ndarray, target_h: int = 430, max_w: int = 1000):
    """Resize image for display (maintaining aspect ratio) so it fits within target_h and max_w.
    Returns the resized PIL image, the scale factor, display size (w,h), and original size (W,H)."""
    H, W = bgr.shape[:2]
    # Do not upscale beyond original size
    scale = min(target_h / float(H), max_w / float(W), 1.0)
    disp_w, disp_h = int(W * scale), int(H * scale)
    disp_bgr = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    return cv_to_pil(disp_bgr), scale, (disp_w, disp_h), (W, H)

# --------- ROI / Bounding Box Utilities ---------
def rect_to_xyxy(rect, fallback_w=None, fallback_h=None):
    """
    Convert various ROI rectangle formats to (x1, y1, x2, y2) coordinates.
    Accepts:
    - dict: keys could be x,y,w,h or left,top,width,height or x1,y1,x2,y2 (xmin,...).
    - list/tuple of 4 numbers: interpreted as [x,y,w,h] if w,h seem like dimensions, otherwise [x1,y1,x2,y2].
    """
    # Dictionary formats
    if isinstance(rect, dict):
        # If keys for top-left and width/height
        if all(k in rect for k in ("x", "y")) and ("w" in rect or "width" in rect):
            x = float(rect.get("x", rect.get("left", 0)))
            y = float(rect.get("y", rect.get("top", 0)))
            w = float(rect.get("w", rect.get("width", 0)))
            h = float(rect.get("h", rect.get("height", 0)))
            return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
        if all(k in rect for k in ("left", "top", "width", "height")):
            x = float(rect["left"]); y = float(rect["top"])
            w = float(rect["width"]); h = float(rect["height"])
            return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
        # If keys for explicit coordinates
        x1 = rect.get("x1", rect.get("xmin"))
        y1 = rect.get("y1", rect.get("ymin"))
        x2 = rect.get("x2", rect.get("xmax"))
        y2 = rect.get("y2", rect.get("ymax"))
        if None not in (x1, y1, x2, y2):
            return int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
    # List/Tuple format
    if isinstance(rect, (list, tuple)) and len(rect) >= 4:
        a, b, c, d = [float(x) for x in rect[:4]]
        # If c,d likely represent width,height (positive and relatively small compared to fallback sizes)
        if fallback_w and fallback_h and 0 < c <= fallback_w and 0 < d <= fallback_h:
            return int(round(a)), int(round(b)), int(round(a + c)), int(round(b + d))
        # Otherwise treat as [x1,y1,x2,y2]
        return int(round(a)), int(round(b)), int(round(c)), int(round(d))
    # If format is unsupported
    raise KeyError("Unsupported rect format for ROI coordinates.")

def rect_to_xyxy_scaled(rect, scale: float, disp_w: int, disp_h: int, orig_W: int, orig_H: int):
    """Convert ROI coordinates from the scaled display image back to original image coordinates."""
    x1d, y1d, x2d, y2d = rect_to_xyxy(rect, fallback_w=disp_w, fallback_h=disp_h)
    inv_scale = 1.0 / max(scale, 1e-9)
    x1 = int(round(x1d * inv_scale));  y1 = int(round(y1d * inv_scale))
    x2 = int(round(x2d * inv_scale));  y2 = int(round(y2d * inv_scale))
    # Clamp coordinates within image bounds
    x1 = clamp(x1, 0, orig_W - 1);  y1 = clamp(y1, 0, orig_H - 1)
    x2 = clamp(x2, 1, orig_W);      y2 = clamp(y2, 1, orig_H)
    if x2 <= x1: x2 = min(orig_W, x1 + 1)
    if y2 <= y1: y2 = min(orig_H, y1 + 1)
    return x1, y1, x2, y2

# ===================== Image Effect Functions =====================
def fx_hsv(img_bgr: np.ndarray, hue_deg: int = 0, sat_pct: int = 0, val_pct: int = 0) -> np.ndarray:
    """Adjust hue, saturation, and value of the image by given percentages."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # OpenCV hue is [0,179]; convert hue shift from degrees ([-180,180]) to OpenCV scale
    h = (h + int(hue_deg / 2)) % 180
    s = np.clip(s + (s * sat_pct // 100), 0, 255)
    v = np.clip(v + (v * val_pct // 100), 0, 255)
    hsv_adjusted = np.stack([h, s, v], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

def fx_canny(img_bgr: np.ndarray, t1: int = 100, t2: int = 200) -> np.ndarray:
    """Apply Canny edge detection to the image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def fx_gaussian(img_bgr: np.ndarray, k: int = 5, sigma: int = 0) -> np.ndarray:
    """Apply Gaussian blur to the image with kernel size k and standard deviation sigma."""
    k = ensure_odd(k)
    return cv2.GaussianBlur(img_bgr, (k, k), sigmaX=sigma, sigmaY=sigma)

def fx_pencil(img_bgr: np.ndarray, blur_k: int = 21) -> np.ndarray:
    """Convert the image to a pencil sketch style image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blur_k = ensure_odd(max(3, blur_k))
    blur = cv2.GaussianBlur(inv_gray, (blur_k, blur_k), 0)
    pencil_sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2BGR)

def fx_adaptive(img_bgr: np.ndarray, block: int = 11, C: int = 2) -> np.ndarray:
    """Apply adaptive thresholding to the image (resulting in a black-and-white effect)."""
    block = ensure_odd(max(3, block))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, C)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def create_histogram_plot(img_bgr: np.ndarray, title: str = "Color Histogram"):
    """Create a histogram plot for the BGR image."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Calculate histograms for each color channel
    colors = ['blue', 'green', 'red']
    bgr_labels = ['Blue', 'Green', 'Red']
    
    for i, (color, label) in enumerate(zip(colors, bgr_labels)):
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, label=label, alpha=0.7)
    
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set style
    plt.style.use('default')
    ax.set_facecolor('#f8f9fa')
    
    return fig

def create_processed_histogram(img_proc: np.ndarray, tool_name: str):
    """Create a histogram for the processed image only."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Colors for BGR channels
    colors = ['blue', 'green', 'red']
    bgr_labels = ['Blue', 'Green', 'Red']
    
    # Processed image histogram only
    for i, (color, label) in enumerate(zip(colors, bgr_labels)):
        hist = cv2.calcHist([img_proc], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, label=f'{label}', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{tool_name} - Color Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

# ===================== Streamlit App Interface =====================
st.set_page_config(page_title="Image Lab (ROI as Tool)", page_icon="📷", layout="wide")
st.title("📷 Explore the rapid prototype in AI Project")

# Initialize session state variables
if "img_pil" not in st.session_state:
    st.session_state.img_pil = None
if "roi_coords" not in st.session_state:
    st.session_state.roi_coords = None

# Sidebar navigation menu (collapsible)
with st.sidebar:
    main_page = option_menu(
        "เมนูหลัก",               # Main Menu title in Thai
        ["หน้าแรก"],              # Options: "Home" in Thai
        icons=["house"],
        menu_icon="menu-up",      # icon for the menu title
        default_index=0,
    )

# Home page content (Image Lab functionality)
if main_page == "หน้าแรก":
    # Sidebar controls (collapsible left navbar)
    with st.sidebar:
        st.markdown("---")
        
        # Control Panel
        st.markdown("### 🎛️ Control Panel")
        
        # Image source selection
        st.markdown("#### 📁 Image Source")
        source_mode = st.radio("เลือกแหล่งรูป", ["Upload Image", "Camera", "Image URL"], index=0)
        
        img_pil = None
        if source_mode == "Upload Image":
            uploaded_file = st.file_uploader("อัปโหลด JPG/PNG", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                img_pil = load_image_from_bytes(uploaded_file.read())
        elif source_mode == "Camera":
            camera_image = st.camera_input("ถ่ายรูปจากกล้อง")
            if camera_image:
                img_pil = load_image_from_bytes(camera_image.getvalue())
        elif source_mode == "Image URL":
            url = st.text_input("วาง URL ของรูป")
            if st.button("ดึงรูป") and url:
                try:
                    with urllib.request.urlopen(url) as response:
                        img_pil = load_image_from_bytes(response.read())
                except Exception as e:
                    st.error(f"ไม่สามารถโหลดรูปจาก URL ได้: {e}")
        
        if img_pil is not None:
            st.session_state.img_pil = img_pil
            st.session_state.roi_coords = None
            # Show thumbnail of uploaded image
            st.image(img_pil, caption="Uploaded Image", width=200)

        st.markdown("---")
        
        # Processing Studio
        st.markdown("#### 🎨 Processing Studio")
        st.markdown("Select transformation:")
        
        tool = st.selectbox(
            "เลือกเครื่องมือ", 
            ["Edge Detection (Canny)", "HSV / Lab", "Gaussian Blur", "Pencil Sketch", "Adaptive Threshold"],
            index=0
        )
        
        # Settings based on selected tool
        st.markdown("##### ⚙️ Settings")
        
        # Edge Detection Settings
        if tool == "Edge Detection (Canny)":
            st.markdown("**Lower Threshold**")
            lower_threshold = st.slider("Lower", 0, 255, 50, label_visibility="collapsed")
            st.markdown("**Upper Threshold**")  
            upper_threshold = st.slider("Upper", 0, 255, 150, label_visibility="collapsed")
        elif tool == "HSV / Lab":
            hue = st.slider("ปรับฮิว (องศา)", -180, 180, 0)
            sat = st.slider("ปรับความอิ่มตัวของสี (%)", -100, 100, 0)
            val = st.slider("ปรับความสว่าง (%)", -100, 100, 0)
        elif tool == "Gaussian Blur":
            k = st.slider("Kernel size", 1, 99, 5, step=2)
            sigma = st.slider("Sigma", 0, 50, 0)
        elif tool == "Pencil Sketch":
            blur_k = st.slider("ระดับความเบลอ", 3, 99, 21, step=2)
        else:  # Adaptive Threshold
            block = st.slider("ขนาดบล็อก", 3, 99, 11, step=2)
            C = st.slider("ค่า C", -20, 20, 2)

    # Main content area
    # Ensure an image is loaded
    if st.session_state.img_pil is None:
        st.info("⬅️ กรุณาเลือกหรืออัปโหลดรูปจากแถบด้านข้างก่อน")
    else:
        # Prepare image data
        img_pil = st.session_state.img_pil
        img_bgr = pil_to_cv(img_pil)
        H, W = img_bgr.shape[:2]

        # Apply selected effect
        if tool == "Edge Detection (Canny)":
            st.subheader("✂️ Edge Detection Processing")
            processed_bgr = fx_canny(img_bgr, t1=lower_threshold, t2=upper_threshold)
        elif tool == "HSV / Lab":
            st.subheader("🌈 HSV Color Adjustment")
            processed_bgr = fx_hsv(img_bgr, hue_deg=hue, sat_pct=sat, val_pct=val)
        elif tool == "Gaussian Blur":
            st.subheader("🌫️ Gaussian Blur")
            processed_bgr = fx_gaussian(img_bgr, k=k, sigma=sigma)
        elif tool == "Pencil Sketch":
            st.subheader("✏️ Pencil Sketch")
            processed_bgr = fx_pencil(img_bgr, blur_k=blur_k)
        else:  # Adaptive Threshold
            st.subheader("📋 Adaptive Threshold")
            processed_bgr = fx_adaptive(img_bgr, block=block, C=C)
        
        # Display comparison
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### 🖼️ Original Image")
            st.image(img_pil, use_container_width=True)
            st.caption(f"Shape: {img_bgr.shape}")
            
        with col2:
            st.markdown(f"#### ✨ Processed Image ({tool})")
            st.image(cv_to_pil(processed_bgr), use_container_width=True)
            st.caption(f"Shape: {processed_bgr.shape}")
        
        # Display comparison histogram in the center below the images
        st.markdown("---")
        st.markdown("### 📊 Processed Image Analysis")
        
        # Single processed image histogram
        fig_processed = create_processed_histogram(processed_bgr, tool)
        st.pyplot(fig_processed)
        plt.close(fig_processed)
        
        # Download button for processed image
        st.download_button(
            "📥 ดาวน์โหลดภาพผลลัพธ์",
            data=cv2.imencode(".png", processed_bgr)[1].tobytes(),
            file_name=f"{tool.lower().replace(' ', '_')}_processed.png",
            mime="image/png",
            use_container_width=True
        )
