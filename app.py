# app.py
import io, urllib.request, os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_img_label import st_img_label

# ===================== Helpers =====================
def load_image_from_bytes(b):
    return Image.open(io.BytesIO(b)).convert("RGB")

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def ensure_odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1

def blend_with_original(orig_bgr, fx_bgr, strength_pct):
    a = float(strength_pct) / 100.0
    return cv2.addWeighted(fx_bgr, a, orig_bgr, 1.0 - a, 0.0)

def bbox_to_xyxy(b):
    """
    รับได้ทั้งรูปแบบ:
    - dict: {"x":..., "y":..., "w":..., "h":...}
    - dict: {"left":..., "top":..., "width":..., "height":...}
    - dict: {"x1":..., "y1":..., "x2":..., "y2":...} / {"xmin","ymin","xmax","ymax"}
    - list/tuple: [x, y, w, h] หรือ [x1, y1, x2, y2]
    คืนค่าเป็น (x1, y1, x2, y2) แบบ int
    """
    # list/tuple
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        x1, y1, a, b2 = b[:4]
        # เดาว่าเป็น [x, y, w, h] ถ้าค่า a<b2 ส่วนใหญ่คือ w,h
        # ปลอดภัยกว่าคือถือเป็น [x,y,w,h] เสมอ:
        x2 = x1 + a
        y2 = y1 + b2
        return int(x1), int(y1), int(x2), int(y2)

    # dict
    if isinstance(b, dict):
        # แบบมี x,y,w,h
        if all(k in b for k in ("x", "y")) and ("w" in b or "width" in b):
            x = b.get("x", b.get("left"))
            y = b.get("y", b.get("top"))
            w = b.get("w", b.get("width"))
            h = b.get("h", b.get("height"))
            return int(x), int(y), int(x + w), int(y + h)

        # แบบมี left,top,width,height
        if all(k in b for k in ("left", "top", "width", "height")):
            x = b["left"]; y = b["top"]; w = b["width"]; h = b["height"]
            return int(x), int(y), int(x + w), int(y + h)

        # แบบมี x1,y1,x2,y2 หรือ xmin,ymin,xmax,ymax
        x1 = b.get("x1", b.get("xmin"))
        y1 = b.get("y1", b.get("ymin"))
        x2 = b.get("x2", b.get("xmax"))
        y2 = b.get("y2", b.get("ymax"))
        if None not in (x1, y1, x2, y2):
            return int(x1), int(y1), int(x2), int(y2)

    raise ValueError(f"Unknown bbox format: {type(b)} -> {b}")


def scale_rect(rect, sx, sy, W, H):
    x1, y1, x2, y2 = rect
    x1 = int(np.clip(x1 * sx, 0, W-1))
    y1 = int(np.clip(y1 * sy, 0, H-1))
    x2 = int(np.clip(x2 * sx, 1, W))
    y2 = int(np.clip(y2 * sy, 1, H))
    if x2 <= x1: x2 = min(W, x1+1)
    if y2 <= y1: y2 = min(H, y1+1)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

def make_mask(h, w, roi):
    m = np.zeros((h, w), dtype=np.uint8)
    m[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]] = 255
    return m

# ===================== Effects =====================
def fx_hsv(img_bgr, hue_deg=0, sat_pct=0, val_pct=0):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    h = (h + int(hue_deg/2)) % 180
    s = np.clip(s + (s*sat_pct//100), 0, 255)
    v = np.clip(v + (v*val_pct//100), 0, 255)
    out = np.stack([h, s, v], axis=-1).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

def fx_canny(img_bgr, t1=100, t2=200):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, t1, t2)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)

def fx_gaussian(img_bgr, k=5, sigma=0):
    k = ensure_odd(k)
    return cv2.GaussianBlur(img_bgr, (k, k), sigmaX=sigma, sigmaY=sigma)

def fx_pencil(img_bgr, blur_k=21):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - g
    blur_k = ensure_odd(max(3, blur_k))
    b = cv2.GaussianBlur(inv, (blur_k, blur_k), 0)
    d = cv2.divide(g, 255 - b, scale=256)
    return cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)

def fx_adaptive(img_bgr, block=11, C=2):
    block = ensure_odd(max(3, block))
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, C
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

# ===================== App =====================
st.set_page_config(page_title="Image Lab + ROI", layout="wide")
st.title("🧩 ROI ➜ Image Processing (Original vs Processed)")

# ---- Sidebar: Main Menu ----
with st.sidebar:
    main_sel = option_menu("Main Menu", ["Home", "Settings"],
                           icons=["house", "gear"], menu_icon="display", default_index=0)

# keep image in state
if "img_pil" not in st.session_state:
    st.session_state.img_pil = None

# ---- Sidebar: load image (Home) ----
if main_sel == "Home":
    with st.sidebar:
        st.markdown("---")
        st.subheader("Source")
        src = st.radio("เลือกแหล่งรูป", ["อัปโหลด", "กล้อง", "URL", "ตัวอย่าง"], index=0)

        img_pil = None
        if src == "อัปโหลด":
            up = st.file_uploader("JPG/PNG", type=["jpg","jpeg","png"])
            if up: img_pil = load_image_from_bytes(up.read())
        elif src == "กล้อง":
            cam = st.camera_input("ถ่ายรูป")
            if cam: img_pil = load_image_from_bytes(cam.getvalue())
        elif src == "URL":
            url = st.text_input("วาง URL ของรูป")
            if st.button("ดึงรูป") and url:
                with urllib.request.urlopen(url) as r:
                    img_pil = load_image_from_bytes(r.read())
        else:
            # fallback demo
            demo = np.full((420,740,3), 220, np.uint8)
            cv2.putText(demo,"DEMO",(260,230),cv2.FONT_HERSHEY_SIMPLEX,2,(70,70,70),5)
            img_pil = Image.fromarray(demo)

        if img_pil is not None:
            st.session_state.img_pil = img_pil

if st.session_state.img_pil is None:
    st.info("⬅️ เลือก/โหลดรูปจาก Sidebar ก่อน")
    st.stop()

img_pil = st.session_state.img_pil
img_bgr = pil_to_cv(img_pil)
H, W = img_bgr.shape[:2]

# ---- Sidebar: Processing Tools (อยู่ใน navbar ด้านข้าง) ----
with st.sidebar:
    st.markdown("---")
    tool_sel = option_menu(
        "Processing Tools",
        ["HSV / Lab", "Canny", "Gaussian Blur", "Pencil Sketch", "Adaptive Threshold"],
        icons=["brightness-high","bounding-box","filter","pencil","ui-checks-grid"],
        menu_icon="magic", default_index=0,
        styles={
            "container":{"padding":"0.5rem","background-color":"#f7f9fc"},
            "icon":{"color":"orange","font-size":"18px"},
            "nav-link":{"font-size":"16px","margin":"2px","--hover-color":"#eee"},
            "nav-link-selected":{"background-color":"green","color":"white"},
        }
    )

# ===================== STEP 1: ROI =====================
st.subheader("Step 1: Select ROI (ลากกรอบเหนือรูป – แล้วค่อยปรับพารามิเตอร์)")
# แสดงภาพย่อเพื่อเลือก ROI ด้วย st_img_label
# เลือกให้ย่อกว้างไม่เกิน 900px เพื่อใช้งานสะดวก
disp_max_w = 900
scale = min(1.0, disp_max_w / float(W))
disp_w, disp_h = int(W * scale), int(H * scale)
disp_img = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
disp_pil = cv_to_pil(disp_img)

# ให้ผู้ใช้ลากกรอบได้ 1–หลายกรอบ (จะใช้กรอบแรกเป็น ROI หลัก)
rects = st_img_label(disp_pil, box_color="red", rects=[])

# คำนวน ROI (ถ้าไม่มีกรอบ -> ใช้ทั้งรูป)
if rects and len(rects) > 0:
    x1r, y1r, x2r, y2r = bbox_to_xyxy(rects[0])
    roi = scale_rect((x1r, y1r, x2r, y2r), sx=1/scale, sy=1/scale, W=W, H=H)
else:
    roi = {"x1": 0, "y1": 0, "x2": W, "y2": H}

# โหมดการใช้ ROI
st.caption(f"ROI pixels: x1={roi['x1']}, y1={roi['y1']}, x2={roi['x2']}, y2={roi['y2']}")
roi_mode = st.radio(
    "โหมดการใช้ ROI",
    ("Process inside ROI", "Process outside ROI (invert mask)", "Crop to ROI"),
    index=0, horizontal=True
)

# ===================== STEP 2: Processing =====================
c1, c2 = st.columns(2, gap="large")

# LEFT: Original
with c1:
    st.markdown("#### 🖼️ Original Image")
    st.image(cv_to_pil(img_bgr), use_container_width=True)
    st.caption(f"Shape: {img_bgr.shape}")

# RIGHT: Processed (พารามิเตอร์อยู่ใต้รูป)
with c2:
    st.markdown(f"#### ✨ Processed Image ({tool_sel})")
    processed_img_box = st.container()
    processed_placeholder = processed_img_box.empty()

    params = st.container()
    with params:
        st.markdown("---")
        st.write("**Parameters**")
        activated = st.slider("Activated", 0, 100, 30, help="ความแรงเอฟเฟกต์ (%)")

        # เตรียมภาพตาม ROI mode
        if roi_mode == "Crop to ROI":
            target = img_bgr[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]].copy()
        else:
            target = img_bgr.copy()

        # เลือกเอฟเฟกต์ + พารามิเตอร์
        if tool_sel == "HSV / Lab":
            hue = st.slider("Hue (°)", -180, 180, 0)
            sat = st.slider("Saturation (%)", -100, 100, 0)
            val = st.slider("Value (%)", -100, 100, 0)
            fx = fx_hsv(target, hue_deg=hue, sat_pct=sat, val_pct=val)

        elif tool_sel == "Canny":
            t1 = st.slider("Threshold 1", 0, 500, 100)
            t2 = st.slider("Threshold 2", 0, 500, 200)
            fx = fx_canny(target, t1=t1, t2=t2)

        elif tool_sel == "Gaussian Blur":
            k = st.slider("Kernel size (odd)", 1, 99, 5, step=2)
            sigma = st.slider("Sigma", 0, 50, 0)
            fx = fx_gaussian(target, k=k, sigma=sigma)

        elif tool_sel == "Pencil Sketch":
            blur_k = st.slider("Blur (odd)", 3, 99, 21, step=2)
            fx = fx_pencil(target, blur_k=blur_k)

        else:  # Adaptive Threshold
            block = st.slider("Block size (odd)", 3, 99, 11, step=2)
            C = st.slider("C (bias)", -20, 20, 2)
            fx = fx_adaptive(target, block=block, C=C)

        # รวมผลลัพธ์ตามโหมด ROI
        if roi_mode == "Crop to ROI":
            processed = blend_with_original(target, fx, activated)
        else:
            # ทำทั้งรูป แล้วผสานด้วยหน้ากาก ROI หรือกลับหน้ากาก
            full_fx = fx_hsv(img_bgr) if False else None  # placeholder
            # แต่เราใช้ fx ที่คำนวณจาก "target=img_bgr.copy()" อยู่แล้วข้างบน
            fx_full = fx  # ผลของเอฟเฟกต์ทั้งภาพ
            mask = make_mask(H, W, roi)
            if roi_mode == "Process outside ROI (invert mask)":
                mask = cv2.bitwise_not(mask)

            a = float(activated) / 100.0
            fx_mix = cv2.addWeighted(fx_full, a, img_bgr, 1.0 - a, 0.0)
            # composite ด้วย mask
            bg = cv2.bitwise_and(img_bgr, img_bgr, mask=cv2.bitwise_not(mask))
            fg = cv2.bitwise_and(fx_mix, fx_mix, mask=mask)
            processed = cv2.add(bg, fg)

    processed_placeholder.image(cv_to_pil(processed), use_container_width=True)
    st.caption(f"Shape: {processed.shape}")

# Download
st.download_button(
    "Download Processed",
    data=cv2.imencode(".png", processed)[1].tobytes(),
    file_name="processed.png",
    mime="image/png",
)

# Settings page (optional)
if main_sel == "Settings":
    st.markdown("---")
    st.subheader("Settings")
    st.write("ใส่ค่าคงที่/ธีมที่ต้องการในอนาคตได้ที่นี่")
