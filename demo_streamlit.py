import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import yaml

st.title("AOI-AI Meiko Automation")

# Hàm load từ config.yaml
def load_model_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_paths = [model["weight_path"] for model in config["models"]]
    model_names = [model["class_name"] for model in config["models"]]
    return model_paths, model_names

yaml_path = "D:/AOI-MKAC/config.yaml"  # Cập nhật đúng đường dẫn file cấu hình

# Sidebar settings
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

# Mật khẩu admin
ADMIN_PASSWORD = "1234"
st.sidebar.title("🔒 Đăng nhập quản trị")
password_input = st.sidebar.text_input("🔑 Nhập mật khẩu để cấu hình", type="password")

if password_input == ADMIN_PASSWORD:
    st.sidebar.success("✅ Đăng nhập thành công!")
    model_paths, model_names = load_model_config(yaml_path)
else:
    st.sidebar.warning("🔐 Nhập mật khẩu để xem cấu hình")
    model_paths, model_names = load_model_config(yaml_path)

# Load models một lần (tối ưu tốc độ)
models = [YOLO(path) for path in model_paths]

# Hàm xử lý từng ảnh
def process_image(image_file, models, model_names, confidence):
    image = Image.open(image_file).convert("RGB")
    image_resized = image.resize((300, 300))
    image_np = np.array(image_resized)

    best_conf = 0
    best_result = None
    best_model_idx = -1

    for i, model in enumerate(models):
        results = model.predict(source=image_np, conf=confidence, device=0)
        boxes = results[0].boxes

        if boxes:
            max_conf = float(max(box.conf[0] for box in boxes))
            if max_conf > best_conf:
                best_conf = max_conf
                best_result = results[0].plot()
                best_model_idx = i

    return {
        "image_id": image_file.name,
        "image_show": best_result if best_model_idx != -1 else image_resized,
        "found": best_model_idx != -1,
        "label": model_names[best_model_idx] if best_model_idx != -1 else "Không phát hiện"
    }

# Upload ảnh
uploaded_files = st.file_uploader("📁 Chọn nhiều ảnh", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
num_cols = 5

if uploaded_files:
    st.markdown("## 📊 Kết quả phát hiện")
    cols = st.columns(num_cols)

    for idx, file in enumerate(uploaded_files):
        result = process_image(file, models, model_names, confidence)

        col = cols[idx % num_cols]
        with col:
            st.markdown(f"**Ảnh {idx+1}**")
            st.image(result["image_show"], caption=result["label"], use_container_width=True)
            if result["found"]:
                st.success("✅ Phát hiện lỗi")
            else:
                st.warning("❌ Không phát hiện")

        if (idx + 1) % num_cols == 0 and idx + 1 != len(uploaded_files):
            cols = st.columns(num_cols)

    # Thêm cột trống nếu ảnh không chia hết cho 5
    remainder = len(uploaded_files) % num_cols
    if remainder != 0:
        for _ in range(num_cols - remainder):
            with st.columns(1)[0]:
                st.empty()
