import streamlit as st
from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np
import yaml
import sys
import asyncio
import torch
import cv2
import torchvision
import time

# Cài đặt tương thích với Windows và Python >= 3.8
start_time = time.time()
if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.title("AOI-AI Meiko Automation")

# Kiểm tra GPU
st.sidebar.markdown("### 🧠 GPU Info")
st.sidebar.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.sidebar.write("GPU:", torch.cuda.get_device_name(0))
else:
    st.sidebar.warning("⚠️ Không tìm thấy GPU, đang chạy trên CPU.")


# Load file YAML cấu hình model
def load_model_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_paths = [model["weight_path"] for model in config["models"]]
    model_names = [model["class_name"] for model in config["models"]]
    model_segment = [model["weight_path"] for model in config["model_segment"]]
    return model_paths, model_names, model_segment


yaml_path = r"D:\AOI-MKAC\MixPL\src\aoi_inspection\configs\config_streamlit.yaml"
st.logo("D:/meiko-logo.webp", size="large", link="https://meiko-elec.com.vn/")

# Sidebar settings
st.sidebar.header("⚙️ Cấu hình")
confidence = st.sidebar.slider("Ngưỡng confidence", 0.1, 1.0, 0.25, 0.05)

# Nút xoá cache ảnh
if st.sidebar.button("🧹 Xử lý lại"):
    st.session_state.processed_images = {}
    st.sidebar.success("Đã xóa cache.")
# Mật khẩu admin
ADMIN_PASSWORD = "1234"
st.sidebar.title("🔒 Đăng nhập quản trị")
password_input = st.sidebar.text_input("🔑 Nhập mật khẩu", type="password")

if password_input == ADMIN_PASSWORD:
    st.sidebar.success("✅ Đăng nhập thành công!")
    yaml_path = st.sidebar.text_input(" 🔧 Đường dẫn file YAML", value=yaml_path)
    model_paths, model_names, model_segment = load_model_config(yaml_path)

else:
    st.sidebar.warning("🔐 Nhập mật khẩu để xem cấu hình")
    model_paths, model_names, model_segment = load_model_config(yaml_path)


# ✅ Load model chỉ 1 lần
@st.cache_resource
def load_models(paths):
    return [YOLO(path) for path in paths]


def load_model_segment(paths):
    return [SAM(path) for path in paths]


models = load_models(model_paths)
models_segment = load_model_segment(model_segment)
# Lưu kết quả đã xử lý vào session_state
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}


# Xử lý ảnh đầu vào
def process_image(image_file, models, model_names, confidence):
    try:
        image = Image.open(image_file).convert("RGB")
        image_array = np.array(image)

        all_boxes = []

        for i, model in enumerate(models):
            try:
                model.eval()
                results = model.predict(
                    source=image, conf=confidence, imgsz=640, device=0, augment=True
                )

                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                            conf_score = float(box.conf.item())
                            cls = int(box.cls.item())
                            if conf_score > 0.7:
                                # Dùng SAM để segment bbox
                                sam_model = models_segment[0]
                                segment_result = sam_model(
                                    source=image_array, bboxes=[[x1, y1, x2, y2]]
                                )

                                # Lấy mask đầu tiên
                                mask = segment_result[0].masks.data[0].cpu().numpy()
                                mask = mask.astype(np.uint8)
                                mask = cv2.resize(
                                    mask,
                                    (image_array.shape[1], image_array.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                
                                pixel_count = len(mask)  # Đếm số pixel
                                print(
                                    f"Bounding box [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] -> {pixel_count} pixels"
                                )

                                all_boxes.append(
                                    {
                                        "box": [x1, y1, x2, y2],
                                        "conf": conf_score,
                                        "model_idx": i,
                                        "labels": [f"{cls}: {conf_score:.2f}"],
                                        "pixel_count": pixel_count,
                                        "mask": mask,
                                    }
                                )

            except Exception as e:
                st.error(f"Lỗi predict model {i}: {e}")

        # Nếu không có box nào được detect
        if not all_boxes:
            return {
                "image_id": image_file.name,
                "image_show": image,
                "found": False,
                "label": "Không phát hiện",
            }

        # Tạo tensor để chạy NMS
        boxes_tensor = torch.tensor([b["box"] for b in all_boxes], dtype=torch.float32)
        confs_tensor = torch.tensor([b["conf"] for b in all_boxes], dtype=torch.float32)
        # Chạy Non-Maximum Suppression
        keep_idxs = torchvision.ops.nms(boxes_tensor, confs_tensor, iou_threshold=0.1)

        # Lấy các box được giữ lại
        final_boxes = [all_boxes[i] for i in keep_idxs.tolist()]
        final_boxes = sorted(final_boxes, key=lambda x: x["conf"], reverse=True)

        # Vẽ kết quả lên ảnh gốc
        draw_img = image_array.copy()
        for b in final_boxes:
            x1, y1, x2, y2 = map(int, b["box"])
            label = f"{model_names[b['model_idx']]}: {b['conf']:.2f} ({b['pixel_count']} px)"

            # Vẽ bbox
            # cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                draw_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            x_min = b["box_mask_x_min"] = np.min(np.where(b["mask"] == 1)[1])
            x_max = b["box_mask_x_max"] = np.max(np.where(b["mask"] == 1)[1])
            y_min = b["box_mask_y_min"] = np.min(np.where(b["mask"] == 1)[0])
            y_max = b["box_mask_y_max"] = np.max(np.where(b["mask"] == 1)[0])

            width = x_max - x_min
            height = y_max - y_min

            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            # Overlay mask
            mask = b.get("mask", None)
            if mask is not None:
                # Resize mask về đúng shape ảnh nếu cần
                mask_resized = cv2.resize(
                    mask.astype(np.uint8), (draw_img.shape[1], draw_img.shape[0])
                )

                # Tạo màu overlay (ví dụ màu đỏ)
                colored_mask = np.zeros_like(draw_img)
                colored_mask[:, :, 2] = (mask_resized * 255).astype(np.uint8)

                # Tô nhẹ mask lên ảnh
                draw_img = cv2.addWeighted(draw_img, 1.0, colored_mask, 0.4, 0)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=cv2.contourArea)
                # Tính rotated bounding box
                rot_rect = cv2.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
                (center_x, center_y), (w, h), angle = rot_rect
                center = (int(center_x), int(center_y))

                # Chuyển angle sang radian
                theta = np.deg2rad(angle)

                # Vector chiều rộng (hướng chính của bbox xoay)
                dx_w = np.cos(theta) * w / 2
                dy_w = np.sin(theta) * w / 2

                # Vector chiều dài (vuông góc với vector rộng)
                dx_h = -np.sin(theta) * h / 2
                dy_h = np.cos(theta) * h / 2

                # Các điểm đầu/cuối của vector width (xanh dương)
                p1 = (int(center_x - dx_w), int(center_y - dy_w))
                p2 = (int(center_x + dx_w), int(center_y + dy_w))

                # Các điểm đầu/cuối của vector height (vàng)
                p3 = (int(center_x - dx_h), int(center_y - dy_h))
                p4 = (int(center_x + dx_h), int(center_y + dy_h))

                # Vẽ bbox xoay (màu xanh lá)
                box = np.intp(cv2.boxPoints(rot_rect))
                print(box[0])
                cv2.drawContours(draw_img, [box], 0, (0, 255, 0), 2)

                # Vẽ tâm bbox (màu đỏ)
                cv2.circle(draw_img, center, 5, (0, 0, 255), -1)

                # Vẽ vector chiều rộng (màu xanh dương)
                cv2.line(draw_img, p1, p2, (255, 0, 0), 2)

                # Vẽ vector chiều dài (màu vàng)
                cv2.line(draw_img, p3, p4, (0, 255, 255), 2)
                

        final_image = Image.fromarray(draw_img)

        return {
            "image_id": image_file.name,
            "image_show": final_image,
            "found": True,
            "label": f"{len(final_boxes)} vùng lỗi",
            "pixel": [f"Pixel: {pixel_count}"],
        }

    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
        return {
            "image_id": image_file.name,
            "image_show": None,
            "found": False,
            "label": "Lỗi xử lý",
        }


# Upload nhiều ảnh
uploaded_files = st.file_uploader(
    "📁 Chọn ảnh đầu vào",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)
num_cols = 1

if uploaded_files:
    st.markdown("## 📊 Kết quả phát hiện")
    cols = st.columns(num_cols)

    for idx, file in enumerate(uploaded_files):
        image_key = file.name

        # Kiểm tra nếu ảnh đã xử lý
        if image_key in st.session_state.processed_images:
            result = st.session_state.processed_images[image_key]
        else:
            result = process_image(file, models, model_names, confidence)
            st.session_state.processed_images[image_key] = result

        col = cols[idx % num_cols]
        with col:
            st.markdown(f"**Ảnh {idx+1}**")

            # Hiển thị ảnh
            if result["image_show"] is not None:
                st.image(
                    result["image_show"], use_container_width=True, output_format="auto"
                )

            # Hiển thị label với chiều cao cố định, ellipsis nếu quá dài
            st.markdown(
                f"""
                    <div style='height: 15px; 
                                display: flex; 
                                align-items: center; 
                                justify-content: center; 
                                text-align: center; 
                                font-size: 12px;
                                font-weight: bold;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                white-space: nowrap;'>
                        {result['label']}
                    </div>
                    """,
                unsafe_allow_html=True,
            )

            # Hiển thị icon ✅/❌ ở dòng riêng, luôn nằm dưới
            st.markdown(
                f"<div style='text-align: center; font-size: 24px;'>{'✅' if result['found'] else '❌'}</div>",
                unsafe_allow_html=True,
            )

        if (idx + 1) % num_cols == 0 and idx + 1 != len(uploaded_files):
            cols = st.columns(num_cols)

    # Bổ sung cột trống nếu số ảnh không chia hết
    remainder = len(uploaded_files) % num_cols
    if remainder != 0:
        for _ in range(num_cols - remainder):
            with st.columns(1)[0]:
                st.empty()
    print(f"Process completed in {time.time() - start_time:.2f} seconds.")
