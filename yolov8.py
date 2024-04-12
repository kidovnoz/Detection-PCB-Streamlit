from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Display model information (optional)
model.info()


# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
# results = model("baocao.avi")


def main():
    st.title("Camera on streamlit")
    st.camera_input("Take a picture")
    name = st.text_input("Enter your name")
    if st.button("Enter") and name == "tường duy":
        st.write(f"Anh yêu em nhiều lắm ")
        st.image("traitim.jpg")
if __name__ == "__main__":
    main()