# main.py
import argparse
import time
import cv2
import os

# Import custom utility functions and processing modules
from utils.config_loader import load_config
from utils.file_utils import get_final_dimensions_from_align_files, read_defect_coordinates
from processing_core_size.image_utils import create_red_mask, extract_roi_from_mask, create_padded_rois
from processing_core_size.tile_utils import calculate_tile_dimensions, assemble_tile_image, create_final_composition
from processing_core_size.defect_annotator import convert_coordinates_with_padding, annotate_and_save_rois
from processing_core_size.line_detection import detect_abnormal_lines

def parse_args():
    """
    Parse command-line arguments.
    Includes options for config file path, Sobel threshold, image override paths, and display toggles.
    """
    parser = argparse.ArgumentParser(description="PCB Compute ROI Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--sobel_thresh", type=int, default=80, help="Sobel threshold for line defect detection")
    parser.add_argument("--show", action="store_true", help="Show intermediate result images")
    parser.add_argument("--input_image", type=str, help="Override input image path")
    parser.add_argument("--reference_image", type=str, help="Override reference image path")
    parser.add_argument("--test_image", type=str, help="Override test image path")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()

    # Prepare override arguments if provided
    override_args = {}
    if args.input_image:
        override_args["input_image"] = args.input_image
    if args.reference_image:
        override_args["reference_image"] = args.reference_image
    if args.test_image:
        override_args["test_image"] = args.test_image

    # Load configuration from YAML file (with optional overrides)
    config = load_config(args.config, override_args)

    # Extract configuration parameters
    final_width, final_height = get_final_dimensions_from_align_files(config)
    input_image = config["input_image"]
    output_image = config["output_image"]
    roi_prefix = config["roi_prefix"]
    defects_file = config["defects_file"]
    padding_outer = config["padding_outer"]
    tile_layout = config["tile_layout"]
    padding_list = config["padding_list"]
    red_color_ranges = config["red_color_ranges"]
    coordinate_box_size = config["coordinate_box_size"]
    reference_image = config["reference_image"]
    test_image = config["test_image"]

    # Run only if the output image does not already exist
    if not os.path.exists(output_image):
        # Read the input image
        image = cv2.imread(input_image)

        # Create a mask for detecting red regions (defect markers)
        mask_red = create_red_mask(image, red_color_ranges)

        # Extract the Region of Interest (ROI) from the red mask
        roi = extract_roi_from_mask(image, mask_red)
        if roi is None:
            print("No ROI found.")
            return

        # Generate padded ROIs based on layout and padding settings
        roi_padded_list, size_matrix = create_padded_rois(roi, padding_list, tile_layout["rows"], tile_layout["cols"])

        # Calculate dimensions for each tile
        tile_w, tile_h, row_heights, col_widths = calculate_tile_dimensions(size_matrix, tile_layout["rows"], tile_layout["cols"])

        # Assemble the final tiled image from padded ROIs
        tile_image = assemble_tile_image(roi_padded_list, tile_layout["rows"], tile_layout["cols"], row_heights, col_widths)

        # Create the final composite image with additional outer padding
        final_result = create_final_composition(tile_image, final_width, final_height, padding_outer)

        # Load defect coordinates and map them to the resized image space
        coords_in_original = read_defect_coordinates(defects_file)
        coords_in_resized = convert_coordinates_with_padding(
            coords_in_original,
            (final_width, final_height),
            (final_width, final_height),
            (0, 0)  # No additional padding offsets applied here
        )

        # Annotate and save ROI image with defect locations
        annotate_and_save_rois(final_result, coords_in_resized, roi_prefix, coordinate_box_size, output_image)

    # Detect line-based defects (e.g., cracks) between reference and test images using the Sobel filter
    '''
    Gọi hàm phát hiện các đường bất thường (vết nứt, hỏng hóc) bằng cách so sánh ảnh chuẩn và ảnh cần kiểm tra
    - Ảnh đầu ra highlight vùng lỗi - highlight_filtered
    - Lọc các contour tương ứng với lỗi - filtered_contours
    - Sinh nhãn (label map) và danh sách chỉ số vùng lỗi - label, index_label

    '''
    highlight_filtered, filtered_contours, label, index_label = detect_abnormal_lines(
        reference_image,      # Ảnh chuẩn không lỗi
        test_image,           # Ảnh cần kiểm tra
        sobel_thresh=args.sobel_thresh,  # Ngưỡng Sobel để phát hiện cạnh
        show=args.show        # Hiển thị ảnh trung gian nếu bật
    )

    print(f"[INFO] Tổng diện tích lỗi (trong ROI) {index_label}: {label} pixel")
    print(f"Process completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
