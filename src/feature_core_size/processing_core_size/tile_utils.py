import cv2
import numpy as np

def calculate_tile_dimensions(size_matrix, rows, cols):
    row_heights = [max(size_matrix[i][j][0] for j in range(cols)) for i in range(rows)]
    col_widths = [max(size_matrix[i][j][1] for i in range(rows)) for j in range(cols)]
    tile_h = sum(row_heights)
    tile_w = sum(col_widths)
    return tile_w, tile_h, row_heights, col_widths

def assemble_tile_image(roi_padded_list, rows, cols, row_heights, col_widths):
    tile_h = sum(row_heights)
    tile_w = sum(col_widths)
    tile_image = np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255
    index = 0
    y_cursor = 0
    for i in range(rows):
        x_cursor = 0
        for j in range(cols):
            roi_padded = roi_padded_list[index]
            h_, w_ = roi_padded.shape[:2]
            tile_image[y_cursor:y_cursor+h_, x_cursor:x_cursor+w_] = roi_padded
            label = f"ROI {index + 1}"
            cv2.putText(tile_image, label, (x_cursor + 20, y_cursor + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            x_cursor += col_widths[j]
            index += 1
        y_cursor += row_heights[i]
    return tile_image

def create_final_composition(tile_image, final_width, final_height, padding_outer):
    pad_top = padding_outer["top"]
    pad_bottom = padding_outer["bottom"]
    pad_left = padding_outer["left"]
    pad_right = padding_outer["right"]

    tile_resized_w = final_width - pad_left - pad_right
    tile_resized_h = final_height - pad_top - pad_bottom
    tile_image_resized = cv2.resize(tile_image, (tile_resized_w, tile_resized_h), interpolation=cv2.INTER_LINEAR)

    final_result = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255
    final_result[pad_top:pad_top+tile_resized_h, pad_left:pad_left+tile_resized_w] = tile_image_resized
    return final_result