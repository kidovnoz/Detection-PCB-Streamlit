import cv2
import numpy as np

# Create a binary mask to detect red regions in the image using HSV color space
def create_red_mask(image, red_color_ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV color space
    # Create a binary mask for each red color range and combine them
    masks = [cv2.inRange(hsv, np.array(lower), np.array(upper)) for lower, upper in red_color_ranges]
    return cv2.bitwise_or(*masks)  # Merge masks with bitwise OR to form a complete red mask

# Extract the region of interest (ROI) that contains red pixels based on the binary mask
def extract_roi_from_mask(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find external contours
    if not contours:
        print("❌ Không tìm thấy vùng màu đỏ.")  # "No red region found."
        return None
    # Compute the bounding box that encloses all detected red regions
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    return image[y:y+h, x:x+w]  # Crop the bounding box from the original image

# Create a list of padded ROI images with specified padding and track their sizes in a matrix
def create_padded_rois(roi, padding_list, rows, cols):
    roi_padded_list = []  # List to store padded ROIs
    size_matrix = [[None for _ in range(cols)] for _ in range(rows)]  # To store padded ROI sizes
    index = 0
    for i in range(rows):
        for j in range(cols):
            top, bottom, left, right = padding_list[index]  # Extract padding values
            # Add padding to the ROI using white (255,255,255) border
            roi_padded = cv2.copyMakeBorder(
                roi, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            roi_padded_list.append(roi_padded)
            size_matrix[i][j] = roi_padded.shape[:2]  # Store dimensions (height, width)
            index += 1
    return roi_padded_list, size_matrix  # Return both the padded ROIs and their size matrix
