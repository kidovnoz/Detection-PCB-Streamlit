import cv2
import numpy as np

# Apply FFT-based denoising by removing horizontal and vertical frequency components
def fft_denoise(image, mask_size_x=3, max_size_y=6):
    f = np.fft.fft2(image)                   # Apply 2D FFT
    fshift = np.fft.fftshift(f)              # Shift the zero frequency component to the center
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2        # Center of the frequency image

    # Create a frequency mask to suppress horizontal and vertical lines
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - mask_size_x:crow + mask_size_x, :] = 0  # Remove horizontal frequencies
    mask[:, ccol - max_size_y:ccol + max_size_y] = 0    # Remove vertical frequencies

    fshift_filtered = fshift * mask         # Apply the mask
    f_ishift = np.fft.ifftshift(fshift_filtered)  # Shift back
    img_back = np.fft.ifft2(f_ishift)       # Inverse FFT to return to spatial domain
    return np.abs(img_back).astype(np.uint8)  # Convert to uint8 image

# Main function to detect abnormal lines between Gerber and actual PCB images
def detect_abnormal_lines(
    img_path_gerber, img_path_actual, 
    sobel_thresh=80, show=True, 
    roi=(100, 100, 100, 100), 
    use_fft=False, fft_mask_size_x=2, fft_mask_size_y=3
):
    # Load images in grayscale
    img1 = cv2.imread(img_path_gerber, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path_actual, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("❌ Không thể đọc ảnh đầu vào.")  # Cannot read input images

    # Resize actual image to match Gerber image dimensions
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Apply optional FFT-based denoising
    if use_fft:
        img1 = fft_denoise(img1, fft_mask_size_x, fft_mask_size_y)
        img2 = fft_denoise(img2, fft_mask_size_x, fft_mask_size_y)

    # Apply Sobel filter (X direction) to both images
    sobel1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    sobel2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)

    # Convert gradients to absolute 8-bit images
    abs_sobel1 = cv2.convertScaleAbs(sobel1)
    abs_sobel2 = cv2.convertScaleAbs(sobel2)

    # Compute absolute difference between edges
    diff = cv2.absdiff(abs_sobel1, abs_sobel2)

    # Threshold the difference to get binary mask of potential defects
    _, mask = cv2.threshold(diff, sobel_thresh, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_diff = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_diff = cv2.morphologyEx(clean_diff, cv2.MORPH_CLOSE, kernel)

    # Morphological closing with wider kernel to connect broken line segments
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    connected = cv2.morphologyEx(clean_diff, cv2.MORPH_CLOSE, kernel_connect)

    # Find external contours of the detected differences
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0
        area = cv2.contourArea(cnt)

        # Filter small noise
        if area < 10:
            continue
        # Filter overly wide and flat shapes
        if 2 < aspect_ratio < 15 and w > 20 and h < 5:
            continue
        # Optional: skip outside ROI (currently commented out)
        # x_min, y_min, x_max, y_max = roi
        # if x + w < x_min or x > x_max or y + h < y_min or y > y_max:
        #     continue

        filtered_contours.append(cnt)

    # Create a BGR version of img1 for annotation
    highlight_filtered = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    for index_label, cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Draw red filled contours
        cv2.drawContours(highlight_filtered, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)

        # Draw green vertical center line
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.line(highlight_filtered, (center_x, y), (center_x, y + h), (0, 255, 0), 1)

        # Draw blue horizontal center line
        cv2.line(highlight_filtered, (x, center_y), (x + w, center_y), (255, 0, 0), 1)

        # Label with defect area
        label = f"{int(area)}"
        cv2.putText(highlight_filtered, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    print(f"[INFO] Tổng contour phát hiện: {len(contours)}")
    print(f"[INFO] Sau lọc còn lại: {len(filtered_contours)}")

    if show:
        cv2.imshow("Khác biệt biên", diff)
        cv2.imshow("Mask đã làm sạch", clean_diff)
        cv2.imshow("Highlight", highlight_filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return annotated image, contours, last label and last index
    return highlight_filtered, filtered_contours, label, index_label
