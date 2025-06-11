import cv2

# Convert coordinates from one image size to another with padding (tile offset)
def convert_coordinates_with_padding(coords, from_size, tile_size, tile_offset):
    from_w, from_h = from_size               # Original image dimensions
    tile_w, tile_h = tile_size               # Target tile dimensions
    offset_x, offset_y = tile_offset         # Offset to be added after scaling
    scale_x = tile_w / from_w                # Horizontal scaling factor
    scale_y = tile_h / from_h                # Vertical scaling factor

    # Scale and shift each coordinate
    return [(int(x * scale_x) + offset_x, int(y * scale_y) + offset_y) for x, y in coords]

# Annotate and save Regions of Interest (ROIs) based on provided coordinates
def annotate_and_save_rois(image, coords, roi_prefix, coordinate_box_size, output_image):
    height, width = image.shape[:2]          # Get image dimensions
    i = 0                                     # Counter for saved ROIs

    for x, y in coords:
        # Ensure the coordinates are within image bounds
        if 0 <= x < width and 0 <= y < height:
            i += 1
            # Define bounding box around each coordinate
            x1 = max(x - coordinate_box_size, 0)
            y1 = max(y - coordinate_box_size, 0)
            x2 = min(x + coordinate_box_size, width)
            y2 = min(y + coordinate_box_size, height)
            # Draw a rectangle around the coordinate
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # Crop the ROI
            roi_small = image[y1:y2, x1:x2]

            # Save ROI if it's not empty
            if roi_small.size > 0:
                cv2.imwrite(f"{roi_prefix}{i}.jpg", roi_small)

    # Optionally resize and save the annotated full image
    # image = cv2.resize(image, (4000, 4000), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_image, image)
    print("✅ Image saved successfully")
