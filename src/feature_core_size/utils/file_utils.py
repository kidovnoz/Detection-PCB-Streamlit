def read_align_file(path):
    max_width = 0
    max_height = 0
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    width = int(parts[1])
                    height = int(parts[2])
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
                except ValueError:
                    continue
    return max_width, max_height

def get_final_dimensions_from_align_files(config):
    align_x_path = config["align_files"]["align_x"]
    align_y_path = config["align_files"]["align_y"]
    width_x, height_x = read_align_file(align_x_path)
    width_y, height_y = read_align_file(align_y_path)
    final_width = width_x + width_y
    final_height = height_x + height_y
    return final_width, final_height

def read_defect_coordinates(defects_file):
    coords_in_original = []
    with open(defects_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                x = int(float(parts[0]))
                y = int(float(parts[1]))
                coords_in_original.append((x, y))
    return coords_in_original
