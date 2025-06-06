from pathlib import Path
import yaml

def load_config(path="config.yaml", override_args=None):
    config_path = Path(path).resolve()
    config_dir = config_path.parent

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Danh sách các key chứa đường dẫn cần chuyển thành tuyệt đối
    path_keys = [
        'input_image', 'output_image', 'tile_output_image', 'roi_prefix',
        'defects_file', 'reference_image', 'test_image'
    ]

    for key in path_keys:
        if key in config and config[key]:
            config[key] = str((config_dir / config[key]).resolve())

    # Chuyển align_files thành tuyệt đối
    if 'align_files' in config:
        for k in ['align_x', 'align_y']:
            if k in config['align_files']:
                config['align_files'][k] = str((config_dir / config['align_files'][k]).resolve())

    return config
