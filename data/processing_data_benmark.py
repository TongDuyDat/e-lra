import os
from .processing_CVC import CVC_CliniCDBDataset
from albumentations import ToTensorV2

class DataBenchmark(CVC_CliniCDBDataset):
    def __init__(self, config_path, phase = "val"):
        self.config_path = config_path
        self.phase = phase
        # Load config từ file .py
        self._load_config(config_path)

        # Load dữ liệu
        self.image_paths = []
        self.mask_paths = []
        self.__load_data()
        self.to_tensor = ToTensorV2()

    def __load_data(self, exts=[".jpg", ".png", '.tif']):
        # Đường dẫn đến thư mục images và masks theo phase (train/val)
        image_dir = os.path.join(self.source, "images")
        mask_dir = os.path.join(self.source, "masks")

        # Kiểm tra thư mục tồn tại
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise ValueError(f"Directory not found: {image_dir} or {mask_dir}")

        # Lấy danh sách file ảnh từ thư mục images
        for file in os.listdir(image_dir):
            file_name, file_extension = os.path.splitext(file)
            if file_extension.lower() in exts:
                img_path = os.path.join(image_dir, file)
                mask_path = os.path.join(mask_dir, file)
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)