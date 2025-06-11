import os
from .processing_CVC import BaseDataset
from albumentations import ToTensorV2
from sklearn.model_selection import train_test_split
from .processing_CVC import BaseDataset


class DataBenchmark(BaseDataset):
    def __init__(self, config_path, phase="val"):
        self.config_path = config_path
        self.phase = phase
        # Load config từ file .py
        self._load_config(config_path)

        # Load dữ liệu
        self.image_paths = []
        self.mask_paths = []
        self.__load_data()
        self.to_tensor = ToTensorV2()

    def __load_data(self, exts=[".jpg", ".png", ".tif"]):
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

    def split_data(self, train_ratio=0.8, seed=0):
        # Tạo danh sách chỉ số và trộn ngẫu nhiên
        train_images, val_images, train_masks, val_masks = train_test_split(
            self.image_paths,
            self.mask_paths,
            test_size=1 - train_ratio,
            random_state=seed,
        )
        train_dataset = DataBenchmark(self.config_path, phase="train")
        val_dataset = DataBenchmark(self.config_path, phase="val")
        train_dataset.image_paths = train_images
        train_dataset.mask_paths = train_masks
        val_dataset.image_paths = val_images
        val_dataset.mask_paths = val_masks
        return train_dataset, val_dataset
    
    def merge_data(self, dataset:BaseDataset):
        self.image_paths.extend(dataset.image_paths)
        self.mask_paths.extend(dataset.mask_paths)
        return self

    def subset(self, indices, phase="train"):
        subset_dataset = DataBenchmark(self.config_path, phase=phase)
        subset_dataset.image_paths = [self.image_paths[i] for i in indices]
        subset_dataset.mask_paths = [self.mask_paths[i] for i in indices]
        return subset_dataset
