import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from albumentations import ToTensorV2


class CVC_CliniCDBDataset(Dataset):
    def __init__(self, config_path, phase="train"):
        self.config_path = config_path
        self.phase = phase

        # Load config từ file .py
        self._load_config(config_path)

        # Load dữ liệu
        self.image_paths = []
        self.mask_paths = []
        self.__load_data()
        self.to_tensor = ToTensorV2()

    def _load_config(self, config_path):
        """
        Đọc nội dung file config bằng exec() mà không dùng import
        """
        with open(config_path, "r") as f:
            code = compile(f.read(), config_path, "exec")
            scope = {}
            exec(code, scope)  # Thực thi toàn bộ script Python trong scope dict

        # Lấy các giá trị từ scope
        self.source = scope.get("DATASET_SOURCE")
        self.image_size = scope.get("IMAGE_SIZE", (256, 256))
        self.normalize = scope.get("NORMALIZE", True)
        self.transforms = scope.get("TRANSFORM_PIPELINE")

    def __load_data(self, exts=[".jpg", ".png"]):
        # Đường dẫn đến thư mục images và masks theo phase (train/val)
        image_dir = os.path.join(self.source, self.phase, "images")
        mask_dir = os.path.join(self.source, self.phase, "masks")

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Đọc ảnh và mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển BGR sang RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Đọc mask ở grayscale

        # Áp dụng transforms nếu có
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Normalize ảnh
        if self.normalize:
            image = (image / 127.5) - 1.0
        else:
            image = image / 255.0

        # Chuyển đổi mask về định dạng nhị phân
        mask = mask / 255.0

        # Chuyển sang tensor
        image = self.to_tensor(image=image)["image"]
        mask = self.to_tensor(image=mask)["image"]
        
        return image, mask


if __name__ == "__main__":
    dataset = CVC_CliniCDBDataset(config_path="data/configs/CVC_ClinicDB_config.py")
    print(f"Number of images: {len(dataset)}")
    # print("Transform pipeline:", dataset.transforms)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    images, masks = next(iter(dataloader))
    for im0, mask in zip(images, masks):
        print(f"Image shape: {im0.shape}, Mask shape: {mask.shape}")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))

        # Hiển thị ảnh gốc
        plt.subplot(1, 2, 1)
        plt.title("Image")
        # Đảo ngược normalize để hiển thị ảnh
        im_display = (im0.permute(1, 2, 0).numpy() + 1.0) * 127.5
        im_display = im_display.astype(np.uint8)
        plt.imshow(im_display)  # Đã ở RGB, không cần chuyển đổi thêm
        plt.axis("off")

        # Hiển thị mask
        plt.subplot(1, 2, 2)
        plt.title("Mask")
        mask_display = mask.squeeze().numpy()  # Loại bỏ chiều channel nếu có
        plt.imshow(mask_display, cmap="gray")  # Hiển thị mask dưới dạng grayscale
        plt.axis("off")

        plt.show()
    print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")