import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import torch
import torch.nn.functional as F

class SEGDataset(Dataset):
    def __init__(self, txt_file, img_size, normalize=True):
        self.txt_file = txt_file
        self.img_size = img_size
        self.normalize = normalize
        self.image_paths = []
        self.mask_paths = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name = line.strip()
                self.image_paths.append(os.path.join('Kvasir-SEG', 'images', img_name))
                self.mask_paths.append(os.path.join('Kvasir-SEG', 'masks', img_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load and normalize image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size[0], self.img_size[1]))
        img = np.array(img)
        if self.normalize:
            img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
        else:
            img = img / 255.0  # Optional: Normalize to [0, 1]
        
        # Load and normalize mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size[0], self.img_size[1]))
        mask = np.array(mask)
        mask = mask / 255.0  # Normalize mask to [0, 1]
        
        # Convert mask to 3 channels
        mask = np.stack([mask]*3, axis=-1)
        
        # Apply augmentations
        img, mask = self.augment(img, mask)
        
        return torch.tensor(img).float(), torch.tensor(mask).float()
    
    def augment(self, image, mask):
        # Apply random zoom
        zoom_factor = random.uniform(0.8, 1.2)
        new_height = int(image.shape[0] * zoom_factor)
        new_width = int(image.shape[1] * zoom_factor)
        if new_height < 256 or new_width < 256:
            new_height, new_width = 256, 256
        image_zoomed = F.interpolate(torch.tensor(image).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze()
        mask_zoomed = F.interpolate(torch.tensor(mask).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze()
        
        # Apply random crop
        crop_size = min(new_height, new_width)
        crop_size = min(crop_size, 256)
        crop = transforms.RandomCrop(crop_size)
        image_zoomed, mask_zoomed = crop(image_zoomed), crop(mask_zoomed)
        
        # Apply intensity transformations
        brightness_factor = random.uniform(0.8, 1.2)
        image_zoomed = transforms.ColorJitter(brightness=brightness_factor)(image_zoomed)
        contrast_factor = random.uniform(0.8, 1.2)
        image_zoomed = transforms.ColorJitter(contrast=contrast_factor)(image_zoomed)
        
        return image_zoomed, mask_zoomed