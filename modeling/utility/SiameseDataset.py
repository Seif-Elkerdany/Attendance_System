import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

class RandomAffineWithSameParams:
    """Custom affine transform for grayscale images."""
    def __init__(self, degrees=10, translate=0.05):
        self.degrees = degrees
        self.translate = translate

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        max_dx = img.shape[2] * self.translate
        max_dy = img.shape[1] * self.translate
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        return TF.affine(img, angle=angle, translate=(tx, ty), scale=1.0, shear=0)

class RandomBrightnessContrast:
    """Custom brightness and contrast adjustment for grayscale images."""
    def __init__(self, brightness=0.3, contrast=0.3):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        return img

class SiameseDataset(Dataset):
    def __init__(self, data_list, train=True):
        """
        Args:
            data_list (list of tuples): List of (img1_path, img2_path, label) tuples
            train (bool): Whether to apply training augmentations (default=True)
        """
        self.data = data_list
        self.train = train
        
        # Define a simple augmentation pipeline for grayscale images
        # This pipeline will be applied with a probability of 0.3
        self.augment = T.RandomApply(
           T.Compose([
               RandomAffineWithSameParams(), 
               RandomBrightnessContrast(brightness=0.3, contrast=0.3),
               T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
           ]),
           p=0.3
       )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img1_path, img2_path, label = self.data[index]

        # Load and preprocess images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        # Apply augmentations if training
        if self.train:
            if label == 1:  # Positive pair: apply same augmentations to both images
                seed = random.randint(0, 2**32)
                img1 = self._apply_augmentations(img1, seed)
                img2 = self._apply_augmentations(img2, seed)
            else:  # Negative pair: apply different augmentations independently
                img1 = self._apply_augmentations(img1)
                img2 = self._apply_augmentations(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def _load_image(self, path):
        """Load and preprocess a single image."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        return TF.to_tensor(img)

    def _apply_augmentations(self, img, seed=None):
        """Apply augmentations with an optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        return self.augment(img)
