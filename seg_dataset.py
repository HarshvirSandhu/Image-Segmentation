import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader


class Human_dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.images[item])
        mask_path = os.path.join(self.mask_dir, self.images[item])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1

        if self.transforms is not None:
            augmentation = self.transforms(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask


img_path = "D:/segmentation dataset/supervisely_person_clean_2667_img/images"
mask_path = "D:/segmentation dataset/supervisely_person_clean_2667_img/masks/ds6_man-person-hat-fur.png"


def load_data(train_dir, train_mask_dir, batch_size, transform):
    train_data = Human_dataset(img_dir=train_dir,
                               mask_dir=train_mask_dir,
                               transforms=transform)

    train_loaded = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loaded

from PIL import Image
import cv2

i = np.array(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2GRAY), dtype=np.float32)
# i = i.reshape((i.shape[0], i.shape[1], 1))
print(i.shape)
