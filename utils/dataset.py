import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import os


class ImageDataset(Dataset):
    def __init__(self, images_dir: str, csv_path: str, transform: transforms = None):
        if not os.path.isfile(csv_path):
            create_csv(images_dir, csv_path)
        self.dataset_df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataset_df.iloc[idx, 0]

        image = Image.fromarray(io.imread(img_path))

        if self.transform:
            image = self.transform(image)
        noise = torch.normal(
            image.mean().item(), image.std().item() * 2, size=image.shape
        )
        sample = (noise, image)
        return sample


def create_csv(images_dir: str, csv_path: str) -> int:
    image_paths = []
    for root, dirs, imgs in os.walk(images_dir):
        if len(imgs) == 0:
            continue
        for img in imgs:
            img_path = os.path.join(root, img)
            image_paths.append(img_path)
    csv_data = {"ImagePaths": image_paths}
    csv_df = pd.DataFrame(csv_data)
    print(f"CSV file saved at: {csv_path}")
    csv_df.to_csv(csv_path, index=False)
    return len(csv_df)
