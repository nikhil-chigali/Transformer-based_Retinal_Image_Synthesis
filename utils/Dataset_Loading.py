import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image
import os
class ImageDataset(Dataset):
    def __init__(self, csv_path: str, transform: transforms = None):
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
        sample = {"X": image}
        return sample

def create_csv(images_dir: str, csv_path: str) -> int:
    image_paths = []
    for root, dirs, imgs in os.walk(images_dir):
        if len(imgs) == 0:
            continue
        for img in imgs:
            img_path = os.path.join(root, img)
            image_paths.append(img_path)
    test_data = {"ImagePaths": image_paths}
    test_df = pd.DataFrame(test_data)
    full_csv_path = os.path.abspath(csv_path)
    print(f"CSV file saved at: {full_csv_path}")
    test_df.to_csv(csv_path, index=False)
    return len(test_df)

# Set the path to the directory containing images - need to edit the paths accordingly
images_directory = "Data/Images/"
# Set the desired CSV file name
csv_file_path = "Image_data.csv"

# Create the CSV file
create_csv(images_directory, csv_file_path)

# Read data from the CSV file - for now I kept 244,488- we need to edit this
transform = transforms.Compose([transforms.Resize((244, 488)), transforms.ToTensor()])
dataset = ImageDataset(csv_path=csv_file_path, transform=transform)
#I have choosen batch size to be 4
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Print the first three batches ka  images
for i, batch in enumerate(loader):
    if i >= 3:
        break
    images = batch["X"]
    print(f"Batch {i+1}, Shape: {images.shape}")
    for j in range(images.shape[0]):
        image = transforms.ToPILImage()(images[j]).convert("RGB")
        image.show()