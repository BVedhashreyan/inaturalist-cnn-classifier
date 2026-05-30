import os
import torch

from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset


generator = torch.Generator().manual_seed(42)
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

default_batch_size = 32

class INaturalistDataset(Dataset):
    def __init__(self, data_path, transform = None):
        self.transform = transform
        
        self.cls_to_idx = {}

        # creating class to index
        classes = sorted(os.listdir(data_path))

        for idx, cls in enumerate(classes):
            self.cls_to_idx[cls] = idx

        self.num_classes = len(self.cls_to_idx)

        self.samples = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(image_extensions):

                    cls_name = os.path.basename(root)
                    cls_idx = self.cls_to_idx[cls_name]
                    self.samples.append((os.path.join(root,file), cls_idx))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

def get_loaders(train_path, test_path, batch_size = default_batch_size, transform = default_transform):
    full_dataset = INaturalistDataset(
        train_path,
        transform=transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator= generator
    )

    test_dataset = INaturalistDataset(
        test_path,
        transform=transform
    )

    # creating loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader