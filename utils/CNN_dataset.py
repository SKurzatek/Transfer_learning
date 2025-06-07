import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DynamicResolutionImageDataset(Dataset):
    def __init__(self, root_dir, class_list, target_size=(512, 512)):
        self.root_dir = root_dir
        self.class_list = class_list
        self.target_size = target_size
        self.filepaths = []
        self.labels = []

        for label_index, label_name in enumerate(class_list):
            class_path = os.path.join(root_dir, label_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.filepaths.append(os.path.join(class_path, file))
                        self.labels.append(label_index)

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.CenterCrop(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.483, 0.477, 0.447],
                                 std=[0.275, 0.274, 0.297])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label