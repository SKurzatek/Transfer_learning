from torchvision import datasets, transforms
from torchsummary import summary

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import tqdm
import sys
import os
import shutil

import utils.CNN_components as components
from utils.CNN_dataset import DynamicResolutionImageDataset

choose_model = "2025-06-09_16-41-41"

class_list = ['inside', 'outside']
  
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.init_block = components.InitBlock(out_channels = 64)
        self.blocks = torch.nn.ModuleList([
            components.Module(
                        conv_blocks_number = 1,
                        in_channels = 64, 
                        internal_channels = 64,
                        out_channels = 64,
                        bypass = True,
                        max_pool = True,
                        batch_norm = True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 1,
                        in_channels = 64, 
                        internal_channels = 64,
                        out_channels = 64,
                        bypass = True,
                        max_pool = True,
                        batch_norm = True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 1,
                        in_channels = 64, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = True,
                        max_pool = False,
                        batch_norm = True,
                        dropout = False
                    )
        ]) 
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.init_block(x)
        # this relu does nothing ?
        #x = torch.nn.functional.relu(x)
        for it in self.blocks:
            x = it(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class DynamicResolutionImageDatasetWithPath(DynamicResolutionImageDataset):
    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        path = self.filepaths[idx]
        return image, path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = DynamicResolutionImageDatasetWithPath(
            root_dir="./test",
            class_list=class_list,
            target_size=(128, 128),
            train = False
        )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 50, shuffle=False, num_workers=4, pin_memory=True)
    
    test_dataset_size = len(test_dataset)

    model = Network()
    model_path = "./models/" + choose_model + "/model.pth"
    
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    #summary(model, (3, 32, 32))
    
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for images, paths in tqdm.tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            for p, path in zip(preds, paths):
                predictions.append((path, class_list[p]))

    output_root = "./labeled_data"
    for label in class_list:
        os.makedirs(os.path.join(output_root, label), exist_ok=True)

    for src_path, pred_label in tqdm.tqdm(predictions):
        dst_dir = os.path.join(output_root, pred_label)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

    print(f"All images sorted under {output_root}/{{{', '.join(class_list)}}}")

if __name__ == "__main__":
    main()
    