from torchvision import datasets, transforms
from torchsummary import summary

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import tqdm
import sys

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = DynamicResolutionImageDataset(
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
    correctly_predicted = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            device_images = images.to(device)
            device_labels = labels.long().to(device)

            outputs = model(device_images)
            predictions = torch.argmax(outputs, dim=1)

            correctly_predicted += (predictions == device_labels).sum().item()

            # Save for confusion matrix
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(device_labels.cpu().numpy())

    print(f"Accuracy: {correctly_predicted / test_dataset_size:.4f}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_list, yticklabels=class_list)
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save to path
    plt.savefig("./conf_mat/matrix.png")
    plt.close()

    with open('./conf_mat/output.txt', 'w') as f:
        for it in all_preds:
            f.write(f"{it}\n")

if __name__ == "__main__":
    main()
    