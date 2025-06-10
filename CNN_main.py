import torch
import time
import tqdm
import copy
import shutil
import os
import datetime 

from torchsummary import summary
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from utils.CNN_dataset import DynamicResolutionImageDataset
import utils.CNN_components as components

class_list = ["inside", "outside"]

def save_txt(history, file_name):
    with open(file_name, "w") as f:
        for it in history:
            f.write(str(it) + " ")

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
        x = torch.nn.functional.relu(x)
        for it in self.blocks:
            x = it(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 25
    F1_history = [0 for _ in range(epochs)]
    loss_history = [0 for _ in range(epochs)]
    time_history = [0 for _ in range(epochs)]

    train_dataset = DynamicResolutionImageDataset(
            root_dir="./dataset/train",
            class_list=class_list,
            target_size=(128, 128),
            train = True
        )
    
    test_dataset = DynamicResolutionImageDataset(
            root_dir="./dataset/valid",
            class_list=class_list,
            target_size=(128, 128),
            train = False
        )

    train_loader = DataLoader(train_dataset, 
                                batch_size = 64, 
                                shuffle=True, 
                                num_workers=4, 
                                pin_memory=True,
                                prefetch_factor=2,
                                persistent_workers=True
    )
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False, num_workers=4, pin_memory=True)
    
    #test_dataset_size = len(test_dataset)

    model = Network()
    model.to(device)

    class_weights = torch.tensor([1.5, 1.0], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    best_F1 = 0

    for epoch in range(epochs):
        print(f"Using device: {device}")
        start_time = time.time()

        model.train()
        scaler = GradScaler()
        total_loss = 0
        
        for images, labels in tqdm.tqdm(train_loader):
            device_samples, device_labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(device_samples)
                loss = criterion(outputs, device_labels)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
                    
        end_time = time.time()

        epoch_f1 = components.evaluate_f1_score(model=model, test_loader=test_loader, device=device)
        F1_history[epoch] = epoch_f1
        loss_history[epoch] = total_loss
        time_history[epoch] = end_time - start_time

        print(f"Time for testing: {time.time() - end_time}")
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, F1_scrore: {epoch_f1}, Time: {end_time - start_time}s")
        #print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, Time: {end_time - start_time}s")
        print()
        
        if(epoch_f1>best_F1):
            torch.save(copy.deepcopy(model.state_dict()), f"./models/checkpoint/model.pth")
            torch.save(optimizer.state_dict(), f"./models/checkpoint/optimizer.pth")
            best_F1 = epoch_f1

    print("F1_score:")
    print(F1_history)
    print("Loss:")
    print(loss_history)
    print("Time:")
    print(time_history)
    print()

    print("Saving history data to history.txt")
    save_txt(F1_history, "./training_data/F1_history.txt")
    save_txt(loss_history, "./training_data/loss_history.txt")
    save_txt(time_history, "./training_data/time_history.txt")
    print(f"Data gathered. Training performed succesfully.")

    # Automatically create timestamped model directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join("./models", timestamp)
    os.makedirs(model_dir, exist_ok=True)

    # Move files with standardized names
    shutil.move("./models/checkpoint/model.pth", os.path.join(model_dir, "model.pth"))
    shutil.move("./models/checkpoint/optimizer.pth", os.path.join(model_dir, "optimizer.pth"))

    print(f"Best model saved to {model_dir}")

    
if __name__ == "__main__":
    main()