import os
from PIL import Image, ImageFile
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

# Suppress EXIF warnings
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_DIR = "dataset"
LABELS = ["inside", "outside"]
BUCKETS = [
    ("<0.5 MP", lambda w, h: (w * h) < 0.5 * 1_000_000),
    ("0.5–1 MP", lambda w, h: 0.5 * 1_000_000 <= (w * h) < 1.0 * 1_000_000),
    ("1–2 MP",   lambda w, h: 1.0 * 1_000_000 <= (w * h) < 2.0 * 1_000_000),
    ("2–4 MP",   lambda w, h: 2.0 * 1_000_000 <= (w * h) < 4.0 * 1_000_000),
    ("4–8 MP",   lambda w, h: 4.0 * 1_000_000 <= (w * h) < 8.0 * 1_000_000),
    ("8+ MP",    lambda w, h: (w * h) >= 8.0 * 1_000_000),
]

def bucket_resolution(width, height):
    for label, condition in BUCKETS:
        if condition(width, height):
            return label
    return "unknown"

def collect_resolution_buckets_split():
    bucket_counts = defaultdict(lambda: {"inside": 0, "outside": 0})

    for label in LABELS:
        subdir = os.path.join(DATASET_DIR, label)
        for fname in os.listdir(subdir):
            fpath = os.path.join(subdir, fname)
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                    bucket = bucket_resolution(w, h)
                    bucket_counts[bucket][label] += 1
            except Exception as e:
                print(f"[ERROR] Skipping {fpath}: {e}")
    return bucket_counts

def plot_resolution_buckets_split(bucket_counts):
    bucket_labels = [b[0] for b in BUCKETS]
    inside_counts = [bucket_counts[b]["inside"] for b in bucket_labels]
    outside_counts = [bucket_counts[b]["outside"] for b in bucket_labels]

    x = range(len(bucket_labels))
    bar_width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x, inside_counts, width=bar_width, label="Inside", color="steelblue")
    plt.bar([i + bar_width for i in x], outside_counts, width=bar_width, label="Outside", color="salmon")

    plt.xlabel("Resolution Bucket (Megapixels)")
    plt.ylabel("Number of Images")
    plt.title("Image Count by Resolution Bucket (Inside vs Outside)")
    plt.xticks([i + bar_width / 2 for i in x], bucket_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig("resolution_buckets_split.png", dpi=200)
    print("Saved resolution_buckets_split.png")

class ImageStatsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_list, transform=None):
        self.filepaths = []
        for label in class_list:
            subdir = os.path.join(root_dir, label)
            for fname in os.listdir(subdir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.filepaths.append(os.path.join(subdir, fname))
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def compute_mean_std(dataset_dir="dataset", class_list=["inside", "outside"], resize=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor()  # Converts to [0, 1] and channels-first (C, H, W)
    ])

    dataset = ImageStatsDataset(dataset_dir, class_list, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_pixels = 0

    print("Computing mean and std ...")
    for batch in tqdm(loader):
        batch = batch.view(batch.size(0), batch.size(1), -1)  # (B, C, H*W)
        n_pixels += batch.size(0) * batch.size(2)
        mean += batch.sum(dim=(0, 2))
        std += (batch ** 2).sum(dim=(0, 2))

    mean /= n_pixels
    std = (std / n_pixels - mean ** 2).sqrt()

    print("\n Dataset mean:", mean.tolist())
    print("Dataset std: ", std.tolist())
    return mean, std

if __name__ == "__main__":
    counts = collect_resolution_buckets_split()

    print("\n--- Resolution Bucket Counts (Inside vs Outside) ---")
    for bucket in [b[0] for b in BUCKETS]:
        i = counts[bucket]["inside"]
        o = counts[bucket]["outside"]
        print(f"{bucket:10}: Inside = {i:4} | Outside = {o:4}")
    compute_mean_std()
    plot_resolution_buckets_split(counts)
