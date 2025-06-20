import os
import re

def build_labels_txt(root_dir='labeled_data',
                     output_file='predictions.txt'):
    label_map = {'inside': 0, 'outside': 1}
    entries = []

    for subdir, label in label_map.items():
        folder = os.path.join(root_dir, subdir)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if os.path.isfile(path):
                entries.append((fname, label))

    def sort_key(item):
        name, _ = item
        stem = os.path.splitext(name)[0]
        m = re.search(r'(\d+)$', stem)
        return int(m.group(1)) if m else float('inf')

    entries.sort(key=sort_key)

    with open(output_file, 'w') as f:
        for _, label in entries:
            f.write(f"{label}\n")

    print(f"Wrote {len(entries)} labels to {output_file}")

if __name__ == "__main__":
    build_labels_txt()