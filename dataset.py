import os
from torch.utils.data import Dataset
from PIL import Image

class CentreDataset(Dataset):
    def __init__(self, root, centres, transform=None):
        self.samples = []
        self.labels = []
        self.centres = []
        self.transform = transform

        for centre in centres:
            centre_path = os.path.join(root, centre)

            for cls in ["positive", "negative"]:
                cls_path = os.path.join(centre_path, cls)

                if not os.path.exists(cls_path):
                    continue

                label = 1 if cls == "positive" else 0

                # Handle both flat and sequence-based folders
                for item in os.listdir(cls_path):
                    item_path = os.path.join(cls_path, item)

                    if os.path.isdir(item_path):
                        # Sequence folder
                        for img in os.listdir(item_path):
                            if img.endswith(".jpg"):
                                self.samples.append(os.path.join(item_path, img))
                                self.labels.append(label)
                                self.centres.append(centre)
                    else:
                        if item.endswith(".jpg"):
                            self.samples.append(item_path)
                            self.labels.append(label)
                            self.centres.append(centre)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]