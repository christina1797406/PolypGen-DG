import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

from dataset import CentreDataset
from model import get_model
from train import train, evaluate

DATA_ROOT = "data"
CENTRES = ["centre_A", "centre_B", "centre_C"]
BATCH_SIZE = 32
EPOCHS = 10

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

aug_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

def run_experiment(transform, label):
    print(f"\n=== Experiment: {label} Model | Epochs: {EPOCHS} ===")

    dataset = CentreDataset(DATA_ROOT, CENTRES, transform=transform)

    # Split: Train on A only
    train_idx = [i for i, c in enumerate(dataset.centres) if c == "centre_A"]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = get_model()
    train(model, train_loader, epochs=EPOCHS)

    results = {}

    # Test on A, B, C
    for centre in CENTRES:
        test_idx = [i for i, c in enumerate(dataset.centres) if c == centre]

        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=BATCH_SIZE
        )

        acc, cm = evaluate(model, test_loader)

        results[centre] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "n_samples": len(test_idx)
        }

        print(f"\nTest on {centre}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix: (rows=true, cols=pred)")
        print(cm)
        
    return results

def compute_drop(results):
    acc_A = results["centre_A"]["accuracy"]

    for centre in ["centre_B", "centre_C"]:
        acc = results[centre]["accuracy"]
        drop = acc_A - acc

        print(f"Drop A → {centre}: {drop:.4f}")

if __name__ == "__main__":

    # Baseline
    results_base = run_experiment(base_transform, "Baseline")
    print("\n=== Baseline Performance Drop ===")
    compute_drop(results_base)

    # Augmented
    results_aug = run_experiment(aug_transform, "Augmented")
    print("\n=== Augmented Performance Drop ===")
    compute_drop(results_aug)