import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
device = torch.device("cpu")

def train(model, loader, epochs=5, lr=1e-3):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

def evaluate(model, loader):
    model.eval()
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.view(-1).cpu().numpy().tolist())

    # Check lengths to avoid mismatch
    print("Number of predictions:", len(all_preds))
    print("Number of labels:", len(all_labels))

    if len(all_preds) != len(all_labels):
        print("Warning: Mismatch in prediction and label counts")

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, cm