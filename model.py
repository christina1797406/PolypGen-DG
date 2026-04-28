import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, confusion_matrix

# Model + training + evaluation
def get_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

def train(model, train_loader, epochs=1, lr=0.001):
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        # Training loop X=images, y=labels
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
def evaluate(model, loader):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return acc, cm