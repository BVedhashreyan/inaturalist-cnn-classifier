import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from model import CNN
import os

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(images)

        # loss calc
        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()


        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0) 
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0) 
            running_loss += loss.item()
        
    val_loss = running_loss / len(val_loader)

    val_acc = correct / total

    return val_loss, val_acc

def main():
    model = CNN(filters=[32,64,128,64,32])

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    model = model.to(device)

    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-4)
    
    if os.path.exists("/content"):
        train_path = "/content/data/inaturalist_12K/train"
        val_path = "/content/data/inaturalist_12K/val"
    else:
        train_path = "data/inaturalist_12K/train"
        val_path = "data/inaturalist_12K/val"

    train_loader, val_loader, test_loader = get_loaders(train_path, val_path, batch_size=32)

    num_epochs = 3
    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "outputs/best_model.pth"
            )

            print("Best model saved")

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss : {train_loss} , Train acc : {train_acc} , Val loss : {val_loss} , Val Acc : {val_acc}")
    

if __name__ == "__main__":
    main()