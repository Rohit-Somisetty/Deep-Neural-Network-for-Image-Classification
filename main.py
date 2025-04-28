import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['cats', 'dogs']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input size: 150x150x3
        # After first conv: 148x148x32
        # After first pool: 74x74x32
        # After second conv: 72x72x64
        # After second pool: 36x36x64
        # After third conv: 34x34x128
        # After third pool: 17x17x128
        # After fourth conv: 15x15x128
        # After fourth pool: 7x7x128
        # Flatten: 7*7*128 = 6272
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 128 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

def organize_dataset():
    """Organize the dataset into train, validation, and test directories."""
    # Create directories if they don't exist
    base_dir = Path('.')
    for dir_name in ['train', 'validation', 'test']:
        for class_name in ['cats', 'dogs']:
            (base_dir / dir_name / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process train directory
    train_dir = base_dir / 'train'
    for img_path in train_dir.glob('*.jpg'):
        if 'cat' in img_path.name.lower():
            shutil.copy2(img_path, train_dir / 'cats' / img_path.name)
        elif 'dog' in img_path.name.lower():
            shutil.copy2(img_path, train_dir / 'dogs' / img_path.name)
    
    # Process test directory
    test_dir = base_dir / 'test'
    for img_path in test_dir.glob('*.jpg'):
        if 'cat' in img_path.name.lower():
            shutil.copy2(img_path, test_dir / 'cats' / img_path.name)
        elif 'dog' in img_path.name.lower():
            shutil.copy2(img_path, test_dir / 'dogs' / img_path.name)
    
    # Split training data into train, validation, and test
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    for class_name in ['cats', 'dogs']:
        # Get all images for this class
        class_dir = train_dir / class_name
        images = list(class_dir.glob('*.jpg'))
        np.random.shuffle(images)
        
        # Calculate split sizes
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split the images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Move validation images
        for img_path in val_images:
            shutil.move(str(img_path), str(base_dir / 'validation' / class_name / img_path.name))
        
        # Move test images
        for img_path in test_images:
            shutil.move(str(img_path), str(base_dir / 'test' / class_name / img_path.name))

def train_model():
    """Train the CNN model using PyTorch."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CatsDogsDataset('train', transform=train_transform)
    val_dataset = CatsDogsDataset('validation', transform=val_transform)
    test_dataset = CatsDogsDataset('test', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
    
    # Create model
    model = CNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 30
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                
                running_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_accs, 'bo-', label='Training acc')
    plt.plot(range(1, num_epochs+1), val_accs, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_losses, 'bo-', label='Training loss')
    plt.plot(range(1, num_epochs+1), val_losses, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f'Test accuracy: {test_acc:.2f}%')
    
    # Save the model
    torch.save(model.state_dict(), 'cats_vs_dogs_cnn.pth')
    return model

def predict_image(model, image_path):
    """
    Make a prediction for a single image using the trained model.
    
    Args:
        model: The trained CNN model
        image_path: Path to the image file
    
    Returns:
        tuple: (prediction, confidence)
            - prediction: 'cat' or 'dog'
            - confidence: probability score (0-1)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Define the same transform used during training
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        probability = output.item()
        prediction = 'dog' if probability > 0.5 else 'cat'
        confidence = probability if prediction == 'dog' else 1 - probability
    
    return prediction, confidence

def main():
    """Main function to run the entire pipeline."""
    print("Step 1: Organizing dataset...")
    organize_dataset()
    
    print("\nStep 2: Training model...")
    model = train_model()
    
    print("\nTraining completed! Results saved in 'training_results.png'")
    print("Model saved as 'cats_vs_dogs_cnn.pth'")

if __name__ == '__main__':
    main() 