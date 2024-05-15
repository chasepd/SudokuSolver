import torch

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class ImageTextDataset(Dataset):
    def __init__(self, img_dir, text_dir, transform=None):
        self.img_dir = img_dir
        self.text_dir = text_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        text_path = os.path.join(self.text_dir, img_name.replace('.png', '.txt'))
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        with open(text_path, 'r') as file:
            text_data = file.read().strip().replace('.', '0').split()
            text_tensor = torch.tensor([int(num) for num in text_data], dtype=torch.long).view(81)  # Flatten the grid
        
        return image, text_tensor

class Reader(nn.Module):
    def __init__(self):
        super(Reader, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Dummy input to calculate flat features
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 500, 500)  # Input image size is 500x500
            dummy_output = self.features(dummy_input)
            flat_features = dummy_output.view(-1).shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flat_features, 800)
        self.fc2 = nn.Linear(800, 5000)
        self.fc3 = nn.Linear(5000, 81*10)

    def features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x



print("Loading dataset...")
transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])

# Load the dataset
dataset = ImageTextDataset('data/train/imgs/class', 'data/train/text', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Dataset loaded")

# Load the validation dataset
validation_dataset = ImageTextDataset('data/validation/imgs/class', 'data/validation/text', transform)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
print("Validation dataset loaded")

print("Creating the reader...")
# Initialize the reader
reader_path = 'models/reader_model.pth'
reader = Reader().to(device)
try:
    if os.path.isfile(reader_path):
        print("Loading saved reader model to continue training...")
        reader.load_state_dict(torch.load(reader_path))
except:
    print(f"Error loading model, initializing new model...")
    pass
print(f"reader initialized on {device}")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(reader.parameters(), lr=0.0001)

# Verify models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

total_batches = len(dataloader)
total_validation_batches = len(validation_loader)

# Train the model
print("Training the model...")
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    reader.train()
    for batch_index, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = reader(images)  # [batch_size, 81, 10]
        loss = criterion(outputs.view(-1, 10), labels.view(-1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the updating percentage
        percentage = (batch_index / total_batches) * 100
        print(f'\rProgress: {percentage:.2f}%', end='')
    # Validation loop
    
    reader.eval()
    with torch.no_grad():
        total_val_loss = 0
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = reader(images)
            val_loss = criterion(outputs.view(-1, 10), labels.view(-1))
            total_val_loss += val_loss.item()
            print(f'\rValidation Progress: {percentage:.2f}%', end='')
    print()

    average_val_loss = total_val_loss / total_validation_batches
    # Print the loss after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] complete, Loss: {loss.item():.4f}, Val Loss: {average_val_loss:.4f}")
    torch.save(reader.state_dict(), f'models/reader_model.pth')

print("Model training complete.")
