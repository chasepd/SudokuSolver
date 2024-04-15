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



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * 125 * 125, 800)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

print("Creating the encoder model...")
encoder = Encoder().to(device)

print("Loading the encoder model weights")
# Load the trained encoder weights
encoder.load_state_dict(torch.load('models/encoder_model.pth'))

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

class Solver(nn.Module):
    def __init__(self, encoder):
        super(Solver, self).__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(800, 256),
            nn.ReLU(),
            nn.Linear(256, 81*10)  #81 cells, each with 10 possible class outputs
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

print("Loading dataset...")
transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])

# Load the dataset
dataset = ImageTextDataset('data/train/imgs', 'data/train/text', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Dataset loaded")

# Load the validation dataset
validation_dataset = ImageTextDataset('data/validation/imgs', 'data/validation/text', transform)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
print("Validation dataset loaded")

print("Creating the solver...")
# Initialize the solver
solver_path = 'models/solver_model.pth'
encoder = Encoder().to(device)
solver = Solver(encoder).to(device)
if os.path.isfile(solver_path):
    print("Loading saved solver model to continue training...")
    solver.load_state_dict(torch.load(solver_path))
print(f"Solver initialized on {device}")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(solver.parameters(), lr=0.001)

# Train the model
print("Training the model...")
num_epochs = 15
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_batches = len(dataloader)
    solver.train()
    for batch_index, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = solver(images)  # [batch_size, 81, 10]
        loss = criterion(outputs.view(-1, 10), labels.view(-1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the updating percentage
        percentage = (batch_index / total_batches) * 100
        print(f'\rProgress: {percentage:.2f}%', end='')
    # Validation loop
    solver.eval()
    with torch.no_grad():
        total_val_loss = 0
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = solver(images)
            val_loss = criterion(outputs.view(-1, 10), labels.view(-1))
            total_val_loss += val_loss.item()
    print()

    average_val_loss = total_val_loss / len(validation_loader)
    # Print the loss after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] complete, Loss: {loss.item():.4f}, Val Loss: {average_val_loss:.4f}")
    torch.save(solver.state_dict(), f'models/solver_model.pth')

print("Model training complete.")
