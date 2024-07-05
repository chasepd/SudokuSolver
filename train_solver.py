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

class SudokuDataset(Dataset):
    def __init__(self, unsolved_dir, solved_dir, transform=None):
        self.unsolved_dir = unsolved_dir
        self.solved_dir = solved_dir
        self.transform = transform
        self.unsolved_filenames = [f for f in os.listdir(unsolved_dir) if f.endswith('.txt')]
        
    def __len__(self):
        return len(self.unsolved_filenames)
    
    def __getitem__(self, idx):
        unsolved_basename = self.unsolved_filenames[idx]
        unsolved_filename = os.path.join(self.unsolved_dir, unsolved_basename)
        solved_filename = os.path.join(self.solved_dir, unsolved_basename)

        with open(unsolved_filename, 'r') as file:
            unsolved_text_data = file.read().strip().replace('.', '0').split()
            unsolved_text_tensor = torch.tensor([int(num) for num in unsolved_text_data], dtype=torch.long).view(81)
        
        with open(solved_filename, 'r') as file:
            solved_text_data = file.read().strip().replace('.', '0').split()
            solved_text_tensor = torch.tensor([int(num) for num in solved_text_data], dtype=torch.long).view(81)
        
        return unsolved_text_tensor, solved_text_tensor

class Solver(nn.Module):
    def __init__(self):
        super(Solver, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(10, 32)  # Assuming 0-9 numbers in Sudoku, embedding size is 32
        
        # Transformer encoder layers with batch_first set to True
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Fully connected layers
        self.fc1 = nn.Linear(81 * 32, 512)
        self.fc2 = nn.Linear(512, 81 * 10)  # Output size is 81 * 10 (each cell has 10 possible values: 0-9)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, 81, 32]
        x = self.transformer_encoder(x)  # [batch_size, 81, 32]
        x = x.view(x.size(0), -1)  # [batch_size, 81 * 32]
        x = torch.relu(self.fc1(x))  # [batch_size, 512]
        x = self.fc2(x)  # [batch_size, 81 * 10]
        x = x.view(-1, 81, 10)  # [batch_size, 81, 10]
        return x

print("Loading dataset...")

# Load the dataset
dataset = SudokuDataset('data/train/text/unsolved', 'data/train/text/solved')
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
print("Dataset loaded")

# Load the validation dataset
validation_dataset = SudokuDataset('data/validation/text/unsolved', 'data/validation/text/solved')
validation_loader = DataLoader(validation_dataset, batch_size=2048, shuffle=False)
print("Validation dataset loaded")

print("Creating the solver...")
# Initialize the solver
solver_path = 'models/solver_model.pth'
solver = Solver().to(device)
try:
    if os.path.isfile(solver_path):
        print("Loading saved solver model to continue training...")
        solver.load_state_dict(torch.load(solver_path))
except:
    print(f"Error loading model, initializing new model...")
    pass
print(f"solver initialized on {device}")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(solver.parameters(), lr=0.0001)

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
            print(f'\rValidation Progress: {percentage:.2f}%', end='')
    print()

    average_val_loss = total_val_loss / total_validation_batches
    # Print the loss after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] complete, Loss: {loss.item():.4f}, Val Loss: {average_val_loss:.4f}")
    torch.save(solver.state_dict(), f'models/solver_model.pth')

print("Model training complete.")
