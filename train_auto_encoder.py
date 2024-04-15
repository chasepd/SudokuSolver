import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
# Define the encoder model
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

# Define the decoder model
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(800, 32*125*125),
            nn.Unflatten(1, (32, 125, 125)),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])

print("Loading dataset...")
# Load the dataset
dataset = ImageFolder("imgs", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Dataset loaded successfully!")

print("Creating the model...")
# Create the autoencoder model
model = Autoencoder().to(device)
print(f"Model created and moved to {device}!")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training the model...")
# Training loop with visualization
num_epochs = 50
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_batches = len(dataloader)
    for batch_index, data in enumerate(dataloader, start=1):
        images, _ = data
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the updating percentage
        percentage = (batch_index / total_batches) * 100
        print(f'\rProgress: {percentage:.2f}%', end='')
    print()

    # Print the loss after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] complete, Loss: {loss.item():.4f}")

    # Visualize some images
    n_images = 3
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Get a batch of data
        images, _ = next(iter(dataloader))
        images = images.to(device)
        outputs = model(images)
        
        # Move images back to cpu for visualization
        images = images.cpu()
        outputs = outputs.cpu()
        
        plt.figure(figsize=(10, 2))
        for i in range(n_images):
            # Display original images
            ax = plt.subplot(2, n_images, i + 1)
            plt.imshow(images[i].permute(1, 2, 0))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstructed images
            ax = plt.subplot(2, n_images, i + 1 + n_images)
            plt.imshow(outputs[i].permute(1, 2, 0))
            ax.axis('off')

            if not os.path.exists("results"):
                os.makedirs("results")
            
            plt.savefig(f'results/epoch_{epoch+1}_output_{i}.png')
            plt.close()

    model.train()  # Set model back to train mode
    if loss.item() < 0.002:
        print("Loss is below 0.002. Training stopped.")
        break

print("Model training complete. Saving the model...")

# Verify the models folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# Save the trained model
torch.save(model.state_dict(), "models/autoencoder_model.pth")

# Save the encoder model separately
encoder = model.encoder
torch.save(encoder.state_dict(), "models/encoder_model.pth")

print("Model saved successfully!")