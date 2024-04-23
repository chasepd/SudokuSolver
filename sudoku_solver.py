import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Reader(nn.Module):
    def __init__(self):
        super(Reader, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Dummy input to calculate flat features
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 500, 500)  # Assuming input image size is 500x500
            dummy_output = self.features(dummy_input)
            flat_features = dummy_output.view(-1).shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flat_features, 800)
        self.fc2 = nn.Linear(800, 81*10)

    def features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model():
    reader = Reader().to(device)
    reader.load_state_dict(torch.load('models/reader_model.pth', map_location=device))
    return reader

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def predict_digits(model, image):
    output = model(image).view(81, 10)
    _, predicted = torch.max(output, 1)
    grid = predicted.view(9, 9).cpu().numpy()
    # Replace 10 with . in the grid
    grid[grid == 10] = 0
    return grid

def is_valid(board, row, col, num):
    for i in range(9):
        if board[i][col] == num or board[row][i] == num:
            return False
    startRow, startCol = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[startRow + i][startCol + j] == num:
                return False
    return True

def board_contains_zero(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return True
    return False

def solve_sudoku(board):
    return True
    # while board_contains_zero(board):
    #     for row in range(9):
    #         for col in range(9):
    #             if board[row][col] == 0:
    #                 for num in range(1, 10):
    #                     if is_valid(board, row, col, num):
    #                         board[row][col] = num
    # return True

def draw_solution(board):
    image = Image.new('RGB', (500, 500), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Update the path to the font file according to your system or installation
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Example path for Linux
    font_size = 42
    font = ImageFont.truetype(font_path, font_size)

    for i in range(9):
        for j in range(9):
            text = str(board[i][j])
            x = j * 50 + 20
            y = i * 50 + 15
            draw.text((x, y), text, fill=(0, 0, 0), font=font)

    image.save('sudoku_solution.png')


if __name__ == "__main__":
    image_path = sys.argv[1]
    model = load_model()
    image = process_image(image_path)
    grid = predict_digits(model, image)
    solve_sudoku(grid)
    draw_solution(grid)
