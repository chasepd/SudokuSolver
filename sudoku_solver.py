import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Solver(nn.Module):
    def __init__(self, encoder):
        super(Solver, self).__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(800, 256),
            nn.ReLU(),
            nn.Linear(256, 81*10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

def load_model():
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load('models/encoder_model.pth'))
    solver = Solver(encoder).to(device)
    solver.load_state_dict(torch.load('models/solver_model.pth'))
    return solver

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
