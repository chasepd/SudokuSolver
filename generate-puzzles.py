import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

def is_valid(puzzle, x, y, num):
    num = str(num)
    # Check row and column
    for i in range(9):
        if puzzle[x][i] == num or puzzle[i][y] == num:
            return False

    # Check box
    box_x, box_y = (x // 3) * 3, (y // 3) * 3
    for i in range(3):
        for j in range(3):
            if puzzle[box_x + i][box_y + j] == num:
                return False

    return True

def solve_sudoku(puzzle):
    for x in range(9):
        for y in range(9):
            if puzzle[x][y] == '.':
                for num in random.sample(range(1, 10), 9):
                    if is_valid(puzzle, x, y, num):
                        puzzle[x][y] = str(num)
                        if solve_sudoku(puzzle):
                            return True
                        puzzle[x][y] = '.'
                return False
    return True

def generate_solved_puzzle():
    puzzle = [['.' for _ in range(9)] for _ in range(9)]
    solve_sudoku(puzzle)
    return puzzle

def generate_puzzle():
    solved_puzzle = generate_solved_puzzle()
    puzzle = [row.copy() for row in solved_puzzle]
    for i in range(9):
        for j in range(9):
            if random.random() < 0.2:
                puzzle[i][j] = '.'
    return puzzle, solved_puzzle

def list_fonts():    
    fonts = os.listdir('fonts/')
    # Filter out non-font files
    fonts = [font for font in fonts if font.endswith('.ttf') or font.endswith('.otf')]
    return fonts  

def high_contrast_color(bg_color):
    # Calculate perceived brightness
    brightness = (0.299*bg_color[0] + 0.587*bg_color[1] + 0.114*bg_color[2]) / 255
    # Choose high contrast color with moderated intensity
    if brightness > 0.5:
        # Darker color but not black
        return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    else:
        # Lighter color but not white
        return (random.randint(155, 255), random.randint(155, 255), random.randint(155, 255))

def random_color(exclude_extremes=True):
    if exclude_extremes:
        # Avoiding very dark and very light colors
        return tuple(random.randint(32, 223) for _ in range(3))
    else:
        return tuple(random.randint(0, 255) for _ in range(3))

def generate_image(puzzle, filepath, available_fonts, img_width=500, img_height=500, padding=10, font_size=42, random_modifier=True):
    if random_modifier:
        modifier = 1 + random.random()
    else:
        modifier = 1

    img_width = int(img_width * modifier)
    img_height = int(img_height * modifier)
    padding = int(padding * modifier)
    cell_size = (img_width - 2 * padding) // 9
    line_width_thin = int(2 * modifier)
    line_width_thick = int(5 * modifier)

    # Generate a random background color
    bg_color = random_color()
    img = Image.new('RGB', (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    font_choice = random.choice(available_fonts)
    try:
        font = ImageFont.truetype(f'fonts/{font_choice}', int(font_size * modifier))
    except IOError:
        print(f'Error loading font {font_choice}, using default font')
        font = ImageFont.load_default()

    # Draw the grid and numbers
    for x in range(9):
        for y in range(9):
            text = puzzle[x][y]
            if text != '.':
                # Calculate cell's top left corner
                cell_x = padding + y * cell_size
                cell_y = padding + x * cell_size

                # Determine high-contrast text color
                text_color = high_contrast_color(bg_color)

                # Measure text size for centering
                text_size = draw.textbbox((0, 0), text, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                text_x = cell_x + (cell_size - text_width) / 2
                text_y = cell_y + (cell_size - text_height) / 2

                draw.text((text_x, text_y), text, fill=text_color, font=font)

    img.save(filepath, 'PNG')



def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate sudoku puzzles')
    parser.add_argument('n', type=int, help='Number of training puzzles to generate')
    parser.add_argument('--validation', type=int, help='Number of validation puzzles to generate')
    parser.add_argument('--images', action='store_true', help='Generate images')
    args = parser.parse_args()

    train_img_dir = 'data/train/imgs/class'
    train_unsolved_text_dir = 'data/train/text/unsolved'
    train_solved_text_dir = 'data/train/text/solved'

    validation_img_dir = 'data/validation/imgs/class'
    validation_unsolved_text_dir = 'data/validation/text/unsolved'
    validation_solved_text_dir = 'data/validation/text/solved'

    generate_images = args.images

    if generate_images:
        available_fonts = list_fonts()    

    # Verify directories exist
    if not os.path.exists(train_img_dir) and generate_images:
        os.makedirs(train_img_dir)

    if not os.path.exists(train_unsolved_text_dir):
        os.makedirs(train_unsolved_text_dir)
    
    if not os.path.exists(train_solved_text_dir):
        os.makedirs(train_solved_text_dir)
    
    if not os.path.exists(validation_img_dir) and generate_images:
        os.makedirs(validation_img_dir)
    
    if not os.path.exists(validation_unsolved_text_dir):
        os.makedirs(validation_unsolved_text_dir)
    
    if not os.path.exists(validation_solved_text_dir):
        os.makedirs(validation_solved_text_dir)


    puzzle_count = args.n

    for i in range(puzzle_count):
        print(f'\rGenerating puzzles... {(i + 1) / args.n * 100:.4f}% complete', end='')
        puzzle, solved_puzzle = generate_puzzle()
        filename = f"puzzle_{i}.txt"

        with open(f'{train_unsolved_text_dir}/{filename}', 'w') as f:
            for row in puzzle:
                f.write(' '.join(row) + '\n')

        with open(f'{train_solved_text_dir}/{filename}', 'w') as f:
            for row in solved_puzzle:
                f.write(' '.join(row) + '\n')
        
        if generate_images:
            filename = f"puzzle_{i}.png"
            generate_image(puzzle, f'{train_img_dir}/{filename}', available_fonts)

    print()

    if args.validation is None:
        validation_puzzle_count = args.n // 10

        if validation_puzzle_count > 2500:
            validation_puzzle_count = 2500
    else:
        validation_puzzle_count = args.validation

    for i in range(validation_puzzle_count):
        print(f'\rGenerating validation puzzles... {(i + 1) / validation_puzzle_count * 100:.4f}% complete', end='')
        puzzle, solved_puzzle = generate_puzzle()
        filename = f"puzzle_{i}.txt"

        with open(f'{validation_unsolved_text_dir}/{filename}', 'w') as f:
            for row in puzzle:
                f.write(' '.join(row) + '\n')

        with open(f'{validation_solved_text_dir}/{filename}', 'w') as f:
            for row in solved_puzzle:
                f.write(' '.join(row) + '\n')
        
        if generate_images:
            filename = f"puzzle_{i}.png"
            generate_image(puzzle, f'{validation_img_dir}/{filename}', available_fonts)
    
    print()


if __name__ == '__main__':
    main()