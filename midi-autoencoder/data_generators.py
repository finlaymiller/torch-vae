import os
import numpy as np
from PIL import Image
from rich.progress import track


def generate_line_images(
    num_images, output_path, img_size=(28, 28), max_lines=5, fixed_line_width=True
):
    """
    Generates a dataset of black and white images composed of vertical and horizontal lines,
    grouped into class folders for use with PyTorch's `ImageFolder`.

    Classes are based on the number of lines in the image.

    Args:
        num_images (int): Number of images to generate.
        output_path (str): Directory to store the generated images.
        img_size (tuple): Height and width of the images as (height, width).
        max_lines (int): Maximum number of lines allowed per image.
        fixed_line_width (bool): If True, all lines will have width 1. If False, lines can have varying widths.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    height, width = img_size

    for i in track(range(num_images)):
        # Create a blank image (black background)
        img = np.zeros((height, width), dtype=np.uint8)

        num_lines = np.random.randint(1, max_lines + 1)

        for _ in range(num_lines):
            # Determine if the line is vertical or horizontal
            is_vertical = np.random.choice([True, False])

            # Determine line width
            line_width = 1 if fixed_line_width else np.random.randint(1, 4)

            if is_vertical:
                x = np.random.randint(0, width)
                start_y = np.random.randint(0, height)
                end_y = np.random.randint(start_y, height)
                img[
                    start_y:end_y,
                    max(0, x - line_width // 2) : min(width, x + line_width // 2 + 1),
                ] = 255
            else:
                y = np.random.randint(0, height)
                start_x = np.random.randint(0, width)
                end_x = np.random.randint(start_x, width)
                img[
                    max(0, y - line_width // 2) : min(height, y + line_width // 2 + 1),
                    start_x:end_x,
                ] = 255

        # Define the class folder based on the number of lines
        class_folder = os.path.join(output_path, f"class_{num_lines}")
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Save the image in the appropriate class folder
        img_path = os.path.join(class_folder, f"image_{i+1}.png")
        Image.fromarray(img).save(img_path)


if __name__ == "__main__":
    generate_line_images(100000, "/media/nova/Datasets/vae-lines")
