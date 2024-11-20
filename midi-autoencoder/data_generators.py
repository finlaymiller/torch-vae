import os
import numpy as np
from PIL import Image
from rich.progress import track


def generate_line_images(
    num_images,
    output_path,
    img_size=(28, 28),
    max_lines=2,
    line_width=2,
    full_length=True,
    filter_duplicates=True,
) -> None:
    """
    Generates a dataset of black and white images composed of vertical and horizontal lines,
    grouped into class folders. Classes are based on the number of lines in the image.

    Args:
        num_images (int): Number of images to generate.
        output_path (str): Directory to store the generated images.
        img_size (tuple): Height and width of the images as (height, width). Default is (28, 28).
        max_lines (int): Maximum number of lines allowed per image. Default is 2.
        line_width (int): Line width in pixels. Random (1-5) if set to 0. Default is 1.
        full_length (bool): Whether all lines should span the entire image. Default is True.
        filter_duplicates (bool): Whether to filter out duplicate images. Default is True.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    height, width = img_size
    generated_images = set()

    # generate images
    for i in track(range(num_images)):
        img = np.zeros((height, width), dtype=np.uint8)
        num_lines = np.random.randint(1, max_lines + 1)

        # generate lines
        for _ in range(num_lines):
            is_vertical = np.random.choice([True, False])
            line_width = np.random.randint(1, 5) if line_width == 0 else line_width

            if is_vertical:
                x = np.random.randint(0, width)
                if full_length:
                    start_y = 0
                    end_y = height
                else:
                    start_y = np.random.randint(0, height)
                    end_y = np.random.randint(start_y, height)
                img[
                    start_y:end_y,
                    max(0, x - line_width // 2): min(width, x + line_width // 2 + 1),
                ] = 255
            else:
                y = np.random.randint(0, height)
                if full_length:
                    start_x = 0
                    end_x = width
                else:
                    start_x = np.random.randint(0, width)
                    end_x = np.random.randint(start_x, width)
                img[
                    max(0, y - line_width // 2): min(height, y + line_width // 2 + 1),
                    start_x:end_x,
                ] = 255

        # filter duplicates
        if filter_duplicates:
            img_tuple = tuple(img.flatten())
            if img_tuple in generated_images:
                continue
            generated_images.add(img_tuple)

        # class folder is based on the number of lines
        class_folder = os.path.join(output_path, f"{num_lines}_lines")
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # save image
        img_path = os.path.join(class_folder, f"image_{i+1}.png")
        Image.fromarray(img).save(img_path)


if __name__ == "__main__":
    generate_line_images(100000, "/media/nova/Datasets/vae-lines")