import os
import matplotlib.pyplot as plt
from PIL import Image

def display_images(directory, num_images=3):
    # Get a list of image file names
    image_files = os.listdir(directory)

    # Display the specified number of images
    for i in range(min(num_images, len(image_files))):
        image_path = os.path.join(directory, image_files[i])
        image = Image.open(image_path)

        # Display the image
        plt.imshow(image)
        plt.title(f"Image {i+1}")
        plt.show()

if __name__ == "__main__":
    # Specify the path to the images directory
    images_directory = "project-dataset/train/images"

    # Display the first 3 images (you can change the number)
    display_images(images_directory, num_images=3)
