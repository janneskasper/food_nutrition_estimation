import os
from PIL import Image

if __name__ == "__main__":
    gif_dir = "out/gif"
    gif_name = "test.gif"

    image_paths = [f"{gif_dir}/{path}" for path in os.listdir(gif_dir)]
    images = [Image.open(img_path) for img_path in image_paths]

    images[0].save(gif_name, save_all=True, append_images=images[1:], optimize=False, duration=400, loop=0)

