import os
import numpy as np
from PIL import Image
from icrawler.builtin import BingImageCrawler

def download_images(keyword, num_images, save_dir):
    crawler = BingImageCrawler(downloader_threads=4, storage={"root_dir": save_dir})
    crawler.crawl(keyword=keyword, max_num=num_images)

def augment_image(image_path):
    img = Image.open(image_path)
    imgs = [img]

    imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    imgs.append(img.transpose(Image.FLIP_TOP_BOTTOM))

    n1 = np.random.randint(0, 10)
    n2 = np.random.randint(0, 10)
    for dx, dy in [(-n1, 0), (n1, 0), (0, -n2), (0, n2)]:
        shifted = np.roll(np.array(img), shift=dx, axis=1)
        shifted = np.roll(shifted, shift=dy, axis=0)
        imgs.append(Image.fromarray(shifted))

    widths, heights = img.size
    for i in range(3):
        startx = np.random.randint(0, widths//10)
        starty = np.random.randint(0, heights//10)
        crop_img = img.crop((startx, starty, widths-startx, heights-starty))
        imgs.append(crop_img.resize((widths, heights)))
  
    return imgs

def save_images(images, base_name, save_dir):
    for i, img in enumerate(images):
        img.save(os.path.join(save_dir, f"{base_name}_{i+1}.png"))

def main():
    keywords = ['dog', 'cat', 'bird', 'hamster', 'goldfish', 'flower', 'car', 'plane', 'ship', 'apartment']
    save_root = './images'
    os.makedirs(save_root, exist_ok=True)

    for keyword in keywords:
        save_dir = f"{save_root}/{keyword}"
        os.makedirs(save_dir, exist_ok=True)
        download_images(keyword, 100, save_dir)
    
        # Get a list of files in a directory
        image_files = [file for file in os.listdir(save_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
        # Process each image file
        for i, file_name in enumerate(image_files):
            img_path = os.path.join(save_dir, file_name)
            augmented_images = augment_image(img_path)
            num_index = i * 10 + 1
            save_images(augmented_images, f"{keyword}_{num_index}", save_dir)

if __name__ == "__main__":
    main()

