import os
import numpy as np
from PIL import Image
import time

#画像読み込みとリサイズ
def load_and_resize_image(image_path, size=(150, 150)):
    img = Image.open(image_path).convert('RGB')#エラー出たので対策用

    return img.resize(size)

#正解画像の準備
def prepare_correct_images(save_root, keywords, num_correct_images=10, size=(150, 150)):
    correct_images = {}

    for keyword in keywords:
        save_dir = os.path.join(save_root, keyword)
        image_files = []
        for file in os.listdir(save_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):#エラー出たので対策用
                image_files.append(file)

        correct_images[keyword] = []
        for file in image_files[:num_correct_images]:
            image_path = os.path.join(save_dir, file)
            resized_image = load_and_resize_image(image_path, size)#エラー出たので対策用
            correct_images[keyword].append(resized_image)
            
    return correct_images

#検証用準備
def prepare_test_dataset(save_root, keywords, num_test_images=1000, size=(150, 150)):
    test_dataset = []
    actual_labels = []
    for keyword in keywords:
        save_dir = f"{save_root}/{keyword}"
        image_files = [file for file in os.listdir(save_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]#エラー出たので対策用
        test_images_per_label = num_test_images // len(keywords)
        skip = 10 #正解画像を省く
        start_index = skip
        end_index = start_index + test_images_per_label
        
        for file in image_files[start_index:end_index]:
            test_dataset.append(load_and_resize_image(os.path.join(save_dir, file), size))
            actual_labels.append(keyword)

    return test_dataset, actual_labels

#画像のビットマップ比較
def classify_image(test_image, correct_images):
    min_distance = float('inf')
    predicted_label = None
    test_image_array = np.array(test_image)
    for label, images in correct_images.items():
        for image in images:
            distance = np.sum((test_image_array - np.array(image))**2)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label
                
    return predicted_label

#精度と時間計測
def evaluate_model(save_root, keywords):
    correct_images = prepare_correct_images(save_root, keywords)
    test_dataset, actual_labels = prepare_test_dataset(save_root, keywords)
    
    correct_p = 0
    total_time = 0

    for test_image, actual_label in zip(test_dataset, actual_labels):
        start_time = time.perf_counter()
        predicted_label = classify_image(test_image, correct_images)
        end_time = time.perf_counter()
        
        total_time += (end_time - start_time)
        if predicted_label == actual_label:
            correct_p += 1

    accuracy = correct_p / len(test_dataset)

    a1 = accuracy * 100
    print("Accuracy: {:.2f}%".format(float(a1)))
    print("Total Classification Time: {:.2f} seconds".format(float(total_time)))

if __name__ == "__main__":
    save_root = './images'
    keywords = ['dog', 'cat', 'bird', 'hamster', 'goldfish', 'flower', 'car', 'plane', 'ship', 'apartment']
    evaluate_model(save_root, keywords)