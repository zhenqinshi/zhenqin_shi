import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time
import re
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def load_and_preprocess_data(base_dir, train_size, validation_size):
    keywords = ['dog', 'cat', 'bird', 'hamster', 'goldfish', 'flower', 'car', 'plane', 'ship', 'apartment']
    all_images = []
    labels = []

    for keyword in keywords:
        keyword_dir = os.path.join(base_dir, keyword)
        if os.path.isdir(keyword_dir):
            for file in os.listdir(keyword_dir):
                if re.match(r'^' + keyword + r'_[0-9]+_[0-9]+\.png$', file):
                    path = os.path.join(keyword, file)
                    #print(path)
                    all_images.append("./images/" + path)
                    labels.append(keyword)

    data = pd.DataFrame({
        'filename': all_images,
        'class': labels
    })

    sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_size / (train_size + validation_size), random_state=42)
    for train_index, val_index in sss.split(data['filename'], data['class']):
        training_data = data.iloc[train_index]
        validation_data = data.iloc[val_index]

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=training_data,
        x_col='filename',
        y_col='class',
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_data,
        x_col='filename',
        y_col='class',
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical'
    )

    return train_generator, validation_generator

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  #10
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, train_generator, validation_generator, train_size):
    steps_per_epoch = train_size // 20
    
    model.fit(train_generator, epochs=10, steps_per_epoch=steps_per_epoch)

    start_time = time.time()
    results = model.evaluate(validation_generator)
    end_time = time.time()

    print(f"Validation time: {end_time - start_time}s")
    print(f"Accuracy: {results[1] * 100}%")

def check_files(base_dir, keywords):
    for keyword in keywords:
        path = os.path.join(base_dir, keyword)
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
        else:
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            if not files:
                print(f"No image files found in {path}")
            else:
                print(f"Found {len(files)} files in {path}")

def main():
    base_dir = './images'
    train_size = 5000#5000
    validation_size = 1000#1000

    #check_files(base_dir, ['dog', 'cat', 'bird', 'hamster', 'goldfish', 'flower', 'car', 'plane', 'ship', 'apartment'])

    train_generator, validation_generator = load_and_preprocess_data(base_dir, train_size, validation_size)

    #try:
    #    x, y = next(train_generator)
    #    print("Data batch shape:", x.shape) 
    #    print("Labels batch shape:", y.shape) 
    #except StopIteration:
    #    print("Train generator is empty.")

    model = build_model()
    train_and_evaluate_model(model, train_generator, validation_generator, train_size)

if __name__ == "__main__":
    main()