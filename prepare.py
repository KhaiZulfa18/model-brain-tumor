import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path ke folder dataset asal
dataset_path = 'dataset_tumor_otak'

# Path untuk folder output
train_path = 'dataset_tumor_otak_split/training'
test_path = 'dataset_tumor_otak_split/testing'

# Buat folder training dan testing jika belum ada
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Rasio data training
train_ratio = 0.8

# Iterasi setiap kelas di dalam folder dataset
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        # Dapatkan semua file gambar di dalam kelas
        images = os.listdir(class_dir)
        random.shuffle(images)

        # Tentukan batas untuk data training
        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        test_images = images[train_size:]

        # Buat directory untuk masing-masing kelas di training dan testing
        train_class_dir = os.path.join(train_path, class_name)
        test_class_dir = os.path.join(test_path, class_name)

        # Cek folder sudah ada atau belum, jika sudah ada kosongkan (empty) folder agar gambar (images) tidak terduplikat
        if os.path.exists(train_class_dir):
            shutil.rmtree(train_class_dir)
        if os.path.exists(test_class_dir):
            shutil.rmtree(test_class_dir)

        # Buat folder untuk masing-masing kelas di training dan testing
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Pindahkan file ke folder training
        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(train_class_dir, image)
            shutil.copyfile(src, dst)

        # Pindahkan file ke folder testing
        for image in test_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(test_class_dir, image)
            shutil.copyfile(src, dst)

print("Dataset berhasil dibagi menjadi training dan testing.")