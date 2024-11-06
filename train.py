import os
import random
import shutil
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path ke folder dataset asal
dataset_path = 'dataset_tumor_otak'

# Path ke folder training dan testing di Google Drive
train_path = 'dataset_tumor_otak_split/training'
test_path = 'dataset_tumor_otak_split/testing'

# Ukuran gambar yang akan digunakan (sesuaikan dengan ResNet-50)
img_size = (224, 224)
batch_size = 32

# Preprocessing untuk data training dengan augmentasi
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,              # Normalisasi gambar
    rotation_range=20,               # Rotasi acak hingga 20 derajat
    width_shift_range=0.1,           # Pergeseran lebar hingga 10%
    height_shift_range=0.1,          # Pergeseran tinggi hingga 10%
    shear_range=0.1,                 # Transformasi geser
    zoom_range=0.1,                  # Zoom in/out
    horizontal_flip=True,            # Flip horizontal
    fill_mode='nearest'              # Isi piksel kosong setelah augmentasi
)

# Preprocessing untuk data testing tanpa augmentasi (hanya rescaling)
test_datagen = ImageDataGenerator(
    rescale=1.0/255.0
)

# Load dataset dari folder training dan testing
train_generator = train_datagen.flow_from_directory(
    directory=train_path,           # Path data training
    target_size=img_size,           # Ukuran gambar yang akan diubah
    batch_size=batch_size,          # Batch size untuk training
    class_mode='categorical'        # Kelas multiple (multi-class)
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,            # Path data testing
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load model ResNet-50 dengan bobot pretrained ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Menambahkan fully connected layer di atasnya
x = base_model.output
x = Flatten()(x)                    # Mengubah fitur menjadi vektor
x = Dense(1024, activation='relu')(x)
x = Dense(4, activation='softmax')(x)  # Sesuaikan output dengan jumlah kelas (4 kelas)

# Membuat model akhir
model = Model(inputs=base_model.input, outputs=x)

# Freeze layer dari base model
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(
    filepath='RESNET50_model.weights.best.hdf5',  # Filepath to save the weights
    monitor='val_loss',                           # Monitor validation loss
    verbose=1,                                    # Print message when saving
    save_best_only=True,                          # Only save the best weights
    save_weights_only=True                        # Save only the weights, not the entire model
)

# Mulai training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=5,  # Sesuaikan dengan kebutuhan
    callbacks=[checkpointer]  # Pass the callback here
)

# Simpan model hasil training
model.save_weights('model_weights.h5')

# Evaluasi performa model
loss, accuracy = model.evaluate(test_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")