import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the image size and other constants
img_size = (224, 224)

# Load the base model (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the model architecture
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(4, activation='softmax')(x)  # Adjust output layer to match the number of classes

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Load the trained model weights
model.load_weights('model_weights.h5')  # Ensure this path points to the saved weights file

# Compile the model (optional if already compiled during training)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Function to predict a new image
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization if needed

    # Predict the class of the image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

# Example usage of the prediction function on a single image
predict_image('single_image.jpg')
