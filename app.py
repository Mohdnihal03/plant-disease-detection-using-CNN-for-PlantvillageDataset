import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
from keras.preprocessing.image import ImageDataGenerator
# Set paths for training and validation datasets
def get_classes_from_dir(directory):
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

# Path to the training directory
train_dir = r'C:/Users/nihall/Documents/plantvillagedataset/PlantVillage/train'

# Get class names
class_names = get_classes_from_dir(train_dir)
num_classes = len(class_names)
print("Class names:", class_names)
print("Number of classes:", num_classes)

# Function to preprocess the images
def preprocess_image(image):
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0  # Normalize to [0, 1] range
    return image

# Load the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    image_size=(128, 128),
    batch_size=32,
    shuffle=True,
    seed=42
)

# Apply preprocessing to the dataset
train_ds = train_ds.map(lambda x, y: (preprocess_image(x), y))

# Augmentation and preprocessing
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, 0.1, 0.2)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

train_ds = train_ds.map(augment_image).prefetch(tf.data.AUTOTUNE)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    epochs=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
)

# Save the trained model (optional)
model.save('plant_disease_model.h5')