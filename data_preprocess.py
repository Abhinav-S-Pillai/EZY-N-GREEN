import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path
dataset_path = "p:/programming/projects/EZY-N-GREEN/dataset-resized"

# Image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ Improved Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=40,  # Increased rotation range for generalization
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],  # Added brightness range
    horizontal_flip=True,
    fill_mode="nearest",  # Prevents losing pixels in transformations
    validation_split=0.2  # Splitting dataset (80% training, 20% validation)
)

# ✅ Separate Data Preprocessing for Validation (NO Augmentations)
val_datagen = ImageDataGenerator(
    rescale=1./255,  # Only normalization, NO augmentation
    validation_split=0.2
)

# ✅ Load Training Data with Augmentation
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Load training data
)

# ✅ Load Validation Data Without Augmentation
val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Load validation data
)

print("✅ Dataset Preprocessing Complete!")
print(f"📂 Training Samples: {train_data.samples}")
print(f"📂 Validation Samples: {val_data.samples}")
