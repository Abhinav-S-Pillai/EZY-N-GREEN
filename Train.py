import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data_preprocess import train_data, val_data

print("✅ Starting training script...")

# Load MobileNetV2 as the base model (excluding top layers)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze base model for initial training
base_model.trainable = False

# Define model architecture
model = Sequential([
    base_model,
    
    # Improved Architecture
    Dropout(0.3),  # Added dropout before BatchNorm to prevent overfitting
    Conv2D(64, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.4),  
    Conv2D(128, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    GlobalAveragePooling2D(),  
    
    Dense(512, activation='relu'),
    Dropout(0.5),  # Adjusted dropout
    Dense(len(train_data.class_indices), activation='softmax')  # Safer way to get num_classes
])

print("✅ Model architecture created!")

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("✅ Model compiled! Training starts now...")

# Compute class weights to handle imbalanced data
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Add early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Dynamic Learning Rate Reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model (Initial training phase)
history = model.fit(train_data, validation_data=val_data, epochs=20, 
                    class_weight=class_weights, callbacks=[early_stop, reduce_lr])

# Save initial trained model
model.save("waste_classifier_model.keras")
model.save("waste_classifier_model.h5")  # Also save in .h5 format
print("✅ Initial model training complete! Model saved.")

# =========================
# 📌 Phase 2: Fine-Tuning
# =========================

print("✅ Starting fine-tuning...")

# Unfreeze last 25% of layers in MobileNetV2 for fine-tuning
unfreeze_layers = int(len(base_model.layers) * 0.25)
base_model.trainable = True
for layer in base_model.layers[:-unfreeze_layers]:  
    layer.trainable = False  # Keep most layers frozen

# Compile again with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),  
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(train_data, validation_data=val_data, epochs=10, 
                         class_weight=class_weights, callbacks=[early_stop, reduce_lr])

# Save the fine-tuned model
model.save("waste_classifier_model_finetuned.keras")
model.save("waste_classifier_model_finetuned.h5")
print("✅ Model fine-tuning complete! Model saved successfully.")
