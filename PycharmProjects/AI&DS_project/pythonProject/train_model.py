import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- 1. Set Up Basic Parameters ---

# Path to the dataset folder
data_dir = "road_crack_data_CE235"

# Image and training parameters
IMG_SIZE = (224, 224)  # ResNet50 expects 224x224 images
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2  # 20% of training data for validation
TEST_SPLIT = 0.1        # 10% of total data for testing

# --- 2. Load and Split the Data ---

print("Loading and splitting data...")

# Create the main training dataset (90% of total data)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=TEST_SPLIT,  # First, take 10% as test data
    subset="training",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'  # Binary classification (crack / no crack)
)

# Create the test dataset (10% of total data)
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=TEST_SPLIT,
    subset="validation",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Further split the training data into training and validation sets
train_batches = tf.data.experimental.cardinality(train_ds)
val_batch_count = int(float(train_batches) * VALIDATION_SPLIT)

# Take a portion of the training set for validation
val_ds = train_ds.take(val_batch_count)
train_ds = train_ds.skip(val_batch_count)

# --- 3. Define Data Augmentation ---
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ],
    name="data_augmentation",
)

# --- 4. Configure Datasets for Better Performance ---
# Caching and prefetching improve training speed
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 5. Quick Data Verification ---

def verify_data():
    print(f"Total batches in Training set: {tf.data.experimental.cardinality(train_ds)}")
    print(f"Total batches in Validation set: {tf.data.experimental.cardinality(val_ds)}")
    print(f"Total batches in Test set: {tf.data.experimental.cardinality(test_ds)}")

    class_names = ['Negative (Uncracked)', 'Positive (Cracked)']

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):  # Take one batch
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(images[i], 0))
            plt.imshow(augmented_image[0].numpy().astype("uint8"))
            plt.title(f"Augmented: {class_names[int(labels[i])]}")
            plt.axis("off")
    plt.suptitle("Sample Augmented Images")

if __name__ == "__main__":
    verify_data()

# --- Phase 2: Build the Model ---

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import models, layers
import tensorflow as tf

# Define input shape
IMG_SHAPE = IMG_SIZE + (3,)

print("Building the ResNet50 model...")

# Load pre-trained ResNet50 without its top layers
base_model = ResNet50(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze the base model so pre-trained weights are not updated
base_model.trainable = False

# Build the complete model
model = models.Sequential([
    layers.Input(shape=IMG_SHAPE),
    data_augmentation,
    tf.keras.layers.Lambda(preprocess_input),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Display model summary
print(model.summary())

# --- Phase 3: Training and Evaluation ---

from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import time

# Compile the model
print("Compiling the model...")
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Set up early stopping to avoid overfitting
early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Train the model
print("Starting model training...")
EPOCHS = 20
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopper]
)

end_time = time.time()
print(f"Model training complete. Time taken: {end_time - start_time:.2f} seconds.")

# Plot training history
print("Plotting training history...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')

ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')

plt.show()

# Evaluate on the test set
print("Evaluating model on the test set...")
loss, accuracy = model.evaluate(test_ds)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# Generate detailed metrics
print("Generating classification report and confusion matrix...")

# Collect true labels
y_true = []
for images, labels in test_ds.unbatch():
    y_true.append(labels.numpy())
y_true = np.array(y_true).astype(int)

# Predict probabilities and convert to class labels
y_pred_probs = model.predict(test_ds)
y_pred = (y_pred_probs > 0.5).astype(int)

# Print classification report
class_names = ['Negative (Uncracked)', 'Positive (Cracked)']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Save the trained model
model_filename = "concrete_crack_detector_resnet50.h5"
print(f"Saving the trained model to {model_filename}...")
model.save(model_filename)
print("Model saved successfully.")
