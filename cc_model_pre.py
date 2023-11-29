import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to your dataset
train_data_dir = "ccd"
test_data_dir = "ccd"

# Image dimensions and batch size
img_height, img_width = 128, 128
batch_size = 32

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # Shearing transformation
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",  # Fill missing pixels using the nearest neighbor
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",  # Assuming you have two classes: Covered and Uncovered
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
)

# Define the CNN model
model = keras.Sequential(
    [
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(
            1, activation="sigmoid"
        ),  # Output layer with a binary sigmoid activation
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
epochs = 20  # You can adjust the number of training epochs
model.fit(train_generator, epochs=epochs)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")

# Save the trained model
model.save("camera_covering_model.h5")
