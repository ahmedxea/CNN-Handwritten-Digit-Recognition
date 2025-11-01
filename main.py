import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping


# Load and preprocess MNIST data
def load_and_preprocess_mnist():
    """Load the MNIST dataset and normalize pixel values."""
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data to range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


# Build and compile the CNN model
def build_model():
    """Define and compile a CNN for digit recognition."""
    model = models.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(28, 28)),

        # Convolution and pooling layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Train and save the model
def train_and_save_model():
    """Train the model on MNIST data and save it."""
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    model = build_model()

    # Use early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    model.fit(
        x_train, y_train,
        epochs=15,
        validation_split=0.1,
        batch_size=128,
        callbacks=[early_stop],
        verbose=2
    )

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/handwritten_cnn.keras")

    # Evaluate model performance
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"\nModel saved to models/handwritten_cnn.keras")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


# Load model from file
def load_model():
    """Load the saved CNN model from disk."""
    return keras.models.load_model("models/handwritten_cnn.keras")


# Preprocess custom image for prediction
def preprocess_image(path):
    """Load, invert, and normalize a custom handwritten image."""
    img = cv2.imread(path)[:, :, 0]
    img = np.invert(np.array([img]))
    img = img.astype("float32") / 255.0
    return img


# Predict custom handwritten digits
def predict_custom_digits():
    """Predict digits from PNG images stored in the digits/ folder."""
    model = load_model()
    image_number = 1

    while os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            img = preprocess_image(f"digits/digit{image_number}.png")
            prediction = model.predict(img)
            predicted_label = np.argmax(prediction)
            print(f"digit{image_number}.png → Predicted: {predicted_label}")

            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.title(f"Prediction: {predicted_label}")
            plt.show()

        except Exception as e:
            print(f"Error processing image {image_number}: {e}")

        image_number += 1


# Main program
if __name__ == "__main__":
    if not os.path.exists("models/handwritten_cnn.keras"):
        print("No model found — training a new one...")
        train_and_save_model()
    else:
        print("Using existing model from models/handwritten_cnn.keras")

    predict_custom_digits()