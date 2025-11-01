import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load the trained Keras model (cached for performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/handwritten_cnn.keras")

model = load_model()

# Set up Streamlit page configuration
st.set_page_config(page_title="✍️ Handwritten Digit Recognition", layout="centered")
st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a handwritten digit (0–9) below, and the model will predict it in real time.")

# Create drawing canvas
canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="#FFFFFF",       # white digit
    background_color="#000000",   # black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Run prediction when drawing is available
if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # Binarize and remove noise
    _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

    # Skip empty drawings
    if np.sum(img) < 50:
        st.info("Draw a digit above to get a prediction.")
        st.stop()

    # Find bounding box around the digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # Resize while keeping aspect ratio to fit 20x20 area
    max_side = max(w, h)
    scale = 20.0 / max_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to make a 28x28 image
    col_pad = (28 - new_w) // 2
    row_pad = (28 - new_h) // 2
    img = cv2.copyMakeBorder(
        img, row_pad, 28 - new_h - row_pad,
        col_pad, 28 - new_w - col_pad,
        cv2.BORDER_CONSTANT, value=0
    )

    # Center the digit based on its center of mass
    M = cv2.moments(img)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        shiftx = np.round(14 - cx).astype(int)
        shifty = np.round(14 - cy).astype(int)
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        img = cv2.warpAffine(img, M, (28, 28))

    # Normalize and reshape for CNN input
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape: (1, 28, 28, 1)

    # Predict digit
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display results
    st.markdown(f"### Prediction: **{predicted_digit}** ({confidence:.2f}% confidence)")
    st.image(img.reshape(28, 28), width=150, caption="Processed Image", clamp=True)
    st.bar_chart(prediction[0])

# Footer
st.markdown("---")
st.caption("Built by Ahmed El Abed using Streamlit and Keras (TensorFlow backend)")