# Handwritten Digit Recognition

This project recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.  
It includes a training script (`main.py`) and an interactive web interface (`app.py`) built with Streamlit.

---

## Project Structure

```markdown

handwritten-digit-recognition/
│
├── models/               # Saved trained model (handwritten_cnn.keras)
├── digits/               # Optional folder for test images
├── main.py               # Trains the CNN model
├── app.py                # Streamlit app for interactive digit drawing
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

````

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedxea/CNN-Handwritten-Digit-Recognition.git
   cd CNN-Handwritten-Digit-Recognition
   ````

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Training the Model

To train the CNN and save it locally:

```bash
python main.py
```

The trained model will be saved in:

```
models/handwritten_cnn.keras
```

---

## Running the Web App

To launch the Streamlit interface:

```bash
python -m streamlit run app.py
```

Open the local URL (usually `http://localhost:8501`) to draw digits on the canvas and see live predictions.

---

## Testing with Saved Images

If you have digit images (e.g., `digit1.png`, `digit2.png`) in the `digits/` folder, you can test predictions by running:

```bash
python main.py
```

---

## Requirements

Dependencies are listed in `requirements.txt`:

* TensorFlow
* NumPy
* OpenCV
* Pillow
* Streamlit
* Matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```