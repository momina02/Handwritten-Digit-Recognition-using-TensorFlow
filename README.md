# Handwritten Digit Recognition using TensorFlow

[![License](https://img.shields.io/github/license/momina02/handwritten-digit-recognition)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-v2.0%2B-blue)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/status-completed-success)](https://github.com/yourusername/handwritten-digit-recognition)

This repository contains a TensorFlow-based implementation of a handwritten digit recognition system using the MNIST dataset. The model is trained on the MNIST dataset, and the goal is to predict the digit in a given image.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/momina02/handwritten-digit-recognition.git
cd handwritten-digit-recognition
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

You can install the required dependencies via pip:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Usage

1. **Training the Model (Optional)**:  
   If you haven't yet trained the model, you can uncomment the training section in the code and run it to train a model on the MNIST dataset.

2. **Predicting Digits**:  
   After training the model, you can use the following code to load the pre-trained model and predict handwritten digits from images in the `digits/` folder.  
   The images should be named as `digit1.png`, `digit2.png`, etc.

   Run the script as follows:

   ```bash
   python digit_recognition.py
   ```

   The model will predict the digit in each image and display it.

## Model Details

The model consists of the following architecture:

- **Input Layer**: 28x28 pixels grayscale image
- **Hidden Layer 1**: Fully connected layer with 128 neurons and ReLU activation
- **Hidden Layer 2**: Fully connected layer with 128 neurons and ReLU activation
- **Output Layer**: Softmax layer with 10 neurons (for digits 0-9)

The model is trained using the **Adam** optimizer and **sparse categorical cross-entropy loss**.

### Training the Model

Uncomment the model training code to train the model on the MNIST dataset. The training process will take approximately 3 epochs to complete.

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
model.save('handwrittendigit.keras')
```

After training, you can save the model using `model.save('handwrittendigit.keras')` and use it for future predictions.

## Project Structure

```
.
├── digits/
│   └── digit1.png
│   └── digit2.png
│   └── ...
├── handwritten_digit_recognition.py
├── requirements.txt
└── handwrittendigit.keras
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

