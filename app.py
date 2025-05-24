import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import gradio as gr
from PIL import Image

# Load and preprocess MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Build model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))

labels = [str(i) for i in range(10)]

def recognize_digit(img):
    # img comes from gr.Sketchpad as a numpy array (H,W,3) RGB
    # Convert to grayscale, resize to 28x28
    img = Image.fromarray(img.astype('uint8'), 'RGB').convert('L').resize((28,28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=(0,-1))  # shape (1,28,28,1)
    preds = model.predict(img)[0]
    top_preds = sorted(zip(labels, preds), key=lambda x: x[1], reverse=True)[:3]
    return {label: float(score) for label, score in top_preds}

iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(),
    outputs=gr.Label(num_top_classes=3),
    live=True
)

iface.launch()
