from flask import Flask, render_template, request

from tensorflow.keras.models import load_model

from PIL import Image

import numpy as np

app = Flask(__name__)

model = load_model('model/cifar10_model.h5')

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

def prepare_image(image):

    image = image.resize((32,32))

    image = np.array(image)

    image = image.astype('float32') / 255.0

    image = np.expand_dims(image, axis=0)

    return image

@app.route('/')

def home():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    file = request.files['file']

    image = Image.open(file)

    image = prepare_image(image)

    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    result = class_names[predicted_class]

    return render_template(
        'index.html',
        prediction=result,
        confidence=round(confidence * 100, 2)
    )

if __name__ == '__main__':

    app.run(debug=True)