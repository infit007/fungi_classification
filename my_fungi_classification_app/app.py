from flask import Flask, render_template, request
import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)

# Load pre-trained model
model_directory = r'C:\Users\jsunt\Downloads\Defungi Project\Model'
model = keras.models.load_model(model_directory)

def classify_fungi(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    # Assume that predictions is an array containing the probabilities for each class
    class_index = np.argmax(predictions)
    fungi_types = ['H1', 'H2', 'H3', 'H5', 'H6']
    fungi_type = fungi_types[class_index]

    return fungi_type

# Return a tuple containing result and fungi type

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['image']

        if uploaded_file.filename != '':
            image_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(image_path)

            fungi_type = classify_fungi(image_path)

            # Pass the fungi type to the result page
            return render_template('result.html', fungi_type=fungi_type)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
