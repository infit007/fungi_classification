from flask import Flask, render_template, request, send_from_directory
import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)

# Path to the model (Ensure you have model.h5 after conversion)
model_directory = './model/model.h5'  # Use relative path for deployment

# Load the pre-trained model
model = keras.models.load_model(model_directory)

def classify_fungi(image_path):
    # Load and preprocess the image for prediction
    img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for MobileNetV2

    # Make prediction using the model
    predictions = model.predict(img_array)

    # Get the class index of the highest predicted class
    class_index = np.argmax(predictions)
    
    # Mapping class index to corresponding fungi type
    fungi_types = ['H1', 'H2', 'H3', 'H5', 'H6']
    fungi_type = fungi_types[class_index]

    return fungi_type

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image file was uploaded
        uploaded_file = request.files['image']

        if uploaded_file.filename != '':
            # Save the uploaded file to the 'static/uploads' folder
            image_path = os.path.join('static', 'uploads', uploaded_file.filename)
            uploaded_file.save(image_path)

            # Classify the fungi and get the result
            fungi_type = classify_fungi(image_path)

            # Pass the result to the result page
            return render_template('result.html', fungi_type=fungi_type)

    # Render the index page for GET request or after uploading an image
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'static/uploads'), filename)

if __name__ == '__main__':
    # Ensure the 'static/uploads' directory exists
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)

    # Run the Flask app in debug mode
    app.run(debug=True)
