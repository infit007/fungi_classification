from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Path to the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.h5')

# Ensure the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the pre-trained model
model = load_model(model_path)

# Fungi type mapping
fungi_types = ['H1', 'H2', 'H3', 'H5', 'H6']

def classify_fungi(image_path):
    """
    Classify fungi based on the uploaded image.
    
    Args:
        image_path (str): Path to the uploaded image file.
    
    Returns:
        str: Predicted fungi type.
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)

    # Get the class index and corresponding fungi type
    class_index = np.argmax(predictions)
    return fungi_types[class_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route for handling image upload and displaying the classification result.
    """
    if request.method == 'POST':
        # Check if an image file was uploaded
        uploaded_file = request.files.get('image')

        if uploaded_file and uploaded_file.filename:
            # Create uploads directory if not exists
            upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            # Save the uploaded file
            image_path = os.path.join(upload_folder, uploaded_file.filename)
            uploaded_file.save(image_path)

            # Classify the fungi and get the result
            fungi_type = classify_fungi(image_path)

            # Render the result page
            return render_template('result.html', fungi_type=fungi_type)

    # Render the index page for GET requests
    return render_template('index.html')

if __name__ == "__main__":
    # Enable support for reverse proxies (e.g., when deployed on platforms like Vercel)
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)

    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)
