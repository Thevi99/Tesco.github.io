from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__, template_folder='.')

# Load the model
MODEL_PATH = 'sand_classification_model20-10.h5'
model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Class labels
class_labels = ['A (Standard)', 'B (Coarse)', 'C (Fine)', 'Notsandimage']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Read and preprocess the image
        image = Image.open(file_path)

        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Resize to match model input
        image = image.resize((765, 1020), Image.Resampling.LANCZOS)

        # Convert to NumPy array and ensure correct shape
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[:, :, :3]  # Remove alpha channel

        image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0  # Normalize

        # Make prediction
        prediction = model.predict(image)
        predicted_class = class_labels[np.argmax(prediction)]
        probabilities = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
        
        result = {
            'prediction': predicted_class,
            'probabilities': probabilities,
            'image_url': f'/uploads/{file.filename}'
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
