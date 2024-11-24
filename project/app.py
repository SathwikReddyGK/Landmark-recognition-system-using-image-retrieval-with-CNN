from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('landmark_recognition_model.h5')

# Define the labels (adjust based on your model's output)
labelsNPY = np.load('../labels.npy') #"['Label1', 'Label2', 'Label3']  # Replace with your actual class labels

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

class_indices = np.argmax(labelsNPY, axis=1)
LABELS = le.inverse_transform(class_indices)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        image = load_img(filepath, target_size=(224, 224))  # Adjust size as per your model
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0  # Normalize if needed

        # Make prediction
        predictions = model.predict(image)
        # predicted_label = LABELS[np.argmax(predictions)]
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = le.inverse_transform([predicted_index])[0]

        print("Predictions", predictions)
        print("Argmax",np.argmax(predictions))
        
        # Render the result
        return render_template('result.html', label=predicted_label, filepath=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)