from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline
from PIL import Image
from gradio_client import Client
import os

# Load the image classification pipeline
pipe = pipeline("image-classification", model="SanketJadhav/PlantDiseaseClassifier-Resnet50")

# Initialize the Gradio client
client = Client("Be-Bo/llama-3-chatbot_70b")

# Function to classify an image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    results = pipe(image)
    return results

# Function to get response from the Gradio model
def get_response(disease):
    result = client.predict(
        message=f"Provide solutions for {disease}.Give the response as easy and understandable. Just give response as i mentioned don't give any extra information.The response should be structured using HTML tags as follows:\n"
                f"<h2>Causes</h2>\n"
                f"<ul>\n<li>List each cause inside a <li> tag.</li>\n</ul>"
                f"<h2>Cures</h2>\n"
                f"<ul>\n<li>List each cure inside a <li> tag.</li>\n</ul>"
                f"<h2>Tips for Growing</h2>\n"
                f"<ul>\n<li>List each tip inside a <li> tag.</li>\n</ul>"
                f"Ensure there are no special characters like asterisks (*) and separate points with proper HTML formatting.",
        api_name="/chat"
    )
    return result


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            results = classify_image(file_path)
            disease = results[0]['label']
            solution = get_response(disease)  # Fetch the solution for the identified disease
            return render_template('index.html', filename=file.filename, disease=disease, solution=solution)
    return render_template('index.html', filename=None, disease=None, solution=None)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
