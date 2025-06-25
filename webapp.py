from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from transformers import pipeline
from PIL import Image
import os
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model once at startup
classifier = pipeline(
    "image-classification",
    model="Organika/sdxl-detector"
)

def detect_image(image):
    results = list(classifier(image))
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['image']
        if not file.filename:
            return render_template('index.html', error='No selected file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath).convert("RGB")
            result = detect_image(image)
    return render_template('index.html', result=result, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 