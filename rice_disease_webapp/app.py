from flask import Flask, render_template, request
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import base64
import re

app = Flask(__name__)

# Load model
model = torch.jit.load('model/rice_model.pt', map_location=torch.device('cpu'))
model.eval()

# Class names
class_names = ['Hispa', 'bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast',
               'leaf_scald', 'leaf_smut', 'neck_blast', 'sheath_blight', 'tungro']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = None
    image_data_url = None

    # Handle file upload
    if 'image' in request.files and request.files['image'].filename != '':
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()

    # Handle camera image
    elif 'camera_image' in request.form:
        base64_data = request.form['camera_image']
        base64_data = re.sub('^data:image/.+;base64,', '', base64_data)
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()

    if image is None:
        return "No image provided", 400

    # Model prediction
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item() * 100
        predicted_class = class_names[predicted_idx] if confidence >= 70 else "Unknown"

    return render_template('result.html',
                           prediction=predicted_class,
                           confidence=confidence,
                           image_data=image_data_url)

if __name__ == '__main__':
    app.run(debug=True)
