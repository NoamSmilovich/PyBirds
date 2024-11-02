from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from classifier import SimpleCNN
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import pandas as pd

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

model = SimpleCNN()
state_dict = torch.load("model_weights/classifier_weights.pth", map_location=torch.device('cpu'), weights_only=True)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

labels_df = pd.read_csv('labels.csv')
class_to_label = dict(zip(labels_df['class_id'], labels_df['label']))

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit model's expected input size
    transforms.ToTensor()
])

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image = Image.open(BytesIO(file.read())).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_class = output.max(1)

            result = {
                # 'species': predicted_class.item(),  
                'species': class_to_label.get(predicted_class.item(), 'Unknown Species'),  # Get the species name or default
                'confidence': torch.softmax(output, dim=1).max().item() 
            }

            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
