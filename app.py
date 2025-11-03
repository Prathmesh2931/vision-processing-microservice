from flask import Flask, request, render_template, jsonify
import cv2
import torch
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read and process image
        image = Image.open(file.stream)
        
        # Run YOLO detection
        results = model(image)
        
        # Get detection results
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        
        # Convert image to base64 for display
        img_array = np.array(image)
        annotated_img = results.render()[0]
        
        # Convert to base64
        pil_img = Image.fromarray(annotated_img)
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'detections': detections,
            'image': img_str,
            'count': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'robot-vision-pipeline'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)