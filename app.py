from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
import os

app = Flask(__name__)
CORS(app)

# ------------------------
# Load YOLO model
# ------------------------
MODEL_PATH = "best.pt"
model = None

def get_model():
    """Lazy load model on first request"""
    global model
    if model is None:
        if Path(MODEL_PATH).exists():
            try:
                model = YOLO(MODEL_PATH)
                print("✅ YOLO model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise
        else:
            print(f"❌ Model not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return model

# ------------------------
# Process image with YOLO
# ------------------------
def process_with_model(img):
    """
    Runs YOLO detection and returns:
    - annotated_img: image with bounding boxes + mango count text
    - mango_count: total number of mangoes detected
    - detections: list of bounding boxes + confidence + label
    """
    if img is None:
        return img, 0, []

    try:
        model = get_model()  # Lazy load
        
        # Run YOLO on CPU
        results = model.predict(source=img, conf=0.01, imgsz=1280, device='cpu', save=False)

        # Annotated image with bounding boxes
        annotated_img = results[0].plot()

        # Count mangoes
        mango_count = len(results[0].boxes)

        # Draw mango count on the image
        text = f"Total Mangoes: {mango_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        color = (255, 255, 255)  # white color
        
        # Place text at top-left corner
        annotated_img = cv2.putText(
            annotated_img,
            text,
            (30, 50),
            font,
            font_scale,
            color,
            thickness
        )

        # Extract detections for JSON
        detections = []
        for r in results[0].boxes.data.tolist():  # x1, y1, x2, y2, conf, cls
            x1, y1, x2, y2, conf, cls = r
            detections.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "confidence": conf,
                "class": int(cls),
                "label": model.names[int(cls)]
            })

        return annotated_img, mango_count, detections

    except Exception as e:
        print("YOLO inference error:", e)
        return img, 0, []


# ------------------------
# Flask endpoints
# ------------------------
@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    model_status = "not loaded"
    try:
        get_model()
        model_status = "loaded"
    except:
        model_status = "error loading"
    
    return jsonify({
        'status': 'API is running',
        'model_status': model_status,
        'endpoints': {
            '/detect': 'POST - Returns annotated image',
            '/predict_json': 'POST - Returns JSON with detections',
            '/health': 'GET - Health check'
        }
    }), 200

@app.route('/detect', methods=['POST'])
def detect_mango():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Run YOLO
        annotated_img, mango_count, _ = process_with_model(img)

        # Encode annotated image
        success, buffer = cv2.imencode('.jpg', annotated_img)
        if not success:
            return jsonify({'error': 'Failed to encode image'}), 500

        io_buf = BytesIO(buffer)
        io_buf.seek(0)

        # Return image + mango count as headers
        response = send_file(io_buf, mimetype='image/jpeg')
        response.headers['X-Mango-Count'] = str(mango_count)
        return response

    except Exception as e:
        print(f"Error in /detect: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_json', methods=['POST'])
def detect_mango_json():
    """
    Endpoint to return JSON with mango count and bounding boxes
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        _, mango_count, detections = process_with_model(img)

        return jsonify({
            "success": True,
            "mango_count": mango_count,
            "detections": detections
        })

    except Exception as e:
        print(f"Error in /predict_json: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        get_model()
        model_loaded = True
    except:
        model_loaded = False
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    }), 200

if __name__ == '__main__':
    # Use PORT environment variable provided by Render
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
