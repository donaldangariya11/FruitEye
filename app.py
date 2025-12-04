from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ------------------------
# Load YOLO model
# ------------------------
MODEL_PATH = "best.pt"
model = None

def load_yolo_model():
    global model
    if Path(MODEL_PATH).exists():
        model = YOLO(MODEL_PATH)
        print("✅ YOLO model loaded successfully!")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")

load_yolo_model()

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
    if model is None or img is None:
        return img, 0, []

    try:
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
        color = (0, 255, 0)
        # Place text at top-left corner
        annotated_img = cv2.putText(annotated_img, text, (30, 50), font, font_scale, color, thickness)

        # Extract detections for JSON
        detections = []
        for r in results[0].boxes.data.tolist():  # x1, y1, x2, y2, conf, cls
            x1, y1, x2, y2, conf, cls = r
            detections.append({
                "x1": x1, "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
                "class": int(cls),
                "label": model.names[int(cls)]
            })
        annotated_img = cv2.putText(
            annotated_img,
            f"Total Mangoes: {mango_count}",
            (30, 100),  # x, y position
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,  # font scale
            (255, 255, 255),  # green color
            3  # thickness
        )

        return annotated_img, mango_count, detections

    except Exception as e:
        print("YOLO inference error:", e)
        return img, 0, []


# ------------------------
# Flask endpoints
# ------------------------
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

        # Return image + mango count as headers (or use JSON endpoint)
        response = send_file(io_buf, mimetype='image/jpeg')
        response.headers['X-Mango-Count'] = str(mango_count)
        return response

    except Exception as e:
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
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
