import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from src.predict import LeafBlightPredictor
from config.settings import MODEL_PATH, CLASSES, CLASS_NAMES, TREATMENTS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = LeafBlightPredictor()
        except FileNotFoundError:
            return None
    return predictor


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    model_loaded = get_predictor() is not None
    class_info = {
        'classes': CLASSES,
        'display_names': CLASS_NAMES,
        'treatments': TREATMENTS
    }
    return render_template('index.html', model_loaded=model_loaded, class_info=class_info)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use PNG, JPG, or JPEG'}), 400

    predictor = get_predictor()
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    filename = secure_filename(file.filename or 'unknown')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predictor.predict(filepath)
        result['image_url'] = f'/static/uploads/{filename}'
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_url', methods=['POST'])
def predict_url():
    data = request.get_json()
    image_path = data.get('image_path', '')

    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Invalid image path'}), 400

    predictor = get_predictor()
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        result = predictor.predict(image_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': CLASSES,
        'display_names': CLASS_NAMES,
        'treatments': TREATMENTS
    })


@app.route('/history')
def history():
    history_path = 'models/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        return render_template('history.html', history=history_data)
    return render_template('history.html', history=None)


if __name__ == '__main__':
    print("Starting Leaf Blight Detection Web Application...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
