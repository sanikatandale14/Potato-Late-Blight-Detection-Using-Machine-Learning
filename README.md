# Leaf Blight Detection System

An intelligent machine learning system for detecting leaf blight disease in tomato and potato plants using image processing and deep learning techniques.

## Features

- **Multi-class Classification**: Detects 4 classes - Healthy Potato, Healthy Tomato, Potato Blight, Tomato Blight
- **CNN-based Deep Learning**: Custom CNN architecture with batch normalization and dropout
- **Transfer Learning Support**: Pre-trained MobileNetV2, ResNet50, and EfficientNetB0 models
- **Test-Time Augmentation (TTA)**: Multiple augmented predictions for more accurate results
- **Data Augmentation**: Rotation, shifting, zooming, brightness, and flipping
- **Class Weight Balancing**: Automatic handling of imbalanced datasets
- **Web Interface**: Flask-based web application with drag-and-drop image upload
- **Treatment Suggestions**: Provides recommended treatments for detected diseases
- **Training Visualization**: Real-time training metrics and history tracking

## Project Structure

```
leaf_blight_detection/
├── app.py                      # Flask web application
├── setup_dataset.py            # Dataset download and setup
├── requirements.txt            # Python dependencies
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── model.py                # CNN model architecture
│   ├── data_preprocessing.py   # Data loading and augmentation
│   ├── train.py                # Model training script
│   └── predict.py              # Prediction/inference module
├── templates/
│   ├── index.html              # Main web interface
│   └── history.html            # Training history page
├── static/
│   ├── css/style.css           # Styles
│   ├── js/main.js              # Frontend logic
│   └── uploads/                # Uploaded images
├── models/                     # Trained model files
└── dataset/                    # Dataset directory
    └── plant_dataset/
        ├── healthy_potato/
        ├── healthy_tomato/
        ├── potato_blight/
        └── tomato_blight/
```

## Installation

1. **Clone or download the project**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   source venv/bin/activate # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

### Download Real Dataset (REQUIRED for accurate predictions)

The model MUST be trained on real leaf images to work correctly. Synthetic images will NOT produce accurate results.

```bash
python setup_dataset.py --download
```

This will download real plant disease images from Kaggle's PlantVillage dataset (~200MB).

### Manual Download

If the automatic download fails, download manually:

1. Go to [Kaggle Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. Download and extract
3. Copy images into the project structure:
   ```
   New Plant Diseases Dataset(Augmented)/train/Potato___healthy/       -> dataset/plant_dataset/healthy_potato/
   New Plant Diseases Dataset(Augmented)/train/Tomato___healthy/       -> dataset/plant_dataset/healthy_tomato/
   New Plant Diseases Dataset(Augmented)/train/Potato___Late_blight/   -> dataset/plant_dataset/potato_blight/
   New Plant Diseases Dataset(Augmented)/train/Tomato___Late_blight/   -> dataset/plant_dataset/tomato_blight/
   ```

### Verify Dataset

```bash
python setup_dataset.py --verify
```

## Usage

### 1. Train the Model

**Transfer Learning with MobileNetV2 (Recommended - best accuracy):**
```bash
python src/train.py --model MobileNetV2 --epochs 30
```

**Transfer Learning with EfficientNetB0:**
```bash
python src/train.py --model EfficientNetB0 --epochs 30
```

**Transfer Learning with ResNet50:**
```bash
python src/train.py --model ResNet50 --epochs 30
```

**Custom CNN (for smaller datasets):**
```bash
python src/train.py --model custom --epochs 30
```

**Disable augmentation (if dataset is already large/augmented):**
```bash
python src/train.py --model MobileNetV2 --no-augmentation
```

### 2. Predict on an Image

```bash
python src/predict.py path/to/leaf_image.jpg
```

**Disable test-time augmentation (faster but less accurate):**
```bash
python src/predict.py path/to/leaf_image.jpg --no-tta
```

### 3. Run Web Application

```bash
python app.py
```

Access the application at: http://localhost:5000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Upload image and get prediction |
| `/predict_url` | POST | Predict using local file path |
| `/api/classes` | GET | Get class information |
| `/history` | GET | View training history |

## Model Architecture

### Custom CNN
- 4 Convolutional blocks with Batch Normalization
- Kernel sizes: 32 -> 64 -> 128 -> 256 filters
- Max Pooling after each block
- Dropout (0.15-0.2) for regularization
- Fully connected layers: 512 -> 256 -> 4 (output)
- Softmax activation for multi-class classification

### Transfer Learning
- Pre-trained backbone (MobileNetV2/ResNet50/EfficientNetB0) on ImageNet
- Global Average Pooling
- Dense layers: 256 -> 128 -> 4 (output)
- L2 regularization (0.0001) for generalization

## Configuration

Edit `config/settings.py` to customize:
- `IMAGE_SIZE`: Input image dimensions (default: 224x224)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 30)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `CLASSES`: List of class names

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV
- Flask
- NumPy
- Matplotlib
- scikit-learn
- kagglehub (for automatic dataset download)

## Output Classes

| Class | Description |
|-------|-------------|
| Healthy Potato | Normal potato leaves, no disease |
| Healthy Tomato | Normal tomato leaves, no disease |
| Potato Blight | Potato leaves with late blight symptoms |
| Tomato Blight | Tomato leaves with early/late blight symptoms |

## Troubleshooting

### Model gives the same prediction for every image

This means the model was trained on synthetic/fake data. You must:
1. Delete existing model files in `models/`
2. Download real dataset: `python setup_dataset.py --download`
3. Retrain: `python src/train.py --model MobileNetV2`

### Not enough memory during training

Try:
- Reduce batch size: Edit `BATCH_SIZE` in `config/settings.py`
- Use `MobileNetV2` (smallest transfer learning model)
- Close other applications

### Download fails

Download the dataset manually from Kaggle and organize as shown in the Dataset Setup section.

## Future Enhancements

- Real-time camera capture integration
- Treatment database with detailed recommendations
- Historical plant health tracking
- Mobile application deployment
- Additional disease classes
- Model optimization for edge devices

## License

This project is for educational and research purposes.
"# Potato-Late-Blight-Detection-Using-Machine-Learning" 
