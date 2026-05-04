CLASSES = [
    "healthy_potato",
    "potato_blight"
]

CLASS_NAMES = {
    "healthy_potato": "Healthy Potato",
    "potato_blight": "Potato Blight"
}

NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

MODEL_PATH = "models/leaf_blight_model.h5"
MODEL_JSON_PATH = "models/leaf_blight_model.json"
MODEL_WEIGHTS_PATH = "models/leaf_blight_weights.weights.h5"
HISTORY_PATH = "models/training_history.json"

DATASET_PATH = "dataset/plant_dataset"
AUGMENTED_DATASET_PATH = "dataset/augmented"

TREATMENTS = {
    "healthy_potato": "No treatment needed. Continue regular care and monitoring.",
    "potato_blight": "Apply fungicide containing chlorothalonil or mancozeb. Remove infected leaves. Ensure proper spacing for air circulation. Avoid overhead watering."
}
