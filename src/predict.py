import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import tensorflow as tf
from config.settings import (
    MODEL_PATH, MODEL_JSON_PATH, IMAGE_SIZE, CLASSES,
    CLASS_NAMES, TREATMENTS
)


class LeafBlightPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please train the model first using: python src/train.py"
            )

        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASSES
        self.class_display_names = CLASS_NAMES
        self.treatments = TREATMENTS

    def preprocess_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, IMAGE_SIZE)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        return image

    def _predict_single(self, image_array):
        predictions = self.model.predict(image_array, verbose=0)
        preds = predictions[0]

        if preds.shape[0] == 4:
            # Map from old 4-class model to new 2-class potato-only
            # Old indices: 0=healthy_potato, 1=healthy_tomato, 2=potato_blight, 3=tomato_blight
            # New indices: 0=healthy_potato, 1=potato_blight
            potato_predictions = np.array([preds[0], preds[2]], dtype=np.float32)
        elif preds.shape[0] == 2:
            potato_predictions = np.array(preds, dtype=np.float32)
        else:
            raise ValueError(
                f"Unexpected number of model output classes: {preds.shape[0]}. "
                "Model must output either 2 or 4 class probabilities."
            )

        total = potato_predictions.sum()
        if total <= 0:
            return potato_predictions

        potato_predictions = potato_predictions / total
        return potato_predictions

    def predict(self, image, use_tta=True):
        processed_image = self.preprocess_image(image)

        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            original = image.copy()

        original_resized = cv2.resize(original, IMAGE_SIZE).astype(np.float32) / 255.0
        original_batch = np.expand_dims(original_resized, axis=0)

        base_probs = self._predict_single(original_batch)

        if use_tta:
            aug_probs_list = [base_probs]

            flip_lr = np.expand_dims(np.fliplr(original_resized), axis=0)
            aug_probs_list.append(self._predict_single(flip_lr))

            rows, cols = original_resized.shape[:2]
            for angle in [10, -10]:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
                rotated = cv2.warpAffine(original_resized, M, (cols, rows))
                rotated_batch = np.expand_dims(rotated, axis=0)
                aug_probs_list.append(self._predict_single(rotated_batch))

            for brightness in [0.85, 1.15]:
                bright_img = np.clip(original_resized * brightness, 0, 1)
                bright_batch = np.expand_dims(bright_img, axis=0)
                aug_probs_list.append(self._predict_single(bright_batch))

            avg_probs = np.mean(aug_probs_list, axis=0)
            avg_probs = avg_probs / avg_probs.sum()
        else:
            avg_probs = base_probs

        predicted_class_idx = np.argmax(avg_probs)
        confidence = float(avg_probs[predicted_class_idx])

        all_probabilities = {
            self.class_names[i]: float(avg_probs[i])
            for i in range(len(self.class_names))
        }

        predicted_class = self.class_names[predicted_class_idx]
        display_name = self.class_display_names[predicted_class]
        treatment = self.treatments.get(predicted_class, "Consult an agricultural expert.")

        result = {
            'predicted_class': predicted_class,
            'display_name': display_name,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'treatment': treatment,
            'is_healthy': 'healthy' in predicted_class,
        }

        return result

    def predict_batch(self, image_paths):
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results

    def get_class_info(self):
        return {
            'classes': self.class_names,
            'display_names': self.class_display_names,
            'treatments': self.treatments
        }


def predict_single_image(image_path, model_path=None):
    predictor = LeafBlightPredictor(model_path=model_path)
    result = predictor.predict(image_path)

    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Prediction: {result['display_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print()
    print("All Probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        bar = "#" * int(prob * 30)
        print(f"  {class_name:20s}: {prob:.2%} {bar}")
    print()
    if result['is_healthy']:
        print(f"Status: HEALTHY")
    else:
        print(f"Status: DISEASE DETECTED")
    print(f"\nRecommended Treatment:")
    print(f"  {result['treatment']}")
    print("=" * 50 + "\n")

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict Leaf Blight from Image')
    parser.add_argument('image', type=str, help='Path to the leaf image')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable test-time augmentation')

    args = parser.parse_args()

    predictor = LeafBlightPredictor(model_path=args.model)
    result = predictor.predict(args.image, use_tta=not args.no_tta)

    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)
    print(f"Image: {args.image}")
    print(f"Prediction: {result['display_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print()
    print("All Probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        bar = "#" * int(prob * 30)
        print(f"  {class_name:20s}: {prob:.2%} {bar}")
    print()
    if result['is_healthy']:
        print(f"Status: HEALTHY")
    else:
        print(f"Status: DISEASE DETECTED")
    print(f"\nRecommended Treatment:")
    print(f"  {result['treatment']}")
    print("=" * 50 + "\n")
