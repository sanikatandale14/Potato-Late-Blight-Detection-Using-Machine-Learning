import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.model import create_custom_cnn, create_transfer_learning_model
from src.data_preprocessing import (
    create_data_generators, compute_class_weights, generator_wrapper
)
from config.settings import (
    EPOCHS, BATCH_SIZE, MODEL_PATH, MODEL_JSON_PATH,
    MODEL_WEIGHTS_PATH, HISTORY_PATH, NUM_CLASSES
)


def plot_training_history(history, save_path='models/training_plots.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to: {save_path}")


def save_history(history, path=HISTORY_PATH):
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }

    with open(path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {path}")


def evaluate_model(model, test_generator):
    if test_generator is None or test_generator['samples'] == 0:
        print("\nNo test data available for evaluation.")
        return None, None

    print("\n" + "=" * 50)
    print("Evaluating Model on Test Data")
    print("=" * 50)

    steps = (test_generator['samples'] + test_generator['batch_size'] - 1) // test_generator['batch_size']
    test_gen = generator_wrapper(test_generator)

    test_loss, test_accuracy = model.evaluate(test_gen, steps=steps, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    predictions = model.predict(test_gen, steps=steps, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.array(test_generator['labels'])

    from sklearn.metrics import classification_report, confusion_matrix

    class_names = list(test_generator['class_indices'].keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes,
                               target_names=class_names))

    cm = confusion_matrix(true_classes, predicted_classes)
    print("\nConfusion Matrix:")
    print(cm)

    return test_accuracy, test_loss


def train_model(model_type='MobileNetV2', use_augmentation=True, epochs=None):
    if epochs is None:
        epochs = EPOCHS

    print("=" * 50)
    print("Leaf Blight Detection Model Training")
    print("=" * 50)
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Augmentation: {'ON' if use_augmentation else 'OFF'}")
    print()

    print("Loading data generators...")
    train_gen, val_gen, test_gen = create_data_generators(
        batch_size=BATCH_SIZE,
        use_augmentation=use_augmentation,
    )

    if train_gen is None:
        print("ERROR: No training data found. Please run: python setup_dataset.py --download")
        return None, None

    train_steps = (train_gen['samples'] + train_gen['batch_size'] - 1) // train_gen['batch_size']
    val_steps = (val_gen['samples'] + val_gen['batch_size'] - 1) // val_gen['batch_size'] if val_gen and val_gen['samples'] > 0 else None

    print(f"Training samples: {train_gen['samples']} ({train_steps} steps/epoch)")
    print(f"Validation samples: {val_gen['samples'] if val_gen else 0}")
    print(f"Test samples: {test_gen['samples'] if test_gen else 0}")
    print(f"Classes: {train_gen['class_indices']}")
    print()

    print("Computing class weights...")
    class_weights = compute_class_weights(train_gen)
    print()

    if model_type == 'custom':
        print("Creating custom CNN model...")
        model = create_custom_cnn()
    elif model_type in ['MobileNetV2', 'ResNet50', 'EfficientNetB0']:
        print(f"Creating transfer learning model ({model_type})...")
        model = create_transfer_learning_model(base_model_name=model_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.summary()

    os.makedirs('models', exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]

    val_data = generator_wrapper(val_gen) if val_gen else None

    print("\nTraining model...")
    history = model.fit(
        generator_wrapper(train_gen),
        steps_per_epoch=train_steps,
        validation_data=val_data,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nSaving model...")
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    model_json = model.to_json()
    with open(MODEL_JSON_PATH, 'w') as f:
        json.dump({
            'model_json': model_json,
            'class_indices': train_gen['class_indices'],
            'model_type': model_type,
        }, f)
    print(f"Model architecture saved to: {MODEL_JSON_PATH}")

    model.save_weights(MODEL_WEIGHTS_PATH)
    print(f"Model weights saved to: {MODEL_WEIGHTS_PATH}")

    save_history(history)
    plot_training_history(history)

    evaluate_model(model, test_gen)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

    return model, history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Leaf Blight Detection Model')
    parser.add_argument('--model', type=str, default='MobileNetV2',
                       choices=['custom', 'MobileNetV2', 'ResNet50', 'EfficientNetB0'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        use_augmentation=not args.no_augmentation,
        epochs=args.epochs,
    )
