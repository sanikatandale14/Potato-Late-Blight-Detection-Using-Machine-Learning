import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.settings import IMAGE_SIZE, CLASSES, DATASET_PATH


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    return image


def preprocess_image(image):
    image = image.astype(np.float32)
    image = image / 255.0
    return image


def load_dataset(dataset_path=None):
    if dataset_path is None:
        dataset_path = DATASET_PATH

    images = []
    labels = []
    class_names = CLASSES

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory not found: {class_path}")
            continue

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                image = load_image(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(class_idx)

    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images across {len(set(labels))} classes")
    return images, labels


def load_all_images_and_labels(dataset_path=None):
    if dataset_path is None:
        dataset_path = DATASET_PATH

    image_paths = []
    labels = []
    class_indices = {}

    for idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: Class directory not found: {class_path}")
            continue
        class_indices[class_name] = idx

        for filename in sorted(os.listdir(class_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_path, filename))
                labels.append(idx)

    print(f"Found {len(image_paths)} images belonging to {len(class_indices)} classes.")
    return image_paths, labels, class_indices


def image_generator_from_paths(image_paths, labels, datagen, batch_size=32,
                                target_size=None, shuffle=False, seed=None):
    if target_size is None:
        target_size = IMAGE_SIZE

    indices = np.arange(len(image_paths))
    if shuffle and seed is not None:
        np.random.seed(seed)
        np.random.shuffle(indices)
    elif shuffle:
        np.random.shuffle(indices)

    n_samples = len(image_paths)

    gen = {
        'filepaths': [image_paths[i] for i in indices],
        'labels': [labels[i] for i in indices],
        'samples': n_samples,
        'class_indices': None,
        'datagen': datagen,
        'batch_size': batch_size,
        'target_size': target_size,
        'n_classes': len(set(labels)),
    }

    def generator():
        i = 0
        while True:
            if i >= n_samples:
                if shuffle:
                    np.random.shuffle(indices)
                    gen['filepaths'] = [image_paths[j] for j in indices]
                    gen['labels'] = [labels[j] for j in indices]
                i = 0

            batch_end = min(i + batch_size, n_samples)
            batch_paths = gen['filepaths'][i:batch_end]
            batch_images = []

            for path in batch_paths:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    batch_images.append(img)

            if len(batch_images) > 0:
                batch_array = np.array(batch_images, dtype=np.float32)
                for idx in range(len(batch_array)):
                    batch_array[idx] = datagen.random_transform(batch_array[idx])
                batch_array = datagen.standardize(batch_array)

                batch_labels_list = gen['labels'][i:batch_end]
                n_labels = len(batch_labels_list)
                n_classes = gen['n_classes']
                batch_labels = np.zeros((n_labels, n_classes))
                for j, lbl in enumerate(batch_labels_list):
                    batch_labels[j, lbl] = 1.0

                yield batch_array, batch_labels

            i = batch_end

    gen['generator'] = generator()
    return gen


def create_data_generators(train_split=0.7, val_split=0.15, test_split=0.15,
                           batch_size=32, use_augmentation=True):
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 0.01:
        print(f"Warning: Splits sum to {total_split}, normalizing to 1.0")
        train_split /= total_split
        val_split /= total_split
        test_split /= total_split

    image_paths, labels, class_indices = load_all_images_and_labels()

    if len(image_paths) == 0:
        print("ERROR: No images found. Please download the dataset first.")
        return None, None, None

    X_paths_train, X_paths_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=(1 - train_split), random_state=42, stratify=labels
    )

    val_ratio = val_split / (val_split + test_split)
    X_paths_val, X_paths_test, y_val, y_test = train_test_split(
        X_paths_temp, y_temp, test_size=(1 - val_ratio), random_state=42, stratify=y_temp
    )

    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = image_generator_from_paths(
        X_paths_train, y_train, train_datagen, batch_size=batch_size, shuffle=True, seed=42
    )
    val_generator = image_generator_from_paths(
        X_paths_val, y_val, val_datagen, batch_size=batch_size, shuffle=False
    )
    test_generator = image_generator_from_paths(
        X_paths_test, y_test, test_datagen, batch_size=batch_size, shuffle=False
    )

    for gen in [train_generator, val_generator, test_generator]:
        gen['class_indices'] = class_indices

    print(f"  Train: {train_generator['samples']} samples")
    print(f"  Val:   {val_generator['samples']} samples")
    print(f"  Test:  {test_generator['samples']} samples")
    print(f"  Augmentation: {'ON' if use_augmentation else 'OFF'}")

    return train_generator, val_generator, test_generator


def generator_wrapper(gen, num_classes=None):
    if num_classes is None:
        num_classes = gen['n_classes']

    while True:
        batch_x, batch_y = next(gen['generator'])
        yield batch_x, batch_y


def create_arrays_from_generator(gen, num_batches=None):
    if num_batches is None:
        num_batches = (gen['samples'] + gen['batch_size'] - 1) // gen['batch_size']

    images, labels = [], []
    for _ in range(num_batches):
        batch_x, batch_y = next(gen['generator'])
        images.append(batch_x)
        labels.append(batch_y)

    return np.vstack(images), np.vstack(labels)


def compute_class_weights(train_generator):
    from sklearn.utils.class_weight import compute_class_weight

    lbls = np.array(train_generator['labels'])
    classes = np.unique(lbls)
    weights = compute_class_weight('balanced', classes=classes, y=lbls)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}

    inverse_indices = {v: k for k, v in train_generator['class_indices'].items()}
    print("\nClass Weights:")
    for idx in sorted(class_weight_dict.keys()):
        name = inverse_indices.get(idx, f"class_{idx}")
        print(f"  {name} (idx {idx}): {class_weight_dict[idx]:.2f}")

    return class_weight_dict


def augment_and_save(source_dir, target_dir, samples_per_class=500):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    os.makedirs(target_dir, exist_ok=True)

    for class_name in CLASSES:
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        if not os.path.exists(source_class_dir):
            continue

        original_images = [f for f in os.listdir(source_class_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in original_images:
            img_path = os.path.join(source_class_dir, img_file)
            target_path = os.path.join(target_class_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(target_path, img)

        current_count = len(original_images)
        augment_count = max(0, samples_per_class - current_count)

        if augment_count > 0 and current_count > 0:
            for i in range(augment_count):
                img_file = original_images[i % current_count]
                img_path = os.path.join(source_class_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = img.reshape((1,) + img.shape)
                aug_iter = datagen.flow(img, batch_size=1)
                augmented_img = next(aug_iter)[0].astype(np.uint8)

                aug_filename = f"aug_{i:04d}_{img_file}"
                aug_path = os.path.join(target_class_dir, aug_filename)
                augmented_img_bgr = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_path, augmented_img_bgr)

        print(f"Class '{class_name}': {samples_per_class} images in augmented dataset")

    print(f"Augmented dataset saved to: {target_dir}")
