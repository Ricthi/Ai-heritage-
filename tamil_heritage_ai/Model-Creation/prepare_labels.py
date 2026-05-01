# prepare_labels.py
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import random

def prepare_data(labels_root: str = "Labels",
                 img_size: int = 50,
                 ignore=None,
                 out_dir: str = "data/processed"):
    if ignore is None:
        ignore = ['1 - Multipart', '2 - Unknown',
                  'Labelled Dataset - Fig 51',
                  'Labelled Dataset - Fig 11']

    IMG_SIZE = img_size
    root_dir = Path(labels_root)
    if not root_dir.exists():
        print(f"Labels directory not found at: {root_dir.resolve()}")
        return

    categories_set = set()

    # pass 1: discover character categories
    for path in root_dir.rglob('*'):
        if path.is_dir() and path.name not in ignore:
            if any(path.glob('*.png')) or any(path.glob('*.jpg')) or any(path.glob('*.jpeg')):
                categories_set.add(path.name)

    categories = sorted(list(categories_set))
    print(f"Found {len(categories)} character categories.")

    training_data = []
    for category in categories:
        class_num = categories.index(category)

        # folders named like the category anywhere under root
        for cat_dir in root_dir.rglob(category):
            if not cat_dir.is_dir():
                continue
            for img_path in cat_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    try:
                        img_array = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if img_array is None:
                            continue
                        resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        training_data.append([resized, class_num])
                    except Exception:
                        # ignore corrupt images
                        pass

    random.shuffle(training_data)

    X, y = [], []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "X.pickle"), "wb") as f:
        pickle.dump(X, f)
    with open(os.path.join(out_dir, "y.pickle"), "wb") as f:
        pickle.dump(y, f)
    with open(os.path.join(out_dir, "categories.pickle"), "wb") as f:
        pickle.dump(categories, f)

    print(f"Processed {len(X)} images across {len(categories)} categories.")


if __name__ == "__main__":
    prepare_data()
