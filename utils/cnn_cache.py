import os
import numpy as np


def features_exist(features_dir: str, cnn_name: str):
    paths = get_feature_paths(features_dir, cnn_name)
    return all(os.path.exists(path) for path in paths.values())


def get_feature_paths(features_dir: str, cnn_name: str):
    return {
        "X_train": os.path.join(features_dir, f"{cnn_name}_X_train.npy"),
        "y_train": os.path.join(features_dir, f"{cnn_name}_y_train.npy"),
        "X_val": os.path.join(features_dir, f"{cnn_name}_X_val.npy"),
        "y_val": os.path.join(features_dir, f"{cnn_name}_y_val.npy"),
        "X_test": os.path.join(features_dir, f"{cnn_name}_X_test.npy"),
        "y_test": os.path.join(features_dir, f"{cnn_name}_y_test.npy"),
    }


def save_features(features_dir: str, cnn_name: str, X_train, y_train, X_val, y_val, X_test, y_test):
    paths = get_feature_paths(features_dir, cnn_name)

    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_test"], X_test)
    np.save(paths["y_test"], y_test)

    print(f"Backbone saved for {cnn_name.upper()} in: {features_dir}")


def load_features(features_dir: str, cnn_name: str):
    paths = get_feature_paths(features_dir, cnn_name)

    X_train = np.load(paths["X_train"])
    y_train = np.load(paths["y_train"])
    X_val = np.load(paths["X_val"])
    y_val = np.load(paths["y_val"])
    X_test = np.load(paths["X_test"])
    y_test = np.load(paths["y_test"])

    print(f"Features loaded for {cnn_name.upper()} of: {features_dir}")

    return X_train, y_train, X_val, y_val, X_test, y_test