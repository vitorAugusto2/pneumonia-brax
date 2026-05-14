import os
import json


def get_best_params_path(base_dir: str, cnn_name: str, clf_name: str) -> str:
    params_dir = os.path.join(base_dir, "utils", "best_params")
    os.makedirs(params_dir, exist_ok=True)

    return os.path.join(params_dir, f"{cnn_name}_{clf_name}_best_params.json")


def save_best_params(base_dir: str, cnn_name: str, clf_name: str, best_params: dict):
    path = get_best_params_path(base_dir, cnn_name, clf_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    print(f"[CV] Best params saved in: {path}")


def load_best_params(base_dir: str, cnn_name: str, clf_name: str):
    path = get_best_params_path(base_dir, cnn_name, clf_name)

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    print(f"[CV] Best params loaded from: {path}")
    return best_params