import os
import time

from src.split_set import csv_to_bin, split_set
from src.data_loaders import get_loaders
from src.cnn_backbone import get_backbone
from src.feature_extractor import get_device, get_extract_features
from src.classifiers import get_classifiers
from src.evaluate import get_evaluate
from utils.transforms import get_transforms
#from utils.plot import plot_confusion_matrix, plot_precision_recall, plot_roc_curve
from utils.format_time import get_format_time


cnns = ["vgg16", "resnet50"]            # "vgg16", "resnet50"
classifiers = ["rf", "xgb", "svm"]      # "rf", "xgb", "svm"

device = get_device()

base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs("./images", exist_ok=True)
os.makedirs("./data", exist_ok=True)
os.makedirs("./dataset", exist_ok=True)
os.makedirs("./plots", exist_ok=True)


def main():
    # Dataset Binary + Split in train/val/test
    path_master_csv = os.path.join(base_dir, "dataset/master_spreadsheet_update.csv")
    csv_to_bin(path_master_csv)

    path_binary_csv = os.path.join(base_dir, "dataset/dataset_binary.csv")
    split_set(path_binary_csv)

    for cnn in cnns:
        # CNN
        model, preprocess, feat_img = get_backbone(cnn)

        # Transforms
        train_transform, eval_transform = get_transforms(preprocess)

        # Create Data Loaders
        loaders = get_loaders(
            base_dir = base_dir,
            train_csv = "./data/train.csv",
            val_csv = "./data/val.csv",
            test_csv = "./data/test.csv",
            train_transform=train_transform,
            eval_transform=eval_transform,
            batch_size = 32,
            num_workers = 2
        )

        # Extract features
        t0_feat = time.time()
        X_train, y_train = get_extract_features(loaders["train"], model, device=device)
        X_val, y_val = get_extract_features(loaders["val"], model, device=device)
        X_test, y_test = get_extract_features(loaders["test"], model, device=device)
        feature_time = time.time() - t0_feat

        for clf in classifiers:
            print(f"\n=> {cnn.upper()} (+) {clf.upper()}")

            # Classification
            clf_name = get_classifiers(clf, y_train)

            # Fit
            t0_train = time.time()
            clf_name.fit(X_train, y_train)
            train_time = time.time() - t0_train

            # Metrics
            val_ans = get_evaluate(clf_name, X_val, y_val)
            test_ans = get_evaluate(clf_name, X_test, y_test)

            # # GridSearchCV
            # # Classification
            # clf_name, param_grid = get_classifiers(clf, y_train)
            #
            # grid_search = GridSearchCV(
            #     clf_name,
            #     param_grid,
            #     scoring="recall",
            #     cv=3,
            #     verbose=1,
            #     n_jobs=-1
            # )
            #
            # # Fit
            # t0_train = time.time()
            # grid_search.fit(X_train, y_train)
            # train_time = time.time() - t0_train
            #
            # print("\nGRID SEARCH")
            # print("Best CV score  =", grid_search.best_score_)
            # print("Best params    =", grid_search.best_params_)
            # print("Best estimator =", grid_search.best_estimator_)
            #
            # final_model = grid_search.best_estimator_
            #
            # # Metrics
            # val_ans = get_evaluate(final_model, X_val, y_val)
            # test_ans = get_evaluate(final_model, X_test, y_test)

            total_time = feature_time + train_time

            # Prints: performance metrics (VAL and TEST) + Time RUN
            print(f"\nEVALUATION METRICS")
            print(f"VAL")
            print(f"Accuracy  = {val_ans['accuracy']:.3f}")
            print(f"Precision = {val_ans['precision']:.3f}")
            print(f"Recall    = {val_ans['recall']:.3f}")
            print(f"F1-Score  = {val_ans['f1']:.3f}")
            print(f"AUC       = {val_ans['auc']:.3f}")
            print("Confusion Matrix\n", val_ans["cm"])

            print(f"\nTEST")
            print(f"Accuracy  = {test_ans['accuracy']:.3f}")
            print(f"Precision = {test_ans['precision']:.3f}")
            print(f"Recall    = {test_ans['recall']:.3f}")
            print(f"F1-Score  = {test_ans['f1']:.3f}")
            print(f"AUC       = {test_ans['auc']:.3f}")
            print("Confusion Matrix\n", test_ans["cm"])

            print(f"\nTIME RUN (MM:SS)")
            print(f"Feature extraction = {get_format_time(feature_time)}")
            print(f"Training           = {get_format_time(train_time)}")
            print(f"Total              = {get_format_time(total_time)}")

            # Plots: Confusion Matrix, Precision, Recall and Curve ROC


if __name__ == "__main__":
    main()