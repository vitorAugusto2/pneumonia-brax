
import os
import time

from sklearn.model_selection import GridSearchCV

from src.data_processing import csv_to_bin
from src.data_balance import split_set
from src.cnn_backbone import (
    HybridFineTuneModel,
    freeze_for_partial_finetune,
    train_partial_finetune,
    extract_features_from_finetuned,
    get_device,
)
from src.data_loaders import get_loaders
from src.classifiers import get_model_and_param_grid
from src.evaluate import get_evaluate, find_best_threshold

from utils.cnn_cache import (
    features_exist,
    save_features,
    load_features
)

from utils.best_params import (
    save_best_params,
    load_best_params
)

from utils.format_time import get_format_time
from utils.transforms import get_transforms
from utils.plot import (
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc_curve
)

# =========================
# CONFIGURACOES
# =========================
cnns = ["vgg16", "resnet50"]        # ["vgg16", "resnet50"]
classifiers = ["rf", "xgb", "svm"]  # ["rf", "xgb", "svm"]
fine_tuning_params = {
    "vgg16": {
        "epochs": 8,
        "lr_backbone": 5e-6,
        "lr_head": 1e-4
    },
    "resnet50": {
        "epochs": 8,
        "lr_backbone": 5e-6,
        "lr_head": 1e-4
    }
}


use_cache = True
force_reextract = False
run_gridsearch = False

device = get_device()

base_dir = os.path.dirname(os.path.abspath(__file__))
cache_cnn = os.path.join(base_dir, "utils", "cache_cnn")

os.makedirs("./images", exist_ok=True)
os.makedirs("./data", exist_ok=True)
os.makedirs("./dataset", exist_ok=True)
os.makedirs("./plots", exist_ok=True)
os.makedirs(cache_cnn, exist_ok=True)


def main():
    # =========================
    # DATASET BINARY + SPLIT
    # =========================
    path_master_csv = os.path.join(base_dir, "dataset", "master_spreadsheet_update.csv")
    csv_to_bin(path_master_csv)

    path_binary_csv = os.path.join(base_dir, "dataset", "dataset_binary.csv")
    split_set(path_binary_csv)

    # =========================
    # LOOP NAS CNNS
    # =========================
    for cnn in cnns:
        if use_cache and features_exist(cache_cnn, cnn) and not force_reextract:
            print(f"\n=> [CNN] Loading backbone {cnn.upper()}")
            X_train, y_train, X_val, y_val, X_test, y_test = load_features(cache_cnn, cnn)

        else:
            print(f"\n=> [CNN] Extract backbone {cnn.upper()}")
            print(f"[INFO] Device={device}")
            ft_model = HybridFineTuneModel(cnn)
            ft_model = freeze_for_partial_finetune(ft_model)
            preprocess = ft_model.get_preprocess()
            train_transform, eval_transform = get_transforms(preprocess)

            loaders = get_loaders(
                base_dir=base_dir,
                train_csv="./data/train.csv",
                val_csv="./data/val.csv",
                test_csv="./data/test.csv",
                train_transform=train_transform,
                eval_transform=eval_transform,
                batch_size=32,
                num_workers=2
            )

            ft_value = fine_tuning_params[cnn]

            # Fine-tuning
            t0_fine_tunning = time.time()
            ft_model = train_partial_finetune(
                model=ft_model,
                loaders=loaders,
                device=device,
                epochs=ft_value["epochs"],
                lr_backbone=ft_value["lr_backbone"],
                lr_head=ft_value["lr_head"]
            )
            fine_tunning_time = time.time() - t0_fine_tunning

            # Features
            t0_feature = time.time()
            X_train, y_train = extract_features_from_finetuned(loaders["train"], ft_model, device)
            X_val, y_val = extract_features_from_finetuned(loaders["val"], ft_model, device)
            X_test, y_test = extract_features_from_finetuned(loaders["test"], ft_model, device)
            feature_time = time.time() - t0_feature

            save_features(
                cache_cnn,
                cnn,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test
            )

            print("\nTIME RUN (MM:SS)")
            print(f"Fine-tunning        = {get_format_time(fine_tunning_time)}")
            print(f"Feature extraction  = {get_format_time(feature_time)}")

        # =========================
        # LOOP NOS CLASSIFICADORES
        # =========================
        for clf in classifiers:
            print(f"\n=> {cnn.upper()} (+) {clf.upper()}")

            best_params = load_best_params(base_dir, cnn, clf)

            model, param_grid = get_model_and_param_grid(clf, y_train)

            if run_gridsearch or best_params is None:
                print(f"[CV] Running GridSearchCV for {cnn.upper()} + {clf.upper()}")

                t0_grid = time.time()

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="f1",
                    cv=3,
                    n_jobs=-1,
                    verbose=2,
                    refit=True
                )

                grid.fit(X_train, y_train)

                grid_time = time.time() - t0_grid
                best_params = grid.best_params_
                best_score = grid.best_score_

                save_best_params(base_dir, cnn, clf, best_params)

                print("\nGRID SEARCH RESULTS")
                print(f"Best params = {best_params}")
                print(f"Best CV F1  = {best_score:.4f}")
                print(f"Time run GridSearch  = {get_format_time(grid_time)}")

            else:
                print(f"[CV] Using saved params for {cnn.upper()} + {clf.upper()}")
                print(f"Best params = {best_params}")

            # aplica os melhores parametros no modelo base
            model.set_params(**best_params)

            # treino final
            t0_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0_train

            # avaliacao
            val_default = get_evaluate(model, X_val, y_val)

            best_threshold, best_val_f1 = find_best_threshold(
                y_true=y_val,
                y_score=val_default["y_score"]
            )

            print(f"Best threshold (VAL) = {best_threshold:.4f}")
            print(f"Best F1 on VAL       = {best_val_f1:.4f}")

            val_ans = get_evaluate(model, X_val, y_val, threshold=best_threshold)
            test_ans = get_evaluate(model, X_test, y_test, threshold=best_threshold)

            print(f"\nEVALUATION METRICS")
            print("VAL")
            print(f"Balanced Accuracy = {val_ans['balanced_accuracy']:.3f}")
            print(f"Accuracy  = {val_ans['accuracy']:.3f}")
            print(f"Precision = {val_ans['precision']:.3f}")
            print(f"Recall    = {val_ans['recall']:.3f}")
            print(f"F1-Score  = {val_ans['f1']:.3f}")
            print(f"AUC       = {val_ans['auc']:.3f}")

            print("\nTEST")
            print(f"Balanced Accuracy = {test_ans['balanced_accuracy']:.3f}")
            print(f"Accuracy  = {test_ans['accuracy']:.3f}")
            print(f"Precision = {test_ans['precision']:.3f}")
            print(f"Recall    = {test_ans['recall']:.3f}")
            print(f"F1-Score  = {test_ans['f1']:.3f}")
            print(f"AUC       = {test_ans['auc']:.3f}")

            print("\nTIME RUN (MM:SS)")
            print(f"Training = {get_format_time(train_time)}")

            # =========================
            # PLOTS - VAL
            # =========================
            plot_confusion_matrix(
                y_true=y_val,
                y_pred=val_ans["y_pred"],
                output_plot=f"./plots/cm_val_{cnn}_{clf}.png"
            )

            plot_roc_curve(
                y_true=y_val,
                y_score=val_ans["y_score"],
                output_plot=f"./plots/roccurve_val_{cnn}_{clf}.png"
            )

            plot_precision_recall(
                y_true=y_val,
                y_score=val_ans["y_score"],
                output_plot=f"./plots/precisionrecall_val_{cnn}_{clf}.png"
            )

            # =========================
            # PLOTS - TEST
            # =========================
            plot_confusion_matrix(
                y_true=y_test,
                y_pred=test_ans["y_pred"],
                output_plot=f"./plots/cm_test_{cnn}_{clf}.png"
            )

            plot_roc_curve(
                y_true=y_test,
                y_score=test_ans["y_score"],
                output_plot=f"./plots/roccurve_test_{cnn}_{clf}.png"
            )

            plot_precision_recall(
                y_true=y_test,
                y_score=test_ans["y_score"],
                output_plot=f"./plots/precisionrecall_test_{cnn}_{clf}.png"
            )


if __name__ == "__main__":
    main()

# import os
# import time
#
# from sklearn.model_selection import GridSearchCV
# from imblearn.over_sampling import SMOTE
#
# from src.data_processing import csv_to_bin
# from src.data_balance import split_set
# from src.cnn_backbone import (
#     HybridFineTuneModel,
#     freeze_for_partial_finetune,
#     train_partial_finetune,
#     extract_features_from_finetuned,
#     get_device,
# )
# from src.data_loaders import get_loaders
# from src.classifiers import get_model_and_param_grid
# from src.evaluate import get_evaluate, find_best_threshold
#
# from utils.cnn_cache import (
#     features_exist,
#     save_features,
#     load_features
# )
#
# from utils.best_params import (
#     save_best_params,
#     load_best_params
# )
#
# from utils.format_time import get_format_time
# from utils.transforms import get_transforms
# from utils.plot import (
#     plot_confusion_matrix,
#     plot_precision_recall,
#     plot_roc_curve
# )
#
# # =========================
# # CONFIGURACOES
# # =========================
# cnns = ["resnet50"]        # ["vgg16", "resnet50"]
# classifiers = ["rf", "xgb", "svm"]  # ["rf", "xgb", "svm"]
# fine_tuning_params = {
#     "vgg16": {
#         "epochs": 3,
#         "lr_backbone": 1e-5,
#         "lr_head": 3e-4
#     },
#     "resnet50": {
#         "epochs": 2,
#         "lr_backbone": 1e-5,
#         "lr_head": 3e-4
#     }
# }
#
# use_cache = True
# force_reextract = True
# run_gridsearch = False
#
# use_smote = True
# smote_strategy = 1.0
# smote_k_neighbors = 5
#
# device = get_device()
#
# base_dir = os.path.dirname(os.path.abspath(__file__))
# cache_cnn = os.path.join(base_dir, "utils", "cache_cnn")
#
# os.makedirs("./images", exist_ok=True)
# os.makedirs("./data", exist_ok=True)
# os.makedirs("./dataset", exist_ok=True)
# os.makedirs("./plots", exist_ok=True)
# os.makedirs(cache_cnn, exist_ok=True)
#
#
# def main():
#     # =========================
#     # DATASET BINARY + SPLIT
#     # =========================
#     path_master_csv = os.path.join(base_dir, "dataset", "master_spreadsheet_update.csv")
#     csv_to_bin(path_master_csv)
#
#     path_binary_csv = os.path.join(base_dir, "dataset", "dataset_binary.csv")
#     split_set(path_binary_csv)
#
#     # =========================
#     # LOOP NAS CNNS
#     # =========================
#     for cnn in cnns:
#         if use_cache and features_exist(cache_cnn, cnn) and not force_reextract:
#             print(f"\n=> [CNN] Loading backbone {cnn.upper()}")
#             X_train, y_train, X_val, y_val, X_test, y_test = load_features(cache_cnn, cnn)
#
#         else:
#             print(f"\n=> [CNN] Extract backbone {cnn.upper()}")
#             print(f"[INFO] Device={device}")
#             ft_model = HybridFineTuneModel(cnn)
#             ft_model = freeze_for_partial_finetune(ft_model)
#             preprocess = ft_model.get_preprocess()
#             train_transform, eval_transform = get_transforms(preprocess)
#
#             loaders = get_loaders(
#                 base_dir=base_dir,
#                 train_csv="./data/train.csv",
#                 val_csv="./data/val.csv",
#                 test_csv="./data/test.csv",
#                 train_transform=train_transform,
#                 eval_transform=eval_transform,
#                 batch_size=32,
#                 num_workers=2
#             )
#
#             ft_value = fine_tuning_params[cnn]
#
#             # Fine-tuning
#             t0_fine_tunning = time.time()
#             ft_model = train_partial_finetune(
#                 model=ft_model,
#                 loaders=loaders,
#                 device=device,
#                 epochs=ft_value["epochs"],
#                 lr_backbone=ft_value["lr_backbone"],
#                 lr_head=ft_value["lr_head"]
#             )
#             fine_tunning_time = time.time() - t0_fine_tunning
#
#             # Features
#             t0_feature = time.time()
#             X_train, y_train = extract_features_from_finetuned(loaders["train"], ft_model, device)
#             X_val, y_val = extract_features_from_finetuned(loaders["val"], ft_model, device)
#             X_test, y_test = extract_features_from_finetuned(loaders["test"], ft_model, device)
#             feature_time = time.time() - t0_feature
#
#             save_features(
#                 cache_cnn,
#                 cnn,
#                 X_train, y_train,
#                 X_val, y_val,
#                 X_test, y_test
#             )
#
#             print("\nTIME RUN (MM:SS)")
#             print(f"Fine-tunning        = {get_format_time(fine_tunning_time)}")
#             print(f"Feature extraction  = {get_format_time(feature_time)}")
#
#         # =========================
#         # SMOTE NAS FEATURES DE TREINO
#         # =========================
#         X_train_clf = X_train
#         y_train_clf = y_train
#
#         if use_smote:
#             print("\n=> Aplicando SMOTE nas features de treino")
#             print(f"Antes do SMOTE: Normal={(y_train == 0).sum()} | Pneumonia={(y_train == 1).sum()}")
#
#             smote = SMOTE(
#                 sampling_strategy=smote_strategy,
#                 k_neighbors=smote_k_neighbors,
#                 random_state=42
#             )
#
#             X_train_clf, y_train_clf = smote.fit_resample(X_train, y_train)
#
#             print(f"Depois do SMOTE: Normal={(y_train_clf == 0).sum()} | Pneumonia={(y_train_clf == 1).sum()}")
#
#         # =========================
#         # LOOP NOS CLASSIFICADORES
#         # =========================
#         for clf in classifiers:
#             print(f"\n=> {cnn.upper()} (+) {clf.upper()}")
#
#             best_params = load_best_params(base_dir, cnn, clf)
#
#             model, param_grid = get_model_and_param_grid(clf, y_train_clf)
#
#             if run_gridsearch or best_params is None:
#                 print(f"[CV] Running GridSearchCV for {cnn.upper()} + {clf.upper()}")
#
#                 t0_grid = time.time()
#
#                 grid = GridSearchCV(
#                     estimator=model,
#                     param_grid=param_grid,
#                     scoring="f1",
#                     cv=3,
#                     n_jobs=-1,
#                     verbose=2,
#                     refit=True
#                 )
#
#                 grid.fit(X_train_clf, y_train_clf)
#
#                 grid_time = time.time() - t0_grid
#                 best_params = grid.best_params_
#                 best_score = grid.best_score_
#
#                 save_best_params(base_dir, cnn, clf, best_params)
#
#                 print("\nGRID SEARCH RESULTS")
#                 print(f"Best params = {best_params}")
#                 print(f"Best CV F1  = {best_score:.4f}")
#                 print(f"Time run GridSearch  = {get_format_time(grid_time)}")
#
#             else:
#                 print(f"[CV] Using saved params for {cnn.upper()} + {clf.upper()}")
#                 print(f"Best params = {best_params}")
#
#             # aplica os melhores parametros no modelo base
#             model.set_params(**best_params)
#
#             # treino final
#             t0_train = time.time()
#             model.fit(X_train_clf, y_train_clf)
#             train_time = time.time() - t0_train
#
#             # avaliacao
#             val_default = get_evaluate(model, X_val, y_val)
#
#             best_threshold, best_val_f1 = find_best_threshold(
#                 y_true=y_val,
#                 y_score=val_default["y_score"]
#             )
#
#             print(f"Best threshold (VAL) = {best_threshold:.4f}")
#             print(f"Best F1 on VAL       = {best_val_f1:.4f}")
#
#             val_ans = get_evaluate(model, X_val, y_val, threshold=best_threshold)
#             test_ans = get_evaluate(model, X_test, y_test, threshold=best_threshold)
#
#             print(f"\nEVALUATION METRICS")
#             print("VAL")
#             print(f"Balanced Accuracy = {val_ans['balanced_accuracy']:.3f}")
#             print(f"Accuracy  = {val_ans['accuracy']:.3f}")
#             print(f"Precision = {val_ans['precision']:.3f}")
#             print(f"Recall    = {val_ans['recall']:.3f}")
#             print(f"F1-Score  = {val_ans['f1']:.3f}")
#             print(f"AUC       = {val_ans['auc']:.3f}")
#
#             print("\nTEST")
#             print(f"Balanced Accuracy = {test_ans['balanced_accuracy']:.3f}")
#             print(f"Accuracy  = {test_ans['accuracy']:.3f}")
#             print(f"Precision = {test_ans['precision']:.3f}")
#             print(f"Recall    = {test_ans['recall']:.3f}")
#             print(f"F1-Score  = {test_ans['f1']:.3f}")
#             print(f"AUC       = {test_ans['auc']:.3f}")
#
#             print("\nTIME RUN (MM:SS)")
#             print(f"Training = {get_format_time(train_time)}")
#
#             # =========================
#             # PLOTS - VAL
#             # =========================
#             plot_confusion_matrix(
#                 y_true=y_val,
#                 y_pred=val_ans["y_pred"],
#                 output_plot=f"./plots/cm_val_{cnn}_{clf}.png"
#             )
#
#             plot_roc_curve(
#                 y_true=y_val,
#                 y_score=val_ans["y_score"],
#                 output_plot=f"./plots/roccurve_val_{cnn}_{clf}.png"
#             )
#
#             plot_precision_recall(
#                 y_true=y_val,
#                 y_score=val_ans["y_score"],
#                 output_plot=f"./plots/precisionrecall_val_{cnn}_{clf}.png"
#             )
#
#             # =========================
#             # PLOTS - TEST
#             # =========================
#             plot_confusion_matrix(
#                 y_true=y_test,
#                 y_pred=test_ans["y_pred"],
#                 output_plot=f"./plots/cm_test_{cnn}_{clf}.png"
#             )
#
#             plot_roc_curve(
#                 y_true=y_test,
#                 y_score=test_ans["y_score"],
#                 output_plot=f"./plots/roccurve_test_{cnn}_{clf}.png"
#             )
#
#             plot_precision_recall(
#                 y_true=y_test,
#                 y_score=test_ans["y_score"],
#                 output_plot=f"./plots/precisionrecall_test_{cnn}_{clf}.png"
#             )
#
#
# if __name__ == "__main__":
#     main()