# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
#
#
# def get_model_and_param_grid(clf_name: str, y_train, random_state: int = 42):
#     if clf_name == "rf":
#         model = RandomForestClassifier(
#             random_state=random_state,
#             n_jobs=-1
#         )
#
#         param_grid = {
#             "n_estimators": [400, 800],
#             "max_depth": [None, 20],
#             "min_samples_split": [2, 5],
#             "min_samples_leaf": [1, 2, 4],
#             "max_features": ["sqrt"],
#             "class_weight": ["balanced", "balanced_subsample"],
#             "criterion": ["gini"]
#         }
#
#         return model, param_grid
#
#     elif clf_name == "xgb":
#         neg_instances = (y_train == 0).sum()
#         pos_instances = (y_train == 1).sum()
#         balanced_spw = neg_instances / pos_instances
#
#         model = XGBClassifier(
#             random_state=random_state,
#             n_jobs=-1,
#             objective="binary:logistic",
#             eval_metric="logloss",
#             tree_method="hist",
#             scale_pos_weight=float(balanced_spw)
#         )
#
#         param_grid = {
#             "n_estimators": [600, 1000],
#             "learning_rate": [0.03, 0.05],
#             "max_depth": [3, 4, 5],
#             "min_child_weight": [1, 3],
#             "gamma": [0],
#             "max_cat_threshold": [1, 2],
#             "subsample": [0.8],
#             "colsample_bytree": [0.8]
#         }
#
#         return model, param_grid
#
#     elif clf_name == "svm":
#         model = Pipeline([
#             ("scaler", StandardScaler()),
#             ("clf", SVC(
#                 random_state=random_state,
#                 probability=False
#             )),
#         ])
#
#         param_grid = {
#             "clf__kernel": ["rbf", "linear"],
#             "clf__C": [0.1, 1, 5],
#             "clf__gamma": ["scale", 0.01],
#             "clf__class_weight": ["balanced"],
#         }
#
#         return model, param_grid
#
#     else:
#         raise ValueError(f"Unknown classifier name: {clf_name}. Use 'rf', 'xgb' or 'svm'.")


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model_and_param_grid(clf_name: str, y_train, random_state: int = 42):
    if clf_name == "rf":
        model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        )

        param_grid = {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced", "balanced_subsample"],
            "criterion": ["gini", "entropy"]
        }

        return model, param_grid

    elif clf_name == "xgb":
        neg_instances = (y_train == 0).sum()
        pos_instances = (y_train == 1).sum()
        balanced_spw = neg_instances / pos_instances

        model = XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=balanced_spw
        )

        param_grid = {
            "n_estimators": [300, 600, 1000],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [3, 4, 5],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        return model, param_grid

    elif clf_name == "svm":
        weights_dict = [
            {0: 1, 1: 2.5},
            {0: 1, 1: 5},
            {0: 1, 1: 10},
            {0: 1, 1: 20}
        ]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                random_state=random_state,
                probability=False
            )),
        ])

        param_grid = [
            {
                "clf__kernel": ["linear"],
                "clf__C": [0.01, 0.1, 1, 5, 10],
                "clf__class_weight": [None, "balanced"] + weights_dict
            },
            {
                "clf__kernel": ["rbf"],
                "clf__C": [0.01, 0.1, 1, 5, 10],
                "clf__gamma": ["scale", 0.1, 0.01, 0.001],
                "clf__class_weight": [None, "balanced"] + weights_dict
            }
        ]

        return model, param_grid

    else:
        raise ValueError(f"Unknown classifier name: {clf_name}. Use 'rf', 'xgb' or 'svm'.")