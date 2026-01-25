from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_classifiers(clf_name: str, y_train, random_state: int = 42):
    if clf_name == "rf":
        return RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            criterion = "gini",
            random_state=random_state,
            n_jobs=-1
        )

    elif clf_name == "xgb":
        neg_instances = (y_train == 0).sum()
        pos_instances = (y_train == 1).sum()
        balanced_spw = neg_instances / pos_instances

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method = "hist",
            n_estimators=800,
            max_depth=None,
            min_child_weight=1,
            gamma=0.0,
            max_cat_threshold=2,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.03,
            scale_pos_weight=balanced_spw,
            random_state = random_state,
            n_jobs = -1,
        )

    elif clf_name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=0.1,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            )),
        ])

    else:
        raise ValueError(f"Unknown classifier name: {clf_name}. Use 'rf' or 'svm' or 'xgb'.")


# #GridSearchCV
# def get_classifiers(clf_name: str, y_train, random_state: int = 42):
#     if clf_name == "rf":
#         clf_rf = RandomForestClassifier(
#             random_state=random_state,
#             n_jobs=-1,
#         )
#         param_grid = {
#             "n_estimators": [400, 800],
#             "max_depth": [None, 20],
#             "min_samples_split": [2, 5],
#             "min_samples_leaf": [1, 2, 4],
#             "max_features": ["sqrt"],
#             "class_weight": ["balanced", "balanced_subsample"],
#             "criterion": ["gini"]
#         }
#         return clf_rf, param_grid
#
#     elif clf_name == "xgb":
#         neg_instances = (y_train == 0).sum()
#         pos_instances = (y_train == 1).sum()
#         balanced_spw = neg_instances / pos_instances
#
#         clf_xgb = XGBClassifier(
#             random_state=random_state,
#             n_jobs=-1,
#             objective="binary:logistic",
#             eval_metric="logloss",
#             tree_method="hist",
#             scale_pos_weight= float(balanced_spw)
#         )
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
#         return clf_xgb, param_grid
#
#     elif clf_name == "svm":
#         clf_svm = Pipeline([
#             ("scaler", StandardScaler()),
#             ("clf", SVC(
#                 random_state=random_state,
#                 probability=False
#             )),
#         ])
#         param_grid = {
#             "clf__kernel": ["rbf", "linear"],
#             "clf__C": [0.1, 1, 5],
#             "clf__gamma": ["scale", 0.01],
#             "clf__class_weight": ["balanced"],
#         }
#         return clf_svm, param_grid
#
#
#     else:
#         raise ValueError(f"Unknown classifier name: {clf_name}. Use 'rf' or 'svm' or 'xgb'.")