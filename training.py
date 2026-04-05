"""
WebPurifier – Multi-Model Training Script
==========================================
Trains six classifiers with hyperparameter tuning (RandomizedSearchCV)
and optional SMOTE oversampling to handle the severe class imbalance
(~94 % noise / ~6 % content).

Models:  Decision Tree, Gaussian Naive Bayes, K-Nearest Neighbors,
         Random Forest, LightGBM, XGBoost

Usage:   python training.py
Outputs: Per-model .pkl files  +  comparison_results.csv
"""

import warnings, time, os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)

# ----- Models -----
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ----- SMOTE -----
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ====================================================================
# 1. CONFIGURATION
# ====================================================================

DATASET_PATH   = "webpurifier_dataset.csv"
OUTPUT_DIR     = "trained_models"
RESULTS_CSV    = "comparison_results.csv"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
CV_FOLDS       = 5
N_ITER_SEARCH  = 40          # number of random search iterations per model
SCORING        = "f1"        # optimise for minority-class F1

CATEGORICAL_FEATURES = ["tag_type"]
NUMERICAL_FEATURES   = [
    "link_density",
    "text_to_tag_ratio",
    "keyword_score",
    "stop_word_density",
    "text_length",
]

# ====================================================================
# 2. MODEL ZOO  – model objects + hyperparameter search spaces
# ====================================================================

# For each model we define:
#   - estimator instance
#   - param_distributions dict  (keys prefixed with 'classifier__')
#   - use_smote: whether to inject SMOTE before this model
#
# SMOTE is applied for models that do NOT have a native class_weight
# or scale_pos_weight equivalent (Naive Bayes, KNN).
# For tree-based models we still use class_weight / scale_pos_weight,
# but we also provide a SMOTE variant to compare.

def _pos_weight(y):
    """Compute scale_pos_weight for XGBoost (neg/pos ratio)."""
    counts = np.bincount(y)
    return counts[0] / counts[1]


def get_model_configs(pos_weight_ratio):
    """Return a dict of {name: config} for every model to train."""

    configs = {}

    # ---------- Decision Tree ----------
    configs["DecisionTree"] = {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "params": {
            "classifier__max_depth":        [5, 10, 15, 20, 30, None],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf":  [1, 2, 5, 10],
            "classifier__class_weight":      ["balanced", None],
            "classifier__criterion":         ["gini", "entropy"],
        },
        "use_smote": False,
    }
    configs["DecisionTree_SMOTE"] = {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "params": {
            "classifier__max_depth":        [5, 10, 15, 20, 30, None],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf":  [1, 2, 5, 10],
            "classifier__criterion":         ["gini", "entropy"],
        },
        "use_smote": True,
    }

    # ---------- Gaussian Naive Bayes ----------
    # GaussianNB has very few hyper-params; var_smoothing is the main one.
    configs["NaiveBayes_SMOTE"] = {
        "estimator": GaussianNB(),
        "params": {
            "classifier__var_smoothing": np.logspace(-12, -1, 50),
        },
        "use_smote": True,   # no native class_weight, SMOTE helps a lot
    }

    # ---------- K-Nearest Neighbors ----------
    configs["KNN_SMOTE"] = {
        "estimator": KNeighborsClassifier(),
        "params": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "classifier__weights":     ["uniform", "distance"],
            "classifier__metric":      ["euclidean", "manhattan", "minkowski"],
            "classifier__p":           [1, 2],
        },
        "use_smote": True,   # KNN is very sensitive to imbalance
    }

    # ---------- Random Forest ----------
    configs["RandomForest"] = {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "classifier__n_estimators":     [100, 200, 300, 500],
            "classifier__max_depth":        [10, 20, 30, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf":  [1, 2, 4],
            "classifier__class_weight":      ["balanced", "balanced_subsample", None],
            "classifier__max_features":      ["sqrt", "log2"],
        },
        "use_smote": False,
    }
    configs["RandomForest_SMOTE"] = {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "classifier__n_estimators":     [100, 200, 300, 500],
            "classifier__max_depth":        [10, 20, 30, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf":  [1, 2, 4],
            "classifier__max_features":      ["sqrt", "log2"],
        },
        "use_smote": True,
    }

    # ---------- LightGBM ----------
    configs["LightGBM"] = {
        "estimator": LGBMClassifier(
            random_state=RANDOM_STATE, verbosity=-1, is_unbalance=True
        ),
        "params": {
            "classifier__n_estimators":   [100, 200, 300, 500],
            "classifier__learning_rate":  [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth":      [3, 5, 7, 10, -1],
            "classifier__num_leaves":     [15, 31, 63, 127],
            "classifier__min_child_samples": [5, 10, 20, 50],
            "classifier__subsample":      [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__reg_alpha":      [0, 0.1, 1.0],
            "classifier__reg_lambda":     [0, 0.1, 1.0],
        },
        "use_smote": False,
    }
    configs["LightGBM_SMOTE"] = {
        "estimator": LGBMClassifier(
            random_state=RANDOM_STATE, verbosity=-1
        ),
        "params": {
            "classifier__n_estimators":   [100, 200, 300, 500],
            "classifier__learning_rate":  [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth":      [3, 5, 7, 10, -1],
            "classifier__num_leaves":     [15, 31, 63, 127],
            "classifier__min_child_samples": [5, 10, 20, 50],
            "classifier__subsample":      [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        },
        "use_smote": True,
    }

    # ---------- XGBoost ----------
    configs["XGBoost"] = {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=pos_weight_ratio,
            use_label_encoder=False,
        ),
        "params": {
            "classifier__n_estimators":   [100, 200, 300, 500],
            "classifier__learning_rate":  [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth":      [3, 5, 7, 10],
            "classifier__min_child_weight": [1, 3, 5, 10],
            "classifier__subsample":      [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__gamma":          [0, 0.1, 0.5, 1],
            "classifier__reg_alpha":      [0, 0.1, 1.0],
            "classifier__reg_lambda":     [0, 0.1, 1.0],
        },
        "use_smote": False,
    }
    configs["XGBoost_SMOTE"] = {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
        "params": {
            "classifier__n_estimators":   [100, 200, 300, 500],
            "classifier__learning_rate":  [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth":      [3, 5, 7, 10],
            "classifier__min_child_weight": [1, 3, 5, 10],
            "classifier__subsample":      [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__gamma":          [0, 0.1, 0.5, 1],
        },
        "use_smote": True,
    }

    return configs


# ====================================================================
# 3. BUILD PIPELINE
# ====================================================================

def build_pipeline(preprocessor, estimator, use_smote):
    """
    Build an (imb)pipeline that preprocesses, optionally applies SMOTE,
    then feeds into the classifier.
    """
    if use_smote:
        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", estimator),
        ])
    else:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
    return pipeline


# ====================================================================
# 4. MAIN
# ====================================================================

def main():
    # ---- Load data ----
    print("=" * 70)
    print("  WebPurifier – Multi-Model Training with Hyperparameter Tuning")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset shape : {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Imbalance ratio (neg/pos): {df['label'].value_counts()[0] / df['label'].value_counts()[1]:.2f}:1\n")

    X = df.drop(columns=["url_hash", "label"])
    y = df["label"].values

    # ---- Preprocessor ----
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])

    # ---- Train/test split (stratified) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print(f"Train label dist: {np.bincount(y_train)}")
    print(f"Test  label dist: {np.bincount(y_test)}\n")

    # ---- Prepare output dir ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- pos weight for XGBoost ----
    pos_w = _pos_weight(y_train)
    model_configs = get_model_configs(pos_w)

    # ---- CV strategy ----
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ---- Train each model ----
    results = []

    for name, cfg in model_configs.items():
        print("-" * 70)
        print(f"  Training: {name}  (SMOTE={'YES' if cfg['use_smote'] else 'NO'})")
        print("-" * 70)

        pipeline = build_pipeline(preprocessor, cfg["estimator"], cfg["use_smote"])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=cfg["params"],
            n_iter=min(N_ITER_SEARCH, _total_combinations(cfg["params"])),
            scoring=SCORING,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True,
            error_score="raise",
        )

        t0 = time.time()
        search.fit(X_train, y_train)
        train_time = time.time() - t0

        best_model = search.best_estimator_
        y_pred  = best_model.predict(X_test)
        y_proba = (
            best_model.predict_proba(X_test)[:, 1]
            if hasattr(best_model, "predict_proba")
            else None
        )

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        cm   = confusion_matrix(y_test, y_pred)

        # Print detailed report
        print(f"\nBest CV {SCORING}: {search.best_score_:.4f}")
        print(f"Best params: {json.dumps(_strip_prefix(search.best_params_), indent=2)}")
        print(f"\nTest Accuracy : {acc:.4f}")
        print(f"Test F1       : {f1:.4f}")
        print(f"Test Precision: {prec:.4f}")
        print(f"Test Recall   : {rec:.4f}")
        if auc is not None:
            print(f"Test ROC-AUC  : {auc:.4f}")
        print(f"Training time : {train_time:.1f}s")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=["Noise (0)", "Content (1)"]))
        print(f"Confusion Matrix:\n{cm}\n")

        # Save model
        model_path = os.path.join(OUTPUT_DIR, f"webpurifier_{name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"Model saved → {model_path}")

        results.append({
            "Model":           name,
            "SMOTE":           cfg["use_smote"],
            "Best_CV_F1":      round(search.best_score_, 4),
            "Test_Accuracy":   round(acc, 4),
            "Test_F1":         round(f1, 4),
            "Test_Precision":  round(prec, 4),
            "Test_Recall":     round(rec, 4),
            "Test_ROC_AUC":    round(auc, 4) if auc is not None else "N/A",
            "Train_Time_s":    round(train_time, 1),
            "Best_Params":     json.dumps(_strip_prefix(search.best_params_)),
        })

    # ---- Summary Table ----
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Test_F1", ascending=False).reset_index(drop=True)
    print(results_df[["Model", "SMOTE", "Best_CV_F1", "Test_Accuracy",
                       "Test_F1", "Test_Precision", "Test_Recall",
                       "Test_ROC_AUC", "Train_Time_s"]].to_string(index=False))

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved → {RESULTS_CSV}")

    # ---- Identify Best Model ----
    best_row = results_df.iloc[0]
    print(f"\n{'*' * 70}")
    print(f"  BEST MODEL: {best_row['Model']}  (Test F1 = {best_row['Test_F1']})")
    print(f"{'*' * 70}\n")

    # Copy best model as the default model for inference
    best_src  = os.path.join(OUTPUT_DIR, f"webpurifier_{best_row['Model']}.pkl")
    best_dst  = "webpurifier_model_best.pkl"
    joblib.dump(joblib.load(best_src), best_dst)
    print(f"Best model also saved as → {best_dst}")


# ====================================================================
# HELPERS
# ====================================================================

def _strip_prefix(params: dict, prefix="classifier__") -> dict:
    """Remove 'classifier__' prefix from param names for readability."""
    return {k.replace(prefix, ""): v for k, v in params.items()}


def _total_combinations(param_grid: dict) -> int:
    """Upper bound on all combinations in a param grid."""
    total = 1
    for v in param_grid.values():
        total *= len(v) if hasattr(v, "__len__") else 10
    return total


if __name__ == "__main__":
    main()
