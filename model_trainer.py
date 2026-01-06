import os
import joblib
import random
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Any, Optional
from dataclasses import dataclass

from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.metrics import (
    make_scorer, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error
)

from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE,
    ADASYN, KMeansSMOTE, SMOTEN
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler


@dataclass
class ModelResult:
    name: str
    estimator: BaseEstimator
    params: dict[str, Any]
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    metrics: dict[str, float]
    cv_score: float
    cv_results: dict[str, Any]
    fit_time: float


def create_estimator(task_type: str, model_type: str, **fixed_model_params):
    model_map = {
        "regression": {
            "rf": RandomForestRegressor,
            "ert": ExtraTreesRegressor,
            "gbdt": GradientBoostingRegressor,
            "svm": SVR,
            "mlp": MLPRegressor,
            "vote": VotingRegressor,
            "adaboost": AdaBoostRegressor,
            "stack": StackingRegressor,
        },
        "classification": {
            "rf": RandomForestClassifier,
            "ert": ExtraTreesClassifier,
            "gbdt": GradientBoostingClassifier,
            "svm": SVC,
            "mlp": MLPClassifier,
            "vote": VotingClassifier,
            "adaboost": AdaBoostClassifier,
            "stack": StackingClassifier,
        }
    }
    
    if task_type not in model_map or model_type not in model_map[task_type]:
        raise ValueError("Task type or model type is invalid")
    
    return model_map[task_type][model_type](**fixed_model_params)


def create_scorer(task_type: str):
    if task_type == "classification":
        return make_scorer(f1_score, average="weighted", zero_division=0, greater_is_better=True)
    elif task_type == "regression":
        return make_scorer(mean_squared_error, greater_is_better=False)
    else:
        raise ValueError("Task type is invalid")


def create_cv(task_type: str, n_splits: int = 5, random_state: int = random.randint(0, 2**32-1)):
    if task_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def create_searcher(search_type: str, estimator: BaseEstimator, search_model_params: dict = {}, searcher_params: dict = {}):
    if search_type == "grid":
        return GridSearchCV(estimator=estimator, param_grid=search_model_params, **searcher_params)
    elif search_type == "random":
        return RandomizedSearchCV(estimator=estimator, param_distributions=search_model_params, **searcher_params)
    else:
        raise ValueError(f"Search type is invalid")


def calculate_regression_metrics(y_true, y_pred):
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }


def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None, average_method: str = "macro"):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        f"precision_{average_method}": float(precision_score(y_true, y_pred, average=average_method, zero_division=0)),
        f"recall_{average_method}": float(recall_score(y_true, y_pred, average=average_method, zero_division=0)),
        f"f1_{average_method}": float(f1_score(y_true, y_pred, average=average_method, zero_division=0)),
    }
    
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes > 2:
                metrics["auc_macro"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'))
                metrics["auc_micro"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='micro'))
                metrics["auc_weighted"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'))
            else:
                y_score = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]
                metrics["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception as e:
            warnings.warn(f"Cannot calculate AUC: {e}")
            if n_classes > 2:
                metrics.update({"auc_macro": np.nan, "auc_micro": np.nan, "auc_weighted": np.nan})
            else:
                metrics["auc"] = np.nan
    
    return metrics


def get_prediction_probabilities(model: BaseEstimator, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        try:
            from scipy.special import expit, softmax
            scores = model.decision_function(X)
            if len(scores.shape) == 1:
                proba = expit(scores)
                return np.column_stack([1 - proba, proba])
            else:
                return softmax(scores, axis=1)
        except:
            warnings.warn("Cannot convert decision_function to probabilities")
            return None
    return None


def create_sampler(sampler_type: str, random_state: int = random.randint(0, 2**32-1)):
    sampler_map = {
        "random_oversample": RandomOverSampler,
        "smote": SMOTE,
        "borderline_smote": BorderlineSMOTE,
        "svm_smote": SVMSMOTE,
        "adasyn": ADASYN,
        "kmeans_smote": KMeansSMOTE,
        "random_undersample": RandomUnderSampler,
        "smote_enn": SMOTEENN,
        "smote_tomek": SMOTETomek,
        "smoten": SMOTEN,
    }
    
    if sampler_type not in sampler_map:
        raise ValueError("Sampler type is invalid")
    
    return sampler_map[sampler_type](random_state=random_state)


def train_models(task_type: str, searchers: list[Tuple[str, BaseEstimator]],
                 # Input data.
                 X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                 # Classification task settings.
                 average_method: str = "macro",
                 # Model train results.
                 save_results: bool = False, save_results_dir: str = "output/models/", save_results_name: str = "model_results.csv",
                 # Trained models.
                 save_models: bool = False, save_best_model_only: bool = False, save_models_dir: str = "output/models/"):
    if task_type not in ["regression", "classification"]:
        raise ValueError("Task type is invalid")
    is_classification = task_type == "classification"
    
    if average_method not in ["macro", "micro", "weighted"]:
        raise ValueError("Average method is invalid")
    
    results = []
    
    for name, searcher in searchers:
        try:
            searcher_clone = clone(searcher)
            searcher_clone.fit(X_train, y_train)
            
            best_estimator = searcher_clone.best_estimator_
            y_pred = best_estimator.predict(X_test)
            y_pred_proba = None
            
            if is_classification:
                y_pred_proba = get_prediction_probabilities(best_estimator, X_test)
                metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba, average_method)
            else:
                metrics = calculate_regression_metrics(y_test, y_pred)
            
            result = ModelResult(
                name=name,
                estimator=best_estimator,
                params=searcher_clone.best_params_,
                predictions=y_pred,
                probabilities=y_pred_proba,
                metrics=metrics,
                cv_score=float(searcher_clone.best_score_),
                cv_results=searcher_clone.cv_results_ if hasattr(searcher_clone, 'cv_results_') else {}
            )
            results.append(result)
            
        except Exception as e:
            warnings.warn(f"Error training model '{name}': {e}")
            continue
    
    if not results:
        raise RuntimeError("All models failed to train")
    
    sort_key = f"f1_{average_method}" if is_classification else "r2"
    valid_results = [r for r in results if sort_key in r.metrics]
    
    if valid_results:
        valid_results.sort(key=lambda x: x.metrics[sort_key], reverse=is_classification)
        results = valid_results + [r for r in results if r not in valid_results]
    
    if save_results:
        os.makedirs(save_results_dir, exist_ok=True)
        
        results_data = [
            {
                "model": result.name,
                **result.metrics,
            } for result in results
        ]
        pd.DataFrame(results_data).to_csv(os.path.join(save_results_dir, save_results_name), index=False)
    
    if save_models:
        os.makedirs(save_models_dir, exist_ok=True)
        models_to_save = [results[0]] if save_best_model_only else results
        
        for result in models_to_save:
            model_path = os.path.join(save_models_dir, f"{result.name}.joblib")
            joblib.dump(result.estimator, model_path)
    
    return results
