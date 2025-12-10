import os
import random
import joblib
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.base import clone, BaseEstimator
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.metrics import (
    make_scorer, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import loguniform


def create_estimator(task_type: str, model_type: str, **fixed_model_params):
    task_type_model_map = {
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
    if task_type not in task_type_model_map or model_type not in task_type_model_map[task_type]:
        raise ValueError(f"Unsupported task type '{task_type}' or model type '{model_type}'")
    return task_type_model_map[task_type][model_type](**fixed_model_params)


def create_scorer(task_type: str, scorer_type: str):
    scorer_map = {
        "regression": {
            "rmse": make_scorer(lambda yt, yp: -root_mean_squared_error(yt, yp)),
            "r2": "r2"
        },
        "classification": {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average='weighted', zero_division=0),
            "recall": make_scorer(recall_score, average='weighted', zero_division=0), 
            "f1": make_scorer(f1_score, average='weighted', zero_division=0)
        }
    }
    if task_type not in scorer_map or scorer_type not in scorer_map[task_type]:
        raise ValueError(f"Unsupported scorer '{scorer_type}' for task '{task_type}'")
    return scorer_map[task_type][scorer_type]


def create_cv(task_type: str, random_state: int = random.randint(0, 4294967296)):
    if task_type == "classification":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=5, shuffle=True, random_state=random_state)


def create_searcher(search_type: str, estimator: BaseEstimator,
                    search_model_params: dict = {}, searcher_params: dict = {}):
    if search_type == "grid":
        return GridSearchCV(param_grid=search_model_params, estimator=estimator, **searcher_params)
    elif search_type == "random":
        return RandomizedSearchCV(param_distributions=search_model_params, estimator=estimator, **searcher_params)
    else:
        raise ValueError(f"Unsupported search type '{search_type}'. Use: grid/random")


def train_models(task_type: str, searchers: Tuple[str, BaseEstimator],
                 X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                 # Feature selection.
                 k_features: int = 0, random_state: int = random.randint(0, 4294967296),
                 # Model train results.
                 save_results: bool = False, save_results_dir: str = "output/models/",
                 # Trained models.
                 save_models: bool = False, save_best_model: bool = False, save_models_dir: str = "output/models/"):
    metric_funcs = {
        "regression": {
            "rmse": lambda yt, yp: round(root_mean_squared_error(yt, yp), 4),
            "r2": lambda yt, yp: round(r2_score(yt, yp), 4)
        },
        "classification": {
            "accuracy": lambda yt, yp: round(accuracy_score(yt, yp), 4),
            "precision": lambda yt, yp: round(precision_score(yt, yp, average='macro', zero_division=0), 4),
            "recall": lambda yt, yp: round(recall_score(yt, yp, average='macro', zero_division=0), 4),
            "f1": lambda yt, yp: round(f1_score(yt, yp, average='macro', zero_division=0), 4)
        }
    }
    
    if k_features > 0:
        if task_type == 'classification':
            score_func = lambda X, y: mutual_info_classif(X, y, random_state=random_state)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=k_features)
        X_train_to_use = selector.fit_transform(X_train, y_train)
        X_test_to_use = selector.transform(X_test)
        # selected_features_indices = selector.get_support(indices=True)
    else:
        X_train_to_use = X_train
        X_test_to_use = X_test
        # selected_features_indices = None
    
    results = []
    
    for searcher_name, searcher in searchers:
        searcher_clone = clone(searcher)
        searcher_clone.fit(X_train_to_use, y_train)
        y_pred = searcher_clone.predict(X_test_to_use)

        results.append({
            "name": searcher_name,
            "best_estimator": searcher_clone.best_estimator_,
            "best_parameters": searcher_clone.best_params_,
            "prediction": y_pred,
            "metrics": {name: func(y_test, y_pred) for name, func in metric_funcs[task_type].items()}
        })
        
    sorted_key = 'f1' if task_type == "classification" else 'rmse'
    ascending = False if sorted_key == "rmse" else True
    results = sorted(results, key = lambda result: result["metrics"][sorted_key], reverse=ascending)
    
    if save_results:
        os.makedirs(os.path.dirname(save_results_dir), exist_ok=True)
        pd.DataFrame([{
            "name": result["name"],
            "best_parameters": result["best_parameters"],
            **result["metrics"]
        } for result in results]).to_csv(os.path.join(save_results_dir, "model_results.csv"), index=False)
    
    if save_models:
        os.makedirs(os.path.dirname(save_models_dir), exist_ok=True)
        if save_best_model:
            joblib.dump(results[0]["best_estimator"], os.path.join(save_models_dir, f"{results[0]['name']}.joblib"))
        else:
            for result in results:
                joblib.dump(result["best_estimator"], os.path.join(save_models_dir, f"{result['name']}.joblib"))
    
    return results


def rf_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 4294967296)):
    fixed_model_params = {
        'random_state': random_state,
        'bootstrap': True
    }
    
    search_model_params = {
        'n_estimators': np.arange(50, 1001, 50).tolist(),
        'max_depth': [None] + np.arange(5, 31, 5).tolist(),
        'min_samples_split': np.concatenate([np.arange(2, 21, 1), [30]]).tolist(),
        'min_samples_leaf': np.concatenate([np.arange(1, 11, 1), [20]]).tolist(),
        'max_features': ['sqrt', 'log2', None] + np.linspace(0.3, 0.9, 7).tolist(),
        'max_samples': [None] + np.linspace(0.6, 1.0, 5).tolist()
    }
    
    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 75
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def ert_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 4294967296)):
    fixed_model_params = {
        'random_state': random_state,
        'bootstrap': True
    }
    
    search_model_params = {
        'n_estimators': np.arange(50, 1001, 50).tolist(),
        'max_depth': [None] + np.arange(5, 31, 5).tolist(),
        'min_samples_split': np.concatenate([np.arange(2, 16, 1), [20]]).tolist(),
        'min_samples_leaf': np.concatenate([np.arange(1, 11, 1), [20]]).tolist(),
        'max_features': ['sqrt', 'log2', None] + np.linspace(0.2, 0.9, 7).tolist(),
        'max_samples': [None] + np.linspace(0.6, 1.0, 5).tolist()
    }
    
    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 75
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def gbdt_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 4294967296)):
    fixed_model_params = {
        'random_state': random_state
    }
    
    search_model_params = {
        'n_estimators': np.arange(100, 1001, 50).tolist(),
        'max_depth': np.arange(2, 9, 1).tolist(),
        'min_samples_split': np.arange(2, 21, 1).tolist(),
        'min_samples_leaf': np.arange(1, 11, 1).tolist(),
        'subsample': np.linspace(0.5, 1.0, 6).tolist(),
        'max_features': ['sqrt', None] + np.linspace(0.3, 0.9, 7).tolist(),
        'learning_rate': loguniform(1e-3, 0.1).rvs(7, random_state=random_state).tolist(),
    }
    
    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 100
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def svm_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 4294967296)):
    fixed_model_params = {
        'max_iter': 10000,
        'cache_size': 1000,
    }
    if task_type == "classification":
        fixed_model_params["random_state"] = random_state
        fixed_model_params["decision_function_shape"] = 'ovr'
    
    search_model_params = {
        'C': loguniform(1e-4, 1e4).rvs(10, random_state=random_state).tolist(),
        'gamma': ['scale', 'auto'] + loguniform(1e-5, 1e1).rvs(6, random_state=random_state).tolist(),
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': np.arange(2, 6, 1).tolist(),
        'coef0': np.linspace(-1.0, 2.0, 7).tolist()
    }
    if task_type == "classification":
        search_model_params['class_weight'] = [None, 'balanced']

    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 150
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def mlp_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 4294967296)):
    fixed_model_params = {
        'max_iter': 10000,
        'random_state': random_state
    }
    
    search_model_params = {
        'hidden_layer_sizes': [(32,), (64,), (128,), (256,), (64, 32), (128, 64), (256, 128)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': loguniform(1e-4, 1e-1).rvs(6, random_state=random_state).tolist(),
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
    }
    
    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 100
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def stack_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 4294967296)):
    is_classifier = task_type == "classification"

    fixed_model_params = {}
    
    search_model_params = {
        'passthrough': [False, True]
    }
    search_model_params = {k: v for k, v in search_model_params.items() if v}

    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if is_classifier else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 100
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params
