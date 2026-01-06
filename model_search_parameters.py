import random
import numpy as np
from scipy.stats import loguniform
from sklearn.base import BaseEstimator

from npt.utils.model_trainer import create_cv, create_scorer, create_estimator


def rf_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 2**32-1)):
    fixed_model_params = {
        "random_state": random_state,
        "bootstrap": True,
    }
    
    search_model_params = {
        "n_estimators": np.arange(50, 1201, 50).tolist(),
        "max_depth": [None] + np.arange(5, 41, 5).tolist(),
        "min_samples_split": np.concatenate([np.arange(2, 31, 1), [40, 50]]).tolist(),
        "min_samples_leaf": np.concatenate([np.arange(1, 21, 1), [30, 40]]).tolist(),
        "max_features": ["sqrt", "log2", None] + np.linspace(0.2, 1.0, 9).tolist(),
        "max_samples": [None] + np.linspace(0.5, 1.0, 6).tolist(),
    }
    
    searcher_params = {
        "cv": create_cv(task_type, 5, random_state),
        "scoring": create_scorer(task_type),
        "refit": True,
        "n_jobs": -1,
        "verbose": 3,
    }
    if search_type == "random":
        searcher_params["n_iter"] = 75
        searcher_params["random_state"] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def ert_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 2**32-1)):
    fixed_model_params = {
        "random_state": random_state,
        "bootstrap": True,
    }
    
    search_model_params = {
        "n_estimators": np.arange(50, 1201, 50).tolist(),
        "max_depth": [None] + np.arange(5, 41, 5).tolist(),
        "min_samples_split": np.concatenate([np.arange(2, 31, 1), [40, 50]]).tolist(),
        "min_samples_leaf": np.concatenate([np.arange(1, 21, 1), [30, 40]]).tolist(),
        "max_features": ["sqrt", "log2", None] + np.linspace(0.2, 1.0, 9).tolist(),
        "max_samples": [None] + np.linspace(0.5, 1.0, 6).tolist(),
    }
    
    searcher_params = {
        "cv": create_cv(task_type, 5, random_state),
        "scoring": create_scorer(task_type),
        "refit": True,
        "n_jobs": -1,
        "verbose": 3,
    }
    if search_type == "random":
        searcher_params["n_iter"] = 75
        searcher_params["random_state"] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def gbdt_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 2**32-1)):
    fixed_model_params = {
        "random_state": random_state,
    }
    
    search_model_params = {
        "n_estimators": np.arange(50, 1201, 50).tolist(),
        "max_depth": np.arange(2, 11, 1).tolist(),
        "min_samples_split": np.arange(2, 31, 1).tolist(),
        "min_samples_leaf": np.arange(1, 21, 1).tolist(),
        "subsample": np.linspace(0.4, 1.0, 7).tolist(),
        "max_features": ["sqrt", "log2", None] + np.linspace(0.2, 1.0, 9).tolist(),
        "learning_rate": loguniform(1e-4, 0.3).rvs(10, random_state=random_state).tolist(),
    }
    
    searcher_params = {
        "cv": create_cv(task_type, 5, random_state),
        "scoring": create_scorer(task_type),
        "refit": True,
        "n_jobs": -1,
        "verbose": 3,
    }
    if search_type == "random":
        searcher_params["n_iter"] = 100
        searcher_params["random_state"] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def svm_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 2**32-1)):
    fixed_model_params = {
        "max_iter": 10000,
        "cache_size": 1000,
    }
    if task_type == "classification":
        fixed_model_params["probability"] = True
        fixed_model_params["random_state"] = random_state
        fixed_model_params["decision_function_shape"] = "ovr"
    
    search_model_params = {
        "C": loguniform(1e-5, 1e5).rvs(15, random_state=random_state).tolist(),
        "gamma": ["scale", "auto"] + loguniform(1e-6, 1e2).rvs(10, random_state=random_state).tolist(),
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "degree": np.arange(2, 7, 1).tolist(),
        "coef0": np.linspace(-2.0, 3.0, 10).tolist(),
    }
    if task_type == "classification":
        search_model_params["class_weight"] = [None, "balanced"]

    searcher_params = {
        "cv": create_cv(task_type, 5, random_state),
        "scoring": create_scorer(task_type),
        "refit": True,
        "n_jobs": -1,
        "verbose": 3,
    }
    if search_type == "random":
        searcher_params["n_iter"] = 150
        searcher_params["random_state"] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def mlp_parameters(task_type: str, search_type: str, random_state: int = random.randint(0, 2**32-1)):
    fixed_model_params = {
        "max_iter": 10000,
        "random_state": random_state,
    }
    
    search_model_params = {
        "hidden_layer_sizes": [(16,), (32,), (64,), (128,), (256,), (512,), 
                               (64, 32), (128, 64), (256, 128),
                               (128, 64, 32), (256, 128, 64)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": loguniform(1e-5, 1e0).rvs(10, random_state=random_state).tolist(),
        "learning_rate": ["constant", "invscaling", "adaptive"],
    }
    
    searcher_params = {
        "cv": create_cv(task_type, 5, random_state),
        "scoring": create_scorer(task_type),
        "refit": True,
        "n_jobs": -1,
        "verbose": 3,
    }
    if search_type == "random":
        searcher_params["n_iter"] = 100
        searcher_params["random_state"] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def vote_parameters(task_type: str, search_type: str,
                    weight_count: int, estmators: list[BaseEstimator],
                    random_state: int = random.randint(0, 2**32-1)):
    fixed_model_params = {
        "estimators": estmators,
    }
    
    rng = random.Random(random_state)
    search_model_params = {
        "voting": ["hard", "soft"] if task_type == "classification" else ["hard"],
        "weights": [[rng.uniform(0.1, 10.0) for _ in range(weight_count)] for _ in range(200)],
        "flatten_transform": [True, False] if task_type == "classification" else [True],
    }
    
    searcher_params = {
        "cv": create_cv(task_type, 5, random_state),
        "scoring": create_scorer(task_type),
        "refit": True,
        "n_jobs": -1,
        "verbose": 3,
    }
    
    if search_type == "random":
        searcher_params["n_iter"] = 100
        searcher_params["random_state"] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def adaboost_parameters(task_type: str, search_type: str,
                        estimator: BaseEstimator, search_estimator_params: dict = {},
                        random_state: int = random.randint(0, 2**32-1)):
    fixed_model_params = {
        "random_state": random_state,
        "estimator": estimator,
    }
    
    search_model_params = {
        "n_estimators": np.arange(50, 1201, 50).tolist(),
        "learning_rate": loguniform(1e-4, 0.3).rvs(10, random_state=random_state).tolist(),
        **{f"estimator__{k}": v for k, v in search_estimator_params.items()},
    }
    if task_type == "regression":
        search_model_params.update({"loss": ["linear", "square", "exponential"]})
    
    searcher_params = {
        "cv": create_cv(task_type, 5, random_state),
        "scoring": create_scorer(task_type),
        "refit": True,
        "n_jobs": 16,
        "verbose": 3,
    }
    if search_type == "random":
        searcher_params["n_iter"] = 100
        searcher_params["random_state"] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def stack_parameters(task_type: str, 
                     search_type: str,
                     estimators: list[tuple[str, BaseEstimator]],
                     final_estimator_type: str,
                     random_state: int = random.randint(0, 2**32-1)):
    parameter_map = {
        "rf": rf_parameters,
        "ert": ert_parameters,
        "gbdt": gbdt_parameters,
        "svm": svm_parameters,
        "mlp": mlp_parameters,
    }

    fixed_final_estimator_params, search_final_estimator_params, _ = parameter_map[final_estimator_type](task_type, "random", random_state)

    fixed_model_params = {
        "estimators": estimators,
        "final_estimator": create_estimator(task_type, final_estimator_type, **fixed_final_estimator_params)
    }
    
    search_model_params = {
        "passthrough": [True, False],
    }
    
    search_model_params.update(**{"final_estimator__" + k: v for k, v in search_final_estimator_params.items()})
    
    searcher_params = {
        "cv": create_cv(task_type, 5, random_state),
        "scoring": create_scorer(task_type),
        "refit": True,
        "n_jobs": -1,
        "verbose": 3,
    }
    if search_type == "random":
        searcher_params["n_iter"] = 100
        searcher_params["random_state"] = random_state
    
    return fixed_model_params, search_model_params, searcher_params
