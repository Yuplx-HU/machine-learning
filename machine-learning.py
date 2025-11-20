import os
import joblib
from tqdm import tqdm

import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import (
    root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.metrics import make_scorer
from sklearn.base import clone, BaseEstimator


def create_estimator(task_type: str, model_type: str, params: dict) -> BaseEstimator:
    model_map = {
        "regression": {
            "RF": RandomForestRegressor,
            "ERT": ExtraTreesRegressor, 
            "GBDT": GradientBoostingRegressor,
            "SVM": SVR
        },
        "classification": {
            "RF": RandomForestClassifier,
            "ERT": ExtraTreesClassifier,
            "GBDT": GradientBoostingClassifier, 
            "SVM": SVC
        }
    }
    
    if task_type in model_map and model_type in model_map[task_type]:
        return model_map[task_type][model_type](**params)
    
    raise ValueError(f"Unsupported task type '{task_type}' or model type '{model_type}'\n"
                     f"Supported: regression/classification and RF/ERT/GBDT/SVM")


def create_searcher(
    estimator: BaseEstimator,
    params: dict,
    search_type: str,
    n_splits: int,
    shuffle: bool,
    scoring: str,
    task_type: str,
    n_iter: int,
    random_state: int,
    n_jobs: int
):
    scoring_map = {
        "regression": {
            "rmse": make_scorer(lambda yt, yp: -root_mean_squared_error(yt, yp)),
            "r2": "r2"
        },
        "classification": {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average='macro'),
            "recall": make_scorer(recall_score, average='macro'), 
            "f1": make_scorer(f1_score, average='macro'),
            "auc": make_scorer(roc_auc_score, average='macro', multi_class='ovr')
        }
    }
    
    if task_type not in scoring_map or scoring not in scoring_map[task_type]:
        supported = list(scoring_map[task_type].keys()) if task_type in scoring_map else []
        raise ValueError(f"Unsupported scoring '{scoring}' for task '{task_type}'. Use: {supported}")
    
    scorer = scoring_map[task_type][scoring]
    
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    param_key = "param_grid" if search_type == "grid" else "param_distributions"
    
    common_params = {
        "estimator": estimator,
        param_key: params,
        "cv": cv,
        "scoring": scorer,
        "n_jobs": n_jobs,
        "verbose": 0
    }
    
    if search_type == "grid":
        return GridSearchCV(**common_params)
    elif search_type == "random":
        return RandomizedSearchCV(**common_params, n_iter=n_iter, random_state=random_state)
    else:
        raise ValueError(f"Unsupported search type '{search_type}'. Use: grid/random")


def train_model(searcher, X_train, y_train, X_test, y_test, task_type: str):
    searcher_clone = clone(searcher)
    searcher_clone.fit(X_train, y_train)
    y_pred = searcher_clone.predict(X_test)
    
    metric_functions = {
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
    
    metrics = {}
    for name, func in metric_functions[task_type].items():
        metrics[name] = func(y_test, y_pred)
    
    if task_type == "classification" and len(set(y_test)) == 2:
        try:
            y_pred_proba = searcher_clone.predict_proba(X_test)[:, 1]
            metrics["auc"] = round(roc_auc_score(y_test, y_pred_proba), 4)
        except Exception:
            pass
    
    return {
        "best_estimator": searcher_clone.best_estimator_,
        "best_params": searcher_clone.best_params_,
        "metrics": metrics,
    }


def _get_best_model(models_result, task_type, scoring):
    if task_type == "regression":
        if scoring == "rmse":
            best_model_name = min(models_result.items(), key=lambda x: x[1]["metrics"]["rmse"])[0]
            sort_ascending, sort_key = True, "rmse"
        elif scoring == "r2":
            best_model_name = max(models_result.items(), key=lambda x: x[1]["metrics"]["r2"])[0]
            sort_ascending, sort_key = False, "r2"
        else:
            raise ValueError(f"Unsupported scoring type '{scoring}'")
    else:
        if scoring in ["accuracy", "precision", "recall", "f1", "auc"]:
            best_model_name = max(models_result.items(), key=lambda x: x[1]["metrics"].get(scoring, -1))[0]
            sort_ascending, sort_key = False, scoring
        else:
            raise ValueError(f"Unsupported scoring type '{scoring}'")
    
    return best_model_name, sort_ascending, sort_key


def _calculate_avg_metrics(best_models_metrics, save_path: str):
    if not best_models_metrics:
        return
    
    total_metrics = {}
    for model_metrics in best_models_metrics:
        for metric, value in model_metrics.items():
            total_metrics[metric] = total_metrics.get(metric, 0) + value
    
    avg_metrics = {metric: round(value / len(best_models_metrics), 4) for metric, value in total_metrics.items()}

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame([avg_metrics]).to_csv(save_path, index=False, encoding='utf-8')
    
    return avg_metrics


def machine_learning(tasks, config):
    task_type = config["type"]
    models_cfg = config["models"]
    verbose = config["verbose"]
    random_state = config["random_state"]
    scoring = config["scoring"]
    
    best_models_metrics = []

    for task in tqdm(tasks, disable=not verbose, desc="Process tasks", unit="task", leave=True):
        task_name = task["name"]
        models_result = {}
        all_models_params = []
        
        for model_name, model_cfg in tqdm(models_cfg.items(), disable=not verbose, desc=f"Train model [{task_name}]", unit="model", leave=False):
            estimator_cfg = model_cfg["estimator"]
            estimator = create_estimator(
                task_type=task_type,
                model_type=estimator_cfg["type"],
                params=estimator_cfg["params"]
            )
            
            searcher_cfg = model_cfg["searcher"]
            searcher = create_searcher(
                estimator=estimator,
                params=searcher_cfg["params"],
                search_type=searcher_cfg["type"],
                n_splits=searcher_cfg["n_splits"],
                shuffle=searcher_cfg["shuffle"],
                scoring=scoring,
                task_type=task_type,
                n_iter=searcher_cfg["n_iter"],
                random_state=random_state,
                n_jobs=searcher_cfg["n_jobs"]
            )
            
            result = train_model(searcher, task["X_train"], task["y_train"], task["X_test"], task["y_test"], task_type)
            
            models_result[model_name] = result
            all_models_params.append({
                "task_name": task_name,
                "model_name": model_name,
                "best_params": result["best_params"],
                "metrics": result["metrics"],
            })
        
        best_model_name, sort_ascending, sort_key = _get_best_model(models_result, task_type, scoring)
        best_model_result = models_result[best_model_name]
        best_models_metrics.append(best_model_result["metrics"])
        
        save_models_dir = config.get("save_models_dir", False)
        if save_models_dir:
            os.makedirs(save_models_dir, exist_ok=True)
            
            joblib.dump(best_model_result["best_estimator"], os.path.join(save_models_dir, f"{task_name}_{best_model_name}.pkl"))
        
            params_df = pd.DataFrame(all_models_params)
            params_df['_temp_sort'] = params_df['metrics'].apply(lambda x: x.get(sort_key, -1 if sort_ascending else 1))
            params_df = params_df.sort_values(by='_temp_sort', ascending=sort_ascending).drop(columns=['_temp_sort'])
            params_df.to_csv(os.path.join(save_models_dir, f"{task_name}_all_model_params.csv"), index=False, encoding='utf-8')
    
    save_avg_results_dir = config.get("save_avg_results_dir", False)
    if save_avg_results_dir:
        os.makedirs(save_avg_results_dir, exist_ok=True)
        _calculate_avg_metrics(best_models_metrics, save_avg_results_dir)
