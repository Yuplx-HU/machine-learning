# Machine Learning Pipeline for Regression & Classification

A flexible, easy-to-use machine learning pipeline that supports automated model training, hyperparameter tuning, and performance evaluation for both regression and classification tasks. It seamlessly integrates with the [feature-engineer](https://github.com/Yuplx-HU/feature-engineer) project to simplify end-to-end ML workflows, while also allowing manual task configuration for users who prefer custom data preprocessing.

## Key Features
- **Dual Task Support**: Handles both regression and classification tasks.
- **Multiple Models**: Integrates popular scikit-learn models (Random Forest, Extra Trees, GBDT, SVM).
- **Hyperparameter Tuning**: Supports Grid Search and Randomized Search with K-fold cross-validation.
- **Comprehensive Metrics**: Built-in evaluation metrics for both task types (e.g., RMSE/R² for regression; Accuracy/F1/AUC for classification).
- **Seamless Integration**: Works out-of-the-box with the `feature-engineer` project for automated data preprocessing and task generation.
- **Flexible Usage**: Allows manual task setup for users who don’t use `feature-engineer`.
- **Result Persistence**: Saves best models (as `.pkl`), model parameters/metrics (as `.csv`), and average performance across tasks.

## Prerequisites
Ensure the following packages are installed:
```bash
pip install pandas scikit-learn joblib tqdm xlrd openpyxl  # `xlrd` and `openpyxl` for Excel file support
```

If using the `feature-engineer` project (recommended for simplified workflows):
```bash
git clone https://github.com/Yuplx-HU/feature-engineer.git
# Add the feature-engineer directory to your Python path
```

## Installation
1. Clone this repository (or copy the `machine_learning.py` file to your project):
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```
2. Install dependencies (see [Prerequisites](#prerequisites)).

## Usage

### Option 1: Using with `feature-engineer` (Simplified Workflow)
The `feature-engineer` project automates data preprocessing (missing value imputation, encoding, standardization) and generates structured `tasks` for direct use with this pipeline.

#### Example Code
```python
import pandas as pd
from feature_engineer import feature_engineer  # From https://github.com/Yuplx-HU/feature-engineer
from machine_learning import machine_learning

# Step 1: Generate preprocessed tasks with feature-engineer
tasks = feature_engineer(
    pd.read_excel("data/raw_data.xlsx"),
    {
        "verbose": True,
        "feature_columns": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'],
        "target_columns": ['Q', 'R', 'S', 'T', 'U'],
        "fill_missing_data": {
            "mean": ['E', 'F', 'G', 'H', 'I', 'L', 'N', 'O', 'P'],
            "mode": ['A', 'B', 'C', 'D', 'J', 'K', 'M']
        },
        "standardize_data_format": {
            "standard": ['E', 'F', 'G', 'H', 'I', 'L', 'N', 'O', 'P']
        },
        "encode_features": {
            "onehot": ['A', 'B', 'C', 'D', 'J', 'K', 'M']
        },
        "generate_tasks": {
            "shuffle": True,
            "test_size": 0.2,
            "save_tasks_path": "./output/tasks/",
            "random_state": 1412
        }
    }
)

# Step 2: Run machine learning pipeline
machine_learning(
    tasks,
    {
        "verbose": True,
        "type": "regression",  # "regression" or "classification"
        "scoring": "rmse",     # Metric for hyperparameter tuning (see Supported Metrics)
        "random_state": 1412,
        "save_models_dir": "output/models/",  # Path to save best models and params
        "models": {
            "rf": {
                "estimator": {
                    "type": "RF",  # Model type (see Supported Models)
                    "params": {}   # Fixed estimator params (overridden by search params)
                },
                "searcher": {
                    "params": {  # Hyperparameters to search
                        "n_estimators": [100, 200, 300],
                        "max_depth": [3, 5, 7],
                        "max_features": ["sqrt", "log2"]
                    },
                    "n_splits": 10,  # K-fold cross-validation splits
                    "type": "random",  # "grid" or "random" search
                    "n_iter": 10,  # Number of random search iterations (ignored for grid)
                    "n_jobs": -1,  # Use all CPU cores
                    "shuffle": True
                }
            },
            # Add more models (ERT, GBDT, SVM) as needed
        }
    }
)
```

### Option 2: Manual Task Setup (Without `feature-engineer`)
If you prefer custom data preprocessing, you can manually create the `tasks` list following the required structure.

#### Task Structure Definition
The `tasks` parameter is a **list of dictionaries**, where each dictionary represents a single ML task (one target variable). Each task must contain the following keys:

| Key         | Type                  | Description                                                                 |
|-------------|-----------------------|-----------------------------------------------------------------------------|
| `name`      | `str`                 | Unique name for the task (e.g., "predict_Q", "classify_category").          |
| `X_train`   | `pd.DataFrame`/`np.array` | Training features (preprocessed).                                           |
| `y_train`   | `pd.Series`/`np.array`   | Training target values.                                                     |
| `X_test`    | `pd.DataFrame`/`np.array` | Test features (preprocessed, same structure as `X_train`).                  |
| `y_test`    | `pd.Series`/`np.array`   | Test target values.                                                         |

#### Example: Manual Task Creation
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from machine_learning import machine_learning

# Step 1: Load and preprocess data (custom logic)
raw_data = pd.read_excel("data/raw_data.xlsx")
# Add your custom preprocessing (imputation, encoding, scaling, etc.)
preprocessed_data = raw_data.dropna().reset_index(drop=True)

# Step 2: Create tasks manually (one task per target variable)
tasks = []
target_columns = ['Q', 'R']  # Example target variables
feature_columns = ['A', 'B', 'C', 'D']  # Example preprocessed features

for target in target_columns:
    X = preprocessed_data[feature_columns]
    y = preprocessed_data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=1412
    )
    
    # Append task to tasks list
    tasks.append({
        "name": f"task_{target}",
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    })

# Step 3: Run machine learning pipeline (same config as Option 1)
machine_learning(tasks, config={...})  # Use your config here
```

## Configuration Details
The `config` parameter for `machine_learning()` is a dictionary with the following keys:

| Key               | Type      | Description                                                                 |
|-------------------|-----------|-----------------------------------------------------------------------------|
| `type`            | `str`     | Task type: `"regression"` or `"classification"`.                            |
| `scoring`         | `str`     | Metric for hyperparameter tuning (see [Supported Metrics](#supported-models--metrics)). |
| `verbose`         | `bool`    | Whether to show progress bars (default: `False`).                           |
| `random_state`    | `int`     | Random seed for reproducibility.                                            |
| `save_models_dir` | `str`/`bool` | Path to save best models (`.pkl`) and model params/metrics (`.csv`). Set to `False` to disable. |
| `save_avg_results_dir` | `str`/`bool` | Path to save average metrics across all tasks (`.csv`). Set to `False` to disable. |
| `models`          | `dict`    | Dictionary of models to train. Each key is a model name (e.g., "rf"), with values as nested dicts for `estimator` and `searcher`. |

### Model Configuration (`models` Key)
Each model in `models` requires two sub-keys:
- `estimator`: 
  - `type`: Model type (see [Supported Models](#supported-models--metrics)).
  - `params`: Fixed parameters for the estimator (overridden by search parameters).
- `searcher`:
  - `type`: Search type: `"grid"` (Grid Search) or `"random"` (Randomized Search).
  - `params`: Dictionary of hyperparameters to search.
  - `n_splits`: Number of K-fold cross-validation splits.
  - `n_iter`: Number of iterations for Randomized Search (ignored for Grid Search).
  - `shuffle`: Whether to shuffle data for K-fold CV.
  - `n_jobs`: Number of CPU cores to use (-1 = all cores).

## Supported Models & Metrics

### Supported Models
| Task Type       | Model Code | Model Name                  |
|-----------------|------------|-----------------------------|
| Regression      | `RF`       | Random Forest Regressor     |
| Regression      | `ERT`      | Extra Trees Regressor       |
| Regression      | `GBDT`     | Gradient Boosting Regressor |
| Regression      | `SVM`      | Support Vector Regressor    |
| Classification  | `RF`       | Random Forest Classifier    |
| Classification  | `ERT`      | Extra Trees Classifier      |
| Classification  | `GBDT`     | Gradient Boosting Classifier|
| Classification  | `SVM`      | Support Vector Classifier   |

### Supported Metrics
| Task Type       | Metrics                                  |
|-----------------|------------------------------------------|
| Regression      | RMSE (Root Mean Squared Error), R²       |
| Classification  | Accuracy, Precision (macro), Recall (macro), F1 (macro), AUC (binary tasks only) |

## Output
The pipeline generates the following outputs (if `save_models_dir` is set):
- **Best Models**: Saved as `{task_name}_{best_model_name}.pkl` (e.g., `task_Q_rf.pkl`).
- **Model Params & Metrics**: Saved as `{task_name}_all_model_params.csv` (sorted by the tuning metric).
- **Average Metrics**: Saved as `avg_metrics.csv` in `save_avg_results_dir` (average of best model metrics across all tasks).

## Acknowledgments
- Integrated with [feature-engineer](https://github.com/Yuplx-HU/feature-engineer) for streamlined data preprocessing and task generation.
- Built on scikit-learn for core machine learning functionality.
