Simple usage:
```Python
rf_fixed_model_params, rf_model_search_params, rf_searcher_params = rf_parameters(task_type, "random", 1412)
rf_estimator = create_estimator(task_type, "rf", **rf_fixed_model_params)
rf_searcher = create_searcher("random", rf_estimator, rf_model_search_params, rf_searcher_params)

ert_fixed_model_params, ert_model_search_params, ert_searcher_params = ert_parameters(task_type, "random", 1412)
ert_estimator = create_estimator(task_type, "ert", **ert_fixed_model_params)
ert_searcher = create_searcher("random", ert_estimator, ert_model_search_params, ert_searcher_params)

gbdt_fixed_model_params, gbdt_model_search_params, gbdt_searcher_params = gbdt_parameters(task_type, "random", 1412)
gbdt_estimator = create_estimator(task_type, "gbdt", **gbdt_fixed_model_params)
gbdt_searcher = create_searcher("random", gbdt_estimator, gbdt_model_search_params, gbdt_searcher_params)

svm_fixed_model_params, svm_model_search_params, svm_searcher_params = svm_parameters(task_type, "random", 1412)
svm_estimator = create_estimator(task_type, "svm", **svm_fixed_model_params)
svm_searcher = create_searcher("random", svm_estimator, svm_model_search_params, svm_searcher_params)

mlp_fixed_model_params, mlp_model_search_params, mlp_searcher_params = mlp_parameters(task_type, "random", 1412)
mlp_estimator = create_estimator(task_type, "mlp", **mlp_fixed_model_params)
mlp_searcher = create_searcher("random", mlp_estimator, mlp_model_search_params, mlp_searcher_params)

searchers = [
    ("rf", rf_searcher),
    ("ert", ert_searcher),
    ("gbdt", gbdt_searcher),
    ("svm", svm_searcher),
    ("mlp", mlp_searcher),
]

trained_model_results = train_models(
    task_type, searchers,
    X_train, X_test, y_train, y_test,
    save_results=True, save_results_dir=model_output_dir, save_results_name="traditional_model_results.csv",
    save_models=True, save_models_dir=model_output_dir, save_best_model_only=False,
)
```
