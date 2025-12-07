simple usage:
```Python
for task in tasks:
    rf_fixed_model_params, rf_model_search_params, rf_searcher_params = rf_parameters("classification", "random", 1412)
    base_rf_estimator = create_estimator("classification", "rf", **rf_fixed_model_params)
    rf_searcher = create_searcher("random", base_rf_estimator, rf_model_search_params, rf_searcher_params)
    trained_rf_estimator = train_model("classification", rf_searcher, task["X_train"], task["y_train"], task["X_test"], task["y_test"])["best_estimator"]
    
    ert_fixed_model_params, ert_model_search_params, ert_searcher_params = ert_parameters("classification", "random", 1412)
    base_ert_estimator = create_estimator("classification", "ert", **ert_fixed_model_params)
    ert_searcher = create_searcher("random", base_ert_estimator, ert_model_search_params, ert_searcher_params)
    trained_ert_estimator = train_model("classification", ert_searcher, task["X_train"], task["y_train"], task["X_test"], task["y_test"])["best_estimator"]
    
    gbdt_fixed_model_params, gbdt_model_search_params, gbdt_searcher_params = gbdt_parameters("classification", "random", 1412)
    base_gbdt_estimator = create_estimator("classification", "gbdt", **gbdt_fixed_model_params)
    gbdt_searcher = create_searcher("random", base_gbdt_estimator, gbdt_model_search_params, gbdt_searcher_params)
    trained_gbdt_estimator = train_model("classification", gbdt_searcher, task["X_train"], task["y_train"], task["X_test"], task["y_test"])["best_estimator"]
    
    svm_fixed_model_params, svm_model_search_params, svm_searcher_params = svm_parameters("classification", "random", 1412)
    base_svm_estimator = create_estimator("classification", "svm", **svm_fixed_model_params)
    svm_searcher = create_searcher("random", base_svm_estimator, svm_model_search_params, svm_searcher_params)
    trained_svm_estimator = train_model("classification", svm_searcher, task["X_train"], task["y_train"], task["X_test"], task["y_test"])["best_estimator"]
    estimators = [
        ("rf", trained_rf_estimator),
        ("ert", trained_ert_estimator),
        ("gbdt", trained_gbdt_estimator),
        ("svm", trained_svm_estimator),
    ]
    stack_fixed_model_params, stack_model_search_params, stack_searcher_params = stack_parameters("classification", "random", 1412)
    stack_estimator = create_estimator("classification", "stack", **stack_fixed_model_params, estimators=estimators, final_estimator=base_gbdt_estimator)
    stack_model_search_params.update({"final_estimator__" + k: v for k, v in gbdt_model_search_params.items()})
    stack_searcher = create_searcher("random", stack_estimator, stack_model_search_params, stack_searcher_params)
    print(train_model("classification", stack_searcher, task["X_train"], task["y_train"], task["X_test"], task["y_test"]))
```
