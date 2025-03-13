"""Nodes for hyperparameter optimization with Optuna."""

import logging
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


def optimize_model_hyperparameters(
        data: pd.DataFrame, parameters: Dict[str, Any]
) -> optuna.Study:
    """Optimize model hyperparameters using Optuna.

    Args:
        data: The model input table.
        parameters: Parameters for hyperparameter optimization.

    Returns:
        An Optuna study with trials.
    """
    logger = logging.getLogger(__name__)
    features = parameters["features"]
    target = parameters["target"]
    n_trials = parameters["n_trials"]

    X = data[features]
    y = data[target]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=parameters["validation_size"], random_state=parameters["random_state"]
    )

    def objective(trial):
        # Select model type
        model_type = trial.suggest_categorical("model_type", ["ridge", "lasso", "elasticnet", "random_forest"])

        if model_type == "ridge":
            alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
            model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
            model = Lasso(alpha=alpha)
        elif model_type == "elasticnet":
            alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        else:  # random_forest
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=parameters["random_state"]
            )

        # Use cross-validation to evaluate the model
        score = cross_val_score(
            model, X_train, y_train,
            cv=parameters.get("cv", 5),
            scoring="neg_mean_squared_error"
        ).mean()

        return -score  # We minimize the negative MSE (maximize MSE)

    # Create and run the study
    study = optuna.create_study(
        study_name=parameters.get("study_name", "price_optimization"),
        direction="minimize"
    )

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"  Value: {best_trial.value:.4f}")
    logger.info("  Params:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    return study


def train_best_model(
        data: pd.DataFrame,
        study: optuna.Study,
        parameters: Dict[str, Any]
) -> Any:
    """Train the best model from the Optuna study.

    Args:
        data: The model input table.
        study: The Optuna study with trials.
        parameters: Model parameters.

    Returns:
        The trained model.
    """
    logger = logging.getLogger(__name__)

    features = parameters["features"]
    target = "price"  # Hardcoded from the spaceflights example

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    # Get the best parameters
    best_params = study.best_params
    best_model_type = best_params.pop("model_type")

    # Create and train the model with best parameters
    if best_model_type == "ridge":
        model = Ridge(**best_params)
    elif best_model_type == "lasso":
        model = Lasso(**best_params)
    elif best_model_type == "elasticnet":
        model = ElasticNet(**best_params)
    else:  # random_forest
        model = RandomForestRegressor(
            **best_params, random_state=parameters["random_state"]
        )

    model.fit(X_train, y_train)
    logger.info(f"Trained best model of type: {best_model_type}")

    return model


def evaluate_best_model(
        model: Any,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
) -> None:
    """Evaluate the best model.

    Args:
        model: The trained model.
        data: The model input table.
        parameters: Model parameters.
    """
    logger = logging.getLogger(__name__)

    features = parameters["features"]
    target = "price"  # Hardcoded from the spaceflights example

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Evaluate on test data
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    logger.info("Model evaluation:")
    logger.info(f"  Training RMSE: {train_rmse:.4f}")
    logger.info(f"  Training R^2: {train_r2:.4f}")
    logger.info(f"  Test RMSE: {test_rmse:.4f}")
    logger.info(f"  Test R^2: {test_r2:.4f}")
