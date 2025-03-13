"""Pipeline for hyperparameter optimization with Optuna."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    optimize_model_hyperparameters,
    train_best_model,
    evaluate_best_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a pipeline for hyperparameter optimization with Optuna.

    Returns:
        A pipeline for hyperparameter optimization.
    """
    return pipeline(
        [
            node(
                func=optimize_model_hyperparameters,
                inputs=["model_input_table", "params:hyperparameter_optimization"],
                outputs="optimization_study",
                name="optimize_model_hyperparameters_node",
            ),
            node(
                func=train_best_model,
                inputs=["model_input_table", "optimization_study", "params:model_options"],
                outputs="optimized_regressor",
                name="train_best_model_node",
            ),
            node(
                func=evaluate_best_model,
                inputs=["optimized_regressor", "model_input_table", "params:model_options"],
                outputs=None,
                name="evaluate_best_model_node",
            ),
        ]
    )