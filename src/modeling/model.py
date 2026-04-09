from classification_model import ClassificationModel


class Model:
    """
    Factory class responsible for creating machine learning model instances.

    This class abstracts the model instantiation process and allows
    dynamic selection of models based on the task type and model name.

    Currently supports:
    - Classification models

    Future extensions may include regression models.
    """

    @staticmethod
    def get_model(task_type: str, model_name: str):
        """
        Create and return a model instance based on task type and model name.

        Parameters
        ----------
        task_type : str
            Type of machine learning task. Supported values:
            - 'classification'

        model_name : str
            Name of the model to instantiate. Supported values:
            - 'logistic'
            - 'random_forest'
            - 'xgboost'

        Returns
        -------
        object
            An instance of the requested model.

        Raises
        ------
        ValueError
            If the task type or model name is not supported.
        """

        if task_type == "classification":
            allowed_models = ["logistic", "random_forest", "xgboost"]

            if model_name not in allowed_models:
                raise ValueError(f"Model '{model_name}' not supported")

            return ClassificationModel(model_name)

        # IF WE WANT TO ADD REGRESSION MODELS IN THE FUTURE, WE CAN DO IT HERE
            '''
        elif task_type == "regression":
            return RegressionModel(config)
            '''

        else:
            raise ValueError("Invalid model type")
