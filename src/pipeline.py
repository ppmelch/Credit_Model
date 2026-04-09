from src.data.data_preparation import Data_preparation
from src.data.data_splitter import DataSplitter
from src.modeling.model import Model
from src.modeling.model_evaluation import ModelEvaluation
from src.modeling.config import MODELS_DIR


class CreditPipeline:
    """
    End-to-end pipeline for credit risk modeling.

    This pipeline handles data preparation, model training,
    evaluation, prediction, and model persistence.
    """

    def __init__(self, data, model_name="logistic"):
        self.data = data
        self.model_name = model_name

    def run(self):
        """
        Execute the full pipeline.

        Returns
        -------
        dict
            Model evaluation results.
        """

        # 1. Data preparation
        prep = Data_preparation(self.data)
        X, y = prep.prepare_data()

        # 2. Split
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # 3. Model
        model = Model.get_model("classification", self.model_name)

        # 4. Train
        model.train(X_train, y_train)

        # 5. Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # 6. Evaluate
        evaluator = ModelEvaluation()
        results = evaluator.evaluate(y_test, y_pred, y_pred_proba)

        # 7. PD for full dataset
        self.data["predicted_pd"] = model.predict_proba(X)

        # 8. Save model
        model.save_model(f"{self.model_name}.pkl", MODELS_DIR)

        return results, self.data