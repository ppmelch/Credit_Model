from src.modeling.model import Model
from src.modeling.config import MODELS_DIR
from src.data.data_splitter import DataSplitter
from src.data.data_preparation import DataPreparation
from src.modeling.business_logic import BusinessLogic
from src.modeling.risk_calculator import RiskCalculator
from src.modeling.model_evaluation import ModelEvaluation


class CreditPipeline:
    """
    End-to-end pipeline for credit risk modeling.

    This pipeline handles data preparation, model training, prediction,
    evaluation, risk metric computation (PD, LGD, EAD, Expected Loss),
    and business decision logic.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing borrower and loan information.

    model_name : str, optional
        Name of the model to be used (default is "logistic").
    """

    def __init__(self, data, model_name="logistic"):
        self.data = data
        self.model_name = model_name

    def run(self):
        """
        Execute the full credit risk pipeline.

        Steps:
        - Prepare and split data
        - Train model
        - Generate predictions and probabilities
        - Evaluate model performance
        - Compute risk metrics (PD, LGD, EAD, Expected Loss)
        - Apply business rules (decision and risk segmentation)
        - Save trained model

        Returns
        -------
        results : dict
            Dictionary containing evaluation metrics, predictions, probabilities,
            and AUC scores.

        data : pd.DataFrame
            Dataset enriched with predicted PD, expected loss, decision,
            and risk bucket segmentation.
        """

        # 1. Data preparation
        prep = DataPreparation(self.data)
        X, y = prep.prepare_data()

        # 2. Train-test split
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # 3. Model selection
        model = Model.get_model("classification", self.model_name)

        # 4. Training
        model.train(X_train, y_train)

        # 5. Predictions (Test)
        y_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        # 7. Predictions (Train)
        y_train_proba = model.predict_proba(X_train)
        y_train_pred = model.predict(X_train)

        # 8. Evaluation
        evaluator = ModelEvaluation()

        results = evaluator.evaluate_full(
            y_train=y_train,
            y_train_pred=y_train_pred,
            y_train_proba=y_train_proba,
            y_test=y_test,
            y_test_pred=y_pred,
            y_test_proba=y_test_proba
        )

        # 9. PD calculation (full dataset)
        risk = RiskCalculator(lgd=0.45)
        pd_values = risk.calculate_pd(model, X)
        self.data["predicted_pd"] = pd_values

        # 10. Risk metrics
        ead = risk.calculate_ead(self.data)
        lgd = risk.calculate_lgd(self.data)

        self.data["expected_loss"] = risk.calculate_expected_loss(
            pd_values, lgd, ead
        )

        # 11. Business logic
        logic = BusinessLogic(threshold=0.4, LGD=0.45 , rf=0.0364,
                 inflation_premium=0.024, liquidity_premium=0.0326, admin_cost=0.02, profit_margin=0.0116)
        self.data["decision"] = logic.credit_decision(pd_values)
        self.data["risk_bucket"] = logic.risk_buckets(pd_values)
        self.data["interest_rate_model"] = logic.calculate_interest_rate(pd_values)

        # 12. Save model
        model.save_model(f"{self.model_name}.pkl", MODELS_DIR)

        return results, self.data
