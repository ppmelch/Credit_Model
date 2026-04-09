from src.modeling.model import Model
from src.modeling.config import MODELS_DIR
from src.data.data_splitter import DataSplitter
from sklearn.preprocessing import StandardScaler
from src.data.data_preparation import DataPreparation
from src.modeling.business_logic import BusinessLogic
from src.modeling.risk_calculator import RiskCalculator
from src.modeling.model_evaluation import ModelEvaluation


class CreditPipeline:

    def __init__(self, data, model_name="logistic"):
        self.data = data
        self.model_name = model_name

    def run(self):

        # 1. Data preparation
        prep = DataPreparation(self.data)
        X, y = prep.prepare_data()

        # 2. Split
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        # 2.5 Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
        results["y_test"] = y_test
        results["y_pred"] = y_pred
        results["y_prob"] = y_pred_proba
        
        
        # 7. PD for full dataset
        X_scaled = scaler.transform(X)
        pd_values = model.predict_proba(X_scaled)
        self.data["predicted_pd"] = pd_values

        # 8. Risk metrics
        risk = RiskCalculator(lgd=0.45)
        ead = risk.calculate_ead(self.data)
        lgd = risk.calculate_lgd(self.data)

        self.data['expected_loss'] = risk.calculate_expected_loss(
            pd_values, lgd, ead)

        # 9. Business logic
        logic = BusinessLogic(threshold=0.4)
        self.data['decision'] = logic.credit_decision(pd_values)
        self.data['risk_bucket'] = logic.risk_buckets(pd_values)

        # 10. Save model
        model.save_model(f"{self.model_name}.pkl", MODELS_DIR)

        return results, self.data