import pandas as pd


class RiskCalculator:
    """
    Class for computing credit risk metrics such as
    Probability of Default (PD), Loss Given Default (LGD),
    Exposure at Default (EAD), and Expected Loss (EL).
    """

    def __init__(self, lgd: float = 0.45):
        """
        Initialize the risk calculator.

        Parameters
        ----------
        lgd : float
            Assumed Loss Given Default (default = 0.45).
        """
        self.lgd = lgd

    def calculate_pd(self, model, X: pd.DataFrame) -> pd.Series:
        """
        Calculate Probability of Default using the trained model.
        """
        return model.predict_proba(X)

    def calculate_ead(self, data: pd.DataFrame) -> pd.Series:
        """
        Exposure at Default (EAD).
        Uses loan amount as proxy.
        """
        return data["loan_amount"]

    def calculate_lgd(self, data: pd.DataFrame) -> pd.Series:
        """
        Loss Given Default (LGD).
        Uses a constant assumption.
        """
        return pd.Series(self.lgd, index=data.index)

    def calculate_expected_loss(self, pd: pd.Series, lgd: pd.Series, ead: pd.Series) -> pd.Series:
        """
        Expected Loss (EL).

        EL = PD * LGD * EAD
        """
        return pd * lgd * ead