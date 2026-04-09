import pandas as pd


class BusinessLogic:
    """
    Class for business rules such as credit decision,
    risk segmentation, and pricing logic.
    """

    def __init__(self, threshold=0.4):
        self.threshold = threshold

    def credit_decision(self, pd_values: pd.Series) -> pd.Series:
        """
        Approve (1) or Reject (0) based on PD threshold.
        """
        return (pd_values < self.threshold).astype(int)

    def risk_buckets(self, pd_values: pd.Series, q=5) -> pd.Series:
        """
        Segment clients into risk buckets.
        """
        return pd.qcut(pd_values, q=q)
