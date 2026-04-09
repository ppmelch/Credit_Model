import pandas as pd


class PrintUtils:
    """
    Utility class for printing results and analyses in a consistent format.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def print_model_results(self, results: dict) -> None:
        print("\n=== Model Evaluation ===")
        for key, value in results.items():
            print(f"{key}: {value}")

    def print_sample(self) -> None:
        print("\n=== Sample Results ===")
        print(self.data[['predicted_pd', 'expected_loss', 'decision']].head())

    def print_risk_analysis(self) -> None:
        print("\n=== Risk Buckets (PD & Expected Loss) ===")
        print(
            self.data.groupby('risk_bucket')[['predicted_pd', 'expected_loss']].mean()
        )

    def print_pricing_analysis(self) -> None:
        print("\n=== PD vs Interest Rate ===")
        print(
            self.data.groupby('risk_bucket')[['predicted_pd', 'interest_rate']].mean()
        )
        
    def print_all(self, results):
        self.print_model_results(results)
        self.print_sample()
        self.print_risk_analysis()
        self.print_pricing_analysis()