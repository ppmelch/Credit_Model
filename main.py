import pandas as pd
from src.pipeline import CreditPipeline


def main():
    """
    Main execution script for the credit risk pipeline.
    """

    # 1. Load data
    data = pd.read_csv("data/dataset.csv")

    # 2. Initialize pipeline
    pipeline = CreditPipeline(data=data, model_name="xgboost")

    # 3. Run pipeline
    results, data_final = pipeline.run()

    # 4. Print results
    print("\n=== Model Evaluation ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    # 5. Show key business outputs
    print("\n=== Sample Results ===")
    print(data_final[['predicted_pd', 'expected_loss', 'decision']].head())

    # 6. Save results
    data_final.to_csv("data/results.csv", index=False)
    
    print("\n=== Risk Buckets (PD & Expected Loss) ===")
    print(
        data_final.groupby('risk_bucket')[['predicted_pd','expected_loss']].mean()
    )

    print("\n=== PD vs Interest Rate ===")
    print(
        data_final.groupby('risk_bucket')[['predicted_pd','interest_rate']].mean()
    )


if __name__ == "__main__":
    main()