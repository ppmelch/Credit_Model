import pandas as pd
from src.pipeline import CreditPipeline


def main():
    """
    Main execution script for the credit risk pipeline.
    """

    # 1. Load data
    df = pd.read_csv("data/dataset.csv")

    # 2. Initialize pipeline
    pipeline = CreditPipeline(data=df, model_name="logistic")

    # 3. Run pipeline
    results, df_final = pipeline.run()

    # 4. Print results
    print("\n=== Model Evaluation ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    # 5. Show key business outputs
    print("\n=== Sample Results ===")
    print(df_final[['predicted_pd', 'expected_loss', 'decision']].head())

    # 6. Save results
    df_final.to_csv("data/results.csv", index=False)


if __name__ == "__main__":
    main()