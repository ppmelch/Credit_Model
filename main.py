import pandas as pd
from src.pipeline import CreditPipeline
from src.utils.prints import PrintUtils
from src.visualization.viz import Visualization


def main():
    """
    Execute the end-to-end credit risk modeling pipeline.

    Workflow
    --------
    1. Load dataset from local storage.
    2. Initialize the credit pipeline with the selected model.
    3. Run the pipeline (data preparation, training, prediction, evaluation).
    4. Display key results.
    5. Generate visualizations for model performance and risk metrics.
    6. (Optional) Save processed results to disk.

    Notes
    -----
    - The pipeline encapsulates the full modeling process, including:
        * Data preprocessing
        * Model training (e.g., XGBoost)
        * Probability of Default (PD) estimation
        * Risk metric computation
    - Ensure that the dataset exists at the specified path before execution.
    """

    # == Load data ==
    data = pd.read_csv("data/dataset.csv")

    # == Initialize pipeline ==
    pipeline = CreditPipeline(data=data, model_name="xgboost")

    # == Run pipeline ==
    results, data_final = pipeline.run()

    # == Print results (optional) ==
    printer = PrintUtils(data_final)
    # printer.print_all(results) 

    # == Visualizations ==
    viz = Visualization()
    viz.plot_all(results, data_final)

    # == Save results (optional) ==
    # data_final.to_csv("data/results.csv", index=False)

if __name__ == "__main__":
    main()