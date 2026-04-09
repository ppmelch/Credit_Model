import pandas as pd
from src.pipeline import CreditPipeline
from src.utils.prints import PrintUtils
from src.visualization.viz import Visualization

def main():
    """
    Main execution script for the credit risk pipeline.
    """

    # 1. Load data
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "data" / "dataset.csv"

    data = pd.read_csv(DATA_PATH)
    print(DATA_PATH)
    print(DATA_PATH.exists())
    # 2. Initialize pipeline
    pipeline = CreditPipeline(data=data, model_name="xgboost")

    # 3. Run pipeline
    results, data_final = pipeline.run()

    # 4. Print results
    printer = PrintUtils(data_final)
    #printer.print_all(results)
    
    # 5. Visualizations results
    viz = Visualization()
    viz.plot_all(results, data_final)
        
    # 6. Save results
    #data_final.to_csv("data/results.csv", index=False)

if __name__ == "__main__":
    main()