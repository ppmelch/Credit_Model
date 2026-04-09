import pandas as pd
from src.pipeline import CreditPipeline
from src.utils.prints import PrintUtils
from src.visualization.viz import Visualization

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
    printer = PrintUtils(data_final)
    printer.print_all(results)
    
    # 5. Visualizations results
    viz = Visualization()
    viz.plot_all(results)
        
    # 6. Save results
    data_final.to_csv("data/results.csv", index=False)

if __name__ == "__main__":
    main()