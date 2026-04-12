# Credit Model
### Credit Models project

**Author:** 
  - José Armando Melchor Soto
  - Rolando Fortanell Canedo
  - David Campos Ambriz 


**Course:** Credit Models  
**Institution:** ITESO  

---

## Table of Contents



---
## Overview

---

## Architecture

### Project Structure
```mermaid
flowchart TD

    ROOT[Credit_Model/]

    ROOT --> DATA
    ROOT --> SRC
    ROOT --> NB
    ROOT --> DOCS
    ROOT --> MAIN[main.py]
    ROOT --> REQ[requirements.txt]
    ROOT --> README[README.md]

    subgraph DATA[data/]
        dataset[dataset.csv]
        results[results.csv]
    end

    subgraph SRC[src/]

        subgraph DATAR[data/]
            dp[data_preparation.py]
            ds[data_splitter.py]
        end

        subgraph MODELING[modeling/]
            bm[base_model.py]
            bl[business_logic.py]
            cm[classification_model.py]
            cfg[config.py]
            me[model_evaluation.py]
            mod[model.py]
            rc[risk_calculator.py]
        end

        subgraph VIZ[visualization/]
            viz[viz.py]
        end

        subgraph UTILS[utils/]
            prints[prints.py]
        end

        subgraph MODELS[models/]
            rf[random_forest.pkl]
        end

    end

    subgraph NB[notebooks/]
        nb1[notebook.ipynb]
        nb2[feature_analysis.ipynb]
    end

    subgraph DOCS[docs/]
        pdf[Credit_Model.pdf]
    end
```

### Functional Architecture

### OOP Architecture
```mermaid
classDiagram

%% ======================
%% ENTRY POINT
%% ======================
class main {
    +main()
}

main --> CreditPipeline

%% ======================
%% PIPELINE (ORCHESTRATOR)
%% ======================
class CreditPipeline {
    -data
    -model_name
    +run()
}

CreditPipeline --> DataPreparation
CreditPipeline --> DataSplitter
CreditPipeline --> Model
CreditPipeline --> ModelEvaluation
CreditPipeline --> RiskCalculator
CreditPipeline --> BusinessLogic

%% ======================
%% DATA LAYER
%% ======================
class DataPreparation {
    +prepare_data()
}

class DataSplitter {
    +split(X, y)
}

%% ======================
%% MODEL LAYER
%% ======================
class BaseModel {
    <<abstract>>
    +train()
    +predict()
    +predict_proba()
}

class ClassificationModel {
    +train()
    +predict()
    +predict_proba()
    +save_model()
    +load_model()
}

BaseModel <|-- ClassificationModel

class Model {
    +get_model()
}

Model --> ClassificationModel

%% ======================
%% EVALUATION
%% ======================
class ModelEvaluation {
    +evaluate()
    +evaluate_full()
}

%% ======================
%% RISK LAYER
%% ======================
class RiskCalculator {
    +calculate_pd()
    +calculate_ead()
    +calculate_lgd()
    +calculate_expected_loss()
}

RiskCalculator --> ClassificationModel

%% ======================
%% BUSINESS LOGIC
%% ======================
class BusinessLogic {
    +credit_decision()
    +risk_buckets()
}

BusinessLogic --> RiskCalculator

%% ======================
%% OUTPUT LAYER
%% ======================
class PrintUtils {
    +print_all()
}

class Visualization {
    +plot_all()
}

main --> PrintUtils
main --> Visualization


```

### Loan Lifecycle

```mermaid
graph LR
    A([Applicant]) --> B[Personal Loan Application]

    subgraph AP [Analysis Phase]
        B --> C{Credit Assessment\nRisk Based Pricing}
        C --> D[Credit History\nCredit Bureau]
        C --> E[Capacity to Pay\nIncome / DTI]
        C --> F[Credit Score\nFICO / VantageScore]
        D --> G{Credit Decision}
        E --> G
        F --> G
    end

    subgraph LL [Loan Lifecycle]
        G -->|Approved| H[Capital Disbursement\nTotal Amount]
        H --> I[Fixed Term Contract\n12 to 84 months]
        I --> J[Fixed Periodic Payments\nPrincipal + Interest + Fees]
        J --> K{Payment Compliance?}
        K -->|Yes| L[Positive Report\nCredit Bureau]
        L --> M([Loan Closure])
        K -->|No| N[Debt Collection Management\nPenalties / Negative Reports]
        N --> O{Recovery?}
        O -->|Yes| J
        O -->|No| P([Loss / Write-off\nLoss Given Default])
    end

    G -->|Rejected| Q([Application Denied])
```


### Flow Diagram


```mermaid
flowchart LR

    A[Raw Dataset] --> B[Data Preparation]

    B --> C[Feature / Target Split]

    C --> D[Train-Test Split]

    D --> E[Training Dataset]
    D --> F[Testing Dataset]

    E --> G[Model Selection]

    G --> H[Model Training]

    H --> I[Prediction Engine]

    I --> J[Classification Prediction]
    I --> K[PD Probability Estimation]

    J --> L[Model Evaluation]
    K --> L
    F --> L

    K --> M[Portfolio PD Estimation]

    M --> N[EAD Calculation]
    M --> O[LGD Calculation]

    M --> P[Expected Loss]
    N --> P
    O --> P

    M --> Q[Credit Decision]
    M --> R[Risk Bucket Assignment]
    M --> S[Interest Rate Pricing]

    P --> T[Final Credit Dataset]
    Q --> T
    R --> T
    S --> T

    H --> U[Model Persistence]

    L --> V[Performance Metrics]

    T --> W[Risk-Enriched Portfolio]

```



---


## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ppmelch/Credit_Model.git
cd Credit_Model

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

---

## Results

---

## Discussion

---

## Assumptions

---

## Limitations

---


## Conclusions

---


## Output

---


## Documentation

The full project report is available at:

- [Credit Model Report](docs/Credit_Model.pdf)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
