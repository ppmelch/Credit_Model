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

    ROOT --> DATA[data/]
    ROOT --> SRC[src/]
    ROOT --> NB[notebooks/]
    ROOT --> DOCS[docs/]
    ROOT --> MAIN[main.py]
    ROOT --> REQ[requirements.txt]
    ROOT --> README[README.md]

    DATA --> dataset[dataset.csv]
    DATA --> results[results.csv]

    SRC --> DATAR[data/]
    SRC --> MODELING[modeling/]
    SRC --> VIZ[visualization/]
    SRC --> UTILS[utils/]
    SRC --> MODELS[models/]

    DATAR --> dp[data_preparation.py]
    DATAR --> ds[data_splitter.py]

    MODELING --> bm[base_model.py]
    MODELING --> bl[business_logic.py]
    MODELING --> cm[classification_model.py]
    MODELING --> cfg[config.py]
    MODELING --> me[model_evaluation.py]
    MODELING --> mod[model.py]
    MODELING --> rc[risk_calculator.py]

    VIZ --> viz[viz.py]

    UTILS --> prints[prints.py]

    MODELS --> rf[random_forest.pkl]

    NB --> notebook[notebook.ipynb]

    DOCS --> pdf[Credit_Model.pdf]

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
flowchart LR

    %% ======================
    %% INPUT
    %% ======================
    A[Raw Data] --> B[Data Preparation]

    %% ======================
    %% DATA
    %% ======================
    B --> C[X, y]
    C --> D[Train/Test Split]

    D --> Xtr[X_train]
    D --> Xte[X_test]
    D --> Ytr[y_train]
    D --> Yte[y_test]

    %% ======================
    %% MODEL
    %% ======================
    Xtr --> E[Model Selection]
    E --> F[Train Model]

    %% ======================
    %% PREDICTIONS
    %% ======================
    F --> G[Predict y_pred (Test)]
    F --> H[Predict PD (Test)]
    F --> I[Predict PD (Train)]

    %% ======================
    %% EVALUATION
    %% ======================
    G --> J[Model Evaluation]
    H --> J
    I --> J
    Yte --> J
    Ytr --> J

    %% ======================
    %% RISK
    %% ======================
    F --> K[PD Full Dataset]
    K --> L[EAD]
    K --> M[LGD]

    K --> N[Expected Loss]
    L --> N
    M --> N

    %% ======================
    %% BUSINESS LOGIC
    %% ======================
    K --> O[Credit Decision]
    K --> P[Risk Buckets]

    %% ======================
    %% OUTPUT
    %% ======================
    N --> Q[Final Dataset]
    O --> Q
    P --> Q

    %% ======================
    %% SAVE
    %% ======================
    F --> R[Save Model]

    %% ======================
    %% RESULTS
    %% ======================
    J --> S[Results Metrics]
    Q --> T[Enriched Dataset]


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
